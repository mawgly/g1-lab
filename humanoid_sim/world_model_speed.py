"""The world model fit's speed (scripts/lab/wm/fit.py --speed, --batch auto). Eager, one Adam step is ~3 k small CUDA
launches against ~15 ms of GEMMs: an RTX 4090 ran 5 steps/s at 16 % GPU with one CPU core at 100 % (27.10). On CUDA the
whole step (the window gather, the 8-step rollout, the loss, backward, gradient clipping, Adam) is one CUDA graph
replayed per step (GraphStep), its windows' indices drawn exactly as the eager loop draws them (IndexFeed: the same
generator calls in the same order), so the stream of windows, checkpoints and resume are the eager run's; the numbers
are meant to differ from it only by float noise (capturable Adam). torch.compile of the forward is opt-in on Linux (no
MSVC/Triton on Windows). On the CPU the eager loop is kept as it was: it is the bit-for-bit reference.
Not yet run on a GPU (27.10): the graph and compile paths, choose_batch, their steps/s, GPU use (the > 60 % target) and
the graph-vs-eager deviation are unmeasured; tests/test_world_model_gpu.py checks them (20 steps within rel 1e-5)."""
import sys

import torch

MODES = ('auto', 'plain', 'graph', 'compile')
WARMUP_STEPS = 3
BATCH_CANDIDATES = (8192, 4096, 2048)
MEMORY_MARGIN, MEMORY_SHARE = 1.3, .7


def speed_mode(device, requested='auto'):
    """(mode, Russian reason) of a requested --speed on `device`: CPU always plain; CUDA auto → graph; compile only on
    Linux CUDA, elsewhere graph."""
    if requested not in MODES:
        raise ValueError(f'неизвестный режим скорости {requested}')
    cuda = str(device).startswith('cuda')
    if not cuda:
        reason = 'процессор: обычный цикл' if requested in ('auto', 'plain') else (
            f'{requested} только на CUDA, считаю обычным циклом')
        return 'plain', reason
    if requested == 'plain':
        return 'plain', 'обычный цикл по запросу'
    if requested == 'compile' and not sys.platform.startswith('linux'):
        return 'graph', 'torch.compile без MSVC/Triton недоступен, считаю графом'
    if requested == 'compile':
        return 'compile', 'torch.compile и граф CUDA'
    return 'graph', 'граф CUDA'


class IndexFeed:
    """The training windows' (starts, worlds) of each step, drawn on the CPU generator with exactly Data.sample's calls
    (randint over the training worlds, then over the starts) and copied without blocking from one of two pinned
    buffers into static device tensors the graph reads; a buffer is reused only after its copy is done (its event)."""

    def __init__(self, data, batch, k, generator, device):
        self.data, self.batch, self.k, self.generator = data, batch, k, generator
        self.train_worlds = data.train_worlds.cpu()
        cuda = str(device).startswith('cuda')
        self.host = [torch.empty((2, batch), dtype=torch.long, pin_memory=cuda) for _ in range(2)]
        self.events = [torch.cuda.Event() if cuda else None for _ in range(2)]
        self.starts = torch.zeros(batch, dtype=torch.long, device=device)
        self.worlds = torch.zeros(batch, dtype=torch.long, device=device)
        self.turn = 0

    def next(self):
        buffer, event = self.host[self.turn], self.events[self.turn]
        if event is not None:
            event.synchronize()
        worlds = self.train_worlds[torch.randint(len(self.train_worlds), (self.batch,), generator=self.generator,
                                                 device='cpu')]
        starts = torch.randint(self.data.total - self.k + 1, (self.batch,), generator=self.generator, device='cpu')
        buffer[0].copy_(worlds)
        buffer[1].copy_(starts)
        self.worlds.copy_(buffer[0], non_blocking=True)
        self.starts.copy_(buffer[1], non_blocking=True)
        if event is not None:
            event.record()
        self.turn ^= 1
        return self.starts, self.worlds


class GraphStep:
    """One Adam step of the window loss on the feed's windows as a CUDA graph: WARMUP_STEPS eager steps on a side
    stream (real, counted training steps), then a capture of the whole step (gather, from_raw, goal, window_loss,
    backward, clip_grad_norm_, Adam) and a replay per step. loss(feed) → (loss, parts) is fit's eager step body. With
    compile the forward is torch.compile'd first (fullgraph, static shapes). Losses are read (synchronising) only when
    asked (log steps, checkpoints). A capture error falls back to eager steps on the same feed (`fallback`, the reason
    kept)."""

    def __init__(self, model, loss, optimizer, feed, clip, compile=False):
        self.model, self.loss, self.optimizer, self.feed, self.clip = model, loss, optimizer, feed, clip
        self.forward = torch.compile(loss, fullgraph=True, dynamic=False, mode='default') if compile else loss
        self.compiled, self.compile_error = compile, None
        self.graph, self.fallback, self.done = None, None, 0
        self.parts = torch.zeros(2, device=feed.starts.device)
        self.side = torch.cuda.Stream()
        self.replay_ms, self._events = 0., []

    def _eager(self, forward):
        self.optimizer.zero_grad(set_to_none=True)
        loss, parts = forward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        self.parts.copy_(torch.stack((parts['mse'], parts['bce'])))

    def _capture(self):
        self.optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss, parts = self.forward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
            self.optimizer.step()
            self.parts.copy_(torch.stack((parts['mse'], parts['bce'])))
        self.graph = graph

    def step(self):
        """One training step on the feed's next windows."""
        self.feed.next()
        if self.fallback is not None:
            self._eager(self.loss)
        elif self.done < WARMUP_STEPS:
            current = torch.cuda.current_stream()
            self.side.wait_stream(current)
            with torch.cuda.stream(self.side):
                try:
                    self._eager(self.forward)
                except Exception as error:  # compile failed: the plain forward, still captured
                    if not self.compiled:
                        raise
                    self.forward, self.compiled = self.loss, False
                    self.compile_error = f'{type(error).__name__}: {error}'
                    self._eager(self.forward)
            current.wait_stream(self.side)
        else:
            if self.graph is None:
                try:
                    self._capture()
                except Exception as error:
                    self.graph = None
                    self.fallback = f'{type(error).__name__}: {error}'
                    torch.cuda.synchronize()
                    self._eager(self.loss)
                    self.done += 1
                    return
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.graph.replay()
            end.record()
            self._events.append((start, end))
        self.done += 1

    def losses(self):
        """{'mse', 'bce'} of the last step (synchronises), and the replays' GPU time so far folded in."""
        mse, bce = self.parts.tolist()
        self.timing()
        return dict(mse=mse, bce=bce)

    def timing(self):
        if self._events:
            self._events[-1][1].synchronize()
            self.replay_ms += sum(start.elapsed_time(end) for start, end in self._events)
            self._events = []
        return self.replay_ms


def choose_batch(model, measure, device, candidates=BATCH_CANDIDATES):
    """(batch, Russian note) for --batch auto on CUDA: the peak memory of one eager forward and backward at 256 windows
    (measure(256), fit's step body on windows of a generator of its own, so the training's stream is untouched), scaled
    ×B/256 ×MEMORY_MARGIN, the largest candidate under MEMORY_SHARE of the free memory with the data loaded; the
    smallest when none fits."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    loss, _ = measure(256)
    loss.backward()
    torch.cuda.synchronize(device)
    per_window = (torch.cuda.max_memory_allocated(device) - base) / 256
    model.zero_grad(set_to_none=True)
    del loss
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    for batch in sorted(candidates, reverse=True):
        need = per_window * batch * MEMORY_MARGIN
        if need < MEMORY_SHARE * free:
            return batch, f'{batch} окон: нужно ~{need / 2 ** 30:.1f} ГБ из {free / 2 ** 30:.1f} ГБ свободных'
    smallest = min(candidates)
    return smallest, f'{smallest} окон: даже он не помещается с запасом ({free / 2 ** 30:.1f} ГБ свободно)'
