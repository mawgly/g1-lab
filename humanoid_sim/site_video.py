"""The site's animated WebP clips as H.264 MP4, for the public copy (humanoid_sim.site_export).

iPhone Safari does not play our clips as animated WebP in <img> (29.10): every frame carries an ALPH chunk (WebKit bug
217897, animated WebP with transparency), it slows down (278507) and 200-1500 frames of up to 1280x810 run into iOS's
image memory. A muted inline <video> plays there and can start by itself, so the export writes each animated WebP as
an MP4 (web/clip.js plays it) and the live lab server keeps serving the WebP.

    .venv\\Scripts\\python.exe -m humanoid_sim.site_video <clip.webp> [--out <clip.mp4>]

Frames are read with Pillow (ffmpeg has no animated WebP decoder), composited on the site's clip background and piped
to the ffmpeg of imageio-ffmpeg: libx264 main, yuv420p, CRF 26, preset medium, faststart (moov before mdat, so a
phone starts playing before the whole file is in), no sound, odd sizes padded to even ones. The frame durations give
the frame rate; uneven ones are kept by repeating frames on the shortest duration's grid. Results are cached by the
source's md5 (and the encoder settings) in .tools/site/mp4-cache, so a publish only encodes the clips that changed.
"""
import argparse
import hashlib
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from PIL import Image

from .core import ROOT

CACHE = ROOT / '.tools' / 'site' / 'mp4-cache'
# Parallel encodes: one ffmpeg process each, at most 6 (the training on the GPU keeps its CPU), each with 2 threads and
# below normal priority on Windows.
WORKERS = 6
THREADS = 2
CRF = 26
PRESET = 'medium'
BACKGROUND = (27, 32, 39)  # #1b2027, the clips' background in site.css (.media)
SHORTEST_FRAME_MS = 10  # a shorter or missing duration is read as this
DEFAULT_FRAME_MS = 40  # a frame without a duration (Pillow's first frame) takes the clip's typical one, else this
ARGUMENTS = ['-c:v', 'libx264', '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-preset', PRESET, '-crf', str(CRF),
             '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=0x1b2027', '-movflags', '+faststart', '-an']
# Part of the cache key, so other encoder settings encode again.
SETTINGS = hashlib.md5(repr((ARGUMENTS, BACKGROUND)).encode()).hexdigest()[:8]


def ffmpeg():
    """The ffmpeg executable bundled with imageio-ffmpeg."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def animated(path):
    """True for an animated WebP (more than one frame); a still WebP stays a picture."""
    try:
        with Image.open(path) as image:
            return image.format == 'WEBP' and getattr(image, 'n_frames', 1) > 1
    except OSError:
        return False


def md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_durations(path):
    """Each frame's duration in ms as the file's ANMF chunks give it (read from the RIFF chunks, which is faster
    than seeking through every frame); empty for a file without them."""
    found = []
    with open(path, 'rb') as file:
        if file.read(12)[8:12] != b'WEBP':
            return found
        while True:
            header = file.read(8)
            if len(header) < 8:
                break
            kind, size = header[:4], int.from_bytes(header[4:], 'little')
            if kind == b'ANMF':
                payload = file.read(16)
                found.append(int.from_bytes(payload[12:15], 'little'))
                file.seek(size + (size & 1) - 16, 1)
            else:
                file.seek(size + (size & 1), 1)
    return found


def durations(path, frames):
    """Each of `frames` frames' duration in ms (a missing or zero one: the clip's most common duration)."""
    found = frame_durations(path)
    if len(found) != frames:
        found = [None] * frames
    known = [value for value in found if value]
    typical = max(set(known), key=known.count) if known else DEFAULT_FRAME_MS
    return [max(SHORTEST_FRAME_MS, value or typical) for value in found]


def timeline(lengths):
    """(frame rate, how many times each frame is written): the shortest duration's grid, rounded cumulatively."""
    step = min(lengths)
    repeats, end, written = [], 0, 0
    for length in lengths:
        end += length
        repeats.append(max(1, round(end / step) - written))
        written += repeats[-1]
    return Fraction(1000, step), repeats


def encode(source, target):
    """Writes the animated WebP `source` as an MP4 at `target`; returns the clip's seconds."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f'{target.name}.{os.getpid()}-{threading.get_ident()}.partial')
    with Image.open(source) as image:
        if getattr(image, 'n_frames', 1) < 2:
            raise ValueError(f'{Path(source).name}: не анимированный WebP')
        rate, repeats = timeline(durations(source, image.n_frames))
        width, height = image.size
        command = [ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-framerate', f'{rate.numerator}/{rate.denominator}', '-i', 'pipe:0',
                   *ARGUMENTS, '-threads', str(THREADS), '-f', 'mp4', str(partial)]
        flags = (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, creationflags=flags)
        background = Image.new('RGB', image.size, BACKGROUND)
        try:
            for index, count in enumerate(repeats):
                image.seek(index)
                frame = image.convert('RGBA')
                flat = background.copy()
                flat.paste(frame, mask=frame.getchannel('A'))
                data = flat.tobytes()
                for _ in range(count):
                    process.stdin.write(data)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # ffmpeg stopped early; its error is read below
        error = process.stderr.read().decode('utf-8', 'replace').strip()
        process.wait()
    if process.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'ffmpeg {process.returncode}: {error[-400:] or "без сообщения"}')
    os.replace(partial, target)
    return float(sum(repeats) / rate)


def cached(source, cache=None):
    """Where the MP4 of `source` is kept in the cache (by its md5 and the encoder settings)."""
    return Path(cache or CACHE) / f'{md5(source)}-{SETTINGS}.mp4'


def convert(source, cache=None, target=None):
    """The MP4 of the animated WebP `source`, from the cache or encoded into it."""
    target = target or cached(source, cache)
    if not target.is_file():
        encode(source, target)
    return target


def convert_all(sources, cache=None, workers=WORKERS):
    """{source: MP4 path or the error text} for the given animated WebP files, up to `workers` encodes at once; files
    with the same content are encoded once."""
    sources = list(dict.fromkeys(Path(source) for source in sources))
    pool_size = max(1, min(workers, WORKERS))
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        targets = dict(zip(sources, pool.map(lambda source: cached(source, cache), sources)))
        first = {}
        for source, target in targets.items():
            first.setdefault(target, source)

        def one(target):
            try:
                return target, convert(first[target], cache, target)
            except Exception as error:  # a clip that fails stays a WebP; the export reports it
                return target, f'{type(error).__name__}: {error}'

        done = dict(pool.map(one, first))
    return {source: done[target] for source, target in targets.items()}


def faststart(path):
    """True when the MP4's moov box comes before its mdat (so playback starts before the whole file is in)."""
    order = []
    with open(path, 'rb') as file:
        while True:
            header = file.read(8)
            if len(header) < 8:
                break
            size, kind = int.from_bytes(header[:4], 'big'), header[4:8]
            if size == 1:
                size = int.from_bytes(file.read(8), 'big')
                file.seek(size - 16, 1)
            elif size == 0:
                order.append(kind)
                break
            else:
                file.seek(size - 8, 1)
            order.append(kind)
    return b'moov' in order and b'mdat' in order and order.index(b'moov') < order.index(b'mdat')


def probe(path):
    """What ffmpeg reads from an MP4: {'video': stream line, 'duration': seconds, 'faststart': bool}."""
    result = subprocess.run([ffmpeg(), '-hide_banner', '-i', str(path)], capture_output=True, text=True,
                            encoding='utf-8', errors='replace',
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    video = next((line.strip() for line in result.stderr.splitlines() if 'Video:' in line), '')
    duration = None
    for line in result.stderr.splitlines():
        if 'Duration:' in line:
            hours, minutes, seconds = line.split('Duration:')[1].split(',')[0].strip().split(':')
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return {'video': video, 'duration': duration, 'faststart': faststart(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', nargs='+', help='animated WebP clips')
    parser.add_argument('--out', help='the MP4 for a single clip (default: the cache)')
    args = parser.parse_args()
    if args.out and len(args.source) != 1:
        parser.error('--out только для одного ролика')
    started = time.time()
    if args.out:
        encode(args.source[0], args.out)
        results = {Path(args.source[0]): Path(args.out)}
    else:
        results = convert_all(args.source)
    for source, result in results.items():
        if isinstance(result, Path):
            print(f'{source.name} → {result} ({result.stat().st_size / 1e6:.2f} МБ): {probe(result)}')
        else:
            print(f'{source.name}: не вышло — {result}')
    print(f'{time.time() - started:.1f} с')


if __name__ == '__main__':
    main()
