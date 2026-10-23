# Local browser interface

Extend existing watch/selection flow with a local-only browser interface. User requested implementation; prior installation permission persists. Existing checkpoints, trainer and normalization remain authoritative. No cloud hosting: runtime must access local MuJoCo and model files.

- Serve a Russian responsive dashboard on 127.0.0.1:8766. Local Three.js renders the actual MuJoCo primitive geometry and streamed transforms, with orbit/zoom camera. All assets vendored for offline use.
- One shared playback session: stage select, first/best/latest, pause, reset and 0.5/1/2x speed. Live-follow switches models between episodes. Paused selection can load a new stage while remaining paused.
- Timeline and fixed-seed metric charts from actual evaluation files; distinguish forward distance and reward, don't describe standing as successful walking. Show training progress/terminal state and model step count separately.
- Continue/stop training from explicit buttons using existing CLI and OS run lock. Server binds loopback, validates Host/Origin and JSON input; does not accept arbitrary paths/commands.
- Tests first for checkpoint selection, pause/reset, invalid requests, no-stage startup, real HTTP static/API/security boundaries. Verify actual browser rendering and controls. Keep local server running for the user and provide launcher.
