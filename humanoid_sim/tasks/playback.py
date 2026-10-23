"""Declarative MuJoCo view settings, independent of robot identifiers."""
from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class PlaybackSpec:
    follow_body: str | None = None
    target: tuple[float, float, float] = (0., 0., .5)
    camera_offset: tuple[float, float, float] = (3.1, -4., 1.6)
    metric_key: str | None = None
    metric_label: str = ''
    fall_height: float | None = None

    def focus(self, model, data):
        if self.follow_body is None:
            return list(self.target)
        # mj_name2id is a direct lookup; model.body(name) builds a costly accessor object per call.
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.follow_body)].tolist()

    def describe(self):
        return dict(camera_offset=list(self.camera_offset), target_height=self.target[2],
                    metric_key=self.metric_key, metric_label=self.metric_label,
                    supports_full_fall=self.fall_height is not None)
