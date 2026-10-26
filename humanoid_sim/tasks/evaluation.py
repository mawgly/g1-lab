"""Task-owned episode metrics; generic evaluation never assumes legs."""


def locomotion_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    return dict(forward_m=float(final['x_position']-initial['x_position']),
                fell=bool(final.get('fell', terminated)))


def reach_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    return dict(distance_m=float(final['distance']), success=bool(final.get('is_success')))


def getup_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    """g1_getup: success = up within 6 s and held 3 s (G1GetupEnv's getup_held), when it got up (s) and the head's
    final height (m); the start class it began from. With task_config track_clip also track_done: whether the episode
    reached the clip's last frame without a tracking termination. With settle_clip success is the settling one instead
    (settle_done: near the nearest target and still, G1GetupEnv._settle_step), with when (settle_s) and into which clip's
    first frame (settle_target)."""
    metrics = dict(success=final.get('getup_held') is True, getup_s=final.get('getup_seconds'),
                   head_m=final.get('head_height'), start_class=initial.get('start_class'))
    if 'track_done' in final:
        metrics['track_done'] = bool(final['track_done'])
    if 'settle_done' in final:
        metrics.update(success=bool(final['settle_done']), settle_done=bool(final['settle_done']),
                       settle_s=final.get('settle_seconds'), settle_target=final.get('settle_target'))
    return metrics
