"""Task-owned episode metrics; generic evaluation never assumes legs."""


def locomotion_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    return dict(forward_m=float(final['x_position']-initial['x_position']),
                fell=bool(final.get('fell', terminated)))


def reach_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    return dict(distance_m=float(final['distance']), success=bool(final.get('is_success')))


def getup_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    """g1_getup: success = up within 6 s and held 3 s (G1GetupEnv's getup_held), when it got up (s) and the head's
    final height (m); the start class it began from."""
    return dict(success=final.get('getup_held') is True, getup_s=final.get('getup_seconds'),
                head_m=final.get('head_height'), start_class=initial.get('start_class'))
