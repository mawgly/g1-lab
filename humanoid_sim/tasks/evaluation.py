"""Task-owned episode metrics; generic evaluation never assumes legs."""


def locomotion_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    return dict(forward_m=float(final['x_position']-initial['x_position']),
                fell=bool(final.get('fell', terminated)))


def reach_metrics(initial: dict, final: dict, terminated: bool) -> dict:
    return dict(distance_m=float(final['distance']), success=bool(final.get('is_success')))
