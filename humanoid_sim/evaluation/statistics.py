"""Descriptive uncertainty across independent evaluation episodes."""
import math
import statistics


def summarize(records: list[dict]) -> dict:
    result = {}
    for key in sorted({k for row in records for k in row}):
        if key in ('seed', 'schema_version', 'termination'):
            continue
        raw = [row.get(key) for row in records]
        values = [v for v in raw if isinstance(v, (int, float)) and math.isfinite(v)]
        if not values and any(v is not None for v in raw):
            continue
        count = len(values)
        summary = dict(count=count, mean=statistics.mean(values) if count else None,
                       std=statistics.stdev(values) if count > 1 else None)
        if count and all(isinstance(v, bool) for v in values):
            p = sum(values) / count
            z = 1.959963984540054
            denominator = 1 + z*z/count
            center = (p + z*z/(2*count)) / denominator
            half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
            summary['wilson95'] = [max(0., center-half), min(1., center+half)]
        result[key] = summary
    return result
