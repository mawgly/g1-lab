import csv, math, sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
print('rows', len(rows))
prev = None
buckets = {}
for r in rows:
    steps = int(float(r['time/total_timesteps']))
    nup = int(float(r['train/n_updates']))
    epochs = None if prev is None else nup - prev
    prev = nup
    ent = -float(r['train/entropy_loss'])
    sigma = math.exp((ent - 29 * 0.5 * math.log(2 * math.pi * math.e)) / 29)
    key = steps // 20_000_000
    b = buckets.setdefault(key, [])
    b.append((steps, epochs, sigma, float(r['train/approx_kl']), float(r['train/clip_fraction']),
              float(r['train/value_loss']), float(r['rollout/ep_len_mean'] or 'nan'),
              float(r['rollout/ep_rew_mean'] or 'nan'), float(r['train/policy_gradient_loss'])))


def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float('nan')


print('M steps | epochs/iter | sigma | kl | clipfrac | vloss | ep_len | ep_rew | early-stop share')
for key in sorted(buckets):
    b = buckets[key]
    ep = [x[1] for x in b if x[1] is not None]
    early = sum(1 for e in ep if e < 5) / len(ep) if ep else float('nan')
    print(f'{b[0][0]/1e6:8.1f} | {mean(ep):4.2f} | {mean([x[2] for x in b]):.3f} | {mean([x[3] for x in b]):.4f} | '
          f'{mean([x[4] for x in b]):.3f} | {mean([x[5] for x in b]):.4f} | {mean([x[6] for x in b]):6.1f} | '
          f'{mean([x[7] for x in b]):8.2f} | {early:.2f}')
