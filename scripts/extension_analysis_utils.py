from __future__ import annotations
import csv, itertools, json, math, random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def percentile(values, q: float) -> float:
    xs = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not xs: return float("nan")
    if len(xs) == 1: return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi: return xs[lo]
    t = pos - lo
    return xs[lo] * (1-t) + xs[hi] * t


def exact_bootstrap_ci(values) -> tuple[float,float]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if not xs: return float("nan"), float("nan")
    if n > 7:
        raise ValueError("exact n^n bootstrap intentionally limited to <=7 independent units")
    vals = [sum(xs[i] for i in draw)/n for draw in itertools.product(range(n), repeat=n)]
    return percentile(vals,.025), percentile(vals,.975)


def mc_bootstrap_ci(values, *, resamples: int=100000, seed: int=20260818) -> tuple[float,float]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if not xs: return float("nan"), float("nan")
    rng = random.Random(int(seed))
    vals = []
    for _ in range(int(resamples)):
        vals.append(sum(xs[rng.randrange(n)] for _ in range(n))/n)
    return percentile(vals,.025), percentile(vals,.975)


def slope(xs, ys) -> float:
    if len(xs) < 2: return float("nan")
    mx, my = mean(xs), mean(ys)
    den = sum((x-mx)**2 for x in xs)
    if den <= 0: return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0]*len(xs); i=0
    while i < len(order):
        j=i+1
        while j < len(order) and xs[order[j]] == xs[order[i]]: j += 1
        r = 0.5*(i+j-1)+1
        for k in range(i,j): out[order[k]]=r
        i=j
    return out


def pearson(xs,ys):
    if len(xs)<2: return float("nan")
    mx,my=mean(xs),mean(ys)
    dx=[x-mx for x in xs]; dy=[y-my for y in ys]
    den=(sum(x*x for x in dx)*sum(y*y for y in dy))**0.5
    return sum(x*y for x,y in zip(dx,dy))/den if den>0 else float("nan")


def spearman(xs,ys):
    return pearson(rankdata(xs),rankdata(ys)) if len(xs)>=2 else float("nan")
