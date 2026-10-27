"""Builds proof_of_concepts/pool_refinement/phase2_2d/analysis.ipynb (cells below, in order)."""
import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 2: adaptive refinement from a 5-point pool (2D Styblinski-Tang)

**Question.** With an extremely sparse starting pool, refinement is the only source of new
candidates. Where can it take the campaign, and does the best-first centroid rule get trapped in a
decoy basin?

**Function.** Styblinski-Tang 2D on [-4, 4]², negated (maximisation). Four corner basins, split
per dimension at x ≈ 0.157:

| Basin | Optimum | y |
|---|---|---|
| `--` (global) | (-2.90, -2.90) | 78.33 |
| `-+`, `+-` | mixed corners | 64.20 |
| `++` | (2.75, 2.75) | 50.06 |

**Setup** (see `README.md`):
- a 5-point LHS pool set by the seed;
- 3 random initial points;
- each cycle acquires 1 point by exploration (mutual info) and 1 by expected improvement;
- 20 cycles, with refinement every 5 acquisitions: 2 cubes × 25 points;
- fps batch selection.

There is no run without refinement: the pool runs out after cycle 1. Each seed `runs/s<seed>/`
holds all variants on the same pool.

| Variant | Centroids | Half-side |
|---|---|---|
| `maximin` | random first, then greedy maximin over the whole history | `density` (shrinks as the pool densifies) |
| `bestfirst` | best y first, then maximin among the top 20% | `density` |
| `maximin_fixed` | as `maximin` | `fixed`, 0.5 scaled units at every step |
| `bestfirst_fixed` | as `bestfirst` | `fixed`, 0.5 |
""")

code(r"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.spatial import cKDTree

RUNS_ROOT = Path('runs')
SEEDS = sorted((p.name for p in RUNS_ROOT.glob('s*') if p.is_dir()), key=lambda s: int(s[1:]))
XS = ['x1', 'x2']
TAU = 5                 # refinement_step of the configs
SPLIT = 0.1567          # per-dimension basin boundary
GLOBAL_OPT = np.array([-2.9035, -2.9035])
GLOBAL_Y = 78.33
REACHED = 70.0          # y above this only exists in the global basin

VARIANTS = {
    'maximin':         dict(label='maximin, density',     color='#2a78d6', ls='-'),
    'bestfirst':       dict(label='best first, density',  color='#eb6834', ls='-'),
    'maximin_fixed':   dict(label='maximin, fixed 0.5',   color='#2a78d6', ls='--'),
    'bestfirst_fixed': dict(label='best first, fixed 0.5', color='#eb6834', ls='--'),
}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e5e4e0', 'grid.linewidth': 0.6,
                     'lines.linewidth': 2, 'font.size': 10})

def styblinski_tang(x1, x2):
    f = lambda x: x**4 - 16 * x**2 + 5 * x
    return -0.5 * (f(x1) + f(x2))

def basin(x1, x2):
    return ('-' if x1 < SPLIT else '+') + ('-' if x2 < SPLIT else '+')
""")

md(r"""
## Load the runs

Training points are stored in the scaled space and mapped back with the run's
`pool_scaler.joblib`. An acquired point is flagged `from_refinement` when it is not in the
starting pool.
""")

code(r"""
def load_variant(d):
    scaler = joblib.load(d / 'pool_scaler.joblib')
    pool = pd.read_csv(d / 'pool_dataset.csv')
    train = pd.read_csv(d / 'train_points_data.csv')
    train[XS] = scaler.inverse_transform(train[XS].to_numpy())
    reps = sorted(train.repetition.unique())
    refined = {r: pd.read_csv(f) for r in reps
               if (f := d / f'collective_refinement_points_rep{r}.csv').exists()}
    cubes = {r: pd.read_csv(f) for r in reps
             if (f := d / f'refinement_cubes_rep{r}.csv').exists()}
    dist, _ = cKDTree(pool[XS].to_numpy()).query(train[XS].to_numpy())
    train['from_refinement'] = dist > 1e-6
    assert not train.loc[train.cycle == 0, 'from_refinement'].any(), 'initial points must come from the pool'
    return dict(pool=pool, train=train, refined=refined, cubes=cubes)

runs = {s: {v: load_variant(RUNS_ROOT / s / v) for v in VARIANTS
            if (RUNS_ROOT / s / v / 'train_points_data.csv').exists()} for s in SEEDS}

# Paired design check: same pool and same initial points across the variants of a seed
for s, rs in runs.items():
    ref = next(iter(rs.values()))
    ok = all(r['pool'][XS].equals(ref['pool'][XS]) and
             np.allclose(r['train'].loc[r['train'].cycle == 0, XS], ref['train'].loc[ref['train'].cycle == 0, XS])
             for r in rs.values())
    print(f"{s:5s} variants={list(rs)}  same pool and initial points: {ok}")
""")

md(r"""
## The starting pools

`nearest_to_global` is the distance from the closest pool point to the global optimum. The first
cube has a half-side of about 1.1–1.35 in original units (0.5 scaled units, with the scaler fitted on
the 5 pool points).
""")

code(r"""
rows = []
for s, rs in runs.items():
    pool = next(iter(rs.values()))['pool']
    best = pool.loc[pool.y.idxmax()]
    rows.append(dict(seed=s, pool_best_y=best.y, pool_best_basin=basin(best.x1, best.x2),
                     nearest_to_global=np.linalg.norm(pool[XS].to_numpy() - GLOBAL_OPT, axis=1).min()))
pools = pd.DataFrame(rows).set_index('seed')
pools.round(2)
""")

md(r"""
## Outcome per seed

- `reached_global`: repetitions (of 5) whose best point has y > 70, i.e. lies in the global basin.
- `first_cycle`: the cycle at which that happened (mean over the repetitions that got there).
""")

code(r"""
rows = []
for s, rs in runs.items():
    for v, r in rs.items():
        for rep, g in r['train'].groupby('repetition'):
            hit = g[g.y > REACHED]
            rows.append(dict(seed=s, variant=v, rep=rep, best_y=g.y.max(),
                             reached=len(hit) > 0, first_cycle=hit.cycle.min() if len(hit) else np.nan,
                             share_from_refinement=g.from_refinement.mean()))
outcome = pd.DataFrame(rows)
summary = outcome.groupby(['seed', 'variant'], sort=False).agg(
    reached_global=('reached', 'sum'), best_y_mean=('best_y', 'mean'), best_y_min=('best_y', 'min'),
    first_cycle=('first_cycle', 'mean'))
summary = summary.join(pools[['nearest_to_global']], on='seed')
display(summary.round(2))

reach = outcome.assign(reachable=outcome.seed.map(pools.nearest_to_global < 1.3)).pivot_table(
    index='reachable', columns='variant', values='reached', aggfunc='sum')[list(VARIANTS)]
print('repetitions reaching the global basin, by whether a pool point lies within 1.3 of it')
reach
""")

md(r"""
## Best y per cycle

Mean over the 5 repetitions; the band spans min to max. The dotted line is the global optimum,
the thin grey line the best y of the starting pool.
""")

code(r"""
fig, axes = plt.subplots(1, len(SEEDS), figsize=(3.4 * len(SEEDS), 3.4), sharey=True, squeeze=False)
for ax, s in zip(axes[0], SEEDS):
    for v, r in runs[s].items():
        cur = r['train'].sort_values('cycle').groupby(['repetition', 'cycle']).y.max() \
            .groupby(level='repetition').cummax().unstack('repetition')
        st = VARIANTS[v]
        ax.plot(cur.index, cur.mean(axis=1), color=st['color'], ls=st['ls'], label=st['label'])
        ax.fill_between(cur.index, cur.min(axis=1), cur.max(axis=1), color=st['color'], alpha=0.08, linewidth=0)
    ax.axhline(GLOBAL_Y, color='black', ls=':', lw=1)
    ax.axhline(pools.loc[s, 'pool_best_y'], color='#9b9a95', lw=1)
    ax.set_title(f"{s}: nearest pool point {pools.loc[s, 'nearest_to_global']:.2f} from global", loc='left', fontsize=10)
    ax.set_xlabel('cycle')
axes[0, 0].set_ylabel('best y found')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
plt.show()
""")

md(r"""
## Cube size per refinement step

Half-side along x1 in original units, as logged (clipped at the domain bounds, so a cube against
a wall reports less). Mean over seeds and repetitions.
""")

code(r"""
fig, ax = plt.subplots(figsize=(6, 3.6))
for v in VARIANTS:
    hs = pd.concat([c.assign(hs=(c.x1_high - c.x1_low) / 2)
                    for s in SEEDS if v in runs[s] for c in runs[s][v]['cubes'].values()])
    m = hs.groupby('refinement_step').hs.mean()
    st = VARIANTS[v]
    ax.plot(m.index, m.values, color=st['color'], ls=st['ls'], marker='o', ms=4, label=st['label'])
ax.set_xlabel('refinement step'); ax.set_ylabel('half-side along x1 (original units)')
ax.legend(frameon=False)
plt.show()
""")

md(r"""
## The campaigns on the landscape

Repetition 1 of a seed where a pool point is close to the global basin (`s13`) and of one where it
is not (`s21`). Cubes are coloured by refinement step (light = early). The four corner optima are
marked with ×; the global one is circled.
""")

code(r"""
g1, g2 = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
landscape = styblinski_tang(g1, g2)
CORNERS = [(-2.9035, -2.9035), (-2.9035, 2.7468), (2.7468, -2.9035), (2.7468, 2.7468)]

def plot_campaign(ax, r, rep):
    ax.contourf(g1, g2, landscape, levels=20, cmap='Greys', alpha=0.35)
    cubes = r['cubes'].get(rep)
    if cubes is not None:
        cmap = plt.get_cmap('Blues')
        n_steps = cubes.refinement_step.max()
        for _, c in cubes.iterrows():
            shade = cmap(0.35 + 0.6 * (c.refinement_step - 1) / max(n_steps - 1, 1))
            ax.add_patch(Rectangle((c.x1_low, c.x2_low), c.x1_high - c.x1_low, c.x2_high - c.x2_low,
                                   fill=False, edgecolor=shade, linewidth=1))
    ref = r['refined'].get(rep)
    if ref is not None:
        ax.scatter(ref.x1, ref.x2, s=4, color='#9b9a95', label='refined candidates', zorder=2)
    ax.scatter(r['pool'].x1, r['pool'].x2, s=60, marker='s', facecolor='none', edgecolor='black',
               linewidths=1.2, label='starting pool', zorder=3)
    t = r['train'][r['train'].repetition == rep]
    ax.scatter(t.loc[~t.from_refinement, 'x1'], t.loc[~t.from_refinement, 'x2'], s=16, color='#2b2b29',
               edgecolors='white', linewidths=0.5, label='acquired, from pool', zorder=4)
    ax.scatter(t.loc[t.from_refinement, 'x1'], t.loc[t.from_refinement, 'x2'], s=16, color='#eb6834',
               edgecolors='white', linewidths=0.5, label='acquired, from refinement', zorder=4)
    for cx, cy in CORNERS:
        ax.scatter([cx], [cy], marker='x', s=50, color='black', linewidths=1.3, zorder=5)
    ax.scatter(*GLOBAL_OPT, s=220, facecolor='none', edgecolor='black', linewidths=1, zorder=5)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')

REP = 1
show = [(s, v) for s in ('s13', 's21') for v in ('maximin', 'bestfirst', 'maximin_fixed') if s in runs and v in runs[s]]
fig, axes = plt.subplots(2, 3, figsize=(14, 9.6), squeeze=False)
for ax, (s, v) in zip(axes.flat, show):
    plot_campaign(ax, runs[s][v], REP)
    best = runs[s][v]['train'].query('repetition == @REP').y.max()
    ax.set_title(f"{s}, {VARIANTS[v]['label']}: best y {best:.1f}", loc='left', fontsize=11)
fig.tight_layout()
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
plt.show()
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, out)
print('wrote', out)
