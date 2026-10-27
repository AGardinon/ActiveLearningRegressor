"""Builds proof_of_concepts/pool_refinement/phase1_6d/analysis.ipynb (cells below, in order)."""
import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 1: adaptive refinement on the 6D Ackley paper setup

**Question.** In 6D, a 50k-point starting pool is sparse. Does adaptive refinement let the
campaign go beyond the best point of the starting pool, and which centroid window works best?

**Setup** (see `README.md`): the paper configuration `6D/GPR/ackley_gpr_40cy_5pts_1stage_3_2`
(50k LHS pool, 40 cycles × 5 points, τ = 10, 3 centroids, 250 points per cube), with GPR
restarts 300 → 50. Each **run set** `runs/<batch method>_s<seed>/` holds the four variants on the
same starting pool (set by the seed). Batch methods (`batch_selection.sampling_method`):
`voronoi` (paper runs) and `fps` (argmax of the acquisition, then greedy maximin).

| Variant | Centroid window |
|---|---|
| `norefine` | no refinement |
| `history` | whole training history (as in the paper runs) |
| `window30` | last 30 points (3τ) |
| `window10` | last 10 points (τ, as in the paper text) |
| `history_bestfirst` | whole history, centroids **best y first**, then maximin among the top 20% |
| `window30_bestfirst` | last 30 points, centroids best y first, then maximin among the top 20% |

The first four use the original centroid rule (random first centroid, then greedy maximin).

Ackley is negated, so the maximum is **0 at the origin**.

**Pool ceiling** = best true y available in the pool at a given cycle. Without refinement it is
constant (the best point of the starting pool), so no model can do better than it.
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
RUN_SETS = sorted(p.name for p in RUNS_ROOT.glob('*_s*') if p.is_dir())
RUN_SET = 'fps_s13'     # run set shown in detail below
XS = [f'x{i}' for i in range(1, 7)]
OPTIMUM = np.zeros(6)   # negated Ackley: maximum f(0) = 0
TAU = 10                # refinement_step of the configs

# Baseline in neutral grey; the three windows take the first three categorical slots.
# Best-first variants share the colour of their window and are drawn dashed.
VARIANTS = {
    'norefine':           dict(label='no refinement',                     color='#6f6e69', ls='-'),
    'history':            dict(label='window: whole history',             color='#2a78d6', ls='-'),
    'window30':           dict(label='window: 30 (3τ)',                   color='#eb6834', ls='-'),
    'window10':           dict(label='window: 10 (τ)',                    color='#1baf7a', ls='-'),
    'history_bestfirst':  dict(label='whole history, best first',         color='#2a78d6', ls='--'),
    'window30_bestfirst': dict(label='window 30, best first',             color='#eb6834', ls='--'),
}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e5e4e0', 'grid.linewidth': 0.6,
                     'lines.linewidth': 2, 'font.size': 10})
""")

md(r"""
## Load the runs

Training points are stored in the scaled space; they are mapped back with the run's
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
    return dict(pool=pool, train=train, refined=refined, cubes=cubes,
                bench=pd.read_csv(d / 'benchmark_data.csv'))

def load_set(run_set):
    root = RUNS_ROOT / run_set
    return {v: load_variant(root / v) for v in VARIANTS if (root / v / 'train_points_data.csv').exists()}

print('run sets:', RUN_SETS)
runs = load_set(RUN_SET)

# Paired design check: same starting pool, and same initial points per repetition
ref = next(iter(runs.values()))
for v, r in runs.items():
    same_pool = r['pool'][XS].equals(ref['pool'][XS])
    same_init = np.allclose(r['train'].loc[r['train'].cycle == 0, XS], ref['train'].loc[ref['train'].cycle == 0, XS])
    print(f"{v:9s} reps={r['train'].repetition.nunique()}  same pool: {same_pool}  same initial points: {same_init}")
""")

md(r"""
## Per-cycle metrics

The refinement steps are replayed with the same trigger as `AdaptiveRefinement`
(fires after a cycle when ≥ τ points were acquired since the last refinement), so that the pool
ceiling at cycle *c* only includes refined points that existed after cycle *c*.
""")

code(r"""
def per_cycle_metrics(r):
    pool_max = r['pool']['y'].max()
    rows = []
    for rep, g in r['train'].groupby('repetition'):
        log = r['refined'].get(rep)
        last, step = 0, 0
        for c in sorted(g.cycle.unique()):
            upto = g[g.cycle <= c]
            n = len(upto)
            if log is not None and c > 0 and n - last >= TAU:
                step, last = step + 1, n
            added = log[log.refinement_step <= step] if log is not None else None
            ceiling = pool_max if added is None or added.empty else max(pool_max, added['y'].max())
            best = upto.loc[upto['y'].idxmax()]
            rows.append(dict(
                repetition=rep, cycle=c, n_acquired=n,
                best_y=best['y'],
                dist_best_to_opt=float(np.linalg.norm(best[XS].to_numpy(float) - OPTIMUM)),
                pool_ceiling=ceiling,
                frac_from_refinement=upto['from_refinement'].mean(),
                pool_size=len(r['pool']) + (0 if added is None else len(added)),
                refinement_steps=step,
            ))
        if log is not None:
            assert step == log.refinement_step.max(), f'replayed {step} steps, log has {log.refinement_step.max()}'
    return pd.DataFrame(rows)

def centroid_origins(runs):
    # Match each centroid back to its training point: acquisition source and true y
    rows = []
    for v, r in runs.items():
        t = r['train']
        for rep, c in r['cubes'].items():
            tr = t[t.repetition == rep].reset_index(drop=True)
            d, idx = cKDTree(tr[XS].to_numpy()).query(c[[f'{x}_centroid' for x in XS]].to_numpy())
            assert d.max() < 1e-6, 'every centroid must be a training point'
            rows.append(pd.DataFrame({'variant': v, 'repetition': rep, 'refinement_step': c.refinement_step,
                                      'source': tr.acquisition_source.to_numpy()[idx], 'y': tr.y.to_numpy()[idx]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['variant', 'source', 'y'])

BASIN = -15   # Ackley's central basin (see the centroid section below)

metrics = pd.concat([per_cycle_metrics(r).assign(variant=v) for v, r in runs.items()], ignore_index=True)

# Consistency with the script's own best-so-far. Note: benchmark_data's `y_best_y` at cycle c is the best
# value BEFORE that cycle's acquisition, i.e. our best_y after cycle c-1.
for v, r in runs.items():
    prev = metrics[metrics.variant == v][['repetition', 'cycle', 'best_y']].assign(cycle=lambda d: d.cycle + 1)
    b = r['bench'].merge(prev, on=['repetition', 'cycle'])
    print(f'{v:9s} best_y(c-1) matches benchmark_data y_best_y(c): {np.allclose(b.best_y, b.y_best_y)}')
""")

md(r"""
## Overview across run sets

Final state (after the last cycle) per run set and variant. `beat_pool` = repetitions whose best y is
above the best y of the starting pool. `centroids_in_basin` = share of centroids with y > −15.
""")

code(r"""
rows = []
for rs in RUN_SETS:
    rs_runs = load_set(rs)
    if not rs_runs:
        continue
    pool_max = next(iter(rs_runs.values()))['pool']['y'].max()
    cen = centroid_origins({v: r for v, r in rs_runs.items() if r['cubes']})
    for v, r in rs_runs.items():
        m = per_cycle_metrics(r)
        last = m[m.cycle == m.cycle.max()]
        c = cen[cen.variant == v]
        rows.append(dict(run_set=rs, variant=v, reps=len(last), pool_max=pool_max,
                         best_y_mean=last.best_y.mean(), best_y_max=last.best_y.max(),
                         beat_pool=int((last.best_y > pool_max + 1e-4).sum()),   # y is stored at float32 precision in the pool CSV
                         share_from_refinement=last.frac_from_refinement.mean(),
                         centroids_in_basin=(c.y > BASIN).mean() if len(c) else np.nan))
overview = pd.DataFrame(rows)
overview['variant'] = pd.Categorical(overview.variant, categories=list(VARIANTS), ordered=True)
overview.sort_values(['run_set', 'variant']).round(3)
""")

code(r"""
# Batch method alone: same seed, same variant, voronoi vs fps (final mean best y)
pivot = overview.assign(method=overview.run_set.str.split('_s').str[0],
                        seed=overview.run_set.str.split('_s').str[1])
pivot.pivot_table(index=['seed', 'variant'], columns='method', values='best_y_mean', observed=True).round(3)
""")

md(r"""
## Detail of one run set

The sections below show the run set `RUN_SET` (set in the first code cell).
""")

code(r"""
PANELS = [
    ('best_y',               'Best y found (optimum = 0)'),
    ('pool_ceiling',         'Pool ceiling: best y in the pool'),
    ('dist_best_to_opt',     'Distance of best point to optimum'),
    ('frac_from_refinement', 'Share of acquired points from refinement'),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (col, title) in zip(axes.flat, PANELS):
    for v, style in VARIANTS.items():
        m = metrics[metrics.variant == v]
        if m.empty:
            continue
        agg = m.groupby('cycle')[col].agg(['mean', 'min', 'max'])
        ax.fill_between(agg.index, agg['min'], agg['max'], color=style['color'], alpha=0.12, linewidth=0)
        ax.plot(agg.index, agg['mean'], color=style['color'], linestyle=style['ls'], label=style['label'])
    ax.set_title(title, loc='left', fontsize=11)
for ax in axes[1]:
    ax.set_xlabel('cycle')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
fig.text(0.5, -0.01, 'Lines: mean over repetitions; bands: min to max.', ha='center', color='#52514e')
fig.tight_layout()
plt.show()
""")

md("## Final state per variant")

code(r"""
last = metrics[metrics.cycle == metrics.cycle.max()]
summary = last.groupby('variant').agg(
    best_y_mean=('best_y', 'mean'), best_y_min=('best_y', 'min'), best_y_max=('best_y', 'max'),
    ceiling_mean=('pool_ceiling', 'mean'),
    dist_to_opt_mean=('dist_best_to_opt', 'mean'),
    share_from_refinement=('frac_from_refinement', 'mean'),
    pool_size_mean=('pool_size', 'mean'),
).reindex([v for v in VARIANTS if v in runs])
pool_max = ref['pool']['y'].max()
print(f'Best y in the starting pool: {pool_max:.3f}')
summary.round(3)
""")

code(r"""
# Per repetition: best y among acquired points from the starting pool vs from refinement
rows = []
for v, r in runs.items():
    for rep, g in r['train'].groupby('repetition'):
        rows.append(dict(variant=v, repetition=rep,
                         best_from_pool=g.loc[~g.from_refinement, 'y'].max(),
                         best_from_refinement=g.loc[g.from_refinement, 'y'].max(),
                         n_from_refinement=int(g.from_refinement.sum()), n_acquired=len(g)))
pd.DataFrame(rows).round(3)
""")

md(r"""
## Hypercube size per refinement step

`half_side` is in the scaled space (StandardScaler units), capped by `max_half_side = 0.5`.
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 3.8))
for v, style in VARIANTS.items():
    if v not in runs or not runs[v]['cubes']:
        continue
    cubes = pd.concat(runs[v]['cubes'].values(), ignore_index=True)
    per_step = cubes.groupby('refinement_step').half_side.mean()
    ax.plot(per_step.index, per_step.values, color=style['color'], linestyle=style['ls'], label=style['label'])
    print(f"{v:9s} share of proposed points kept after filtering: {cubes.n_added.sum() / cubes.n_proposed.sum():.4f}")
ax.axhline(0.5, color='#8a8984', linewidth=1, linestyle='--')
ax.text(1, 0.5, ' max_half_side', va='bottom', color='#52514e', fontsize=9)
ax.set_ylim(0, 0.55)
ax.set_title('Mean cube half-side (scaled space)', loc='left', fontsize=11)
ax.set_xlabel('refinement step')
ax.legend(frameon=False, loc='lower left')
fig.tight_layout()
plt.show()
""")

md(r"""
## Where do the centroids come from?

Each centroid is a training point. Matching it back gives its acquisition source and its true y.
For 6D Ackley almost the whole domain is a plateau around y ≈ −21; the informative region is the
central basin, here taken as **y > −15** (0.03% of the starting pool).
""")

code(r"""
centroids = centroid_origins({v: r for v, r in runs.items() if r['cubes']})
acq = {v: r['train'][r['train'].cycle > 0] for v, r in runs.items()}
table = pd.DataFrame({
    'centroids in basin': centroids.groupby('variant').y.apply(lambda y: (y > BASIN).mean()),
    'acquired points in basin': pd.Series({v: (a.y > BASIN).mean() for v, a in acq.items()}),
    'EI points in basin': pd.Series({v: (a.loc[a.acquisition_source == 'expected_improvement', 'y'] > BASIN).mean() for v, a in acq.items()}),
    'centroids from exploration': centroids.groupby('variant').source.apply(lambda s: (s == 'exploration_mutual_info').mean()),
    'acquired points from exploration': pd.Series({v: (a.acquisition_source == 'exploration_mutual_info').mean() for v, a in acq.items()}),
}).reindex([v for v in VARIANTS if v in centroids.variant.unique()])
print(f"starting pool in basin: {(ref['pool'].y > BASIN).mean():.4%}")
table.round(3)
""")

md(r"""
**Reading.** The good points are clustered in one small basin, while greedy maximin picks the
points that are *farthest apart*. After a random first pick, at most one of the three centroids
can land in the cluster, and the others go to spread-out (mostly exploration, mostly plateau)
points. Shortening the window helps a little, but does not change this. The `*_bestfirst`
variants change the rule itself: best observed y first, then maximin among the top 20%.
""")

md(r"""
## Projection onto two coordinates

A hypercube projects onto any pair of coordinates as an exact rectangle, so cubes and points can
be drawn faithfully in (x1, x2). **Caveat:** points that look close here can be far apart in the
other four coordinates. Grey: a random 5% of the starting pool. Rectangles: cubes, light → dark
with the refinement step. Dots: acquired points from the starting pool (dark) and from refinement
(orange). Cross: the optimum.
""")

code(r"""
def plot_projection(ax, r, rep, dims=('x1', 'x2'), lim=None, pool_frac=0.05):
    a, b = dims
    pool = r['pool'].sample(frac=pool_frac, random_state=0)
    ax.scatter(pool[a], pool[b], s=2, color='#d4d3cf', linewidths=0, rasterized=True)
    cubes = r['cubes'].get(rep)
    if cubes is not None:
        cmap = plt.get_cmap('Blues')
        n_steps = cubes.refinement_step.max()
        for _, c in cubes.iterrows():
            shade = cmap(0.3 + 0.7 * (c.refinement_step - 1) / max(n_steps - 1, 1))
            ax.add_patch(Rectangle((c[f'{a}_low'], c[f'{b}_low']),
                                   c[f'{a}_high'] - c[f'{a}_low'], c[f'{b}_high'] - c[f'{b}_low'],
                                   fill=False, edgecolor=shade, linewidth=1))
    t = r['train'][r['train'].repetition == rep]
    ax.scatter(t.loc[~t.from_refinement, a], t.loc[~t.from_refinement, b], s=14, color='#2b2b29',
               edgecolors='white', linewidths=0.5, label='acquired, from pool', zorder=3)
    ax.scatter(t.loc[t.from_refinement, a], t.loc[t.from_refinement, b], s=14, color='#eb6834',
               edgecolors='white', linewidths=0.5, label='acquired, from refinement', zorder=3)
    ax.scatter([0], [0], marker='x', s=60, color='black', linewidths=1.5, label='optimum', zorder=4)
    if lim is not None:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(a); ax.set_ylabel(b); ax.set_aspect('equal')

REP = 1
show = [v for v in ('history', 'history_bestfirst', 'window10') if v in runs]
fig, axes = plt.subplots(len(show), 2, figsize=(11, 5.2 * len(show)), squeeze=False)
for row, v in zip(axes, show):
    plot_projection(row[0], runs[v], REP)
    plot_projection(row[1], runs[v], REP, lim=8)
    row[0].set_title(f"{RUN_SET}, {VARIANTS[v]['label']}, rep {REP}: full domain", loc='left', fontsize=11)
    row[1].set_title('zoom around the optimum', loc='left', fontsize=11)
fig.tight_layout()
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.035))
plt.show()
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, out)
print('wrote', out)
