#!

# --------------------------------------------------------------------------------
# SI figure: morphology of the three benchmark landscapes (Hartmann, Ackley,
# Styblinski-Tang) used in the study, at the 6D benchmark dimensionality.
#
# Recipe (uniform across all three panels): the LOWER ENVELOPE of the 6D function
# over the plane (x1, x2),
#
#     g(x1, x2) = min_{x3..x6} f(x1, x2, x3, ..., x6)
#
# i.e. the best value attainable at a given (x1, x2). A plain slice with x3..x6
# pinned at the optimizer was rejected: it renders Hartmann as a single monotone
# ramp, because slicing through the global optimizer cuts the deepest basin and
# cannot reach the others (all 15 dimension pairs show exactly 1 local minimum).
#
# The envelope is computed analytically where the argmin over the free dims is
# independent of (x1, x2), and numerically otherwise:
#
#   - Ackley: f decreases as sum(xi^2) -> 0 and as sum(cos(c*xi)) -> max, both at
#     xi = 0. The free dims therefore sit at 0 regardless of (x1, x2), so the
#     envelope coincides exactly with the slice through the optimizer.
#   - Styblinski-Tang: additively separable, so each free dim independently sits
#     at -2.903534 and only contributes a constant offset. The envelope is the 2D
#     function up to that offset.
#   - Hartmann: not separable; the argmin over x3..x6 genuinely depends on
#     (x1, x2), so the envelope is minimised numerically (multi-start Adam).
#     Verified to recover the true optimum -3.32237 to 4 decimals.
#
# Note on Hartmann: the 6D function has 6 local minima, but only 2 survive as
# distinguishable basins under projection to (x1, x2). The caption should say so.
#
# Output is PNG only: the surfaces emit one path per quad, which makes the
# vector version ~110 MB.

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from activereg.beauty import get_axes

from botorch.test_functions import Hartmann, Ackley, StyblinskiTang

FIG_DIR = Path(__file__).resolve().parent
CACHE = FIG_DIR / "_hartmann6d_envelope.npz"

DIM = 6

# The campaigns negate the objectives so that acquisition maximises (see
# target_function_config.yaml: negate: true). With NEGATE = True the panels are
# drawn in that same maximisation convention: the envelope becomes an UPPER
# envelope, max_{x3..x6} -f = -min_{x3..x6} f, i.e. the exact mirror of the
# minimisation surface. The argmax over the free dims is the same point as the
# argmin, so every analytic argument above carries over unchanged and the cached
# Hartmann envelope stays valid (it is negated on load).
NEGATE = True

HILIGHT_OPT = True  # draw a crimson star at the optimizer on each panel

CMAP = cm.plasma  # colormap for the surfaces and contours

# --------------------------------------------------------------------------------
# Layout / sizing
#
# Panels are wrapped at MAX_COL columns by activereg.beauty.get_axes. With 3
# landscapes and MAX_COL = 2 the layout is 2x2 and the fourth slot is blanked.
MAX_COL = 3

# Target width of the saved figure, in inches. FIG_WIDTH_IN is the full width of
# the rendered PNG, split evenly across MAX_COL panels; 7.09 in = 180 mm is the
# usual double-column (full text width) spec for an SI page. Use ~3.35 in
# (85 mm) for a single-column figure.
FIG_WIDTH_IN = 7.09
PANEL_HEIGHT_IN = 3.1

# NOTE: the figure is saved WITHOUT bbox_inches="tight", because tight recomputes
# the bbox from the content and would silently override FIG_WIDTH_IN. Margins are
# therefore set explicitly in main() and must leave room for the 3d z-labels --
# a 3d axes reports a tight bbox that omits its own z-label, so with tight
# cropping the right-hand column loses it.

# --------------------------------------------------------------------------------
# Fonts (pt)
BASE_FONT = 8        # fallback for anything not listed below
AXES_LABEL_FONT = 9  # x / y / z axis labels
TITLE_FONT = 9       # per-panel titles
SUBTITLE_FONT = 8    # the morphology line under each panel title
TICK_FONT = 7        # tick labels on all three axes
SUPTITLE_FONT = 10   # figure-level title

plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.labelsize": AXES_LABEL_FONT,
    "axes.titlesize": TITLE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": BASE_FONT - 1,
})


def analytic_envelope(func, dim: int, n_grid: int):
    """Envelope for functions whose free-dim argmin is the optimizer coordinate."""
    lo, hi = func._bounds[0]
    optimizer = np.asarray(func._optimizers[0], dtype=float)

    x = np.linspace(lo, hi, n_grid)
    X1, X2 = np.meshgrid(x, x)

    pts = np.tile(optimizer, (X1.size, 1))
    pts[:, 0] = X1.ravel()
    pts[:, 1] = X2.ravel()

    with torch.no_grad():
        Z = func(torch.from_numpy(pts)).numpy().reshape(X1.shape)

    return X1, X2, Z, optimizer


def numeric_envelope(func, dim: int, n_grid: int, n_starts: int = 256,
                     n_steps: int = 150, lr: float = 0.05, seed: int = 0):
    """Envelope by multi-start gradient minimisation over the free dimensions."""
    lo, hi = func._bounds[0]
    optimizer = np.asarray(func._optimizers[0], dtype=float)

    x = np.linspace(lo, hi, n_grid)
    X1, X2 = np.meshgrid(x, x)

    rng = np.random.default_rng(seed)
    starts = torch.from_numpy(lo + (hi - lo) * rng.random((n_starts, dim - 2)))

    grid = np.stack([X1.ravel(), X2.ravel()], 1)
    Z = np.empty(grid.shape[0])

    for s in range(0, grid.shape[0], 500):
        blk = torch.from_numpy(grid[s:s + 500])
        B = blk.shape[0]

        free = starts.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True)
        fixed = blk.unsqueeze(1).expand(B, n_starts, 2)
        opt = torch.optim.Adam([free], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            pts = torch.cat([fixed, free.clamp(lo, hi)], dim=2).reshape(-1, dim)
            func(pts).reshape(B, n_starts).sum().backward()
            opt.step()

        with torch.no_grad():
            pts = torch.cat([fixed, free.clamp(lo, hi)], dim=2).reshape(-1, dim)
            Z[s:s + 500] = func(pts).reshape(B, n_starts).min(dim=1).values.numpy()

    return X1, X2, Z.reshape(X1.shape), optimizer


def hartmann_envelope(n_grid: int):
    func = Hartmann(dim=DIM)
    if CACHE.exists():
        d = np.load(CACHE)
        if d["Z"].shape[0] == n_grid:
            return d["X1"], d["X2"], d["Z"], np.asarray(func._optimizers[0])

    X1, X2, Z, optimizer = numeric_envelope(func, DIM, n_grid)
    np.savez_compressed(CACHE, X1=X1, X2=X2, Z=Z)
    print(f"  Hartmann envelope min = {Z.min():.5f} (true {func._optimal_value:.5f})")
    return X1, X2, Z, optimizer


LANDSCAPES = [
    {
        "title": r"$f(\mathcal{X}_P): \mathrm{Hartmann}$",
        "subtitle": "smooth, broad peaks" if NEGATE else "smooth, broad basins",
        "build": lambda: hartmann_envelope(n_grid=160),
    },
    {
        "title": r"$f(\mathcal{X}_P): \mathrm{Ackley}$",
        "subtitle": ("sharp global spike above a rippled plain" if NEGATE
                     else "global funnel under a dense ripple field"),
        "build": lambda: analytic_envelope(Ackley(dim=DIM), DIM, n_grid=600),
    },
    {
        "title": r"$f(\mathcal{X}_P): \mathrm{Styblinski\text{-}Tang}$",
        "subtitle": "$2^d$ symmetric peaks" if NEGATE else "$2^d$ symmetric wells",
        "build": lambda: analytic_envelope(StyblinskiTang(dim=DIM), DIM, n_grid=300),
    },
]


def main():
    # 3D axes for the surfaces
    fig, axes = get_axes(
        len(LANDSCAPES),
        max_col=MAX_COL,
        fig_frame=(FIG_WIDTH_IN / MAX_COL, PANEL_HEIGHT_IN),
        res=300,
        subplot_kw={"projection": "3d", "computed_zorder": False},
    )

    # Simple 2D axes
    fig1, axes1 = get_axes(
        len(LANDSCAPES),
        max_col=MAX_COL,
        fig_frame=(FIG_WIDTH_IN / MAX_COL, PANEL_HEIGHT_IN),
        res=300,
    )

    for i, spec in enumerate(LANDSCAPES):
        print(f"building {spec['title']} ...")
        X1, X2, Z, optimizer = spec["build"]()

        if NEGATE:
            Z = -Z

        ax = axes[i]
        ax1 = axes1[i]

        ax.plot_surface(
            X1, X2, Z,
            cmap=CMAP,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )

        z_floor = Z.min() - 0.45 * (Z.max() - Z.min())
        ax.contourf(X1, X2, Z, levels=30, zdir="z", offset=z_floor,
                    cmap=CMAP, alpha=0.7)
        ax.set_zlim(z_floor, Z.max())

        ax1.contourf(X1, X2, Z, levels=30, cmap=CMAP, alpha=0.7)

        if HILIGHT_OPT:
            ax.scatter(
                optimizer[0], optimizer[1], z_floor,
                marker="*", s=180, color="crimson", edgecolor="white",
                linewidth=0.6, depthshade=False, zorder=10,
            )

            ax1.scatter(
                optimizer[0], optimizer[1],
                marker="*", s=180, color="crimson", edgecolor="white",
                linewidth=0.6, zorder=10, alpha=0.5
            )

        ax.set_xlabel("$x_1$", labelpad=-2)
        ax.set_ylabel("$x_2$", labelpad=-2)
        ax.set_zlabel("$-f$" if NEGATE else "$f$", labelpad=-4)
        
        ax1.set_xlabel("$x_1$", labelpad=-2)
        ax1.set_ylabel("$x_2$", labelpad=-2)

        ax.set_title(spec["title"], fontsize=TITLE_FONT, pad=0)
        ax1.set_title(spec["title"], fontsize=TITLE_FONT, pad=0)
        # subtitle as a separate artist so it can take its own size
        ax.text2D(0.5, 1.0, spec["subtitle"], transform=ax.transAxes,
                  ha="center", va="top", fontsize=SUBTITLE_FONT)
        ax.view_init(elev=32, azim=-52)
        # there is no ztick.labelsize rcParam, so the z axis needs setting here
        ax.tick_params(labelsize=TICK_FONT, pad=-2)
        ax.zaxis.set_rotate_label(False)

    envelope_expr = ("$\\max_{x_3..x_6}\\, -f(\\mathbf{x})$" if NEGATE
                     else "$\\min_{x_3..x_6} f(\\mathbf{x})$")
    fig.suptitle(
        "Morphology of the three benchmark landscapes (6D)\n"
        f"best attainable {envelope_expr} over the $(x_1, x_2)$ plane",
        fontsize=SUPTITLE_FONT, y=0.985,
    )
    # right margin must clear the 3d z-labels (see NOTE at the top)
    fig.subplots_adjust(left=0.0, right=0.93, wspace=0.0, hspace=0.16,
                        top=0.88, bottom=0.0)
    fig1.tight_layout()

    png = FIG_DIR / "benchmark_landscapes_6D.png"
    fig.savefig(png, dpi=300)
    w_in = fig.get_size_inches()[0]
    print(f"wrote {png} ({png.stat().st_size / 1e6:.2f} MB, "
          f"{w_in:.2f} in = {w_in * 25.4:.0f} mm wide)")

    png1 = FIG_DIR / "benchmark_landscapes_6D_contours.png"
    svg1 = FIG_DIR / "benchmark_landscapes_6D_contours.svg"
    fig1.savefig(png1, dpi=300)
    fig1.savefig(svg1)
    w_in1 = fig1.get_size_inches()[0]
    print(f"wrote {png1} ({png1.stat().st_size / 1e6:.2f} MB, "
          f"{w_in1:.2f} in = {w_in1 * 25.4:.0f} mm wide)")
    print(f"wrote {svg1} ({svg1.stat().st_size / 1e6:.2f} MB, "
          f"{w_in1:.2f} in = {w_in1 * 25.4:.0f} mm wide)")
    

if __name__ == "__main__":
    main()
