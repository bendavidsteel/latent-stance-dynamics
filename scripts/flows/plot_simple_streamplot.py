import os

import matplotlib.pyplot as plt
import numpy as np


ATTRACTOR = ( 1.1,  1.1, -1.6, 0.55)
REPELLER  = (-1.1,  0.0,  1.6, 0.55)


def grad_potential(x, y):
    """Analytic gradient of a sum of two Gaussians: one attractor, one repeller."""
    dphi_dx = np.zeros_like(x)
    dphi_dy = np.zeros_like(y)
    for x0, y0, amp, sigma2 in [ATTRACTOR, REPELLER]:
        g = amp * np.exp(-(((x - x0) ** 2) + ((y - y0) ** 2)) / sigma2)
        dphi_dx = dphi_dx + (-2.0 / sigma2) * (x - x0) * g
        dphi_dy = dphi_dy + (-2.0 / sigma2) * (y - y0) * g
    return dphi_dx, dphi_dy


def plot_simple_streamplot(fig_path):
    extent = 2.2
    res = 200
    x = np.linspace(-extent, extent, res)
    y = np.linspace(-extent, extent, res)
    xs, ys = np.meshgrid(x, y)

    dphi_dx, dphi_dy = grad_potential(xs, ys)
    fu, fv = -dphi_dx, -dphi_dy
    speed = np.sqrt(fu ** 2 + fv ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

    lw = 0.5 + 2.5 * (speed / (speed.max() + 1e-9))
    ax.streamplot(
        xs, ys, fu, fv,
        density=1.4,
        color='red',
        linewidth=lw,
        arrowsize=1.2,
        arrowstyle='->',
    )

    ax.set_xticks([-extent, extent])
    ax.set_xticklabels(['Left', 'Right'], fontsize=12)
    for label, ha in zip(ax.get_xticklabels(), ['left', 'right']):
        label.set_ha(ha)
    ax.set_yticks([-extent, extent])
    ax.set_yticklabels(['Libertarian', 'Authoritarian'], fontsize=12, rotation=90)
    for label, va in zip(ax.get_yticklabels(), ['bottom', 'top']):
        label.set_va(va)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_box_aspect(1)

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out = os.path.join(fig_path, 'simple_streamplot.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.2)
    print(f'wrote {out}')


def main():
    fig_path = './figs'
    os.makedirs(fig_path, exist_ok=True)
    plot_simple_streamplot(fig_path)


if __name__ == '__main__':
    main()
