"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call TODO: add functions
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

from siclib.geometry.perspective_fields import get_perspective_field
from siclib.utils.conversions import rad2deg

# flake8: noqa
# mypy: ignore-errors


def cm_ranking(sc, ths=None):
    if ths is None:
        ths = [512, 1024, 2048, 4096]

    ls = sc.shape[0]
    colors = ["red", "yellow", "lime", "cyan", "blue"]
    out = ["gray"] * ls
    for i in range(ls):
        for c, th in zip(colors[: len(ths) + 1], ths + [ls]):
            if i < th:
                out[i] = c
                break
    sid = np.argsort(sc, axis=0).flip(0)
    return np.array(out)[sid]


def cm_RdBl(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 1.0, 0, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    return np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)


def plot_images(imgs, titles=None, cmaps="gray", dpi=200, pad=0.5, adaptive=True):
    """Plot a list of images.

    Args:
        imgs (List[np.ndarray]): List of images to plot.
        titles (List[str], optional): Titles. Defaults to None.
        cmaps (str, optional): Colormaps. Defaults to "gray".
        dpi (int, optional): Dots per inch. Defaults to 200.
        pad (float, optional): Padding. Defaults to 0.5.
        adaptive (bool, optional): Whether to adapt the aspect ratio. Defaults to True.

    Returns:
        plt.Figure: Figure of the images.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, axs = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)

    return fig


def plot_image_grid(
    imgs,
    titles=None,
    cmaps="gray",
    dpi=100,
    pad=0.5,
    fig=None,
    adaptive=True,
    figs=3.0,
    return_fig=False,
    set_lim=False,
) -> plt.Figure:
    """Plot a grid of images.

    Args:
        imgs (List[np.ndarray]): List of images to plot.
        titles (List[str], optional): Titles. Defaults to None.
        cmaps (str, optional): Colormaps. Defaults to "gray".
        dpi (int, optional): Dots per inch. Defaults to 100.
        pad (float, optional): Padding. Defaults to 0.5.
        fig (_type_, optional): Figure to plot on. Defaults to None.
        adaptive (bool, optional): Whether to adapt the aspect ratio. Defaults to True.
        figs (float, optional): Figure size. Defaults to 3.0.
        return_fig (bool, optional): Whether to return the figure. Defaults to False.
        set_lim (bool, optional): Whether to set the limits. Defaults to False.

    Returns:
        plt.Figure: Figure and axes or just axes.
    """
    nr, n = len(imgs), len(imgs[0])
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs[0]]  # W / H
    else:
        ratios = [4 / 3] * n

    figsize = [sum(ratios) * figs, nr * figs]
    if fig is None:
        fig, axs = plt.subplots(
            nr, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
        )
    else:
        axs = fig.subplots(nr, n, gridspec_kw={"width_ratios": ratios})
        fig.figure.set_size_inches(figsize)

    if nr == 1 and n == 1:
        axs = [[axs]]
    elif n == 1:
        axs = axs[:, None]
    elif nr == 1:
        axs = [axs]

    for j in range(nr):
        for i in range(n):
            ax = axs[j][i]
            ax.imshow(imgs[j][i], cmap=plt.get_cmap(cmaps[i]))
            ax.set_axis_off()
            if set_lim:
                ax.set_xlim([0, imgs[j][i].shape[1]])
                ax.set_ylim([imgs[j][i].shape[0], 0])
            if titles:
                ax.set_title(titles[j][i])
    if isinstance(fig, plt.Figure):
        fig.tight_layout(pad=pad)
    return (fig, axs) if return_fig else axs


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=4,
    ha="left",
    va="top",
    axes=None,
    **kwargs,
):
    """Add text to a plot.

    Args:
        idx (int): Index of the axes.
        text (str): Text to add.
        pos (tuple, optional): Text position. Defaults to (0.01, 0.99).
        fs (int, optional): Font size. Defaults to 15.
        color (str, optional): Text color. Defaults to "w".
        lcolor (str, optional): Line color. Defaults to "k".
        lwidth (int, optional): Line width. Defaults to 4.
        ha (str, optional): Horizontal alignment. Defaults to "left".
        va (str, optional): Vertical alignment. Defaults to "top".
        axes (List[plt.Axes], optional): Axes to put text on. Defaults to None.

    Returns:
        plt.Text: Text object.
    """
    if axes is None:
        axes = plt.gcf().axes

    ax = axes[idx]

    t = ax.text(
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=ax.transAxes,
        zorder=5,
        **kwargs,
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )
    return t


def plot_heatmaps(
    heatmaps,
    vmin=-1e-6,  # include negative zero
    vmax=None,
    cmap="Spectral",
    a=0.5,
    axes=None,
    contours_every=None,
    contour_style="solid",
    colorbar=False,
):
    """Plot heatmaps with optional contours.

    To plot latitude field, set vmin=-90, vmax=90 and contours_every=15.

    Args:
        heatmaps (List[np.ndarray | torch.Tensor]): List of 2D heatmaps.
        vmin (float, optional): Min Value. Defaults to -1e-6.
        vmax (float, optional): Max Value. Defaults to None.
        cmap (str, optional): Colormap. Defaults to "Spectral".
        a (float, optional): Alpha value. Defaults to 0.5.
        axes (List[plt.Axes], optional): Axes to plot on. Defaults to None.
        contours_every (int, optional): If not none, will draw contours. Defaults to None.
        contour_style (str, optional): Style of the contours. Defaults to "solid".
        colorbar (bool, optional): Whether to show colorbar. Defaults to False.

    Returns:
        List[plt.Artist]: List of artists.
    """
    if axes is None:
        axes = plt.gcf().axes
    artists = []

    for i in range(len(axes)):
        a_ = a if isinstance(a, float) else a[i]

        if isinstance(heatmaps[i], torch.Tensor):
            heatmaps[i] = heatmaps[i].detach().cpu().numpy()

        alpha = a_
        # Plot the heatmap
        art = axes[i].imshow(
            heatmaps[i],
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        if colorbar:
            cmax = vmax or np.percentile(heatmaps[i], 99)
            art.set_clim(vmin, cmax)
            cbar = plt.colorbar(art, ax=axes[i])
            artists.append(cbar)

        artists.append(art)

        if contours_every is not None:
            # Add contour lines to the heatmap
            contour_data = np.arange(vmin, vmax + contours_every, contours_every)

            # Get the colormap colors for contour lines
            contour_colors = [
                plt.colormaps.get_cmap(cmap)(plt.Normalize(vmin=vmin, vmax=vmax)(level))
                for level in contour_data
            ]
            contours = axes[i].contour(
                heatmaps[i],
                levels=contour_data,
                linewidths=2,
                colors=contour_colors,
                linestyles=contour_style,
            )

            contours.set_clim(vmin, vmax)

            fmt = {
                level: f"{label}°"
                for level, label in zip(contour_data, contour_data.astype(int).astype(str))
            }
            t = axes[i].clabel(contours, inline=True, fmt=fmt, fontsize=16, colors="white")

            for label in t:
                label.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=1, foreground="k"),
                        path_effects.Normal(),
                    ]
                )
            artists.append(contours)

    return artists


def plot_horizon_lines(
    cameras, gravities, line_colors="orange", lw=2, styles="solid", alpha=1.0, ax=None
):
    """Plot horizon lines on the perspective field.

    Args:
        cameras (List[Camera]): List of cameras.
        gravities (List[Gravity]): Gravities.
        line_colors (str, optional): Line Colors. Defaults to "orange".
        lw (int, optional): Line width. Defaults to 2.
        styles (str, optional): Line styles. Defaults to "solid".
        alpha (float, optional): Alphas. Defaults to 1.0.
        ax (List[plt.Axes], optional): Axes to draw horizon line on. Defaults to None.
    """
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(cameras)

    if not isinstance(styles, list):
        styles = [styles] * len(cameras)

    fig = plt.gcf()
    ax = fig.gca() if ax is None else ax

    if isinstance(ax, plt.Axes):
        ax = [ax] * len(cameras)

    assert len(ax) == len(cameras), f"{len(ax)}, {len(cameras)}"

    for i in range(len(cameras)):
        _, lat = get_perspective_field(cameras[i], gravities[i])
        # horizon line is zero level of the latitude field
        lat = lat[0, 0].cpu().numpy()
        contours = ax[i].contour(lat, levels=[0], linewidths=lw, colors=line_colors[i])
        for contour_line in contours.collections:
            contour_line.set_linestyle(styles[i])


def plot_vector_fields(
    vector_fields,
    cmap="lime",
    subsample=15,
    scale=None,
    lw=None,
    alphas=0.8,
    axes=None,
):
    """Plot vector fields.

    Args:
        vector_fields (List[torch.Tensor]): List of vector fields of shape (2, H, W).
        cmap (str, optional): Color of the vectors. Defaults to "lime".
        subsample (int, optional): Subsample the vector field. Defaults to 15.
        scale (float, optional): Scale of the vectors. Defaults to None.
        lw (float, optional): Line width of the vectors. Defaults to None.
        alphas (float | np.ndarray, optional): Alpha per vector or global. Defaults to 0.8.
        axes (List[plt.Axes], optional): List of axes to draw on. Defaults to None.

    Returns:
        List[plt.Artist]: List of artists.
    """
    if axes is None:
        axes = plt.gcf().axes

    vector_fields = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in vector_fields]

    artists = []

    H, W = vector_fields[0].shape[-2:]
    if scale is None:
        scale = subsample / min(H, W)

    if lw is None:
        lw = 0.1 / subsample

    if alphas is None:
        alphas = np.ones_like(vector_fields[0][0])
        alphas = np.stack([alphas] * len(vector_fields), 0)
    elif isinstance(alphas, float):
        alphas = np.ones_like(vector_fields[0][0]) * alphas
        alphas = np.stack([alphas] * len(vector_fields), 0)
    else:
        alphas = np.array(alphas)

    subsample = min(W, H) // subsample
    offset_x = ((W % subsample) + subsample) // 2

    samples_x = np.arange(offset_x, W, subsample)
    samples_y = np.arange(int(subsample * 0.9), H, subsample)

    x_grid, y_grid = np.meshgrid(samples_x, samples_y)

    for i in range(len(axes)):
        # vector field of shape (2, H, W) with vectors of norm == 1
        vector_field = vector_fields[i]

        a = alphas[i][samples_y][:, samples_x]
        x, y = vector_field[:, samples_y][:, :, samples_x]

        c = cmap
        if not isinstance(cmap, str):
            c = cmap[i][samples_y][:, samples_x].reshape(-1, 3)

        s = scale * min(H, W)
        arrows = axes[i].quiver(
            x_grid,
            y_grid,
            x,
            y,
            scale=s,
            scale_units="width" if H > W else "height",
            units="width" if H > W else "height",
            alpha=a,
            color=c,
            angles="xy",
            antialiased=True,
            width=lw,
            headaxislength=3.5,
            zorder=5,
        )

        artists.append(arrows)

    return artists


def plot_latitudes(
    latitude,
    is_radians=True,
    vmin=-90,
    vmax=90,
    cmap="seismic",
    contours_every=15,
    alpha=0.4,
    axes=None,
    **kwargs,
):
    """Plot latitudes.

    Args:
        latitude (List[torch.Tensor]): List of latitudes.
        is_radians (bool, optional): Whether the latitudes are in radians. Defaults to True.
        vmin (int, optional): Min value to clip to. Defaults to -90.
        vmax (int, optional): Max value to clip to. Defaults to 90.
        cmap (str, optional): Colormap. Defaults to "seismic".
        contours_every (int, optional): Contours every. Defaults to 15.
        alpha (float, optional): Alpha value. Defaults to 0.4.
        axes (List[plt.Axes], optional): Axes to plot on. Defaults to None.

    Returns:
        List[plt.Artist]: List of artists.
    """
    if axes is None:
        axes = plt.gcf().axes

    assert len(axes) == len(latitude), f"{len(axes)}, {len(latitude)}"
    lat = [rad2deg(lat) for lat in latitude] if is_radians else latitude
    return plot_heatmaps(
        lat,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        a=alpha,
        axes=axes,
        contours_every=contours_every,
        **kwargs,
    )


def plot_confidences(
    confidence,
    as_log=True,
    vmin=-4,
    vmax=0,
    cmap="turbo",
    alpha=0.4,
    axes=None,
    **kwargs,
):
    """Plot confidences.

    Args:
        confidence (List[torch.Tensor]): Confidence maps.
        as_log (bool, optional): Whether to plot in log scale. Defaults to True.
        vmin (int, optional): Min value to clip to. Defaults to -4.
        vmax (int, optional): Max value to clip to. Defaults to 0.
        cmap (str, optional): Colormap. Defaults to "turbo".
        alpha (float, optional): Alpha value. Defaults to 0.4.
        axes (List[plt.Axes], optional): Axes to plot on. Defaults to None.

    Returns:
        List[plt.Artist]: List of artists.
    """
    if axes is None:
        axes = plt.gcf().axes

    confidence = [c.cpu() if isinstance(c, torch.Tensor) else torch.tensor(c) for c in confidence]

    assert len(axes) == len(confidence), f"{len(axes)}, {len(confidence)}"

    if as_log:
        confidence = [torch.log10(c.clip(1e-5)).clip(vmin, vmax) for c in confidence]

    # normalize to [0, 1]
    confidence = [(c - c.min()) / (c.max() - c.min()) for c in confidence]
    return plot_heatmaps(confidence, vmin=0, vmax=1, cmap=cmap, a=alpha, axes=axes, **kwargs)


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
