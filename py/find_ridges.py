import argparse
import os

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from param_manager.param_manager import ParamManager
from scipy.ndimage import gaussian_filter
from skimage.filters import meijering

mpl.use("qtagg")


def detect_ridges(
    hist,
    n_bins_x,
    n_bins_y,
    extent,
    sigmas=[3.0],
    thresh=0.2,
    non_max_suppress=True,
):
    ridges = meijering(hist, sigmas=sigmas, black_ridges=False)
    ridges_neg = meijering(hist, sigmas=sigmas, black_ridges=True)
    xs = []
    ys = []

    mask = np.full_like(ridges, False)
    for x_bin in range(n_bins_x):
        for y_bin in range(n_bins_y):
            if ridges[x_bin, y_bin] > thresh:
                mask[x_bin, y_bin] = True

            if ridges_neg[x_bin, y_bin] > thresh:
                mask[x_bin, y_bin] = True

    blurred_hist = gaussian_filter(hist, sigma=sigmas[0])

    for x_bin in range(1, n_bins_x - 1):
        for y_bin in range(1, n_bins_y - 1):
            if (
                ridges_neg[x_bin, y_bin]
                or ridges_neg[x_bin - 1, y_bin]
                or ridges_neg[x_bin + 1, y_bin]
                or ridges_neg[x_bin, y_bin - 1]
                or ridges_neg[x_bin, y_bin + 1]
            ):
                mask[x_bin, y_bin] = False

    while non_max_suppress:
        mask_copy = mask.copy()
        non_max_suppress = False
        for x_bin in range(1, n_bins_x - 1):
            for y_bin in range(1, n_bins_y - 1):
                if not mask[x_bin, y_bin]:
                    continue
                value = blurred_hist[x_bin, y_bin]

                passes_ns = (
                    value > blurred_hist[x_bin, y_bin - 1]
                    and value > blurred_hist[x_bin, y_bin + 1]
                )
                passes_ne_sw = (
                    value > blurred_hist[x_bin + 1, y_bin + 1]
                    and value > blurred_hist[x_bin - 1, y_bin - 1]
                )
                passes_nw_se = (
                    value > blurred_hist[x_bin + 1, y_bin - 1]
                    and value > blurred_hist[x_bin - 1, y_bin + 1]
                )
                passes_ew = (
                    value > blurred_hist[x_bin - 1, y_bin]
                    and value > blurred_hist[x_bin + 1, y_bin]
                )
                if passes_ew or passes_ns or passes_ne_sw or passes_nw_se:
                    continue
                mask_copy[x_bin, y_bin] = False
                non_max_suppress = True
        mask = mask_copy

    for x_bin in range(n_bins_x):
        for y_bin in range(n_bins_y):
            if mask[x_bin, y_bin]:
                xs.append(
                    extent[0] + ((x_bin + 0.5) / n_bins_x) * (extent[1] - extent[0])
                )
                ys.append(
                    extent[2] + ((y_bin + 0.5) / n_bins_y) * (extent[3] - extent[2])
                )
    return pl.DataFrame(data={"x": xs, "y": ys})


def plot_histogram(ax, hist, colorbar=True, **kwargs):
    hist_vals, x_edges, y_edges = hist
    x_values = []
    y_values = []
    weights = []
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    for x_idx in range(len(x_edges) - 1):
        for y_idx in range(len(y_edges) - 1):
            x_bin = x_edges[x_idx]
            y_bin = y_edges[y_idx]

            x_values.append(x_bin + dx / 2)
            y_values.append(y_bin + dy / 2)

            weights.append(hist_vals[x_idx, y_idx])

    art = ax.hist2d(
        x_values, y_values, weights=weights, bins=(x_edges, y_edges), **kwargs
    )
    if colorbar:
        _cb = plt.colorbar(
            art[3], ax=ax, pad=0, fraction=0.05, aspect=20, location="top"
        )


def main():
    """
    todo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type="str")
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()

    input_file = "raw.parquet"
    output_file = "ridges.parquet"

    param_manager = ParamManager(args.directory)

    #
    # data cleaning
    #
    df_file_name = os.path.join(args.directory, input_file)
    print(f"Reading df {df_file_name}")
    df = (
        pl.scan_parquet(df_file_name)
        .select([param_manager.x_col, param_manager.y_col])
        .drop_nulls()
        .collect()
    )

    #
    # Finding the maxima
    #
    # PARAM USED: Number of bins in both x and y.
    initial_hist, x_edges, y_edges = np.histogram2d(
        df[param_manager.x_col],
        df[param_manager.y_col],
        bins=(param_manager.num_bins_x, param_manager.num_bins_y),
    )

    # This is a crude way of tracking edges. Someone really should use a good histogram class.
    # In the future, root histograms are probably the correct tool here, but I can't link cleanly to
    # ROOT at the moment, so numpy hists for now. You would also have to handroll your own image processing
    # algorithms... I'm not sure what the best move is
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    #
    # Find the bins where a bin is maximal along the y-axis
    #
    maximal_y_pnts = detect_ridges(
        hist=initial_hist,
        n_bins_x=param_manager.num_bins_x,
        n_bins_y=param_manager.num_bins_y,
        extent=extent,
        sigmas=[param_manager.sigma],
        thresh=param_manager.threshold,
    )

    #
    # Output
    #
    os.path.join(args.directory, output_file)
    output_file = os.path.join(args.directory, output_file)

    print(f"Writing output points to {output_file}")
    maximal_y_pnts.write_parquet(output_file)

    if args.plot:
        print("Plotting")
        fig, axs = plt.subplots(1, 1, figsize=(8, 16))
        ax = axs
        plot_histogram(
            ax,
            (initial_hist, x_edges, y_edges),
            cmap=cmr.neutral,
            cmin=1,
            norm="log",
            alpha=0.2,
        )
        ax.plot(
            maximal_y_pnts["x"],
            maximal_y_pnts["y"],
            marker=".",
            ls="",
            color="steelblue",
            ms=0.5,
        )
        plt.show()

    return


if __name__ == "__main__":
    main()
