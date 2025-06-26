#!python
import argparse
import os

import cmasher as cmr
import cmocean
import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from datashader.mpl_ext import dsshow
from param_manager.param_manager import ParamManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default="data/faust_det_60")
    parser.add_argument("-i", "--lin_file", default="linearized.parquet")
    parser.add_argument("-r", "--raw_file", default="raw.parquet")

    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()

    pm = ParamManager(args.directory)

    linearized_data = os.path.join(args.directory, args.lin_file)
    raw_data = os.path.join(args.directory, args.raw_file)
    extend_dir = os.path.join(args.directory, "extended_curves/")
    print(extend_dir)

    raw_df = (
        pl.scan_parquet(raw_data)
        .select([pm.x_col, pm.y_col])
        .with_columns(x=pm.x_col, y=pm.y_col)
        .collect()
    )

    lin_df = (
        pl.scan_parquet(linearized_data).collect().with_columns(z_lin=pl.col("z_lin"))
    )

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), layout="tight")
    ax = axs
    raw_df = raw_df.to_pandas()
    dsshow(
        raw_df,
        ds.Point("x", "y"),
        aspect="auto",
        cmap=cmr.neutral,
        ax=ax,
        norm="log",
    )

    for file_name in os.listdir(extend_dir):
        file_name = os.path.join(extend_dir, file_name)
        df = pl.read_parquet(file_name)
        ax.plot(df["x"], df["y"], color="steelblue", lw=1)

    smooth_dir = os.path.join(args.directory, "smoothed_lines/")
    for file_name in os.listdir(smooth_dir):
        file_name = os.path.join(smooth_dir, file_name)
        df = pl.read_parquet(file_name).filter(
            pl.col("extrapolated_left").not_() & pl.col("extrapolated_right").not_()
        )
        ax.plot(df["x"], df["y"], color="red", lw=1)

    ax = axs
    print(lin_df)
    lin_df = lin_df.select(["x", "y", "z_lin"])
    print(f"z_lin nulls {lin_df.filter(pl.col('z_lin').is_nan()).height}")

    lin_df = (
        lin_df.drop_nans().drop_nulls().with_columns(z_lin_sqr=pl.col("z_lin").pow(2))
    )
    lin_df = lin_df.to_pandas()

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), layout="tight")

    ax = axs[0]
    dsshow(
        lin_df,
        ds.Point("z_lin", "x"),
        aspect="auto",
        # cmap=cmr.neutral,
        cmap=cmr.amber,
        ax=ax,
        # alpha=0.7,
        norm="eq_hist",
    )
    x = 20
    while x < 200:
        ax.axvline(x, color="k")
        x += 5
    ax = axs[1]
    dsshow(
        lin_df,
        ds.Point("z_lin", "x"),
        aspect="auto",
        # cmap=cmr.neutral,
        cmap=cmr.amber,
        ax=ax,
        # alpha=0.7,
        norm="log",
    )
    x = 20
    while x < 200:
        ax.axvline(x, color="k")
        x += 5
    ax = axs[2]
    ax.set_yscale("log")
    ax.hist(
        lin_df["z_lin"],
        bins=1024 * 4,
        histtype="step",
        color="k",
    )
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])

    plt.show()


if __name__ == "__main__":
    main()
