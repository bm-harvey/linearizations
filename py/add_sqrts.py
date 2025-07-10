import argparse
import os

import polars as pl


def main():
    """
    simple script to add si_sqrt and csi_sqrt columns
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", default="data/faust_det_60")
    parser.add_argument("-i", "--input_file", default="raw.parquet")
    args = parser.parse_args()

    df_file_name = os.path.join(args.directory, args.input_file)

    print(f"Reading df {df_file_name}")

    lf = pl.read_parquet(df_file_name)
    lf = lf.with_columns(
        si_sqrt=pl.col("si").sqrt().cast(pl.Float64),
        csi_sqrt=pl.col("csi").sqrt().cast(pl.Float64),
    ).filter(pl.col("csi_sqrt").ge(6.3))

    print(f"Writing df {df_file_name}")
    lf.write_parquet(df_file_name)
    return


if __name__ == "__main__":
    print(f"Executing {os.path.abspath(os.path.dirname(__file__))}")
    main()
