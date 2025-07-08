import argparse
import os

import polars as pl


def main():
    """
    simple script to add si_sqrt and csi_sqrt columns
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-i", "--input_file", default="raw.parquet")
    args = parser.parse_args()

    df_file_name = os.path.join(args.directory, args.input_file)

    print(f"Reading df {df_file_name}")

    lf = pl.read_parquet(df_file_name)
    lf = (
        lf.with_columns(
            fast_shifted=(pl.col("fast") - pl.col("fast").min()).cast(pl.Float64),
            slow_shifted=(pl.col("slow") - pl.col("slow").min()).cast(pl.Float64),
            # slow=pl.col("slow").cast(pl.Float64),
        )
        .with_columns(
            fast_over_slow=(pl.col("fast_shifted") / pl.col("slow_shifted")).cast(
                pl.Float64
            ),
            fast_plus_slow=(pl.col("fast_shifted") + pl.col("slow_shifted")).cast(
                pl.Float64
            ),
            fast_minus_slow=(pl.col("fast_shifted") - pl.col("slow_shifted")).cast(
                pl.Float64
            ),
        )
        .with_columns(
            diff_over_sum=pl.col("fast_minus_slow") / pl.col("fast_plus_slow")
        )
        .with_columns(
            sqrt_diff_over_sum=pl.col("diff_over_sum").sqrt(),
            sqrt_sum=pl.col("fast_plus_slow").sqrt(),
        )
        .with_columns(
            scaled_fps=pl.col("fast_plus_slow") / pl.col("fast_plus_slow").std(),
            scaled_dos=pl.col("diff_over_sum") / pl.col("diff_over_sum").std(),
        )
        .with_columns(
            sqrt_fps=(pl.col("scaled_fps") - pl.col("scaled_fps").min()).sqrt(),
            sqrt_dos=(pl.col("diff_over_sum") - pl.col("diff_over_sum").min()).sqrt(),
        )
        .filter(pl.col("scaled_dos").gt(-0.3))
        .filter(pl.col("sqrt_fps").gt(0.164))
    )

    print(f"Writing df {df_file_name}")
    lf.write_parquet(df_file_name)
    return


if __name__ == "__main__":
    print(f"Executing {os.path.abspath(os.path.dirname(__file__))}")
    main()
