import argparse
import os

import polars as pl


def main():
    """
    simple script to add si_sqrt and csi_sqrt columns
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    output_file = os.path.join(args.directory, "curve.dat")

    extend_dir = os.path.join(args.directory, "extended_curves/")

    with open(output_file, "w") as file:
        for file_idx, file_name in enumerate(os.listdir(extend_dir)):
            file_name = os.path.join(extend_dir, file_name)
            df = pl.read_parquet(file_name)

            xs = df["x"]
            ys = df["y"]

            for idx in range(len(xs)):
                x = xs[idx]
                y = ys[idx]

                if idx == 0 and file_idx != 0:
                    file.write("\n")

                file.write(f"{x} {y}\n")


if __name__ == "__main__":
    main()
