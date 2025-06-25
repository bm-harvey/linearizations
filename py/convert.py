import argparse
import os

from hepconvert import merge_root, root_to_parquet


def main():
    """
    simple script to add si_sqrt and csi_sqrt columns
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default="data/trav_det_9/")
    parser.add_argument("-i", "--input_file", default="raw.parquet")
    args = parser.parse_args()

    # root_files_name = os.path.join(args.directory, "root/*.root")
    root_files_dir = os.path.join(args.directory, "root")
    out_file_name = os.path.join(args.directory, "raw.parquet")

    in_files = [
        os.path.join(root_files_dir, file) for file in os.listdir(root_files_dir)
    ]
    merge_root(
        destination="data/trav_det_9/raw.root",
        files=in_files,
        force=True,
        progress_bar=False
    )
    root_to_parquet(
        in_file="data/trav_det_9/raw.root",
        out_file=out_file_name,
        force=True,
        tree="T",
    )
    # root_to_parquet(in_file=root_files_name, out_file=out_file_name)


if __name__ == "__main__":
    print(f"Executing {os.path.abspath(os.path.dirname(__file__))}")
    main()
