#!/bin/bash

dir=$1 

#python py/find_ridges.py -d $dir --x_bins 2048 --y_bins 2048 --sigma 2 --threshold 3.e-3
python py/find_ridges.py -d $dir --x_bins 1024 --y_bins 1024 --sigma 3  --threshold 3.e-4
python py/pick_clusters.py -d $dir
