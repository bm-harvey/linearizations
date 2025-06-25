#!/bin/bash

dir=$1 

python py/find_ridges.py -d $dir
#python py/find_ridges.py -d $dir --x_bins 1536 --y_bins 1536 --sigma 3  --threshold 3.e-3
#python py/find_ridges.py -d $dir --x_bins 1536 --y_bins 1536 --sigma 3  --threshold 3.e-4
#python py/find_ridges.py -d $dir --x_bins 1024 --y_bins 1024 --sigma 3  --threshold 3.e-4
python py/pick_clusters.py -d $dir
