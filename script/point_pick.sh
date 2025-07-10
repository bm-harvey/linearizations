#!/bin/bash

dir=$1 

python py/find_ridges.py $dir
python py/pick_clusters.py $dir
