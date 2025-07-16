#!/bin/bash

# ensure the rust scripts are up to date 
cargo build --release

dir=$1

./target/release/lowess_smooth -d $dir
./target/release/linearize -d $dir  

python ./py/make_limits.py $dir 
