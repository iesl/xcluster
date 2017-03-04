#!/usr/bin/env bash


pushd $XCLUSTER_DATA
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/aloi.scale.bz2
bzcat aloi.scale.bz2 > aloi.scale.raw
python $XCLUSTER_ROOT/bin/data_processing/process_aloi.py aloi.scale.raw aloi.tsv
popd
