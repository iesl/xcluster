#!/usr/bin/env bash

set -exu

tree=$1
algorithm=$2
dataset=$3
threads=${4:-24}
expected_dp_point_file=${5:-"None"}

java -Xmx50G -cp $XCLUSTER_JARPATH xcluster.eval.EvalDendrogramPurity \
--input $tree --algorithm $algorithm --dataset $dataset --threads $threads \
--print true --id-file $expected_dp_point_file