#!/usr/bin/env bash

set -exu

predicted=$1
gold=$2
algorithm=$3
dataset=$4
expected_dp_point_file=${5:-"None"}

java -Xmx50G -cp $XCLUSTER_JARPATH xcluster.eval.EvalPairwise \
--predicted $predicted --gold $gold --algorithm $algorithm \
--dataset $dataset --id-file $expected_dp_point_file --threads 24