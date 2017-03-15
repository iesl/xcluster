#!/usr/bin/env bash

set -exu

output_dir=${1:-"experiments_out"}

dataset_file=$XCLUSTER_ROOT/data/glass.tsv
dataset_name=glass

num_runs=10
num_threads=24

mkdir -p $output_dir

# Shuffle
sh bin/util/shuffle_dataset.sh $dataset_file $num_runs

expected_dp_point_file="None"

for i in `seq 1  $num_runs`
    do
        algorithm_name="Perch-py"
        shuffled_data="${dataset_file}.$i"
        exp_output_dir="$output_dir/$dataset_name/$algorithm_name/run_$i"

        python -m xcluster.eval.eval_dataset --input $shuffled_data --outdir $exp_output_dir \
        --algorithm $algorithm_name --dataset $dataset_name --max_leaves None --clusters None

        sh bin/util/score_tree.sh \
        $exp_output_dir/tree.tsv $algorithm_name $dataset_name $num_threads $expected_dp_point_file \
         > $exp_output_dir/score.txt

        cat $exp_output_dir/score.txt
done
