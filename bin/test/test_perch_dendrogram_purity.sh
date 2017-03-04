#!/usr/bin/env bash

set -exu

output_dir=${1:-"test_out"}

dataset_files=( $XCLUSTER_ROOT/data/separated-2.tsv $XCLUSTER_ROOT/data/separated-10.tsv $XCLUSTER_ROOT/data/separated-100.tsv $XCLUSTER_ROOT/data/separated-1000.tsv )

num_runs=1
num_threads=4
par_max_frontier=50

mkdir -p $output_dir

# Shuffle

expected_dp_point_file="None"


for dataset_file in "${dataset_files[@]}"
do
    for i in `seq 1  $num_runs`
        do
            algorithm_name="Perch"
            dataset_name=`basename $dataset_file`
            sh bin/util/shuffle_dataset.sh $dataset_file $num_runs
            shuffled_data="${dataset_file}.$i"
            exp_output_dir="$output_dir/$dataset_name/$algorithm_name/run_$i"

            java -Xmx20G -cp $XCLUSTER_JARPATH xcluster.eval.RunPerch --input $shuffled_data --outdir $exp_output_dir \
            --algorithm $algorithm_name --dataset $dataset_name --max-leaves None --clusters None --threads $num_threads \
            --max-frontier-par $par_max_frontier

            sh bin/util/score_tree.sh \
            $exp_output_dir/tree.tsv $algorithm_name $dataset_name $num_threads $expected_dp_point_file \
             > $exp_output_dir/score.txt

            cat $exp_output_dir/score.txt
    done
done

find $output_dir -name score.txt -exec cat {} \; > $output_dir/all_scores.txt

set +x

echo ""
echo ""
echo ""
echo "------------------------------"
echo "Dendrogram purity results: "
column -t $output_dir/all_scores.txt
echo ""
echo "All tests should have perfect (1.0) dendrogram purity"
echo "------------------------------"
echo ""