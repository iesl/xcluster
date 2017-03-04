#!/usr/bin/env bash

set -xu

exp_dir=$1

find $exp_dir -name score.txt -exec cat {} \; > $exp_dir/all_scores.txt

python bin/util/format_result_table.py $exp_dir/all_scores.txt > $exp_dir/dendrogram_purity.tex

cat $exp_dir/dendrogram_purity.tex

echo "Dendrogram Purity Result table saved here: $exp_dir/dendrogram_purity.tex"


find $exp_dir -name "running_time.*" -exec cat {} \; > $exp_dir/all_running_times.txt

python bin/util/format_result_table.py $exp_dir/all_running_times.txt > $exp_dir/running_times.tex

cat $exp_dir/running_times.tex

echo "Running Time Result table saved here: $exp_dir/running_times.tex"


find $exp_dir -name "pairwise.*" -exec cat {} \; > $exp_dir/all_pairwise.txt

cut -f 1,2,5 $exp_dir/all_pairwise.txt > $exp_dir/all_f1.txt

python bin/util/format_result_table.py $exp_dir/all_f1.txt > $exp_dir/all_f1.tex

cat $exp_dir/all_f1.tex

echo "F1 result table saved here: $exp_dir/f1.tex"