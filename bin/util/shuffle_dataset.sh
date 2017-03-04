#!/usr/bin/env bash

set -exu

dataset=$1
num_shuffles=$2

seed=33
rseed=$seed
pshuf() { perl -MList::Util=shuffle -e "srand($1); print shuffle(<>);" "$2"; }
for i in `seq 1  $num_shuffles`
do
    shuffled_data="${dataset}.$i"
    if ! [ -f $shuffled_data ]; then
        echo "Shuffling $dataset > $shuffled_data"
        pshuf $rseed $dataset > $shuffled_data
    fi
    rseed=$((seed + i))
done