#!/usr/bin/env bash

set -exu

pushd $XCLUSTER_DATA
wget https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data
cat glass.data | tr ',' '\t' | cut -f 2- | awk '{print $10 "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8 "\t" $9 "\t" }' | awk '{print NR-1 "\t" $0}' > glass.tsv
popd