#!/usr/bin/env bash

set -exu

tree=$1
output=$2

java -Xmx20G -cp $XCLUSTER_JARPATH xcluster.utils.Graphviz $tree $output
