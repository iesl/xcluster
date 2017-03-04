#!/usr/bin/env bash

set -exu

$XCLUSTER_ROOT/bin/data_processing/download_aloi.sh
$XCLUSTER_ROOT/bin/data_processing/download_glass.sh