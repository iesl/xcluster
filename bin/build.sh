#!/usr/bin/env bash

set -exu

pushd $XCLUSTER_ROOT
mvn clean package
popd