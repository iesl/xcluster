#!/usr/bin/env bash

set -xu

mkdir $XCLUSTER_ROOT/dep
pushd $XCLUSTER_ROOT/dep
wget http://mirror.cc.columbia.edu/pub/software/apache/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
tar -xvf apache-maven-3.3.9-bin.tar.gz
popd