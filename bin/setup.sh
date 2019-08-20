#!/usr/bin/env bash

export XCLUSTER_ROOT=`pwd`
export XCLUSTER_DATA=$XCLUSTER_ROOT/data
export XCLUSTER_JARPATH=$XCLUSTER_ROOT/target/xcluster-0.1-SNAPSHOT-jar-with-dependencies.jar
export PYTHONPATH=$XCLUSTER_ROOT/src/python:$PYTHONPATH
export PATH=$XCLUSTER_ROOT/dep/apache-maven-3.6.1/bin:$PATH

if [ ! -f $XCLUSTER_ROOT/.gitignore ]; then
    echo ".gitignore" > $XCLUSTER_ROOT/.gitignore
    echo "target" >> $XCLUSTER_ROOT/.gitignore
    echo ".idea" >> $XCLUSTER_ROOT/.gitignore
    echo "__pycache__" >> $XCLUSTER_ROOT/.gitignore
    echo "dep" >> $XCLUSTER_ROOT/.gitignore
    echo "data" >> $XCLUSTER_ROOT/.gitignore
    echo "test_out" >> $XCLUSTER_ROOT/.gitignore
    echo "experiments_out" >> $XCLUSTER_ROOT/.gitignore
    echo ".DS_STORE" >> $XCLUSTER_ROOT/.gitignore
    echo "*.iml" >> $XCLUSTER_ROOT/.gitignore
fi
