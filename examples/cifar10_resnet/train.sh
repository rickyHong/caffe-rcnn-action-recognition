#!/usr/bin/env sh
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is use first 4 gpus"
    gpu="0,1,3,4"
else
    echo "use $1-th gpu"
    gpu=$1
fi

# Batch = 64, use four gpus
TOOLS=./build/tools

GLOG_log_dir=examples/cifar10_resnet/log $TOOLS/caffe train --gpu $gpu \
    --solver=examples/cifar10_resnet/solver.proto
