#!/bin/bash

export seeds=(14 60 71 92)
export ranks=(2 4 8 16)


for rank in ${ranks[@]}; do
    for seed in ${seeds[@]}; do
        exp_file="ackley_5-DBO-async-RF-qUCB-qUCB-128-$rank-1800-$seed.sh";
        echo $exp_file;
        qsub $exp_file
    done;
done