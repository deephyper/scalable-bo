#!/bin/bash

set -xe

all_datasets=("APSFailure" "Amazon_employee_access"  "Australian" "Fashion-MNIST" "KDDCup09_appetency" "MiniBooNE" "adult" "airlines" "albert" "bank-marketing" "blood-transfusion-service-center" "car" "christine" "cnae-9" "connect-4" "covertype" "credit-g" "dionis" "fabert" "helena" "higgs" "jannis" "jasmine" "jungle_chess_2pcs_raw_endgame_complete" "kc1" "kr-vs-kp" "mfeat-factors" "nomao" "numerai28.6" "phoneme" "segment" "shuttle" "sylvine" "vehicle" "volkert")


for dataset in ${all_datasets[@]}; do
    export dataset=$dataset
    echo "Processing $dataset"
    ./sequential-random.sh
done