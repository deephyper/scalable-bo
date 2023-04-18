#!/bin/bash

cat /etc/os-release &> "infos/$1-os.out"
./collect_environment.sh &> "infos/$1-env.out"
