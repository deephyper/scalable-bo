#!/bin/bash
#COBALT -n 1
#COBALT -t 10
#COBALT -q full-node
#COBALT -A datascience
#COBALT --attrs mig-mode=True

export DEVICES=(0 1 2 3 4 5 6 7)

for device_id in ${DEVICES[@]}; do
    # Creates GPU instances (GI)
    nvidia-smi mig -i $device_id -cgi 9,9;

    # Creates Compute instances (CI)
    nvidia-smi mig -i $device_id -gi 1 -cci 0;
    nvidia-smi mig -i $device_id -gi 2 -cci 0;
done

touch nvidia-smi.txt
nvidia-smi -L > nvidia-smi.txt

# GPU 0: A100-SXM4-40GB (UUID: GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c)
#   MIG 1c.3g.20gb Device 0: (UUID: MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/1/0)
#   MIG 1c.3g.20gb Device 1: (UUID: MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/2/0)

# CUDA_VISIBLE_DEVICES=MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/1/0 python myapp1.py &
# CUDA_VISIBLE_DEVICES=MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/2/0 python myapp2.py &