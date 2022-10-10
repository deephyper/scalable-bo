#!/bin/bash
gpustat --no-color -i >> "${GPUSTAT_LOG_DIR}/gpustat.${PMI_RANK}.txt"
