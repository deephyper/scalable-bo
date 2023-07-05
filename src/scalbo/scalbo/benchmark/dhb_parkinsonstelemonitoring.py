import os

os.environ["DEEPHYPER_BENCHMARK_TASK"] = "parkinsonstelemonitoring"
os.environ["DEEPHYPER_BENCHMARK_MOO"] = "1"

import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular")

from deephyper_benchmark.lib.hpobench.tabular import hpo

hp_problem = hpo.problem
run = hpo.run
