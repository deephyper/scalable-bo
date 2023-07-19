import os

os.environ["DEEPHYPER_BENCHMARK_TASK"] = "proteinstructure"

import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular")

from deephyper_benchmark.lib.hpobench.tabular import hpo

hp_problem = hpo.problem
run = hpo.run
