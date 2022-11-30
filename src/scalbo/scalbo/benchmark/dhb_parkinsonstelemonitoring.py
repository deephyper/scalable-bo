import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular/parkinsonstelemonitoring")

from deephyper_benchmark.lib.hpobench.tabular.parkinsonstelemonitoring import hpo

hp_problem = hpo.problem
run = hpo.run
