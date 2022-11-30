import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular/proteinstructure")

from deephyper_benchmark.lib.hpobench.tabular.proteinstructure import hpo

hp_problem = hpo.problem
run = hpo.run