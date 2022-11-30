import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular/slicelocalization")

from deephyper_benchmark.lib.hpobench.tabular.slicelocalization import hpo

hp_problem = hpo.problem
run = hpo.run