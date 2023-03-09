import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular/navalpropulsion")

from deephyper_benchmark.lib.hpobench.tabular.navalpropulsion import hpo

hp_problem = hpo.problem
run = hpo.run