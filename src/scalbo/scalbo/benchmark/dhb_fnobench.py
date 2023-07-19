import deephyper_benchmark as dhb

dhb.load("FNOBench")

from deephyper_benchmark.lib.fnobench import hpo

hp_problem = hpo.problem
run = hpo.run
