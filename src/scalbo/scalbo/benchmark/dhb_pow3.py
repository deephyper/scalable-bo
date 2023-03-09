import deephyper_benchmark as dhb

dhb.load("LCu/pow3")

from deephyper_benchmark.lib.lcu.pow3 import hpo

hp_problem = hpo.problem
run = hpo.run
