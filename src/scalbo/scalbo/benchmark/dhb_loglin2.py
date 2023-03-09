import deephyper_benchmark as dhb

dhb.load("LCu/loglin2")

from deephyper_benchmark.lib.lcu.loglin2 import hpo

hp_problem = hpo.problem
run = hpo.run
