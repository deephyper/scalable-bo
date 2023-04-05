import deephyper_benchmark as dhb

dhb.load("LCBench/lcbench")

from deephyper_benchmark.lib.lcbench.lcbench import hpo

hp_problem = hpo.problem
run = hpo.run