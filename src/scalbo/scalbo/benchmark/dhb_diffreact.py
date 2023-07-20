import deephyper_benchmark as dhb

dhb.load("PINNBench/DiffusionReaction")

from deephyper_benchmark.lib.pinnbench.diffusionreaction import hpo

hp_problem = hpo.problem
run = hpo.run
