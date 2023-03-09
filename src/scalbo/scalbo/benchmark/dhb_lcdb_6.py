import functools

import deephyper_benchmark as dhb

dhb.load("LCu/lcdb")

from deephyper_benchmark.lib.lcu.lcdb import hpo

hp_problem = hpo.problem
run = functools.partial(hpo.run, task_id=6)
