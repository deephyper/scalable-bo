import deephyper_benchmark as dhb

dhb.load("ECP-Candle/Pilot1/Combo")

from deephyper_benchmark.lib.ecp_candle.pilot1.combo import hpo

hp_problem = hpo.problem
run = hpo.run