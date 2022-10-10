import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

import deephyper_benchmark as dhb


dhb.load("ECP-Candle/Pilot1/Combo")


# Run training with Testing scores
from deephyper_benchmark.lib.ecp_candle.pilot1.combo import model
from deephyper_benchmark.lib.ecp_candle.pilot1.combo import hpo


params = {
    "epochs": 100,
    "timeout": 60 * 60, # 60 minutes per model
    "verbose": True
}
config = hpo.problem.default_configuration
params.update(config)

res = model.run_pipeline(params, mode="test")
print(f"{res=}")


