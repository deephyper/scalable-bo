import json
import logging
import os

HERE = os.path.dirname(os.path.abspath(__file__))

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


def load_json(f):
    with open(f, "r") as f:
        js_data = json.load(f)
    return js_data


path = os.path.join(
    HERE,
    "output",
    "dhb_combo-DBOS4M-async-RF-qUCB-qUCB-128-4-10800-42",
    "best-config.json",
)
config = load_json(path)["0"]
config = {k[2:]:v for k,v in config.items() if "p:" == k[:2]} # filter the hyperparameters


params = {"epochs": 100, "timeout": 60 * 60, "verbose": True}  # 60 minutes per model
params.update(config)

res = model.run_pipeline(params, mode="test")
print(f"{res=}")
