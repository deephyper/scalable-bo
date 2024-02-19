import warnings

warnings.simplefilter("ignore")

import argparse
import pathlib
import importlib
import sys

PROBLEMS = {
    "fast_ackley_2": "scalbo.benchmark.fast_ackley_2",
    "fastest_ackley_2": "scalbo.benchmark.fastest_ackley_2",
    "hb_sim": "scalbo.benchmark.hb_sim",
    "dackley_2": "scalbo.benchmark.dackley_2",
    "dackley_5": "scalbo.benchmark.dackley_5",
    "ackley_5": "scalbo.benchmark.ackley_5",
    "ackley_10": "scalbo.benchmark.ackley_10",
    "ackley_30": "scalbo.benchmark.ackley_30",
    "ackley_50": "scalbo.benchmark.ackley_50",
    "ackley_100": "scalbo.benchmark.ackley_100",
    "hartmann6D": "scalbo.benchmark.hartmann6D",
    "levy": "scalbo.benchmark.levy",
    "griewank": "scalbo.benchmark.griewank",
    "schwefel": "scalbo.benchmark.schwefel",
    "frnn": "scalbo.benchmark.frnn",
    "minimalistic-frnn": "scalbo.benchmark.minimalistic_frnn",
    "molecular": "scalbo.benchmark.molecularmpnn",
    "candle_attn": "scalbo.benchmark.candle_attn",
    "candle_attn_sim": "scalbo.benchmark.candle_attn_sim",
    "candle_combo": "scalbo.benchmark.candle_combo",
    "dhb_combo": "scalbo.benchmark.dhb_combo",
    "dhb_slicelocalization": "scalbo.benchmark.dhb_slicelocalization",
    "dhb_navalpropulsion": "scalbo.benchmark.dhb_navalpropulsion",
    "dhb_proteinstructure": "scalbo.benchmark.dhb_proteinstructure",
    "dhb_parkinsonstelemonitoring": "scalbo.benchmark.dhb_parkinsonstelemonitoring",
    "dhb_pow3": "scalbo.benchmark.dhb_pow3",
    "dhb_loglin2": "scalbo.benchmark.dhb_loglin2",
    "dhb_lcdb_6": "scalbo.benchmark.dhb_lcdb_6",
    "dhb_lcdb_293": "scalbo.benchmark.dhb_lcdb_293",
    "dhb_lcdb_351": "scalbo.benchmark.dhb_lcdb_351",
    "dhb_lcdb_354": "scalbo.benchmark.dhb_lcdb_354",
    "dhb_lcdb_41150": "scalbo.benchmark.dhb_lcdb_41150",
    "dhb_lcbench": "scalbo.benchmark.dhb_lcbench",
    "test": "scalbo.benchmark.test",
    "dhb_fnobench": "scalbo.benchmark.dhb_fnobench",
    "dhb_diffreact": "scalbo.benchmark.dhb_diffreact",
}

SEARCHES = {
    "BO": "scalbo.search.bo",  # Serial Bayesian Optimization
    "CBO": "scalbo.search.cbo",  # Centralized Model-Based Search (Master-Worker)
    "DBO": "scalbo.search.dbo",  # Fully Distributed Model-Based Search
    "OPT-TPE": "scalbo.search.optuna_tpe",  # TPE
    "OPT-NSGAII": "scalbo.search.optuna_nsgaii",  # NSGAII
    "OPT-RDM": "scalbo.search.optuna_random",  # RANDOM
}


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--problem",
        type=str,
        choices=list(PROBLEMS.keys()),
        required=True,
        help="Problem on which to experiment.",
    )
    parser.add_argument(
        "--search",
        type=str,
        choices=list(SEARCHES.keys()),
        required=True,
        help="Search the experiment must be done with.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["RF", "GP", "DUMMY", "MF", "ET"],
        default=None,
        help="Surrogate model used by the Bayesian optimizer.",
    )
    parser.add_argument(
        "--sync",
        type=bool,
        default=False,
        help="If the search workers must be syncronized or not.",
    )
    parser.add_argument(
        "--acq-func",
        type=str,
        default="UCB",
        help="Acquisition funciton to use.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="cl_max",
        help="The strategy for multi-point acquisition.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Search maximum duration (in min.) for each optimization.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=-1,
        help="Number of iterations to run for each optimization.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Control the random-state of the algorithm.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="output",
        help="Logging directory to store produced outputs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to use to cache logged outputs (e.g., /dev/shm/).",
    )
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     type=bool,
    #     default=False,
    #     help="Wether to activate or not the verbose mode.",
    # )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="The number of parallel processes to use to fit the surrogate model.",
    )
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default="mpi",
        help="Communication backend to use when using DBO",
    )
    parser.add_argument(
        "--pruning-strategy",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--scheduler",
        type=bool,
        default=False,
    )
    parser.add_argument("--scheduler-periode", type=int, default=25)
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument(
        "--filter-duplicated",
        type=bool,
        default=False,
    )
    parser.add_argument("--objective-scaler", type=str, default="identity")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--interval-steps", type=int, default=1)
    parser.add_argument("--scalar-func", type=str, default="Chebyshev")
    parser.add_argument("--lower-bounds", type=str, default=None)
    parser.add_argument("--acq-func-optimizer", type=str, default="sampling")
    return parser


def main(args):
    args = vars(args)

    # load the problem
    args["problem"] = importlib.import_module(PROBLEMS.get(args["problem"]))
    search = importlib.import_module(SEARCHES.get(args.pop("search")))

    pathlib.Path(args["log_dir"]).mkdir(parents=True, exist_ok=True)

    search.execute(**args)


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    # delete arguments to avoid conflicts
    sys.argv = [sys.argv[0]]

    main(args)
