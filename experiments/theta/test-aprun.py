import warnings

warnings.simplefilter("ignore")

import argparse
import pathlib

import os

import yaml

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

print("scalbo.benchmark.ackley")
from scalbo.benchmark import ackley

print("scalbo.search.ambs")
from scalbo.search import ambs, dmbs

PROBLEMS = {"ackley": ackley}

SEARCHES = {
    "ambs": ambs,  # Centralized Model-Based Search (Master-Worker)
    "dmbs": dmbs,  # Fully Distributed Model-Based Search
}


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Problem on which to experiment.",
    )
    parser.add_argument(
        "--search",
        type=str,
        required=True,
        help="Search the experiment must be done with.",
    )
    parser.add_argument(
        "--sync",
        type=int,
        choices=[0, 1],
        default=0,
        help="If the search workers must be syncronized or not.",
    )
    parser.add_argument(
        "--liar-strategy",
        type=str,
        default="boltzmann",
        choices=["cl_max", "topk", "boltzmann"],
        help="The liar strategy the optimizer must use.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
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
        help="Path to use to cache logged outputs (e.g., /dev/shm/)."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="Wether to activate or not the verbose mode.",
    )

    return parser


def main(args):

    argments= [
        args.problem,
        args.search,
        str(args.sync),
        args.liar_strategy,
        str(args.timeout),
        str(args.max_evals),
        str(args.random_state),
        args.log_dir,
        args.cache_dir
    ]

    print(f"args: {'-'.join(argments)}")


if __name__ == "__main__":

    parser = create_parser()

    args = parser.parse_args()

    main(args)
