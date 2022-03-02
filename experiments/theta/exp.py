import warnings

warnings.simplefilter("ignore")

import argparse

import scalbo.benchmark.ackley

PROBLEMS = {
    "ackley": scalbo.benchmark.ackley
}


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--log-dir",
        type=str,
        default="output",
        help="Logging directory to store produced outputs.",
    )
    parser.add_argument(
        "--timeout"
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=-1,
        help="Number of iterations to run for each optimization.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=list(PROBLEMS.keys()),
        required=True,
        help="Problem on which to experiment.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="Wether to activate or not the verbose mode.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Control the random-state of the algorithm."
    )

    return parser


def main(args):

    ...


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    main(args)