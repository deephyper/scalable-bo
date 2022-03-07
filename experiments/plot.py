import argparse
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml
import pandas as pd
import inspect
from scipy import stats
import numpy as np

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

width = 8
height = width / 1.618

matplotlib.rcParams.update({
    'font.size': 21,
    'figure.figsize': (width, height),
    'figure.facecolor': 'white',
    'savefig.dpi': 72,
    'figure.subplot.bottom': 0.125,
    'figure.edgecolor': 'white',
    'xtick.labelsize': 21,
    'ytick.labelsize': 21
})

HERE = os.path.dirname(os.path.abspath(__file__))
FILE_EXTENSION = "pdf"


def yaml_load(path):
    with open(path, "r") as f:
        yaml_data = yaml.load(f, Loader=Loader)
    return yaml_data

def yaml_dump(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)


def load_results(exp_root: str, exp_config: dict) -> dict:
    data = {}
    for exp_prefix in exp_config["data"]:
        if "rep" in exp_config["data"][exp_prefix]:
            dfs = []
            for rep in exp_config["data"][exp_prefix].get("rep"):
                exp_results_path = os.path.join(exp_root,
                                                f"{exp_prefix}-{rep}/results.csv")
                df = pd.read_csv(exp_results_path)
                dfs.append(df)
                data[exp_prefix] = dfs
        else:
            exp_results_path = os.path.join(exp_root, f"{exp_prefix}/results.csv")
            df = pd.read_csv(exp_results_path)
            data[exp_prefix] = df
    return data


@ticker.FuncFormatter
def hour_major_formatter(x, pos):
    x = float(f"{x/3600:.2f}")
    if x % 1 == 0:
        x = str(int(x))
    else:
        x = f"{x:.2f}"
    return x


def plot_scatter_multi(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            for i, exp_df in enumerate(exp_dfs):
                x, y = exp_df.timestamp_end.to_numpy(
                ), -exp_df.objective.to_numpy()

                plt_kwargs = dict(color=exp_config["data"][exp_name]["color"],
                                  s=10,
                                  alpha=0.5)
                if i == 0:
                    plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

                plt.scatter(x, y, **plt_kwargs)
        else:
            x, y = exp_df.timestamp_end.to_numpy(), -exp_df.objective.to_numpy()

            plt.scatter(x,
                        y,
                        color=exp_config["data"][exp_name]["color"],
                        label=exp_config["data"][exp_name]["label"],
                        s=10)

    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    # ax.xaxis.set_major_formatter(hour_major_formatter)

    if exp_config.get("title"):
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel("Instance run time (sec)")
    plt.xlabel("Search time (hour)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def only_min(values):
    res = [values[0]]
    for value in values[1:]:
        res.append(min(res[-1], value))
    return np.array(res)


def plot_objective_multi(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:

            exp_dfs = exp_df

            times = np.unique(
                np.concatenate([df.timestamp_end.to_numpy() for df in exp_dfs],
                               axis=0))
            times = np.concatenate([[0], times, [3600]])

            series = []
            for exp_df in exp_dfs:

                exp_df = exp_df.sort_values("timestamp_end")
                x, y = exp_df.timestamp_end.to_numpy(
                ), -exp_df.objective.to_numpy()
                y = only_min(y)

                s = pd.Series(data=y, index=x)
                s = s.reindex(times).fillna(method="ffill").fillna(method="bfill")
                series.append(s)

            array = np.array([s.to_numpy() for s in series])
            loc = np.nanmean(array, axis=0)
            # scale = np.nanstd(array, axis=0)
            loc_max = np.nanmax(array, axis=0)
            loc_min = np.nanmin(array, axis=0)

            plt.plot(
                times,
                loc,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )
            plt.fill_between(times,
                             loc_min,
                             loc_max,
                             facecolor=exp_config["data"][exp_name]["color"],
                             alpha=0.3)
        else:
            exp_df = exp_df.sort_values("timestamp_end")
            x, y = exp_df.timestamp_end.to_numpy(), -exp_df.objective.to_numpy()
            y = only_min(y)

            plt.plot(x,
                     y,
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    # ax.xaxis.set_major_formatter(hour_major_formatter)

    if exp_config.get("title"):
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel("Instance run time (sec)")
    plt.xlabel("Search time (hour)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_objective_multi_iter(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            for i, exp_df in enumerate(exp_dfs):
                exp_df = exp_df.sort_values("timestamp_end")
                x, y = list(range(1,
                                  len(exp_df.timestamp_end.to_list()) +
                                  1)), (-exp_df.objective).to_list()

                y = only_min(y)

                plt_kwargs = dict(color=exp_config["data"][exp_name]["color"],
                                  linestyle=exp_config["data"][exp_name].get(
                                      "linestyle", "-"))

                if i == 0:
                    plt_kwargs["label"] = label = exp_config["data"][exp_name][
                        "label"]

                plt.plot(x, y, **plt_kwargs)
        else:
            exp_df = exp_df.sort_values("timestamp_end")
            x, y = list(range(1,
                              len(exp_df.timestamp_end.to_list()) +
                              1)), (-exp_df.objective).to_list()

            y = only_min(y)

            plt.plot(x,
                     y,
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    if exp_config.get("title"):
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel("Experiment Duration (sec.)")
    plt.xlabel("#Evaluation")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def compile_profile(df):
    history = []

    df = df.sort_values("timestamp_end")

    for _, row in df.iterrows():
        history.append((row['timestamp_start'], 1))
        history.append((row['timestamp_end'], -1))

    # history = sorted(history, key=lambda v: v[0])
    nb_workers = 0
    timestamp = []
    n_jobs_running = []
    for time, incr in history:
        nb_workers += incr
        timestamp.append(time)
        n_jobs_running.append(nb_workers)
    
    return timestamp, n_jobs_running


def plot_utilization_multi_iter(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            ...
            # exp_dfs = exp_df
            # for i, exp_df in enumerate(exp_dfs):
            #     exp_df = exp_df.sort_values("timestamp_end")
            #     x, y = list(range(1,
            #                       len(exp_df.timestamp_end.to_list()) +
            #                       1)), (-exp_df.objective).to_list()

            #     y = only_min(y)

            #     plt_kwargs = dict(color=exp_config["data"][exp_name]["color"],
            #                       linestyle=exp_config["data"][exp_name].get(
            #                           "linestyle", "-"))

            #     if i == 0:
            #         plt_kwargs["label"] = label = exp_config["data"][exp_name][
            #             "label"]

            #     plt.plot(x, y, **plt_kwargs)
        else:
            # exp_df = exp_df.sort_values("timestamp_end")
            x, y = compile_profile(exp_df)

            plt.step(x,
                     y,
                     where="pre",
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    # ax.xaxis.set_major_formatter(hour_major_formatter)

    if exp_config.get("title"):
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel("Worker Utilization")
    plt.xlabel("Time (sec.)")

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def write_infos(df, exp_config, output_dir):
    output_file_name = f"infos.yaml"
    output_path = os.path.join(output_dir, output_file_name)

    infos = {}

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
        else:
            infos[exp_name] = {}
            infos[exp_name]["num_evaluations"] = len(exp_df)
            
            yaml_dump(output_path, infos)


def generate_figures(config):

    exp_root = config["data-root"]
    figures_dir = config.get("figures-root", "figures")
    show = config.get("show", False)

    for exp_num, exp_config in config["experiments"].items():
        exp_dirname = str(exp_num)
        output_dir = os.path.join(figures_dir, exp_dirname)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        df = load_results(exp_root, exp_config)

        write_infos(df, exp_config, output_dir)

        plot_functions = [
            plot_scatter_multi,
            plot_objective_multi,
            plot_objective_multi_iter,
            plot_utilization_multi_iter,

        ]

        for plot_func in plot_functions:
            plot_func(df, exp_config, output_dir, show)


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to plot experiments results.")

    parser.add_argument(
        "--config",
        type=str,
        default="plot.yaml",
        help="Plotter configuration YAML file.",
    )

    return parser

if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    # yaml_path = os.path.join(HERE, "plot.yaml")
    yaml_path = args.config
    config = yaml_load(yaml_path)
    generate_figures(config)
    print("Done!")
