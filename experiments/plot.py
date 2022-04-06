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
from scipy.interpolate import interp1d   
import numpy as np

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

width = 5
height = width / 1.618
# height = width * 0.5

matplotlib.rcParams.update({
    'font.size': 12,
    'figure.figsize': (width, height),
    'figure.facecolor': 'white',
    'savefig.dpi': 72,
    'figure.subplot.bottom': 0.125,
    'figure.edgecolor': 'white',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

HERE = os.path.dirname(os.path.abspath(__file__))
FILE_EXTENSION = "pdf"
PRINT_TITLE = False
NEGATIVE = True
MODE = "max"


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
                exp_results_path = os.path.join(exp_root,f"{exp_prefix}-{rep}/results.csv")
                df = pd.read_csv(exp_results_path)

                if NEGATIVE:
                    df.objective = -df.objective

                dfs.append(df)
                data[exp_prefix] = dfs
        else:
            exp_results_path = os.path.join(exp_root, f"{exp_prefix}/results.csv")
            df = pd.read_csv(exp_results_path)
            if NEGATIVE:
                df.objective = -df.objective
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

@ticker.FuncFormatter
def minute_major_formatter(x, pos):
    x = float(f"{x/60:.2f}")
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
            x, y = exp_df.timestamp_end.to_numpy(), exp_df.objective.to_numpy()

            if NEGATIVE:
                y = -y

            plt.scatter(x,
                        y,
                        color=exp_config["data"][exp_name]["color"],
                        label=exp_config["data"][exp_name]["label"],
                        s=1, alpha=0.5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel("Objective")
    plt.xlabel("Search time (min.)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def best_objective(values):

    if MODE == "max":
        f = max
    else:
        f = min

    res = [values[0]]
    for value in values[1:]:
        res.append(f(res[-1], value))
    return res

def count_better(values, baseline_perf):

    count = 0
    res = []

    if MODE == "max":
        for value in values:
            if value > baseline_perf:
                count += 1
            res.append(count)
    else:
        for value in values:
            if value < baseline_perf:
                count += 1
            res.append(count)

    return res


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
            times = np.concatenate([[0], times, [exp_config["t_max"]]])

            series = []
            for exp_df in exp_dfs:

                exp_df = exp_df.sort_values("timestamp_end")
                x, y = exp_df.timestamp_end.to_numpy(
                ), exp_df.objective.to_numpy()
                y = best_objective(y)

                s = pd.Series(data=y, index=x)
                s = s.reindex(times).fillna(method="ffill").fillna(method="bfill")
                series.append(s)

            array = np.array([s.to_numpy() for s in series])
            loc = np.nanmean(array, axis=0)
            scale = np.nanstd(array, axis=0)
            loc_max = loc + scale
            loc_min = loc - scale

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
            x, y = exp_df.timestamp_end.to_numpy(), exp_df.objective.to_numpy()
            if "hartmann6D" in exp_name:
                y = y+3.32237 # hartmann6D
            y = best_objective(y)

            plt.plot(x,
                     y,
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     marker=exp_config["data"][exp_name].get("marker", None),
                     markevery=len(x)//5,
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    if MODE == "min":
        plt.legend(loc="upper right")
        # plt.legend(loc="lower left")
    else:
        plt.legend(loc="lower right")

    plt.ylabel("Objective")
    plt.xlabel("Search time (min.)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    # plt.yscale("log")

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
            max_n_iters = 0
            y_list = []
            for i, exp_df in enumerate(exp_dfs):
                exp_df = exp_df.sort_values("timestamp_end")
                x, y = list(range(1,
                                  len(exp_df.timestamp_end.to_list()) +
                                  1)), (-exp_df.objective).to_list()
                max_n_iters = max(len(x), max_n_iters)

                y = best_objective(y)
                y_list.append(y)

            for i, y in enumerate(y_list):
                y = y + [y[-1]] * (max_n_iters - len(y))
                y_list[i] = y

            y_list = np.asarray(y_list)
            y_mean = y_list.mean(axis=0)
            y_std = y_list.std(axis=0)

            plt_kwargs = dict(color=exp_config["data"][exp_name]["color"],
                                linestyle=exp_config["data"][exp_name].get(
                                    "linestyle", "-"))

            plt_kwargs["label"] = exp_config["data"][exp_name][
                    "label"]

            x = np.arange(max_n_iters)
            plt.plot(x, y_mean, **plt_kwargs)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25, color=exp_config["data"][exp_name]["color"])
        else:
            exp_df = exp_df.sort_values("timestamp_end")
            x, y = list(range(1,
                              len(exp_df.timestamp_end.to_list()) +
                              1)), (-exp_df.objective).to_list()

            y = best_objective(y)

            max_n_iters = len(x)

            plt.plot(x,
                     y,
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel("Objective")
    plt.xlabel("#Evaluation")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    plt.xlim(0, max_n_iters)

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def compile_profile(df):
    history = []

    for _, row in df.iterrows():
        history.append((row['timestamp_start'], 1))
        history.append((row['timestamp_end'], -1))

    history = sorted(history, key=lambda v: v[0])
    nb_workers = 0
    timestamp = [0]
    n_jobs_running = [0]
    for time, incr in history:
        nb_workers += incr
        timestamp.append(time)
        n_jobs_running.append(nb_workers)
    
    return timestamp, n_jobs_running


def compute_num_workers(exp_name):
    exp_name = exp_name.split("-")
    alg_name = exp_name[1]
    num_nodes = int(exp_name[5])
    num_ranks_per_node = int(exp_name[6])
    
    if alg_name == "AMBS":
        return num_nodes * num_ranks_per_node - 1
    else:
        return num_nodes * num_ranks_per_node

def plot_utilization_multi_iter(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            T = np.linspace(0, exp_config["t_max"], 1000)
            y_list = []
            for i, exp_df in enumerate(exp_dfs):
                x, y = compile_profile(exp_df)
                f = interp1d(x, y, kind="previous", fill_value="extrapolate")
                y = f(T)
                y_list.append(y)

            y_list = np.asarray(y_list)
            y_mean = y_list.mean(axis=0)
            y_std = y_list.std(axis=0)

            plt_kwargs = dict(color=exp_config["data"][exp_name]["color"],
                                linestyle=exp_config["data"][exp_name].get(
                                    "linestyle", "-"))

            plt_kwargs["label"] = exp_config["data"][exp_name][
                    "label"]

            plt.plot(T, y_mean, **plt_kwargs)
            plt.fill_between(T, y_mean - y_std, y_mean + y_std, alpha=0.25, color=exp_config["data"][exp_name]["color"])
        else:
            x, y = compile_profile(exp_df)

            num_workers = compute_num_workers(exp_name)
            y = np.asarray(y) / num_workers * 100

            plt.step(x,
                     y,
                     where="post",
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     marker=exp_config["data"][exp_name].get("marker", None),
                     markevery=len(x)//5,
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend(loc="lower left")
    plt.ylabel("Workers Evaluating $f(x)$ (%)")
    plt.xlabel("Search time (min.)")

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])
    
    plt.ylim(0,100)
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

# def plot_strong_scaling(df, exp_config, output_dir, show):
#     output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
#     output_path = os.path.join(output_dir, output_file_name)

#     infos = {}
#     base_exp_name = None

#     plt.figure()

#     for exp_name, exp_df in df.items():

#         if "rep" in exp_config["data"][exp_name]:
#             exp_dfs = exp_df
#         else:
#             infos[exp_name] = {}

#             infos[exp_name]["num_evaluations"] = len(exp_df)

#             num_workers = compute_num_workers(exp_name)
#             infos[exp_name]["num_workers"] = num_workers

#             # available compute time
#             T_avail = exp_config["t_max"] * num_workers
#             T_eff = float((exp_df.timestamp_end - exp_df.timestamp_start).to_numpy().sum())
#             infos[exp_name]["utilization"] = T_eff/T_avail

#             if exp_config.get("baseline", False):
#                 base_exp_name = exp_name

#     # baseline 
#     base_num_workers = infos[base_exp_name]["num_workers"]
#     base_num_evaluations = infos[base_exp_name]["num_evaluations"] / base_num_workers
#     num_workers = [2**i for i in range(13)]
#     linear_scaling = [base_num_evaluations*w for w in num_workers]

#     plt.plot(num_workers, linear_scaling, linestyle="--", color="black", label="baseline")

#     for exp_name, exp_infos in df.items():

#             infos[exp_name] = {}

#             infos[exp_name]["num_evaluations"] = len(exp_df)

#             num_workers = compute_num_workers(exp_name)
#             infos[exp_name]["num_workers"] = num_workers

#             # available compute time
#             T_avail = exp_config["t_max"] * num_workers
#             T_eff = float((exp_df.timestamp_end - exp_df.timestamp_start).to_numpy().sum())
#             infos[exp_name]["utilization"] = T_eff/T_avail

#             if exp_config.get("baseline", False):
#                 base_exp = exp_name
    
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(output_path)
#     if show:
#         plt.show()
#     plt.close()


def write_infos(df, exp_config, output_dir):
    output_file_name = f"infos.yaml"
    output_path = os.path.join(output_dir, output_file_name)

    infos = {}

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
        else:
            infos[exp_name] = {}
            exp_df = exp_df[exp_df.timestamp_end < exp_config["t_max"]]

            infos[exp_name]["num_evaluations"] = len(exp_df)

            num_workers = compute_num_workers(exp_name)
            infos[exp_name]["num_workers"] = num_workers

            # available compute time
            T_avail = exp_config["t_max"] * num_workers
            T_eff = float((exp_df.timestamp_end - exp_df.timestamp_start).to_numpy().sum())
            infos[exp_name]["utilization"] = T_eff/T_avail

            # compute best objective found
            if MODE == "max":
                idx_best = exp_df.objective.argmax()
            else:
                idx_best = exp_df.objective.argmin()

            obj_best = exp_df.objective.iloc[idx_best]
            obj_best_timestamp = exp_df.timestamp_end.iloc[idx_best]

            infos[exp_name]["best_objective"] = float(obj_best)
            infos[exp_name]["best_objective_timestamp"] = float(obj_best_timestamp)
            
    #         if exp_config.get("baseline", False):
    #             base_exp_name = exp_name

    # # baseline 
    # base_num_workers = infos[base_exp_name]["num_workers"]
    # base_num_evaluations = infos[base_exp_name]["num_evaluations"] / base_num_workers
    # num_workers = [2**i for i in range(13)]
    # linear_scaling = [base_num_evaluations*w for w in num_workers]

    yaml_dump(output_path, infos)

def plot_count_better_than_best(df, exp_config, output_dir):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:

            exp_dfs = exp_df
            ...

        else:
            exp_df = exp_df.sort_values("timestamp_end")


            x, y = exp_df.timestamp_end.to_numpy(), exp_df.objective.to_numpy()
            y = count_better(y, exp_config["baseline_best"])

            plt.plot(x,
                     y,
                     label=exp_config["data"][exp_name]["label"],
                     color=exp_config["data"][exp_name]["color"],
                     marker=exp_config["data"][exp_name].get("marker", None),
                     markevery=len(x)//5,
                     linestyle=exp_config["data"][exp_name].get(
                         "linestyle", "-"))

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend(loc="upper left")

    plt.ylabel("Models $>$ Baseline")
    plt.xlabel("Search time (min.)")

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_figures(config):
    global NEGATIVE, MODE

    exp_root = config["data-root"]
    figures_dir = config.get("figures-root", "figures")
    show = config.get("show", False)
    NEGATIVE = config.get("negative", True)
    MODE = config.get("mode", "min")

    for exp_num, exp_config in config["experiments"].items():
        exp_dirname = str(exp_num)
        output_dir = os.path.join(figures_dir, exp_dirname)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        df = load_results(exp_root, exp_config)

        write_infos(df, exp_config, output_dir)

        plot_functions = [
            # plot_scatter_multi,
            plot_objective_multi,
            # plot_objective_multi_iter,
            plot_utilization_multi_iter,

        ]

        for plot_func in plot_functions:
            plot_func(df, exp_config, output_dir, show)

        if "baseline_best" in exp_config:
            plot_count_better_than_best(df, exp_config, output_dir)


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

    yaml_path = args.config
    config = yaml_load(yaml_path)
    generate_figures(config)
    print("Done!")
