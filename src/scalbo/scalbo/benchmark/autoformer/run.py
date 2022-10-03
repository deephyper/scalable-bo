import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from deephyper.evaluator import profile
from torch import optim

from .model import Model as GNN_autoformer
from .utils.data import load_graph_data, trainingset_construct
from .utils.train_test import eval, train

HERE = os.path.dirname(os.path.abspath(__file__))


@profile
def run(config):
    # define random seed
    torch.manual_seed(42)

    config_filename = HERE + "/configs.yaml"
    device = torch.device("cuda")

    with open(config_filename) as f:
        setting = yaml.load(f, Loader=Loader)

        setting["data"]["speed_data_path"] = (
            HERE + "/" + setting["data"]["speed_data_path"]
        )
        setting["data"]["graph_data_path"] = (
            HERE + "/" + setting["data"]["graph_data_path"]
        )

        for k in config:
            d = [
                (dict_name, sub_dict)
                for dict_name, sub_dict in setting.items()
                if k in sub_dict
            ]
            if d:
                assert len(d) == 1
                dict_name, sub_dict = d[0]
                sub_dict[k] = config[k]

    # define set up
    parser = argparse.ArgumentParser(description="G Autoformer")
    # fixed parameters
    parser.add_argument(
        "--pred_len", type=int, default=setting["fixed_parameter"]["pred_len"]
    )
    parser.add_argument(
        "--time_steps", type=int, default=setting["fixed_parameter"]["input_len"]
    )
    parser.add_argument(
        "--n_heads", type=int, default=setting["fixed_parameter"]["n_heads"]
    )
    parser.add_argument("--topk", type=int, default=setting["fixed_parameter"]["topk"])
    parser.add_argument(
        "--num_nodes", type=int, default=setting["fixed_parameter"]["num_nodes"]
    )

    # searched parameters
    parser.add_argument(
        "--enc_width", type=int, default=setting["searched_parameter"]["encoder_width"]
    )
    parser.add_argument(
        "--enc_depth", type=int, default=setting["searched_parameter"]["encoder_depth"]
    )
    parser.add_argument(
        "--dec_width", type=int, default=setting["searched_parameter"]["decoder_width"]
    )
    parser.add_argument(
        "--dec_depth", type=int, default=setting["searched_parameter"]["decoder_depth"]
    )
    parser.add_argument(
        "--d_model", type=int, default=setting["searched_parameter"]["d_model"]
    )
    parser.add_argument(
        "--CAM_width", type=int, default=setting["searched_parameter"]["CAM_width"]
    )
    parser.add_argument(
        "--CAM_depth", type=int, default=setting["searched_parameter"]["CAM_depth"]
    )
    args, unknown = parser.parse_known_args()

    # load the adjacent matrix
    """ dense adj mat """
    sensor_ids, sensor_id_to_ind, W = load_graph_data(
        setting["data"]["graph_data_path"]
    )
    adj = torch.from_numpy(W).float().to(device)

    # define model, optimizer and loss function
    model = GNN_autoformer(adj=adj, configs=args, DEVICE=device).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=setting["train"]["lr"])
    criterion = nn.MSELoss()

    # define data loader
    data = (
        pd.read_hdf(setting["data"]["speed_data_path"]).to_numpy()
    ).T  # return (N,T)
    time_step_per_week = 7 * 24 * 12
    """ The first week of each dataset is only for input and not for target value"""
    train_data = data[:, : 12 * time_step_per_week]
    val_data = data[:, 12 * time_step_per_week : 14 * time_step_per_week]
    mean_val = np.mean(train_data)
    std_val = np.std(train_data)
    train_loader = trainingset_construct(
        args=args,
        traffic_data=train_data,
        batch_val=setting["train"]["train_batch"],
        num_data_limit=np.inf,
        Shuffle=True,
        mean=mean_val,
        std=std_val,
    )
    val_loader = trainingset_construct(
        args=args,
        traffic_data=val_data,
        batch_val=setting["train"]["test_batch"],
        num_data_limit=np.inf,
        Shuffle=False,
        mean=mean_val,
        std=std_val,
    )

    # perform training and testing
    for epoch in range(setting["train"]["epochs"]):
        print("Current epoch", epoch)
        train_loss = train(
            train_loader, model, optimizer, criterion, device, mean_val, std_val, 1
        )
        print("training loss:", train_loss)
        ma = eval(val_loader, model, device, args, mean_val, std_val)
        # torch.save(model.state_dict(), r'./trained_models/model_12_to_{}_epoch{}.pkl'.format(args.pred_len, epoch))

    return ma


if __name__ == "__main__":
    from scalbo.benchmark.autoformer.problem import hp_problem

    config = hp_problem.default_configuration
    run(config)

# ma = hyper_search(r'./configs.yaml')
# print('final MAE:', ma)
