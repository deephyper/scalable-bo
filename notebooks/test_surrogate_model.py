import argparse
import pickle

import pandas as pd
from sklearn.metrics import r2_score


def create_parser():
    parser = argparse.ArgumentParser(description="Test a surrogate model.")

    parser.add_argument(
        "-i", "--input", type=str, default="results.csv", help="Path to the *.csv file."
    )
    parser.add_argument(
        "-m", "--model", type=str, default="surrogate.pkl", help="Path to the pickle file containing the surrogate model."
    )

    return parser


def main(args):

    # load the data
    real_data = pd.read_csv(args.input, index_col=0)
    real_data["elapsed"] = real_data["timestamp_end"] - real_data["timestamp_start"]
    real_data = real_data.drop(
        columns=["worker_rank", "timestamp_start", "timestamp_end"]
    )

    with open(args.model, "rb") as f:
        saved_pipeline = pickle.load(f)

    data_pipeline_model = saved_pipeline["data"]
    regr = saved_pipeline["model"]

    preprocessed_data = data_pipeline_model.transform(real_data)
    preds = regr.predict(preprocessed_data[:, 2:])
    r2 = r2_score(preprocessed_data[:, 0],preds[:,0])
    print(f"full data objective {r2=}")

    r2 = r2_score(preprocessed_data[:, 1],preds[:,1])
    print(f"full data elapsed   {r2=}")


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    main(args)
