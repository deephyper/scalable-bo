import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_parser():
    parser = argparse.ArgumentParser(description="Create a surrogate model.")

    parser.add_argument(
        "-i", "--input", type=str, default="results.csv", help="Path to the *.csv file."
    )

    return parser


def main(args):

    # load the data
    real_data = pd.read_csv(args.input, index_col=0)
    real_data["elapsed"] = real_data["timestamp_end"] - real_data["timestamp_start"]
    real_data = real_data.drop(
        columns=["worker_rank", "timestamp_start", "timestamp_end"]
    )
    real_df = real_data
    real_df_cut = real_df

    # preprocessing
    quantiles = np.quantile(
        real_df_cut["objective"].values, [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    real_df_cut_0 = real_df_cut[real_df_cut.objective >= quantiles[4]]
    real_df_cut_1 = real_df_cut[
        (real_df_cut["objective"] >= quantiles[3])
        & (real_df_cut["objective"] < quantiles[4])
    ]
    real_df_cut_2 = real_df_cut[
        (real_df_cut["objective"] >= quantiles[2])
        & (real_df_cut["objective"] < quantiles[3])
    ]
    real_df_cut_3 = real_df_cut[
        (real_df_cut["objective"] >= quantiles[1])
        & (real_df_cut["objective"] < quantiles[2])
    ]
    real_df_cut_4 = real_df_cut[
        (real_df_cut["objective"] >= quantiles[0])
        & (real_df_cut["objective"] < quantiles[1])
    ]
    real_df_cut_5 = real_df_cut[(real_df_cut["objective"] < quantiles[0])]

    nresamp = 200
    real_df_cut_0_r = real_df_cut_0.sample(nresamp, replace=True)
    real_df_cut_1_r = real_df_cut_1.sample(nresamp, replace=True)
    real_df_cut_2_r = real_df_cut_2.sample(nresamp, replace=True)
    real_df_cut_3_r = real_df_cut_3.sample(nresamp, replace=True)
    real_df_cut_4_r = real_df_cut_4.sample(nresamp, replace=True)
    real_df_cut_5_r = real_df_cut_5.sample(nresamp, replace=True)

    real_df_cut_r = pd.concat(
        [
            real_df_cut_0_r,
            real_df_cut_1_r,
            real_df_cut_2_r,
            real_df_cut_3_r,
            real_df_cut_4_r,
            real_df_cut_5_r,
        ]
    )

    # model fitting
    cat_vars = [
        "attention:activation",
        "dense_0:activation",
        "dense_1:activation",
        "dense_3:activation",
        "dense_4:activation",
        "dense_5:activation",
        "dense_6:activation",
        "optimizer",
    ]
    num_vars = [
        "attention:units",
        "batch_size",
        "dense_0:units",
        "dense_3:units",
        "dense_4:units",
        "dense_5:units",
        "dense_6:units",
        "learning_rate",
        "momentum",
    ]
    response_vars = ["objective", "elapsed"]

    num_pipeline = Pipeline(
        [
            ("std_scaler", StandardScaler()),
        ]
    )

    data_pipeline = ColumnTransformer(
        [
            ("response", num_pipeline, response_vars),
            ("numerical", num_pipeline, num_vars),
            ("categorical", OneHotEncoder(sparse=False), cat_vars),
        ]
    )

    data_pipeline_model = data_pipeline.fit(real_df_cut_r)

    

    preprocessed_data = data_pipeline_model.transform(real_df_cut_r)

    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_data[:, 2:],
        preprocessed_data[:, [0, 1]],
        test_size=0.10,
        random_state=42,
    )

    regr = RandomForestRegressor()
    regr.fit(X_train,y_train)
    preds = regr.predict(X_test)
    r2 = r2_score(y_test[:,0],preds[:,0])
    print(f"training objective {r2=}")
    r2 = r2_score(y_test[:,1],preds[:,1])
    print(f"training elapsed   {r2=}")

    preprocessed_data = data_pipeline_model.transform(real_df_cut_r)
    preds = regr.predict(preprocessed_data[:, 2:])
    r2 = r2_score(preprocessed_data[:, 0],preds[:,0])
    print(f"full data objective {r2=}")

    r2 = r2_score(preprocessed_data[:, 1],preds[:,1])
    print(f"full data elapsed   {r2=}")

    saved_pipeline = {
        "data": data_pipeline_model,
        "model": regr,
    }
    with open("surrogate.pkl", "wb") as f:
        pickle.dump(saved_pipeline, f)


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    main(args)
