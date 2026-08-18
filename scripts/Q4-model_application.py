import pandas as pd
import numpy as np
import logging
import sys
import joblib
from sklearn.preprocessing import StandardScaler

import params.params as params

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Stream to stdout
    ],
    force=True,
)


def check_well_predicted(group):
    if group["pred_label"].any():
        # Get first row where pred_label is True
        first_true = group[group["pred_label"]].iloc[0]
        # Check if both id1 and id2 are the same
        if first_true["id1"] == first_true["id2"]:
            # If they are the same, return the is_in_db value of the first true prediction
            return pd.Series([first_true["is_in_db"]] * len(group), index=group.index)
        else:
            # If they are not, return False for all
            return pd.Series([False] * len(group), index=group.index)
    else:
        # No prediction is True → check if all is_in_db are also False
        return pd.Series([not group["is_in_db"].any()] * len(group), index=group.index)


def model_predictions(
    matches_processed_file_path,
    new_matches_processed_file_path,
    best_model_path,
    scaler_path,
    logreg_model_path,
    threshold_path,
    top10_results_path,
):
    logging.info("Loading matches data...")
    matches = pd.read_parquet(matches_processed_file_path)
    matches = matches.reset_index(drop=True)
    matches["id1"] = matches["img1_full"].map(lambda x: x.split("/")[0])
    matches["id2"] = matches["img2_full"].map(lambda x: x.split("/")[0])

    logging.info("Loading new matches data...")
    new_matches = pd.read_parquet(new_matches_processed_file_path)
    new_matches = new_matches.reset_index(drop=True)
    new_matches["id1"] = new_matches["img1_full"].map(lambda x: x.split("/")[0])
    new_matches["id2"] = new_matches["img2_full"].map(lambda x: x.split("/")[0])

    rf_model = joblib.load(best_model_path)
    logging.info("Random forest model loaded successfully.")
    scaler = joblib.load(scaler_path)
    logging.info("Scaler loaded successfully.")
    regression_model = joblib.load(logreg_model_path)
    logging.info("Logistic regression model loaded successfully.")
    with open(threshold_path, "r") as f:
        best_threshold = float(f.read().strip())
    logging.info(f"Best threshold: {best_threshold}")

    # Define the features to use.
    features = [
        "num_nonzero_points",
        "mean_point_prob",
        "num_nonzero_lines",
        "mean_line_prob",
    ]

    # Standardize the features.
    new_matches_scaled = new_matches[features]
    # scaler = StandardScaler()
    # scaler.fit_transform(matches[features])
    new_matches_scaled = scaler.transform(new_matches[features])

    # Get predicctions and probabilities
    logging.info("Getting predictions and probabilities...")
    y_val_pred = rf_model.predict(new_matches_scaled)

    # Get predictions and probabilities.
    if hasattr(rf_model, "predict_proba"):
        y_val_proba = rf_model.predict_proba(new_matches_scaled)[:, 1]
    else:
        y_val_proba = rf_model.decision_function(new_matches_scaled)

    # Add the probabilities to the new matches dataframe.
    new_matches["prob"] = y_val_proba

    df_grouped_predictions = []
    for target_img in new_matches["img1_full"].unique():
        print(f"Target: {target_img}")

        grouped_predictions = (
            new_matches[new_matches["img1_full"] == target_img]
            .groupby("id2")
            .agg(
                mean_predicted_probability=("prob", "mean"),
                max_predicted_probability=("prob", "max"),
                count=("id2", "count"),
            )
            .reset_index()
        )
        grouped_predictions["img1_full"] = target_img

        df_grouped_predictions.append(
            pd.merge(new_matches, grouped_predictions, on=["img1_full", "id2"])
        )
    df_grouped_predictions = pd.concat(df_grouped_predictions)

    results = df_grouped_predictions.copy()
    results["same"] = results["id1"] == results["id2"]
    # Extract features and target
    X = results[
        [
            "prob",
            "max_predicted_probability",
            "mean_predicted_probability",
            # "count",
        ]
    ].values

    # Predict probabilities
    y_proba = regression_model.predict_proba(X)[:, 1]  # Probability for positive class

    results["logreg_proba"] = y_proba
    results["pred_label"] = (results["logreg_proba"] >= best_threshold).astype(int)

    top10_results = (
        results[
            [
                "id1",
                "img1_full",
                "id2",
                "num_nonzero_points",
                "mean_point_prob",
                "num_nonzero_lines",
                "mean_line_prob",
                "max_predicted_probability",
                "mean_predicted_probability",
                "logreg_proba",
                "pred_label",
            ]
        ]
        .drop_duplicates()
        .groupby(["img1_full", "id2"], as_index=False)
        .agg(logreg_proba=("logreg_proba", "mean"))
        .assign(pred_label=lambda df: df.logreg_proba > best_threshold)
        .sort_values(by=["img1_full", "logreg_proba"], ascending=False)
        .groupby("img1_full")
        .head(10)
        .reset_index(drop=True)
    )
    ext = top10_results["img1_full"].map(lambda x: x.split(".")[-1])[0]
    imgs = top10_results["img1_full"].map(lambda x: x.split("/")[1].split(f".{ext}")[0])
    # Get all the unique ids from the matches dataframe sorted
    id1 = matches["id1"].unique()
    id2 = matches["id2"].unique()
    ids = list(set(np.concatenate((id1, id2))))
    # Sort the ids: B1, B2, ..., B10, B11, ..., B20, B21, ...
    # ids.sort(key=lambda x: (x[0], int(x[1:])))

    top10_results["id1"] = np.nan
    top10_results["is_in_db"] = np.nan
    for img in top10_results["img1_full"].unique()[::-1]:
        print(f"Processing image: {img}")
        # Get the first value of logreg_proba and id2
        first_value = (
            top10_results[top10_results["img1_full"] == img]
            # .reset_index(drop=True)
            .iloc[0]
        )
        # Get the id2 value
        id2_value = first_value["id2"]
        # Get the logreg_proba value
        logreg_proba_value = first_value["logreg_proba"]
        # print(f"Image: {img}, ID2: {id2_value}, LogReg Proba: {logreg_proba_value:.4f}")

        if logreg_proba_value >= best_threshold:
            # Set the id2 value in the top10_results dataframe
            top10_results.loc[top10_results["img1_full"].str.contains(img), "id1"] = (
                id2_value
            )
            top10_results.loc[
                top10_results["img1_full"].str.contains(img), "is_in_db"
            ] = True
        else:
            # Get the next id
            # next_id = "B" + str(int(ids[-1][1:]) + 1)
            next_id = "new"
            # Set the id2 value in the top10_results dataframe
            top10_results.loc[top10_results["img1_full"].str.contains(img), "id1"] = (
                next_id
            )
            top10_results.loc[
                top10_results["img1_full"].str.contains(img), "is_in_db"
            ] = False
            # Add the next id to the ids list
            ids.append(next_id)

    print(top10_results)
    top10_results.to_csv(top10_results_path, index=False)
    logging.info("Top10 results saved successfully.")


if __name__ == "__main__":

    logging.info("-----------------------------------------------")
    logging.info("------- Running predict_new_results.py  -------")
    logging.info("-----------------------------------------------")

    database_path = params.DATABASE_PATH
    processed_matches_file_path = params.PROCESSED_MATCHES_FILE_PATH
    new_processed_matches_file_path = params.NEW_PROCESSED_MATCHES_FILE_PATH
    best_model_path = params.BEST_MODEL_PATH
    scaler_path = params.SCALER_PATH
    logreg_model_path = params.LOGREG_MODEL_PATH
    threshold_path = params.BEST_THRESHOLD_PATH
    top10_results_path = params.TOP10_RESULTS_PATH

    model_predictions(
        processed_matches_file_path,
        new_processed_matches_file_path,
        best_model_path,
        scaler_path,
        logreg_model_path,
        threshold_path,
        top10_results_path,
    )

    logging.info("-----------------------------------------------")
    logging.info("------- Finished predict_new_results.py -------")
    logging.info("-----------------------------------------------")
