import joblib
import logging
import sys

import numpy as np
import pandas as pd

# Imported before xgboost on purpose. On macOS the xgboost wheel links
# libomp.dylib by @rpath and finds it only if some other library has already
# loaded one; without Homebrew's libomp installed, importing xgboost first dies
# with "Library not loaded: @rpath/libomp.dylib". params.params pulls in torch,
# which ships its own copy, so this import order is what keeps P4 working on a
# stock Mac. On Linux and Windows the order makes no difference.
import params.params as params
import utils.utils as utils

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Stream to stdout
    ],
    force=True,
)


def train_val_split(matches, train_frac=0.70, val_frac=0.30, splits=5, seed=42):
    """Image-disjoint rollout splits, sized as fractions of the dataset.

    Split 1 is a plain **70/30** train/validation partition. Every later split
    folds one chunk of the validation pool into training and validates on what
    is left of it, so the training database grows from one split to the next the
    way it does in the field, where a re-identification catalogue is built up
    batch by batch and each new batch is identified against everything already
    in it.

    Splitting on *images* rather than on pairs matters: a single photograph
    appears in many pairs, so a pair-level split would put near-copies of the
    same comparison on both sides and inflate the score.

    Returns (train_splits, val_splits), each a list of image-name arrays with
    one entry per split.
    """
    imgs = pd.Series(pd.concat([matches.img1_full, matches.img2_full]).unique())
    n = len(imgs)

    total = train_frac + val_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train and val fractions must sum to 1, got {total}")
    if n < 6:
        raise ValueError(
            f"Only {n} images with matches - too few to split. Use the demo "
            "notebooks, which size their folds for small galleries."
        )

    shuffled = imgs.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_val = max(1, round(n * val_frac))
    base_train = shuffled.iloc[n_val:].values
    val_pool = shuffled.iloc[:n_val].values

    # One chunk per rollout step; never more chunks than images to give out.
    splits = max(1, min(splits, len(val_pool)))
    chunks = np.array_split(val_pool, splits)

    print(f"Total unique images: {n}")
    train_splits, val_splits = [], []
    for i in range(splits):
        train_i = np.concatenate([base_train, *chunks[:i]]) if i else base_train
        val_i = np.concatenate(chunks[i:])
        train_splits.append(train_i)
        val_splits.append(val_i)
        print(
            f"Split {i+1} - Train: {len(train_i)} images ({len(train_i)/n:.0%}), "
            f"Val: {len(val_i)} images ({len(val_i)/n:.0%})"
        )

    return train_splits, val_splits


def train(matches, train_splits, val_splits):
    features = [
        "num_nonzero_points",
        "mean_point_prob",
        "num_nonzero_lines",
        "mean_line_prob",
        # "same_flank",
    ]

    # Define a dictionary of models to evaluate.
    # Adjusted XGBoost parameters for improvement:
    xgb_params = {
        "learning_rate": 0.01,  # Lower learning rate
        "n_estimators": 150,  # More boosting rounds
        "max_depth": 10,  # Increase as needed
        "subsample": 0.8,  # Use 80% of data per tree
        "colsample_bytree": 0.8,  # Use 80% of features per tree
        # "scale_pos_weight": 4000,  # Ratio of negative to positive cases (adjust accordingly)
        "eval_metric": "auc",  # Or use "auc" if that's more relevant
        "random_state": 42,
    }
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=25, max_features="sqrt", n_jobs=32, random_state=42
        ),
        "XGBoost": XGBClassifier(**xgb_params),
        "CatBoost": CatBoostClassifier(
            learning_rate=0.1,
            iterations=100,
            depth=5,
            random_state=42,
            verbose=0,
        ),
    }

    # Dictionary to store aggregated metrics per model.
    results = {
        model_name: {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for model_name in models.keys()
    }
    best_model = None
    best_f1 = 0.0

    # Outer loop: iterate over models.
    for model_name, model in models.items():
        print(f"===== Model: {model_name} =====")

        # Iterate over each training/validation split.
        for i, (train_imgs, val_imgs) in enumerate(zip(train_splits, val_splits)):
            print(f"--- Split {i+1} ---")

            # Filter matches such that both images are in the training set.
            train_mask = matches["img1_full"].isin(train_imgs) & matches[
                "img2_full"
            ].isin(train_imgs)
            # For validation, allow pairs with one image from train and one from validation.
            val_mask = (
                matches["img1_full"].isin(train_imgs)
                & matches["img2_full"].isin(val_imgs)
            ) | (
                matches["img1_full"].isin(val_imgs)
                & matches["img2_full"].isin(train_imgs)
            )

            train_matches = matches[train_mask]
            val_matches = matches[val_mask]

            # Check if we have enough data in the split.
            if train_matches.empty or val_matches.empty:
                print("Not enough data in this split. Skipping...")
                continue

            # Create binary target: 1 if same indv, 0 otherwise.
            train_matches = train_matches.copy()
            val_matches = val_matches.copy()

            # Ensure same_flank is numeric (0 or 1)
            if "same_flank" in features:
                train_matches["same_flank"] = train_matches["same_flank"].astype(int)
                val_matches["same_flank"] = val_matches["same_flank"].astype(int)

            train_matches["same"] = (
                train_matches["id1"] == train_matches["id2"]
            ).astype(int)
            val_matches["same"] = (val_matches["id1"] == val_matches["id2"]).astype(int)

            # Define X and y for training and validation.
            X_train = train_matches[features]
            y_train = train_matches["same"]
            X_val = val_matches[features]
            y_val = val_matches["same"]

            numerical_features = [f for f in features if f != "same_flank"]

            # Scale numerical features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train[numerical_features]),
                columns=numerical_features,
                index=X_train.index,
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val[numerical_features]),
                columns=numerical_features,
                index=X_val.index,
            )

            # Add the binary feature back
            if "same_flank" in features:
                X_train_scaled["same_flank"] = X_train["same_flank"].astype(int).values
                X_val_scaled["same_flank"] = X_val["same_flank"].astype(int).values

            # Train the model.
            model.fit(X_train_scaled, y_train)

            # Get predictions.
            y_val_pred = model.predict(X_val_scaled)

            # Get predictions and probabilities.
            if hasattr(model, "predict_proba"):
                y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                y_val_proba = model.decision_function(X_val_scaled)

            mask = (val_matches["img2_full"].isin(val_imgs)) & (
                ~val_matches["img1_full"].isin(val_imgs)
            )
            # Swap the columns when the condition is met
            matches.loc[val_matches.index[mask], ["img1_full", "img2_full"]] = (
                val_matches.loc[mask, ["img2_full", "img1_full"]].values
            )
            matches.loc[val_matches.index[mask], ["id1", "id2"]] = val_matches.loc[
                mask, ["id2", "id1"]
            ].values
            matches.loc[val_matches.index, f"prob_{model_name}"] = y_val_proba

            # Compute metrics.
            acc = accuracy_score(y_val, y_val_pred)
            prec = precision_score(y_val, y_val_pred, average="macro", zero_division=0)
            rec = recall_score(y_val, y_val_pred, average="macro", zero_division=0)
            f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

            # Append metrics to the results dictionary.
            results[model_name]["accuracy"].append(acc)
            results[model_name]["precision"].append(prec)
            results[model_name]["recall"].append(rec)
            results[model_name]["f1"].append(f1)

            print(
                f"Split {i+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
            )

            # Evaluate the model.
            acc = accuracy_score(y_val, y_val_pred)
            report = classification_report(y_val, y_val_pred)
            print("Validation Accuracy:", acc)
            print("Classification Report:")
            print(report)
            print("\n")

        # Compute mean metrics for the current model across splits.
        mean_accuracy = np.mean(results[model_name]["accuracy"])
        mean_precision = np.mean(results[model_name]["precision"])
        mean_recall = np.mean(results[model_name]["recall"])
        mean_f1 = np.mean(results[model_name]["f1"])

        print(f"===== Average Metrics for {model_name} =====")
        print(f"Mean Accuracy:  {mean_accuracy:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall:    {mean_recall:.4f}")
        print(f"Mean F1-score:  {mean_f1:.4f}")
        print("\n\n")

        # Check if this model is the best so far.
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model = model
            best_model_name = model_name
            print(f"New best model: {best_model_name} with F1: {best_f1:.4f}\n")

    return results, best_model, best_model_name, scaler


def obtain_val_results(val_splits, val_matches):
    results = []
    for split, val_images in enumerate(val_splits):
        print(f"Split {split+1}:")
        df_grouped_predictions = []
        for target_img in val_images:
            print(f"Target: {target_img}")

            grouped_predictions = (
                val_matches[val_matches["img1_full"] == target_img]
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
                pd.merge(val_matches, grouped_predictions, on=["img1_full", "id2"])
            )
        df_grouped_predictions = pd.concat(df_grouped_predictions)
        results.append(df_grouped_predictions)
    results = pd.concat(results)
    return results


def train_logreg_and_compute_thres(results):
    X = results[
        [
            "prob",
            "max_predicted_probability",
            "mean_predicted_probability",
            # "count",
        ]
    ].values
    y = results["same"].values

    # Train logistic regression
    clf = LogisticRegression()
    clf.fit(X, y)

    # Predict probabilities
    y_proba = clf.predict_proba(X)[:, 1]  # Probability for positive class

    # Optimize threshold based on F1 score
    best_threshold = 0.5
    best_f1 = -1
    thresholds = np.linspace(0.0, 1.0, 101)
    f1_scores = []
    recall_scores = []
    precision_scores = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        current_f1 = f1_score(y, y_pred)
        current_recall = recall_score(y, y_pred)
        current_precision = precision_score(y, y_pred)
        f1_scores.append(current_f1)
        recall_scores.append(current_recall)
        precision_scores.append(current_precision)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    print("Best threshold:", best_threshold)
    print("Best F1 at that threshold:", best_f1)

    return clf, best_threshold


if __name__ == "__main__":
    logging.info("-----------------------------------------------")
    logging.info("---------- Running model_training.py ----------")
    logging.info("-----------------------------------------------")

    # Everything P4 needs is in the parquet. It used to also read
    # results/unique_ids.txt and never use it, which made the script fail for
    # anyone who produced the matches from a notebook rather than from P1.
    matches = pd.read_parquet(params.PROCESSED_MATCHES_FILE_PATH)

    print("Splitting images into training and validation sets...")
    train_splits, val_splits = train_val_split(matches)

    print("Training models...")
    results, best_model, best_model_name, scaler = train(matches, train_splits, val_splits)

    print(
        f"Best model: {best_model_name} with F1: {results[best_model_name]['f1'][-1]:.4f}"
    )
    joblib.dump(best_model, params.BEST_MODEL_PATH)
    joblib.dump(scaler, params.SCALER_PATH)
    print(f"Best model saved to {params.BEST_MODEL_PATH}")

    val_matches = matches[~matches[f"prob_{best_model_name}"].isna()]
    val_matches = val_matches.rename(columns={f"prob_{best_model_name}": "prob"})

    results = obtain_val_results(val_splits, val_matches)
    results["same"] = results["id1"] == results["id2"]

    # Train logistic regression and compute threshold
    clf, threshold = train_logreg_and_compute_thres(results)

    # Save the model and threshold
    joblib.dump(clf, params.LOGREG_MODEL_PATH)
    with open(params.BEST_THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))

    logging.info("-----------------------------------------------")
    logging.info("--------- model_training.py finished! ---------")
    logging.info("-----------------------------------------------")
