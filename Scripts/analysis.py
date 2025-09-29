import os
import seaborn as sns
from shutil import rmtree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unweighted_mlp_model import unweighted_mlp_model
from unweighted_random_forest import unweighted_random_forest
from unweighted_xgb_chain import unweighted_xgb_chain
from weighted_mlp_model import weighted_mlp_model
from weighted_random_forest import weighted_random_forest
from weighted_xgb_chain import weighted_xgb_chain
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    balanced_accuracy_score, hamming_loss, cohen_kappa_score
)

def results_to_dataframe(results_dict):
    def to_1d(arr):
        a = np.array(arr)
        if a.ndim > 1 and a.shape[1] == 1:
            a = a.flatten()
        return a.astype(int)

    rows = []
    for model_key, info in results_dict.items():
        for i, label in enumerate(LABELS):
            y_true = np.array(info['y_test'][i]).astype(int)
            y_pred = to_1d(info['y_pred_bins'][i])

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            ham_loss = hamming_loss(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)

            rows.append({
                "Model_key": model_key,
                "Label": label,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Balanced_accuracy": bal_acc,
                "Hamming_loss": ham_loss,
                "Kappa": kappa,
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                "Total_pos": int(np.sum(y_true==1)),
                "Total_neg": int(np.sum(y_true==0)),
            })
    return pd.DataFrame(rows)

save_df_path = "metrics_table.csv"
model_saved_dir = "all_models"

if not os.path.exists(save_df_path):

    if os.path.exists(model_saved_dir): 
        rmtree(model_saved_dir)
        print(f"Removed existing {model_saved_dir} directory. Creating a new one.")
    os.mkdir(model_saved_dir)

    data_results = {
        "mlp_weighted": weighted_mlp_model(),
        "mlp_unweighted": unweighted_mlp_model(),
        "xgb_weighted": weighted_xgb_chain(),
        "xgb_unweighted": unweighted_xgb_chain(),
        "random_forest_weighted": weighted_random_forest(),
        "random_forest_unweighted": unweighted_random_forest()
    }

    LABELS = ["Valence", "Arousal", "Dominance", "Liking"]

    metrics_df = results_to_dataframe(data_results)
    metrics_df.to_csv(save_df_path, index=False)
else: 
    metrics_df = pd.read_csv(save_df_path)

# Add model_type + weighting columns
metrics_df["model_type"] = metrics_df["Model_key"].str.extract(r"(mlp|xgb|random_forest)")
metrics_df["weighting"] = metrics_df["Model_key"].str.extract(r"(weighted|unweighted)")

# Aggregate metrics
summary = (
    metrics_df
    .groupby(["model_type", "weighting", "Label"])
    .agg({
        "F1": "mean",
        "Balanced_accuracy": "mean",
        "Kappa": "mean",
        "Hamming_loss": "mean"
    })
    .reset_index()
)

def plot_metric(metric: str, title: str, ylabel: str):
    # Weighted vs Unweighted
    g = sns.catplot(
        data=summary, kind="bar",
        x="Label", y=metric,
        hue="weighting", col="model_type",
        height=5, aspect=1, errorbar=None
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Label", ylabel)
    plt.subplots_adjust(top=0.85)
    g.figure.suptitle(f"{title} by Label: Weighted vs Unweighted")
    plt.show()

    # Model comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x="Label", y=metric, hue="model_type", errorbar=None)
    plt.title(f"{title} Comparison Across Models")
    plt.xlabel("Label")
    plt.ylabel(ylabel)
    plt.legend(title="Model Type")
    plt.show()

    # Overall heatmap
    avg_summary = (
        summary.groupby(["model_type", "weighting"])
        .agg({metric: "mean"}).reset_index()
        .pivot(index="model_type", columns="weighting", values=metric)
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(avg_summary, annot=True, cmap="Blues", fmt=".3f")
    plt.title(f"Average {title} Across Labels")
    plt.ylabel("Model Type")
    plt.xlabel("Weighting")
    plt.show()

if __name__ == "__main__":

    ans = input("Want F1 plots? Press Enter to continue. If not type 'next':")
    if ans.lower() != 'next':
        plot_metric("F1", "F1 Score", "F1 Score")

    ans = input("Want Balanced Accuracy plots? Press Enter to continue. If not type 'next':")
    if ans.lower() != 'next':
        plot_metric("Balanced_accuracy", "Balanced Accuracy", "Balanced Accuracy")

    ans = input("Want Kappa plots? Press Enter to continue. If not type 'next':")
    if ans.lower() != 'next':
        plot_metric("Kappa", "Kappa", "Kappa")

    ans = input("Want Hamming Loss plots? Press Enter to continue. If not type 'next':")
    if ans.lower() != 'next':
        plot_metric("Hamming_loss", "Hamming Loss", "Hamming Loss")