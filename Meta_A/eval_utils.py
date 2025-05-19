import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from alive_progress import alive_bar
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

def plot_metacell_umap(adata, save_name, meta_size=10, cell_size=0.5):
    umap = (
        pd.DataFrame(adata.obsm["X_umap"])
        .set_index(adata.obs_names)
        .join(adata.obs["metacell"])
    )
    umap["metacell"] = umap["metacell"].astype("category")
    mcs = umap.groupby("metacell").mean().reset_index()

    plt.figure(figsize=(7, 7), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    sns.scatterplot(x=0, y=1, hue="metacell", data=umap, s=cell_size, legend=None)
    sns.scatterplot(
        x=0,
        y=1,
        s=meta_size,
        hue="metacell",
        data=mcs,
        edgecolor="black",
        linewidth=1,
        legend=None,
    )

    plt.xlabel(f"UMAP1")
    plt.ylabel(f"UMAP2")
    plt.title("Metacell Assignment")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_umap.png", transparent=False)
    plt.close()


def plot_metacell_size(adata, save_name, bins=50):
    label_df = adata.obs["metacell"].reset_index()
    count = label_df.groupby("metacell").count().iloc[:, 0]

    plt.figure(figsize=(7, 3), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    sns.histplot(data=count, stat="density", bins=bins, kde=True)

    plt.xlabel("Number of Cells per Metacell")
    plt.title("Metacell Size")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_size.png", transparent=False)
    plt.close()


def plot_celltype_purity(adata, annotations, save_name):
    celltypes, class_size = np.unique(annotations, return_counts=True)
    celltype2int = {celltype: i for i, celltype in enumerate(celltypes)}

    assignments = adata.obs["metacell"]
    assignment_ids = np.unique(assignments)

    majority_type_pred = annotations.copy()
    for i in assignment_ids:
        idx = assignments == i
        majority_type = annotations[idx].value_counts().idxmax()
        majority_type_pred[idx] = majority_type

    annotations_int = np.array([celltype2int[celltype] for celltype in annotations])
    majority_type_pred_int = np.array(
        [celltype2int[celltype] for celltype in majority_type_pred]
    )
    balanced_acc = balanced_accuracy_score(annotations_int, majority_type_pred_int)
    print("* Balanced Cell Type Purity:", balanced_acc)

    data = pd.DataFrame(columns=["Cell Type", "Prediction by Majority", "Fraction"])

    sorted_indices = np.argsort(class_size)[::-1]
    sorted_celltypes = celltypes[sorted_indices]

    for type in celltypes:
        type_idx = annotations == type
        preds = majority_type_pred[type_idx]
        for type_ in celltypes:
            total_number = (preds == type_).sum()
            frac = total_number / type_idx.sum()
            data = data.append(
                {
                    "Cell Type": type,
                    "Prediction by Majority": type_,
                    "Fraction": frac,
                },
                ignore_index=True,
            )
    data["Cell Type"] = pd.CategoricalIndex(
        data["Cell Type"], categories=sorted_celltypes
    )
    data.sort_values("Cell Type")
    data["Prediction by Majority"] = pd.CategoricalIndex(
        data["Prediction by Majority"],
        categories=sorted_celltypes,
    )
    data.sort_values("Prediction by Majority")
    data = data.pivot("Cell Type", "Prediction by Majority", "Fraction")
    data = data.fillna(0)

    plt.figure(figsize=(7.3, 7), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    ax = sns.heatmap(
        data=data,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("Reds", as_cmap=True),
        cbar=False,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

    plt.title("Balanced Cell Type Purity: {:.2f}%".format(balanced_acc * 100))
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_purity_heatmap.png", transparent=False)
    plt.close()

    data = pd.DataFrame(columns=["Cell Type", "Metacell Purity"])
    purities = []
    for i in assignment_ids:
        idx = assignments == i
        majority_type = annotations[idx].value_counts().idxmax()
        purity = np.sum(annotations[idx] == majority_type) / len(annotations[idx])
        purities.append(purity)
        data = data.append(
            {
                "Cell Type": majority_type,
                "Metacell Purity": purity,
            },
            ignore_index=True,
        )
    avg_purity = data["Metacell Purity"].mean()
    print("* Average Cell Type Purity:", avg_purity)
    np.savetxt("./save/" + save_name + "_purity.txt", np.array(purities))

    plt.figure(figsize=(7, 4), dpi=300)

    ax = sns.boxplot(
        data=data,
        x="Cell Type",
        y="Metacell Purity",
        saturation=0.55,
        fliersize=0.5,
        linewidth=0.5,
        width=0.87,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

    plt.title("Metacell Purity")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_purity_box.png", transparent=False)
    plt.close()
