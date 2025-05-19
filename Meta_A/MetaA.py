import warnings

warnings.filterwarnings("ignore")

import os
import torch
import random
import argparse
import matplotlib
import numpy as np
import scanpy as sc
import seaborn as sns
from model import MetaA, snn_contrastive_loss
from matplotlib import rcParams
from alive_progress import alive_bar
from engine import train_one_epoch, inference
from data_utils import load_data, compute_metacell
from eval_utils import (
    plot_metacell_umap,
    plot_metacell_size,
    plot_celltype_purity,
)


def main(args):
    device = torch.device(args.device)
    
    dataloader_train, dataloader_eval, input_dim_projection, original_data = load_data(args)
    print("Target metacell number:", args.metacell_num)

    # --- Model Instantiation ---
    # Instantiate the new MetaA model
    
    net = MetaA(
        input_dim_projection=input_dim_projection,
        output_dim_projection=args.output_dim_projection,
        entry_num=args.metacell_num,
        entry_dim=args.entry_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    # --- Codebook Initialization ---
    print("Initializating Codebook...")

    print("======= Training Start =======")

    with alive_bar(args.train_epoch, enrich_print=False) as bar:
        snn_loss_weight=args.snn_loss_weight
        codebook_loss_weight=args.codebook_loss_weight
        for epoch in range(args.train_epoch):
            bar()
            snn_loss, codebook_loss= train_one_epoch(
                model=net,
                dataloader=dataloader_train,
                optimizer=optimizer,
                epoch=epoch,
                device=device,  
                snn_loss_weight=snn_loss_weight,
                codebook_loss_weight=codebook_loss_weight,
            )
            
            
            

    print("======= Training Done =======")

    print("")

    print("======= Inference Start =======")
    all_embeddings, all_metacell_ids = inference(
        model=net,
        data_loader=dataloader_eval,
        device=device,
    )
    
    if not os.path.exists("./save/"):
        os.makedirs("./save/")

    #Saving metacell assignments
    assignment_path= (
        "./save/" + args.save_name + "_" + str(args.metacell_num) + "metacell_ids.h5ad"
    )

    #create anndata obj for the embeddings and metacell ids
    adata_assignments= sc.AnnData(all_embeddings, dtype=np.float32)
    adata_assignments.obs["metacell"]= all_metacell_ids.astype(str)
    original_adata = sc.read_h5ad(args.original_data_paths[0]) # Assuming the first file has the common metadata
    if args.original_data_paths is not None:
        original_adata = sc.read_h5ad(args.original_data_paths[0])
        if args.type_key in original_adata.obs_keys():
            adata_assignments.obs[args.type_key] = original_adata.obs[args.type_key].values[:len(all_metacell_ids)]

    sc.set_figure_params(figsize=(7, 7), dpi=300)
    sc.pp.neighbors(adata_assignments, use_rep="X", metric="cosine")
    sc.tl.umap(adata_assignments)
    
    if args.type_key in adata_assignments.obs_keys():
        sc.pl.umap(
            adata_assignments,
            color=[args.type_key],
            save="_" + args.save_name + "_embedding.png",
            palette=sns.color_palette(
                "husl", np.unique(adata_assignments.obs[args.type_key].values).size
            ),
            show=False,
        )
    rcParams.update(matplotlib.rcParamsDefault)
    adata_assignments.write_h5ad(assignment_path)
    print("Metacell assignment saved at:", assignment_path)
    print("")
    if not os.path.exists("./figures/"):
        os.makedirs("./figures/")

    fig_save_name = args.save_name + "_" + str(args.metacell_num) + "metacell"
    plot_metacell_umap(adata_assignments, fig_save_name)
    plot_metacell_size(adata_assignments, fig_save_name)
    if args.type_key in adata_assignments.obs_keys():
        plot_celltype_purity(adata_assignments, adata_assignments.obs[args.type_key], fig_save_name)
    

    print("======= Inference Done =======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data configs
    # data_path will now point to files containing e_combined_pca, snn_matrix, and potentially original data info
    # You might need new arguments to specify paths to these pre-computed files.
    parser.add_argument("--e_pca_path", type=str, required=True) # Path to E_combined_pca
    parser.add_argument("--snn_matrix_path", type=str, required=True) # Path to SNN matrix (sparse format recommended)
    #If you still need original data for metacell aggregation/some evaluations:
    parser.add_argument("--original_data_paths", nargs="+", type=str, help="Paths to original anndata files for aggregation")
    parser.add_argument("--original_data_types", nargs="+", type=str, choices=["RNA", "ADT", "ATAC"], help="Types of original data")

    parser.add_argument("--save_name", type=str)
    parser.add_argument("--metacell_num", type=int)
    parser.add_argument("--type_key", type=str, default="celltype") # Still useful for evaluation if available
    # Model configs (Add arguments for projection layer and quantizer dimensions/weights)
    parser.add_argument("--output_dim_projection", type=int, default=64) # Output dimension of projection layer
    parser.add_argument("--entry_dim", type=int, default=64) # Dimension of quantizer codebook entries (should match output_dim_projection)

    # Training configs (Recommend to use the default)
    parser.add_argument(
        "--codebook_init",
        type=str,
        choices=["Random", "Kmeans", "Geometric"],
        default="Kmeans",
    )
    parser.add_argument("--train_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--converge_threshold", type=int, default=10)

    parser.add_argument(
        "--snn_loss_weight", type=float, default=1.0
    ) 
    parser.add_argument("--codebook_loss_weight", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Randomization
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)

    main(args)
