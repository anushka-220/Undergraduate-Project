import torch
import numpy as np
import scanpy as sc
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
import h5py
from scipy.sparse import csr_matrix, load_npz


class MetaADataset(Dataset):

    def __init__(self, e_pca, snn_matrix, global_indices):
        super().__init__()
        self.e_pca= torch.from_numpy(e_pca).float()
        self.snn_matrix= snn_matrix
        self.global_indices= torch.arange(len(e_pca))
        self.cell_num = self.e_pca.shape[0]

    def __len__(self):
        return int(self.cell_num)

    def __getitem__(self, index):
        #return e_pca for the batch and their global indices
        snn_row = self.snn_matrix[index].toarray().squeeze()
        snn_row=torch.from_numpy(snn_row).float()
        data={
            "e_pca": self.e_pca[index],
            "batch_global_indices": self.global_indices[index],
            "snn_matrix": snn_row,  
            'indices': torch.tensor(self.global_indices[index]),    
        }

        return data


def load_data(args):
    print("=======Loading Pre-Computed Data=======")
    #Loading E_combined_pca saved as h5ad
    try:
        adata_pca=sc.read_h5ad(args.e_pca_path)
        e_pca= adata_pca.X
    except FileNotFoundError:
        print("Error loading E_combined_pca h5ad file")
        exit()
    #Loading SNN matrix saved as npz
    try:
        snn_matrix= load_npz(args.snn_matrix_path)
    except FileNotFoundError:
        print("Error loading SNN matrix npz file")
        exit()

    #checking if dimensions match
    if e_pca.shape[0] != snn_matrix.shape[0]:
        print("Error: E_combined_pca and SNN matrix dimensions do not match")
        exit()

    cell_num= e_pca.shape[0]
    input_dim_projection= e_pca.shape[1] #this is the D_pca
    #create the new dataset
    dataset= MetaADataset(e_pca, snn_matrix, np.arange(cell_num))
    if args.metacell_num > 1000 and args.batch_size <= 512:
        args.batch_size = 4096
    dataloader_train = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_eval = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # If original data is needed for metacell aggregation or specific evaluations, load it here
    original_adata_list = []

    if args.original_data_paths is not None:
        print("Loading original data for metacell aggregation...")
        if len(args.original_data_paths) != len(args.original_data_types):
            print("Number of original data paths must match number of data types.")
            exit()
        for i, data_path in enumerate(args.original_data_paths):
            try:
                adata = sc.read_h5ad(data_path)
                original_adata_list.append(adata)
                print(f"Loaded original data from {data_path} with shape {adata.shape}")
            except FileNotFoundError:
                print(f"Error loading original data from {data_path}")
                exit()
    
    return dataloader_train, dataloader_eval, input_dim_projection, original_adata_list

def compute_metacell(original_adata, meta_ids, args):
    meta_ids = meta_ids.astype(int)
    if len(meta_ids)!= original_adata.shape[0]:
        print("Error: meta_ids length does not match original data length")
        pass
    non_empty_metacell = np.zeros(meta_ids.max() + 1).astype(bool)
    non_empty_metacell[np.unique(meta_ids)] = True

    data = original_adata.X
    if isinstance(data, sparse.csr.csr_matrix) or isinstance(data, sparse.csc.csc_matrix):
        data = data.toarray()
    data_meta = np.stack(
        [data[meta_ids == i].mean(axis=0) for i in range(meta_ids.max() + 1)]
    )
    data_meta = data_meta[non_empty_metacell]
    metacell_adata = sc.AnnData(data_meta)

    if args.type_key in original_adata.obs_keys():
        type_int = torch.from_numpy(original_adata.obs[args.type_key].cat.codes.values).long()
        type_map = {
            i: original_adata.obs[args.type_key].cat.categories[i]
            for i in range(type_int.max() + 1)
        }
        type_one_hot = torch.zeros(type_int.shape[0], type_int.max() + 1)
        type_one_hot.scatter_(1, type_int.unsqueeze(1), 1)
        type_meta_one_hot = (
            torch.stack(
                [
                    type_one_hot[meta_ids == i].mean(dim=0)
                    for i in range(meta_ids.max() + 1)
                ]
            )
            
        )
        type_meta=type_meta_one_hot.argmax(dim=1).numpy()
        type_meta = np.array([type_map[i] for i in type_meta])
        type_meta = type_meta[non_empty_metacell]

        metacell_adata.obs[args.type_key] = type_meta

    return metacell_adata
