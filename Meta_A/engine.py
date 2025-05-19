import torch
import numpy as np
from model import snn_contrastive_loss

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    epoch,
    device,
    snn_loss_weight,
    codebook_loss_weight,
):
    model.train(True)
    optimizer.zero_grad()

    snn_loss_epoch=0.0
    codebook_loss_epoch=0.0

    for batch_idx, data in enumerate(dataloader):
        e_pca_batch = data['e_pca'].to(device) 
        snn_matrix_batch = data['snn_matrix'].to(device)
        batch_global_indices = data['indices'].to(device)

        optimizer.zero_grad()

        #Forward pass : getting the projected embeddings and codebook loss
        e_projected_batch, _, codebook_loss= model(e_pca_batch)
        #calculate the SNN contrastive loss
        snn_loss= snn_contrastive_loss(e_projected_batch, snn_matrix_batch, batch_global_indices)
        #calculate the total loss
        total_loss= snn_loss_weight * snn_loss + codebook_loss_weight * codebook_loss
        # Backward pass
        total_loss.backward()
        optimizer.step()

        snn_loss_epoch += snn_loss.item()
        codebook_loss_epoch += codebook_loss.item()

    #print the epoch statistics
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(
            f"[Epoch {epoch + 1}] | SNN Loss: {snn_loss_epoch / len(dataloader):.4f} "
            f"| Codebook Loss: {codebook_loss_epoch / len(dataloader):.4f}"
        )

    return snn_loss_epoch/ len(dataloader), codebook_loss_epoch / len(dataloader) 

@torch.no_grad()
def inference(model, data_loader, device):
    model.eval()
    all_embeddings = []
    all_metacell_ids = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            e_pca_batch = data['e_pca'].to(device)  # Access the 'e_pca' tensor
            # snn_matrix_batch = data['snn_matrix'].to(device) # You might not need this in inference
            # batch_global_indices = data['indices'].to(device) # You might not need this in inference
            e_projected_batch, metacell_ids_batch, _ = model(e_pca_batch)
            all_embeddings.append(e_projected_batch.cpu().numpy())
            all_metacell_ids.append(metacell_ids_batch.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_metacell_ids = np.concatenate(all_metacell_ids, axis=0)
    return all_embeddings, all_metacell_ids

    