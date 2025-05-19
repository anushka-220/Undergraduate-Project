import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



#Implementation of the Porjection Layer 
#The projection layer's job is to transform the initial embeddings
#into a new space where the SNN relationships are better reflected.
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class Quantizer(nn.Module):
    def __init__(self, entry_num, entry_dim):
        super().__init__()

        self.entry_num = entry_num
        self.entry_dim = entry_dim
        self.decay = 0.9
        self.entry = nn.Embedding(self.entry_num, self.entry_dim)
        self.register_buffer("entry_prob", torch.zeros(self.entry_num))

    def init_codebook(self, z, method):
        if method == "Random":
            self.entry.weight.data.uniform_(-1.0 / self.entry_num, 1.0 / self.entry_num)
        elif method == "Kmeans":
            import faiss

            d = z.shape[1]
            kmeans = faiss.Kmeans(d, self.entry_num, spherical=True, gpu=True)
            kmeans.train(z)
            D, I = kmeans.index.search(z, 1)
            assignments = I.reshape(-1)
            centers = np.zeros((self.entry_num, d))
            for i in range(self.entry_num):
                centers[i] = z[assignments == i].mean(axis=0)
            self.entry.weight.data.copy_(torch.from_numpy(centers))
        elif method == "Geometric":
            from geosketch import gs

            sketch_index = gs(z, self.entry_num, replace=False)
            self.entry.weight.data.copy_(torch.from_numpy(z[sketch_index]))

    def forward(self, e, return_assignment=False):
        normed_e = F.normalize(e, dim=1).detach()
        normed_c = F.normalize(self.entry.weight, dim=1)
        sim = torch.einsum("bd,dn->bn", normed_e, rearrange(normed_c, "n d -> d n"))
        assignment_indices = torch.argmax(sim, dim=1)
        assignments = torch.zeros(assignment_indices.unsqueeze(1).shape[0], self.entry_num, device=e.device)
        assignments.scatter_(1, assignment_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(assignments, dim=0)
        e_q = torch.matmul(assignments, self.entry.weight)
        loss = torch.mean((e_q - e.detach()) ** 2)  # This is the codebook loss

        if self.training:
            self.entry_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            norm_distance = F.softmax(1 - sim, dim=1)
            norm_distance = torch.max(norm_distance, dim=1).values
            dis_indices = torch.multinomial(norm_distance, num_samples=self.entry_num, replacement=True).view(-1)
            random_feat = e.detach()[dis_indices]
            beta_s = (torch.exp(-self.entry_prob * self.entry_num * 100 - 1e-3).unsqueeze(1).repeat(1, self.entry_dim))
            self.entry.weight.data = (self.entry.weight.data * (1 - beta_s) + random_feat * beta_s)
            if self.entry_prob.sum() + 1e-4 >= 1:
                sim_t = sim.t()
                median_distance = torch.median(sim_t, dim=1).values
                median_distance = torch.abs(sim_t - median_distance[:, None])
                dis_indices = torch.multinomial(F.softmax(-median_distance, dim=1), num_samples=1).view(-1)
                random_feat = e.detach()[dis_indices]
                beta_l = (torch.exp(-self.entry_prob.mean() / self.entry_prob * 10 - 1e-3).unsqueeze(1).repeat(1, self.entry_dim))
                self.entry.weight.data = (self.entry.weight.data * (1 - beta_l) + random_feat * beta_l)

        if return_assignment:
            top2_sim = torch.topk(sim, 2, dim=1).values
            delta_conf = top2_sim[:, 0] - top2_sim[:, 1]
            loss_c_sum = torch.sum((e_q - e.detach()) ** 2)
            return e_q, loss, assignment_indices  # Return e_q, loss, and assignment_indices
        else:
            return e_q, loss


class MetaA(nn.Module):
    def __init__(self, input_dim_projection, output_dim_projection, entry_num, entry_dim=32):
        super(MetaA, self).__init__()
        self.encoders = None
        self.projection_layer= ProjectionLayer(input_dim_projection, output_dim_projection)
        self.quantizer = Quantizer(entry_num, output_dim_projection)
        self.decoders = None
        self.decoders_q = None

    def quantize(self, e_projected , return_assignment=False):
        return self.quantizer(e_projected, return_assignment)

    def forward(self, inputs):
        # inputs should be the E_combined_pca embeddings directly
        e_projected, codebook_loss, metacell_ids = self.quantizer(self.projection_layer(inputs), return_assignment=True)
        return e_projected, metacell_ids, codebook_loss
        
    def init_codebook(self, z, method): # new init function
        self.quantizer.init_codebook(z, method)
    

   
def snn_contrastive_loss(e_projected, snn_matrix, batch_global_indices, temperature=0.2):
    """
    Calculates the SNN contrastive loss.

    Args:
        e_projected:  Projected embeddings (batch_size, output_dim_projection).
        snn_matrix:   Sparse SNN matrix (batch_size, total_cells).
        batch_global_indices: Indices of the cells in the batch within the full dataset (LongTensor).
        temperature:  Temperature parameter for the contrastive loss.

    Returns:
        The SNN contrastive loss (scalar).
    """

    batch_size = e_projected.shape[0]
    loss = 0.0

    for i in range(batch_size):
        anchor_idx = i
        anchor_embedding = e_projected[anchor_idx].unsqueeze(0)
        global_anchor_idx = batch_global_indices[i].item()  # Get the global index of the anchor

        # Get SNN neighbors for the anchor cell (global indices)
        positive_global_indices = snn_matrix[i].nonzero().flatten()
        positive_global_indices = positive_global_indices[positive_global_indices != global_anchor_idx]

        if len(positive_global_indices) > 0:
            # Find the indices of these positive neighbors within the current batch
            mask = torch.isin(batch_global_indices, positive_global_indices)
            positive_batch_indices = torch.where(mask)[0]

            if len(positive_batch_indices) > 0:
                positive_embeddings = e_projected[positive_batch_indices]

                # Calculate cosine similarity
                anchor_norm = F.normalize(anchor_embedding, p=2, dim=1)
                positive_norm = F.normalize(positive_embeddings, p=2, dim=1)
                similarity = torch.matmul(anchor_norm, positive_norm.transpose(0, 1)) / temperature

                # Create labels (similarity with positive examples)
                labels = torch.zeros(similarity.shape[0], dtype=torch.long, device=similarity.device)  # Single label for anchor

                # Calculate loss
                loss += F.cross_entropy(similarity, labels)

    return loss / batch_size if batch_size > 0 else torch.tensor(0.0, requires_grad=True, device=e_projected.device)