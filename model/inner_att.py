import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

BIG_CONSTANT = 1e8
def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):
    del is_query
    if projection_matrix is None:
        return F.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / torch.sqrt(
            torch.tensor(projection_matrix.shape[0], torch.float32)
        )
        data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
        return F.relu(data_dash) + numerical_stabilizer

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash

def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs)
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones)
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs)
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)

def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones)
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)
def vister(position_embedding, value):
    return torch.einsum("bnvh,nbhd->bnhd", position_embedding, value)
def kernelized_softmax(query, key, value, kernel_transformation, projection_matrix=None):

    query_prime = kernel_transformation(query, True, projection_matrix)
    key_prime = kernel_transformation(key, False, projection_matrix)
    query_prime = query_prime.permute(1, 0, 2, 3)
    key_prime = key_prime.permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)

    z_num = numerator(query_prime, key_prime, value)
    z_den = denominator(query_prime, key_prime)


    z_num = z_num.permute(1, 0, 2, 3)
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den
    return z_output

class innerConv(nn.Module):
    def __init__(self,n_heads,win_size,kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',nb_random_features=90,relative_position_embedding=True):
        super(innerConv, self).__init__()
        self.projection_matrix_type = projection_matrix_type
        self.kernel_transformation = kernel_transformation
        self.nb_random_features = nb_random_features

        self.relative_position_embedding = relative_position_embedding

    def forward(self, query, key, value, attn_mask):
        B, N = query.size(0), query.size(1)

        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
            projection_matrix = create_projection_matrix(
                self.nb_random_features, dim, seed=seed).to(query.device)
        z_next = kernelized_softmax(query, key, value, self.kernel_transformation, projection_matrix)

        return z_next
