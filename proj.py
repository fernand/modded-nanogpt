import torch

def make_proj_matrix_coo(N, M, s, device, dtype):
    scaling_factor = torch.sqrt(torch.tensor(1.0 / s, device=device, dtype=dtype))

    row_indices = torch.randint(0, M, (N, s), device=device)
    signs = torch.randint(0, 2, (N, s), device=device) * 2 - 1
    values = signs * scaling_factor

    row_indices = row_indices.flatten()
    col_indices = torch.arange(N, device=device).repeat_interleave(s)
    indices = torch.stack([row_indices, col_indices], dim=0)
    values = values.flatten()

    return torch.sparse_coo_tensor(indices, values, size=(M, N), device=device, dtype=dtype)


def make_proj_matrix_csr(N, M, s, device, dtype):
    scaling_factor = torch.sqrt(torch.tensor(1.0 / s, device=device, dtype=dtype))

    row_indices = torch.stack([torch.randperm(M, device=device)[:s] for _ in range(N)])  # Shape: (N, s)
    signs = torch.randint(0, 2, (N, s), device=device) * 2 - 1  # Shape: (N, s)
    values = signs * scaling_factor  # Shape: (N, s)

    row_indices = row_indices.flatten()  # Shape: (N * s,)
    col_indices = torch.arange(N, device=device).repeat_interleave(s)  # Shape: (N * s,)

    # Sort row indices for CSR
    sorted_row_indices, sorted_order = torch.sort(row_indices)
    sorted_col_indices = col_indices[sorted_order]
    sorted_values = values.flatten()[sorted_order]

    return torch.sparse_csr_tensor(
        sorted_row_indices,
        sorted_col_indices,
        sorted_values,
        size=(M, N),
        device=device,
        dtype=dtype
    )

def sjlt_projection_sparse(x, R_sparse):
    return torch.sparse.mm(R_sparse, x)