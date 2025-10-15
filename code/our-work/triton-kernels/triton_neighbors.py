import torch
import triton
import triton.language as tl


@triton.jit
def compute_neighbor_list_kernel(
    positions_ptr,
    batch_ptr,
    neighbors_ptr,
    distances_ptr,
    num_neighbors_ptr,
    N,
    cutoff_sq,
    max_neighbors,
    BLOCK_SIZE: tl.constexpr,
):
    atom_idx = tl.program_id(0)

    if atom_idx >= N:
        return

    atom_pos_x = tl.load(positions_ptr + atom_idx * 3 + 0)
    atom_pos_y = tl.load(positions_ptr + atom_idx * 3 + 1)
    atom_pos_z = tl.load(positions_ptr + atom_idx * 3 + 2)
    atom_batch = tl.load(batch_ptr + atom_idx)

    neighbor_count = 0

    for other_idx in range(N):
        if other_idx != atom_idx:
            other_batch = tl.load(batch_ptr + other_idx)
            if atom_batch == other_batch:
                other_pos_x = tl.load(positions_ptr + other_idx * 3 + 0)
                other_pos_y = tl.load(positions_ptr + other_idx * 3 + 1)
                other_pos_z = tl.load(positions_ptr + other_idx * 3 + 2)

                dx = atom_pos_x - other_pos_x
                dy = atom_pos_y - other_pos_y
                dz = atom_pos_z - other_pos_z
                dist_sq = dx * dx + dy * dy + dz * dz

                if dist_sq < cutoff_sq and neighbor_count < max_neighbors:
                    edge_offset = atom_idx * max_neighbors + neighbor_count
                    tl.store(neighbors_ptr + edge_offset * 2 + 0, atom_idx)
                    tl.store(neighbors_ptr + edge_offset * 2 + 1, other_idx)
                    tl.store(distances_ptr + edge_offset, tl.sqrt(dist_sq))
                    neighbor_count += 1

    tl.store(num_neighbors_ptr + atom_idx, neighbor_count)


def compute_neighbors_triton(pos, batch, cutoff, max_neighbors=32):
    N = pos.shape[0]

    neighbors = torch.full((N * max_neighbors, 2), -1, dtype=torch.long, device=pos.device)
    distances = torch.zeros(N * max_neighbors, dtype=pos.dtype, device=pos.device)
    num_neighbors = torch.zeros(N, dtype=torch.long, device=pos.device)

    grid = (N,)

    compute_neighbor_list_kernel[grid](
        pos, batch,
        neighbors, distances, num_neighbors,
        N, cutoff * cutoff, max_neighbors,
        BLOCK_SIZE=1,
    )

    valid_mask = neighbors[:, 0] >= 0
    edge_index = neighbors[valid_mask].t().contiguous()
    edge_weight = distances[valid_mask]

    return edge_index, edge_weight


@triton.jit
def compute_distances_kernel(
    positions_ptr,
    edge_index_ptr,
    distances_ptr,
    edge_vecs_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    edge_idx = tl.program_id(0)

    if edge_idx >= num_edges:
        return

    src = tl.load(edge_index_ptr + edge_idx * 2 + 0)
    dst = tl.load(edge_index_ptr + edge_idx * 2 + 1)

    src_x = tl.load(positions_ptr + src * 3 + 0)
    src_y = tl.load(positions_ptr + src * 3 + 1)
    src_z = tl.load(positions_ptr + src * 3 + 2)

    dst_x = tl.load(positions_ptr + dst * 3 + 0)
    dst_y = tl.load(positions_ptr + dst * 3 + 1)
    dst_z = tl.load(positions_ptr + dst * 3 + 2)

    dx = src_x - dst_x
    dy = src_y - dst_y
    dz = src_z - dst_z

    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)

    tl.store(distances_ptr + edge_idx, dist)
    tl.store(edge_vecs_ptr + edge_idx * 3 + 0, dx)
    tl.store(edge_vecs_ptr + edge_idx * 3 + 1, dy)
    tl.store(edge_vecs_ptr + edge_idx * 3 + 2, dz)
