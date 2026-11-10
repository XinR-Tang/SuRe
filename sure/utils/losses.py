import math

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
loss_fn = torch.nn.L1Loss()

def unshuffled_loss(
    q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None
):
    loss = 0
    attn_mask = None
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask=attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale,
            torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            attn_mask=attn_mask
        )
        loss += loss_fn(self_out, target_out.detach())
    return loss

def shuffled_loss(
    q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None,
    num_shuffle=5, seed=42
):
    loss = 0
    attn_mask = None

    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask = attn_mask.to(q.device)

        shuffle_losses = []

        for i in range(num_shuffle):
            cur_ks = ks[i]  # [num_heads, seq_len, dim]
            cur_vs = vs[i]  # [num_heads, seq_len, dim]
            num_heads, seq_length, dim = cur_vs.shape

            # # Initialize the permutation matrix
            perm = torch.randperm(seq_length, device=cur_vs.device)
            P = torch.eye(seq_length, device=cur_vs.device, dtype=cur_vs.dtype)[perm]

            # V~ = P V
            vs_rand = torch.einsum('ij,hjd->hid', P, cur_vs)

            q_input = q * scale
            k_input = cur_ks.unsqueeze(0).repeat(q.shape[0], 1, 1, 1)
            v_input = vs_rand.unsqueeze(0).repeat(q.shape[0], 1, 1, 1)

            target_out = F.scaled_dot_product_attention(
                q_input,
                k_input,
                v_input,
                attn_mask=attn_mask
            )

            shuffle_losses.append(
                loss_fn(
                    self_out,
                    target_out.detach()
                )
            )

            seed += 1
            set_seed(seed)

        loss += sum(shuffle_losses) / len(shuffle_losses)

    return loss

def q_loss(q_list, qc_list):
    loss = 0
    for q, qc in zip(q_list, qc_list):
        loss += loss_fn(q, qc.detach())
    return loss