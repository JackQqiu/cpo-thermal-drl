"""
cross_attention_actor.py
========================

Stage C — Cross-attention policy with a **factored placement / delay**
head, supporting all three v2 action modes.

Architecture (matches plan §4.2)
--------------------------------
1. Take the encoder's task and proc embeddings.
2. **Query**: the embedding of the *current* task being scheduled
   (``x_task[current_task_idx]`` in batched form, gathered per graph).
3. **Placement key/value**: the proc embeddings of this graph's procs.
4. Multi-head dot-product attention -> placement logits, one per proc.
5. Apply the action mask (mask out procs whose temp ≥ ``mask_temp``).
6. **Delay head** (optional): an MLP that takes the same query (plus a
   pooled context) and outputs ``K_delay`` logits.

Action modes
------------
* ``auto_only``: only placement head is sampled.  Delay head is built but
  receives **no gradient** — it produces a uniform output that's
  discarded.  This keeps the model architecture identical between
  Stage 1 (auto_only training) and Stage 2 (hybrid warm-start fine-tuning),
  so the encoder and placement head can be loaded directly.
* ``agent_only`` / ``hybrid``: both heads are sampled, log-probs and
  entropy are summed across heads (factored policy).

Output of ``forward(...)``
--------------------------
A dict with keys::

    placement_logits : (B, num_proc_max)        — per-graph proc logits
    placement_mask   : (B, num_proc_max)        — bool, True = valid
    delay_logits     : (B, K_delay)             — per-graph delay logits
    proc_embed_pad   : (B, num_proc_max, hidden) — for caller diagnostics
    proc_count       : (B,)                     — actual proc count per graph

The ``placement_logits``-where-mask-is-False are set to a large negative
value, so a softmax over them gives zero probability to invalid actions.

Sampling and log-prob computation are done by the wrapper module in
``ppo_actor_critic.py``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Helper: ragged-pad a (sum_proc, hidden) tensor into (B, max_proc, hidden)
# =====================================================================
def _pad_ragged(
    x:        torch.Tensor,        # (sum_n, h)
    batch:    torch.Tensor,        # (sum_n,) graph indices in [0, B)
    batch_size: int,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flat (sum_n, h) tensor into (B, max_n, h) by zero-padding.

    Returns
    -------
    padded : (B, max_n, h)
    counts : (B,) — true (un-padded) length per graph
    """
    counts = torch.bincount(batch, minlength=batch_size)
    max_n  = int(counts.max().item()) if counts.numel() > 0 else 0
    h      = x.size(-1)
    padded = x.new_full((batch_size, max_n, h), pad_value)

    # Build per-graph offsets via a cumulative sum of counts.
    # Within each graph, we want positions 0..count[g]-1 of `padded[g]`.
    # `arange_within` constructs that index for the flat layout.
    arange_within = torch.arange(x.size(0), device=x.device) - \
                    torch.cat([
                        x.new_zeros(1, dtype=torch.long),
                        counts.cumsum(0)[:-1],
                    ])[batch]
    padded[batch, arange_within] = x
    return padded, counts


# =====================================================================
# Cross-attention factored actor
# =====================================================================
class CrossAttentionActor(nn.Module):
    """Placement + delay actor.

    Parameters
    ----------
    hidden : encoder embedding dim (default 128).
    K_delay : number of delay levels.  Set to 1 for action_mode=auto_only
        (the head still exists but is sampled trivially).
    num_heads : multi-head attention heads for placement scoring.
    """

    def __init__(
        self,
        hidden:   int = 128,
        K_delay:  int = 5,
        num_heads: int = 4,
        delay_hidden: int = 64,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden    = hidden
        self.K_delay   = K_delay
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads

        # Placement head: scaled dot-product cross-attention
        # (multi-head; output is per-proc score, NOT per-proc embedding)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        # No v_proj or output proj: we use attention scores DIRECTLY as
        # logits.  This is the "scoring head" pattern from Decima/Pointer
        # networks: the goal isn't a new representation but a per-key
        # selection score.

        # Delay head: small MLP on (query, mean(proc), max(proc))
        self.delay_mlp = nn.Sequential(
            nn.Linear(hidden * 3, delay_hidden), nn.GELU(),
            nn.Linear(delay_hidden,    K_delay),
        )

    # -----------------------------------------------------------------
    def forward(
        self,
        x_task:        torch.Tensor,       # (sum_tasks, h)
        x_proc:        torch.Tensor,       # (sum_proc, h)
        batch_task:    torch.Tensor,       # (sum_tasks,)
        batch_proc:    torch.Tensor,       # (sum_proc,)
        current_idx_within_graph: torch.Tensor,   # (B,) — see below
        action_mask:   torch.Tensor,       # (B, max_proc) bool
    ) -> Dict[str, torch.Tensor]:
        """Run the actor heads.

        ``current_idx_within_graph[g]`` is the *within-graph* index of
        the task being scheduled for graph ``g``.  The caller's job is to
        compute it from ``data['task'].current_idx`` (which is per-graph
        local) — we do NOT do that here because the ragged offsets depend
        on the batch object, not just the encoder output.
        """
        B, max_proc = action_mask.shape
        device      = x_task.device

        # 1. Pad procs into (B, max_proc, h) with zero pad.  Pads are
        # masked out below via the action_mask, so their value doesn't
        # leak into softmax probabilities.
        proc_pad, proc_counts = _pad_ragged(x_proc, batch_proc, B)

        # 2. Look up the query embedding (current task's encoded embed)
        # in the flat x_task tensor.  We need the GLOBAL flat index, which
        # is `(start_of_graph_g_in_flat) + current_idx_within_graph[g]`.
        if x_task.size(0) == 0:
            # All graphs in the batch are empty (no uncompleted tasks).
            # This shouldn't happen mid-rollout under normal play because
            # the env always has a "current task" to schedule, but it can
            # happen briefly at episode boundaries.  Fall back to a zero
            # query — the resulting action is meaningless because the
            # episode is about to end anyway, and the rollout buffer
            # discards this transition's gradient via the done mask.
            q_emb = torch.zeros(B, self.hidden, device=device, dtype=proc_pad.dtype)
        else:
            task_counts = torch.bincount(batch_task, minlength=B)
            task_starts = torch.cat([
                x_task.new_zeros(1, dtype=torch.long),
                task_counts.cumsum(0)[:-1],
            ])
            global_q_idx = task_starts + current_idx_within_graph.to(device)
            global_q_idx = global_q_idx.clamp(min=0, max=x_task.size(0) - 1)
            q_emb = x_task[global_q_idx]               # (B, h)

        # 3. Multi-head scaled dot-product placement logits
        #    Q : (B, num_heads, head_dim)
        #    K : (B, max_proc, num_heads, head_dim)
        Q = self.q_proj(q_emb).view(B, self.num_heads, self.head_dim)
        K = self.k_proj(proc_pad).view(B, max_proc, self.num_heads, self.head_dim)
        # scores : (B, num_heads, max_proc) = Q · K
        scores = torch.einsum("bhd,bnhd->bhn", Q, K) / (self.head_dim ** 0.5)
        # average over heads -> per-proc logit
        placement_logits = scores.mean(dim=1)      # (B, max_proc)

        # 4. Mask invalid positions (padded slots OR T_i ≥ mask_temp)
        # Use a large-negative value so softmax assigns ~0 probability.
        # We use -1e9 (not -inf) to keep gradients well-defined when ALL
        # actions are masked (shouldn't happen, but a defensive cushion).
        mask_neg = -1e9
        placement_logits = placement_logits.masked_fill(~action_mask, mask_neg)

        # 5. Delay head: query + pooled-proc context
        # Use mean and max pooling for richer context (Decima-style).
        # Pool only over real procs, not padding, by replacing padding
        # rows with -inf for max-pool / 0 for sum (and dividing by counts).
        valid = action_mask.unsqueeze(-1).float()  # (B, max_proc, 1)
        proc_sum = (proc_pad * valid).sum(dim=1)   # (B, h)
        proc_mean = proc_sum / valid.sum(dim=1).clamp(min=1.0)  # (B, h)
        # Max pool: set masked positions to -inf
        masked_for_max = proc_pad.masked_fill(~action_mask.unsqueeze(-1), -1e9)
        proc_max = masked_for_max.max(dim=1).values   # (B, h)

        delay_input = torch.cat([q_emb, proc_mean, proc_max], dim=-1)  # (B, 3h)
        delay_logits = self.delay_mlp(delay_input)                     # (B, K_delay)

        return {
            "placement_logits": placement_logits,   # (B, max_proc)
            "placement_mask":   action_mask,        # (B, max_proc)
            "delay_logits":     delay_logits,       # (B, K_delay)
            "proc_embed_pad":   proc_pad,           # (B, max_proc, h)
            "proc_count":       proc_counts,        # (B,)
            "query_embed":      q_emb,              # (B, h) — for critic
        }