"""
NESS — our model — as a unified benchmark entry.

NEighborhood Statistics Self-supervision. Predicts neighborhood feature
statistics from graph structure (neighborhood spread + centroid) as SSL objectives,
alongside edge reconstruction, Barlow-Twins contrastive, and classification.

Wraps the model components from models/NESS.py so it runs through the SAME
benchmark runner / logging / saving as the baselines. Small graphs (<100k) train
full-batch; OGBN uses cluster-based training. SSL targets are computed once on
the full graph and indexed per cluster.
"""

import sys
import json
import time
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import add_self_loops, negative_sampling

sys.path.insert(0, '..')
from models.NESS import (
    GNNEncoder, EdgeDecoder, Projector, Con_Projector, Model,
)
from src.utils import MaskEdge


def _compute_ssl_targets(adj, features, masked_features, observable_id, device, ssl_hops=1):
    """
    Compute the two SSL targets on the full graph:
      - stats:    neighborhood embedding spread (std)
      - centroid: mean of observable neighbors' embeddings

    ssl_hops=1 uses direct (1-hop) neighbors. ssl_hops>1 aggregates over a k-hop
    ball — needed at high missingness where a 1-hop neighborhood is mostly empty.
    Computed on CPU (one-time precompute); only results move to device.
    """
    adj_cpu = adj.cpu()
    feats_cpu = features.cpu()
    obs_cpu = observable_id.cpu()

    if ssl_hops > 1:
        from src.fast_ssl_compute import compute_khop_neighborhood_targets
        target_centroid, target_stats = compute_khop_neighborhood_targets(
            adj_cpu, feats_cpu, obs_cpu, k=ssl_hops)
    else:
        from src.fast_ssl_compute import (
            compute_neighborhood_embedding_stats as stats_fn,
            compute_neighborhood_centroid as centroid_fn,
        )
        target_stats = stats_fn(adj_cpu, feats_cpu, obs_cpu)
        target_centroid = centroid_fn(adj_cpu, feats_cpu, obs_cpu)

    return target_stats.cpu(), target_centroid.cpu()


def _compute_label_histogram(adj, labels, observable_id, num_classes, k=2):
    """
    Deterministic SSL target: for each node, the class distribution of its
    OBSERVABLE neighbors within a k-hop ball (proximity-weighted).

    Uses only observable-node labels (no val/test leakage). Returns [N, C]
    probability distributions (rows sum to 1; nodes with no observable neighbor -> 0).
    """
    import torch as _t
    device = 'cpu'
    N = labels.size(0)
    adj = adj.coalesce()
    idx, val = adj.indices(), adj.values()
    deg = _t.zeros(N).scatter_add_(0, idx[0], val).clamp(min=1)
    A = _t.sparse_coo_tensor(idx, val / deg[idx[0]], (N, N)).coalesce()

    # one-hot labels, zeroed for non-observable nodes
    onehot = _t.zeros(N, num_classes)
    onehot[observable_id, labels[observable_id]] = 1.0

    acc = _t.zeros(N, num_classes)
    cur = onehot
    for _ in range(k):
        cur = _t.sparse.mm(A, cur)
        acc += cur

    row = acc.sum(dim=1, keepdim=True)
    hist = acc / row.clamp(min=1e-12)   # normalize to a distribution
    hist[row.squeeze(1) < 1e-12] = 0.0  # no observable neighbor within k hops
    return hist


def _build_csr(edge_index, num_nodes):
    """CSR (rowptr, col) for O(1) random-neighbor sampling. Built once per cluster."""
    src = edge_index[0]
    order = torch.argsort(src)
    col = edge_index[1][order]
    counts = torch.bincount(src, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(counts, 0)
    return rowptr, col


def _random_walk(rowptr, col, start, walk_len):
    """Vectorized random walk: [B] start nodes -> [B, walk_len+1] node sequences.
    Isolated nodes stay put. Fresh each call (stochastic)."""
    cur = start
    seq = [cur]
    ncol = col.size(0)
    for _ in range(walk_len):
        deg = (rowptr[cur + 1] - rowptr[cur])
        rand = (torch.rand(cur.size(0), device=cur.device) * deg.clamp(min=1).float()).long()
        nxt_idx = (rowptr[cur] + rand).clamp(max=ncol - 1)
        nxt = col[nxt_idx]
        cur = torch.where(deg > 0, nxt, cur)
        seq.append(cur)
    return torch.stack(seq, dim=1)


def _within_khop_mask(edge_index, num_nodes, k=3):
    """
    Boolean [N, N] mask: True if j is reachable from i within k hops (incl. self).
    Built once per cluster (cached). Complement = the 'far set' used for
    guaranteed-true negatives in the path objective. Dense; fine for cluster sizes.
    """
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    reach = (A + torch.eye(num_nodes)) > 0
    cum = reach.float()
    for _ in range(k - 1):
        cum = (cum @ A > 0).float()
        reach = reach | (cum > 0)
    return reach  # [N, N] bool: within-k-hop


def _landmark_distances(sp_adj, landmarks, num_nodes, device, max_hops=None, raw_hops=False):
    """
    Multi-source BFS hop-distance from each landmark to all nodes -> [num_nodes, K],
    normalized by the longest finite shortest path (diameter proxy). Unreachable -> 1.0.
    Vectorized: propagate a [N, K] reached-frontier via one sparse matmul per hop;
    newly-reached (node, landmark) pairs get the current hop as their distance.
    K = len(landmarks). Cheap (K small, few hops); used by the anchor-diff objective.
    """
    K = landmarks.numel()
    INF = float(num_nodes + 1)
    dist = torch.full((num_nodes, K), INF, device=device)
    reached = torch.zeros(num_nodes, K, device=device)
    ar = torch.arange(K, device=device)
    dist[landmarks, ar] = 0.0
    reached[landmarks, ar] = 1.0
    frontier = reached.clone()
    if max_hops is None:
        max_hops = num_nodes
    for hop in range(1, max_hops + 1):
        # who is reachable in exactly one more hop from the current frontier
        nxt = torch.sparse.mm(sp_adj, frontier)          # [N, K]
        newly = (nxt > 0) & (reached == 0)
        if not newly.any():
            break
        dist[newly] = float(hop)
        reached[newly] = 1.0
        frontier = newly.float()
    finite = dist[dist < INF]
    diameter = max(finite.max().item() if finite.numel() > 0 else 1.0, 1.0)
    if raw_hops:
        # integer hop distance; unreachable -> diameter+1 (caller usually buckets/clamps)
        dist[dist >= INF] = diameter + 1.0
        return dist  # [num_nodes, K] raw hops (float-valued integers)
    dist = dist.clamp(max=diameter) / diameter          # unreachable -> 1.0
    return dist  # [num_nodes, K] in [0,1]


def _ppr_diffuse(edge_index, x, num_nodes, alpha=0.15, iters=10):
    """Personalized-PageRank diffusion of features: z = (1-α) Ã z + α x, iterated.
    Static per cluster (precomputed once). Returns diffused feature matrix [N, D]."""
    dev = x.device
    ei = edge_index
    deg = torch.zeros(num_nodes, device=dev).scatter_add_(
        0, ei[0], torch.ones(ei.size(1), device=dev)).clamp(min=1)
    vals = (1.0 / deg[ei[0]].sqrt()) * (1.0 / deg[ei[1]].sqrt())  # sym-norm
    A = torch.sparse_coo_tensor(ei, vals, (num_nodes, num_nodes)).coalesce()
    h = x.clone()
    for _ in range(iters):
        h = (1 - alpha) * torch.sparse.mm(A, h) + alpha * x
    return h


def make_views(view2, x, edge_index, num_nodes, encoder, mask_edge, device,
               deep_encoder=None, x_raw=None, ppr_cache=None, single_view=False):
    """
    Build two encoded views (z1, z2) plus the view-1 masked edges used by edge/path loss.

    view1 is always the standard edge-masked encoding of x.
    view2 depends on `view2`:
      edge_mask         — second independent random edge-mask (default; current behavior)
      dropout           — same graph+features, second stochastic forward pass (dropout only)
      feat_mask         — view2 randomly zeroes a fraction of feature dims
      ppr               — view2 uses PPR-diffused features (static, precomputed in ppr_cache)
      prefill_contrast  — view1 = prefilled x, view2 = raw (zero-filled) x_raw
      deep              — view2 uses a deeper encoder (imbalanced depth)

    single_view=True  — ablation: build ONE view only (z1==z2). Removes the two-view
    structure entirely (contrastive becomes trivial → caller must disable w_con).

    Returns (z1, z2, masked_edges).
    """
    rem1, masked_edges = mask_edge(edge_index)

    if single_view:
        z = encoder(x, rem1)
        return z, z, masked_edges

    if view2 == 'edge_mask':
        rem2, _ = mask_edge(edge_index)
        z1 = encoder(x, rem1)
        z2 = encoder(x, rem2)
    elif view2 == 'dropout':
        # same graph+features; diversity comes only from dropout stochasticity
        z1 = encoder(x, rem1)
        z2 = encoder(x, rem1)
    elif view2 == 'feat_mask':
        rem2, _ = mask_edge(edge_index)
        fmask = (torch.rand(x.size(1), device=device) > 0.3).float()  # drop ~30% dims
        z1 = encoder(x, rem1)
        z2 = encoder(x * fmask, rem2)
    elif view2 == 'ppr':
        rem2, _ = mask_edge(edge_index)
        z1 = encoder(x, rem1)
        z2 = encoder(ppr_cache, rem2)         # diffused features (precomputed)
    elif view2 == 'prefill_contrast':
        # view1 = prefilled x, view2 = raw zero-filled features
        rem2, _ = mask_edge(edge_index)
        z1 = encoder(x, rem1)
        z2 = encoder(x_raw, rem2)
    elif view2 == 'deep':
        rem2, _ = mask_edge(edge_index)
        z1 = encoder(x, rem1)
        z2 = deep_encoder(x, rem2)            # deeper encoder (imbalanced depth)
    else:
        raise ValueError(f'unknown view2: {view2}')

    return z1, z2, masked_edges


def _cluster_to(c, device):
    """Move a cached cluster's tensors to `device` for a single step."""
    out = {
        'x': c['x'].to(device), 'y': c['y'].to(device),
        'edge_index': c['edge_index'].to(device),
        'aug_edge_index': c['aug_edge_index'].to(device),
        'global_indices': c['global_indices'],  # only used on CPU for eval gather
        'obs_mask': c['obs_mask'].to(device),
        'num_nodes': c['num_nodes'],
    }
    for key in ('target_stats', 'target_centroid', 'target_hist', 'x_raw'):
        if key in c:
            out[key] = c[key].to(device)
    return out


def _barlow_twins(z1, z2, lambda_param=0.005):
    N = z1.size(0)
    D = z1.size(1)
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = torch.mm(z1n.T, z2n) / N
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.fill_diagonal_(0).pow_(2).sum()
    return (on_diag + lambda_param * off_diag) / D


def _info_nce(z1, z2, temp=0.5, max_nodes=8192):
    # GRACE-style node-level InfoNCE: same node across views = positive; all other
    # nodes = negatives. Subsample rows for tractable N x N similarity.
    N = z1.size(0)
    if N > max_nodes:
        idx = torch.randperm(N, device=z1.device)[:max_nodes]
        z1, z2 = z1[idx], z2[idx]
        N = max_nodes
    h1 = F.normalize(z1, dim=1)
    h2 = F.normalize(z2, dim=1)
    sim = torch.mm(h1, h2.t()) / temp          # [N, N]
    labels = torch.arange(N, device=z1.device)
    # symmetric: view1->view2 and view2->view1
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def _proto_info_nce(z1, z2, y, obs_mask, num_classes, temp=0.5):
    # Class-aware (D2PT-style) contrastive: build per-class prototypes from OBSERVABLE
    # nodes in each view, contrast class-j prototype across views (positive) vs other
    # classes (negatives). Semi-supervised; O(C^2) not O(N^2). No leakage (obs only).
    idx = obs_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return z1.new_zeros(())
    yo = y[idx]
    h1 = F.normalize(z1[idx], dim=1)
    h2 = F.normalize(z2[idx], dim=1)
    C, D = num_classes, h1.size(1)
    oh = F.one_hot(yo, C).float()                      # [n_obs, C]
    cnt = oh.sum(0).clamp(min=1).unsqueeze(1)          # [C,1]
    p1 = F.normalize((oh.t() @ h1) / cnt, dim=1)       # [C, D] class prototypes, view1
    p2 = F.normalize((oh.t() @ h2) / cnt, dim=1)       # [C, D] view2
    present = oh.sum(0) > 0
    if present.sum() < 2:
        return z1.new_zeros(())
    p1, p2 = p1[present], p2[present]                  # [C', D] present classes only
    sim = torch.mm(p1, p2.t()) / temp                  # [C', C']
    labels = torch.arange(p1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def _build_model(num_features, num_classes, encoder_channels, hidden, decoder_channels,
                 dropout, p, device, encoder_layer='gcn', num_layers=2):
    encoder = GNNEncoder(num_features, encoder_channels, hidden,
                         num_layers=num_layers, dropout=dropout, layer=encoder_layer, activation='elu')
    edge_decoder = EdgeDecoder(hidden, decoder_channels, num_layers=2, dropout=0.3)
    projector = Projector(hidden, encoder_channels, out_channels=num_features,
                          num_layers=2, dropout=0.3)
    con_projector = Con_Projector(hidden, encoder_channels, out_channels=num_features,
                                  num_layers=2, dropout=0.3)
    mask_edge = MaskEdge(p=p)
    model = Model(encoder, edge_decoder, projector, con_projector,
                  temp=0.2, pos_weight_tensor=None, neg_weight_tensor=None,
                  mask=mask_edge, feature_dim=num_features, hidden_dim=hidden,
                  num_classes=num_classes)
    return model.to(device)


def train_NESS(graph, features, labels, observable_id, masked_id, vali_id, test_id,
               num_classes, device, adj=None, hidden=128, encoder_channels=256,
               decoder_channels=64, dropout=0.5, lr=0.001, weight_decay=5e-5,
               epochs=200, patience=20, num_parts=50, p=0.7,
               cache_device='cpu', prefill='fp', fp_iterations=40, ssl_hops=2,
               view2='edge_mask',        # second-view construction (see make_views)
               encoder_layer='gcn',      # encoder conv type: gcn / sage / gat
               num_layers=2,             # encoder depth (message-passing layers)
               single_view=False,        # ablation: one view only (forces w_con=0; no two-view structure)
               con_loss='barlow',         # contrastive loss: 'barlow' or 'infonce'
               no_view_avg=False,         # z=z1 (primary) instead of (z1+z2)/2; contrastive still pulls z1<->z2
               ssl_proj_head=False,       # SimCLR-style: SSL objectives act on proj(z), cls uses z directly
               ssl_warmup=0,              # epochs of SSL-first: cls ramps 0->1 over these epochs (SSL shapes z first)
               ppr_on_raw=True,          # ppr view diffuses raw zero-filled feats (True) vs FP-prefilled (False)
               ssl_objective=('hist', 'path'), # list/tuple of objectives, e.g. ['recon','path']
               walk_len=2,               # hops for 'path' objective (short = tight, discriminative locality)
               num_anchors=16, anchor_resample=5,  # 'anchor' objective: K landmarks, resample every N epochs
               # --- E8 factor-flip toggles (all on the anchor-distance quantity) ---
               freeze_landmarks=False,   # F2: sample landmarks ONCE, never resample (static)
               anchor_local=False,       # F3: use NEAR (k-hop-capped) reference distances vs far landmarks
               anchor_fixed=False,       # F1: per-node regression (predict z_u's K distances) vs pairwise diff
               anchor_local_hops=2,      # F3: hop cap when anchor_local=True
               w_edge=1.0, w_cls=1.0, w_con=1.0, w_recon=1.0, w_hist=1.0, w_path=1.0,
               w_triplet=1.0, w_anchor=1.0, w_stats=1.0, w_centroid=1.0,
               log_path=None, weights_path=None):
    """
    Args:
        graph: PyG Data with edge_index
        features: [N, D] ground-truth features
        labels: [N] labels
        observable_id, masked_id, vali_id, test_id: node splits
        num_classes: number of classes
        adj: sparse adjacency (needed for SSL target computation)
        ssl_objective: comma-separated list of active SSL objectives. Supported:
            'recon' (masked-feature reconstruction), 'hist' (neighbor label
            histogram — semi-supervised), 'path' (stochastic multi-hop link
            prediction), 'none'.

    Returns:
        val_f1, test_f1
    """
    from data_loader import is_small

    # Active SSL objectives as a set (accepts list/tuple; 'none' or empty -> no SSL)
    if ssl_objective is None:
        ssl_set = set()
    elif isinstance(ssl_objective, str):
        ssl_set = set() if ssl_objective in ('none', '') else {ssl_objective}
    else:
        ssl_set = {s for s in ssl_objective if s and s != 'none'}

    # Single-view ablation: z1==z2, so the contrastive (Barlow-Twins) term is trivial
    # and meaningless — force it off regardless of the passed w_con.
    if single_view and w_con != 0:
        print('  single_view=True -> forcing w_con=0 (contrastive is trivial with one view)')
        w_con = 0.0

    num_nodes = features.size(0)
    num_features = features.size(1)

    # Zero out missing nodes
    masked_features = features.clone()
    masked_features[masked_id] = 0.0
    raw_features = masked_features.clone()  # zero-filled (pre-prefill); for prefill_contrast view

    # Optional: prefill missing nodes via Feature Propagation (long-range reach),
    # instead of leaving them at 0. Helps at high missingness where 2-hop
    # aggregation starves.
    if prefill == 'fp':
        from FP import feature_propagation
        obs_mask_bool = torch.zeros(num_nodes, dtype=torch.bool)
        obs_mask_bool[observable_id.cpu()] = True
        print(f'  Prefilling missing nodes via Feature Propagation ({fp_iterations} iters)...')
        propagated = feature_propagation(
            graph.edge_index, masked_features.cpu(), obs_mask_bool, num_nodes,
            num_iterations=fp_iterations, device=device
        ).cpu()
        # Keep observable rows exact; fill only masked rows with propagated values
        masked_features[masked_id] = propagated[masked_id.cpu()]

    elif prefill == 'mean':
        # Fill missing nodes with the mean of their OBSERVED neighbors (one hop of
        # neighbor averaging), a simple baseline prefill vs FP's iterative diffusion.
        print('  Prefilling missing nodes via observed-neighbor mean...')
        ei = graph.edge_index
        obs_bool = torch.zeros(num_nodes, dtype=torch.bool)
        obs_bool[observable_id.cpu()] = True
        obs_feat = masked_features.clone()  # observed rows real, missing rows zero
        vals = torch.ones(ei.size(1))
        A = torch.sparse_coo_tensor(ei, vals, (num_nodes, num_nodes)).coalesce()
        obs_ind = obs_bool.float().unsqueeze(1)                       # [N,1]
        nbr_sum = torch.sparse.mm(A, obs_feat)                        # sum of observed-neighbor feats
        nbr_cnt = torch.sparse.mm(A, obs_ind).clamp(min=1)           # count of observed neighbors
        nbr_mean = nbr_sum / nbr_cnt
        masked_features[masked_id] = nbr_mean[masked_id.cpu()]

    # SSL targets (once, on full graph) — only compute if a feature-neighborhood
    # objective (stats/centroid) is actually active (they are fixed-input negatives).
    target_stats = target_centroid = None
    if 'stats' in ssl_set or 'centroid' in ssl_set:
        print(f'  Computing SSL targets (neighborhood stats + centroid, {ssl_hops}-hop)...')
        target_stats, target_centroid = _compute_ssl_targets(
            adj, features, masked_features, observable_id, device, ssl_hops=ssl_hops)

    obs_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    obs_mask_full[observable_id.cpu()] = True

    # Neighbor label-histogram target (deterministic; observable labels only)
    target_hist = None
    if 'hist' in ssl_set:
        print(f'  Computing neighbor label-histogram target ({ssl_hops}-hop, observable labels)...')
        target_hist = _compute_label_histogram(
            adj.cpu(), labels.cpu(), observable_id.cpu(), num_classes, k=ssl_hops)

    mask_edge = MaskEdge(p=p)
    model = _build_model(num_features, num_classes, encoder_channels, hidden,
                         decoder_channels, dropout, p, device, encoder_layer=encoder_layer,
                         num_layers=num_layers)
    # Deeper encoder for the 'deep' view2 (imbalanced-depth contrast); built only if needed.
    # One layer deeper than the main encoder.
    deep_encoder = (GNNEncoder(num_features, encoder_channels, hidden,
                               num_layers=num_layers + 1, dropout=dropout, layer=encoder_layer,
                               activation='elu')
                    .to(device) if view2 == 'deep' else None)
    # Extra prediction heads per objective
    hist_head = torch.nn.Linear(hidden, num_classes).to(device) if 'hist' in ssl_set else None
    # Dedicated path-decoder (separate from edge_decoder, which learns 1-hop edges;
    # sharing would conflict since a k-hop-reachable pair is a negative for edge loss).
    path_head = (torch.nn.Sequential(
        torch.nn.Linear(hidden, decoder_channels), torch.nn.ReLU(),
        torch.nn.Linear(decoder_channels, 1)
    ).to(device) if 'path' in ssl_set else None)
    # Anchor head. Two forms (F1 flip):
    #   pairwise (default): concat(z_u, z_v) -> 1  (signed distance diff; input-varying)
    #   fixed (anchor_fixed): z_u -> K            (per-node distance regression; fixed input)
    if 'anchor' in ssl_set:
        if anchor_fixed:
            anchor_head = torch.nn.Sequential(
                torch.nn.Linear(hidden, decoder_channels), torch.nn.ReLU(),
                torch.nn.Linear(decoder_channels, num_anchors)
            ).to(device)
        else:
            anchor_head = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden, decoder_channels), torch.nn.ReLU(),
                torch.nn.Linear(decoder_channels, 1)
            ).to(device)
    else:
        anchor_head = None
    # anchorcls: per-node hop-bucket CLASSIFICATION of landmark distances (harder,
    # class-relevant target — probe showed anchor-distance predicts class at F1~0.21).
    # Predicts, for each of K landmarks, which hop-bucket the node falls in.
    ANCHORCLS_BUCKETS = 6   # hop buckets: 0,1,2,3,4,>=5
    anchorcls_head = (torch.nn.Sequential(
        torch.nn.Linear(hidden, decoder_channels), torch.nn.ReLU(),
        torch.nn.Linear(decoder_channels, num_anchors * ANCHORCLS_BUCKETS)
    ).to(device) if 'anchorcls' in ssl_set else None)
    # Optional SSL projection head (SimCLR-style): SSL objectives act on proj(z),
    # classification/edge/contrastive keep using z directly. Decouples the pretext
    # geometry from the representation used downstream. Off by default.
    ssl_proj = (torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden)
    ).to(device) if ssl_proj_head else None)
    params = list(model.parameters())
    if anchorcls_head is not None:
        params += list(anchorcls_head.parameters())
    if ssl_proj is not None:
        params += list(ssl_proj.parameters())
    if hist_head is not None:
        params += list(hist_head.parameters())
    if path_head is not None:
        params += list(path_head.parameters())
    if anchor_head is not None:
        params += list(anchor_head.parameters())
    if deep_encoder is not None:
        params += list(deep_encoder.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # cache_dev: where cached cluster tensors live.
    #   'gpu' -> preload all clusters on GPU (fast; needs full capacity)
    #   'cpu' -> hold on CPU, move one cluster to GPU per step (frugal)
    cache_dev = device if cache_device == 'gpu' else torch.device('cpu')

    # ---- Build cluster cache (single full-graph cluster if small) ----
    # num_parts=1 forces the full-batch path even for large graphs (METIS can't
    # partition into 1 part). Useful for diagnosing clustering-induced edge loss.
    if is_small(num_nodes) or num_parts == 1:
        print('  Full-batch (single cluster, no partitioning)')
        ei = graph.edge_index
        aug_ei, _ = add_self_loops(ei)
        gidx = torch.arange(num_nodes)
        single = {
            'x': masked_features.to(cache_dev), 'y': labels.to(cache_dev),
            'edge_index': ei.to(cache_dev), 'aug_edge_index': aug_ei.to(cache_dev),
            'global_indices': gidx.to(cache_dev),
            'obs_mask': obs_mask_full.to(cache_dev),
            'num_nodes': num_nodes,
        }
        if view2 == 'prefill_contrast' or (view2 == 'ppr' and ppr_on_raw):
            single['x_raw'] = raw_features.to(cache_dev)
        if target_stats is not None:
            single['target_stats'] = target_stats.to(cache_dev)
            single['target_centroid'] = target_centroid.to(cache_dev)
        if target_hist is not None:
            single['target_hist'] = target_hist.to(cache_dev)
        cluster_cache = [single]
    else:
        graph_cpu = graph.clone()
        graph_cpu.x = masked_features.cpu()
        graph_cpu.y = labels.cpu()
        if view2 == 'prefill_contrast' or (view2 == 'ppr' and ppr_on_raw):
            graph_cpu.x_raw = raw_features.cpu()  # ClusterData permutes it alongside x
        print(f'  Partitioning graph into {num_parts} clusters (cache_device={cache_device})...')
        cluster_data = ClusterData(graph_cpu, num_parts=num_parts,
                                   save_dir=None, log=False)
        cache_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=0)
        node_perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr
        cluster_cache = []
        for i, batch in enumerate(cache_loader):
            gidx_cpu = node_perm[partptr[i]:partptr[i+1]]
            aug_ei, _ = add_self_loops(batch.edge_index)
            entry = {
                'x': batch.x.to(cache_dev), 'y': batch.y.to(cache_dev),
                'edge_index': batch.edge_index.to(cache_dev),
                'aug_edge_index': aug_ei.to(cache_dev),
                'global_indices': gidx_cpu.to(cache_dev),
                'obs_mask': obs_mask_full[gidx_cpu].to(cache_dev),
                'num_nodes': batch.num_nodes,
            }
            if view2 == 'prefill_contrast' or (view2 == 'ppr' and ppr_on_raw):
                entry['x_raw'] = batch.x_raw.to(cache_dev)
            if target_stats is not None:
                entry['target_stats'] = target_stats[gidx_cpu].to(cache_dev)
                entry['target_centroid'] = target_centroid[gidx_cpu].to(cache_dev)
            if target_hist is not None:
                entry['target_hist'] = target_hist[gidx_cpu].to(cache_dev)
            cluster_cache.append(entry)
        print(f'  Cached {len(cluster_cache)} clusters')

    encoder = model.encoder
    edge_decoder = model.edge_decoder

    print(f'  Training NESS for up to {epochs} epochs (patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        el = {'total': 0.0, 'edge': 0.0, 'con': 0.0, 'cls': 0.0,
              'recon': 0.0, 'hist': 0.0, 'path': 0.0, 'triplet': 0.0, 'anchor': 0.0,
              'anchorcls': 0.0, 'stats': 0.0, 'centroid': 0.0}
        random.shuffle(cluster_cache)

        for c0 in cluster_cache:
            # Move this cluster to GPU for the step (no-op if already on GPU)
            c = _cluster_to(c0, device) if cache_device == 'cpu' else c0
            edge_index = c['edge_index']

            # PPR-diffused features (static) — precompute once per cluster, memoize on c0
            ppr_cache = None
            if view2 == 'ppr':
                if 'ppr' not in c0:
                    # Diffuse RAW zero-filled features (ppr_on_raw) so the view is a genuine
                    # alternative completion — not a re-diffusion of the FP-prefilled x
                    # (which would double-smooth and make the two views near-redundant).
                    ppr_src = c['x_raw'] if (ppr_on_raw and 'x_raw' in c) else c['x']
                    c0['ppr'] = _ppr_diffuse(edge_index, ppr_src, c['num_nodes']).to(
                        torch.device('cpu') if cache_device == 'cpu' else device)
                ppr_cache = c0['ppr'].to(device)

            # Build the two views (view1 = edge-masked; view2 per --view2)
            z1, z2, masked_edges = make_views(
                view2, c['x'], edge_index, c['num_nodes'], encoder, mask_edge, device,
                deep_encoder=deep_encoder, x_raw=c.get('x_raw'), ppr_cache=ppr_cache,
                single_view=single_view)
            z = z1 if no_view_avg else (z1 + z2) * 0.5

            num_neg = min(masked_edges.size(1), 50000)
            neg_edges = negative_sampling(c['aug_edge_index'], num_nodes=c['num_nodes'],
                                          num_neg_samples=num_neg, method='sparse')

            # Edge loss (subsample edges)
            max_edges = 100000
            if masked_edges.size(1) > max_edges:
                perm = torch.randperm(masked_edges.size(1), device=device)[:max_edges]
                masked_edges = masked_edges[:, perm]
            if neg_edges.size(1) > max_edges:
                perm = torch.randperm(neg_edges.size(1), device=device)[:max_edges]
                neg_edges = neg_edges[:, perm]

            pos1 = edge_decoder(z1, z2, masked_edges, sigmoid=False)
            pos2 = edge_decoder(z2, z1, masked_edges, sigmoid=False)
            neg1 = edge_decoder(z1, z2, neg_edges, sigmoid=False)
            neg2 = edge_decoder(z2, z1, neg_edges, sigmoid=False)
            loss_edge = (
                F.binary_cross_entropy_with_logits(pos1, torch.ones_like(pos1)) +
                F.binary_cross_entropy_with_logits(pos2, torch.ones_like(pos2)) +
                F.binary_cross_entropy_with_logits(neg1, torch.zeros_like(neg1)) +
                F.binary_cross_entropy_with_logits(neg2, torch.zeros_like(neg2))
            ) / 4

            # Classification — observable nodes only (no label leakage)
            y = c['y'].squeeze() if c['y'].dim() > 1 else c['y']
            obs = c['obs_mask']

            if con_loss == 'infonce':
                loss_con = _info_nce(z1, z2)
            elif con_loss == 'proto':
                loss_con = _proto_info_nce(z1, z2, y, obs, num_classes)
            else:
                loss_con = _barlow_twins(z1, z2)

            logits = model.forward_classifier(z)
            loss_cls = F.cross_entropy(logits[obs], y[obs])

            # --- SSL objectives (any subset active via ssl_set) ---
            # SSL objectives operate on z_ssl: proj(z) if a projection head is enabled,
            # else z itself. Classification/edge/contrastive above always use z directly.
            z_ssl = ssl_proj(z) if ssl_proj is not None else z
            loss_ssl = torch.tensor(0.0, device=device)

            ssl_vals = {'recon': 0.0, 'hist': 0.0, 'path': 0.0, 'triplet': 0.0, 'anchor': 0.0,
                        'anchorcls': 0.0, 'stats': 0.0, 'centroid': 0.0}  # per-step raw

            if 'centroid' in ssl_set:
                # Fixed-input per-node: predict neighborhood centroid from z (redundant
                # with aggregation — kept as a negative-result baseline for the ablation).
                loss_centroid = F.mse_loss(model.residual_predictor(z_ssl), c['target_centroid'])
                loss_ssl = loss_ssl + w_centroid * loss_centroid
                ssl_vals['centroid'] = loss_centroid.detach().item()

            if 'stats' in ssl_set:
                # Fixed-input per-node: predict neighborhood spread (std) from z.
                loss_stats = F.mse_loss(model.stats_predictor(z_ssl), c['target_stats'])
                loss_ssl = loss_ssl + w_stats * loss_stats
                ssl_vals['stats'] = loss_stats.detach().item()

            if 'recon' in ssl_set:
                # Masked-feature self-reconstruction: hide half the observable nodes,
                # reconstruct their TRUE features from neighbors.
                obs_idx = obs.nonzero(as_tuple=True)[0]
                if obs_idx.numel() > 1:
                    perm = torch.randperm(obs_idx.numel(), device=device)
                    held = obs_idx[perm[:obs_idx.numel() // 2]]
                    x_recon = c['x'].clone()
                    target_feat = x_recon[held].clone()
                    x_recon[held] = 0.0
                    z_recon = encoder(x_recon, edge_index)
                    pred_feat = model.projector(z_recon[held])
                    loss_recon = F.mse_loss(pred_feat, target_feat)
                    loss_ssl = loss_ssl + w_recon * loss_recon
                    ssl_vals['recon'] = loss_recon.detach().item()

            if 'hist' in ssl_set:
                # Neighbor label-histogram (semi-supervised): predict class distribution
                # of a node's observable k-hop neighbors. Observable nodes only.
                pred_logp = F.log_softmax(hist_head(z_ssl[obs]), dim=1)
                target = c['target_hist'][obs]
                valid = target.sum(dim=1) > 0
                if valid.any():
                    loss_hist = F.kl_div(pred_logp[valid], target[valid], reduction='batchmean')
                    loss_ssl = loss_ssl + w_hist * loss_hist
                    ssl_vals['hist'] = loss_hist.detach().item()

            if 'path' in ssl_set:
                # Stochastic multi-hop link prediction:
                #   positive = (start, endpoint) from a fresh walk_len-hop random walk
                #   negative = (start, node beyond 3 hops) — guaranteed NOT reachable,
                #              so no false negatives; re-sampled each step (stochastic).
                nnodes = c['num_nodes']
                # CSR + far-mask are fixed per cluster — build once, memoize on c0.
                # far = >=3 hops (complement of within-2-hop), so the 3-hop ring counts
                # as far and nothing is discarded between near (<=2) and far (>=3).
                if 'csr' not in c0:
                    rp, cl = _build_csr(edge_index.cpu(), nnodes)
                    c0['csr'] = (rp, cl)
                    within2 = _within_khop_mask(edge_index.cpu(), nnodes, k=2)
                    c0['far_mask'] = (~within2)  # True where node is >=3 hops away
                rowptr, col = c0['csr']
                rowptr, col = rowptr.to(device), col.to(device)
                far_mask = c0['far_mask'].to(device)      # [N, N] bool

                B = min(nnodes, 4096)
                starts = torch.randint(0, nnodes, (B,), device=device)
                walk = _random_walk(rowptr, col, starts, walk_len)
                ends = walk[:, -1]                         # reachable ≤ walk_len hops (positive)

                # Negatives: random node in each start's >=3-hop far set. Resample a few
                # rounds (vectorized) to replace any candidate that isn't actually far.
                cand = torch.randint(0, nnodes, (B,), device=device)
                for _ in range(5):
                    bad = ~far_mask[starts, cand]
                    if not bad.any():
                        break
                    cand[bad] = torch.randint(0, nnodes, (int(bad.sum()),), device=device)
                neg_ends = cand

                # Dedicated path head (separate from edge decoder)
                p_out = path_head(z_ssl[starts] * z_ssl[ends]).squeeze(-1)
                n_out = path_head(z_ssl[starts] * z_ssl[neg_ends]).squeeze(-1)
                loss_path = (
                    F.binary_cross_entropy_with_logits(p_out, torch.ones_like(p_out)) +
                    F.binary_cross_entropy_with_logits(n_out, torch.zeros_like(n_out))
                ) / 2
                loss_ssl = loss_ssl + w_path * loss_path
                ssl_vals['path'] = loss_path.detach().item()

            if 'triplet' in ssl_set:
                # Distance-ranking triplet: anchor A, near node (≤walk_len hops),
                # far node (>=3 hops). Embedding of A must be CLOSER to near than far.
                # Fresh triplets each step (stochastic, input-varying); label-free.
                nnodes = c['num_nodes']
                if 'csr' not in c0:
                    rp, cl = _build_csr(edge_index.cpu(), nnodes)
                    c0['csr'] = (rp, cl)
                    within2 = _within_khop_mask(edge_index.cpu(), nnodes, k=2)
                    c0['far_mask'] = (~within2)  # >=3 hops
                rowptr, col = c0['csr']
                rowptr, col = rowptr.to(device), col.to(device)
                far_mask = c0['far_mask'].to(device)

                B = min(nnodes, 4096)
                anchor = torch.randint(0, nnodes, (B,), device=device)
                near = _random_walk(rowptr, col, anchor, walk_len)[:, -1]  # ≤ walk_len hops
                far = torch.randint(0, nnodes, (B,), device=device)
                for _ in range(5):
                    bad = ~far_mask[anchor, far]
                    if not bad.any():
                        break
                    far[bad] = torch.randint(0, nnodes, (int(bad.sum()),), device=device)

                # Distances in embedding space; near should be closer than far by a margin
                d_near = (z_ssl[anchor] - z_ssl[near]).pow(2).sum(dim=1)
                d_far = (z_ssl[anchor] - z_ssl[far]).pow(2).sum(dim=1)
                loss_triplet = F.relu(d_near - d_far + 1.0).mean()   # margin = 1.0
                loss_ssl = loss_ssl + w_triplet * loss_triplet
                ssl_vals['triplet'] = loss_triplet.detach().item()

            if 'anchor' in ssl_set:
                # Anchor-diff (input-varying, global): sample K landmarks (resampled every
                # `anchor_resample` epochs), BFS their normalized distances to all nodes.
                # Each step, sample fresh node pairs (u, v); predict the SIGNED difference
                # d(u,anchor) - d(v,anchor) from concat(z_u, z_v). Relational → input varies.
                nnodes = c['num_nodes']
                # normalized sparse adjacency for multi-source BFS (built once per cluster)
                if 'bfs_adj' not in c0:
                    ei_cpu = edge_index.cpu()
                    vals = torch.ones(ei_cpu.size(1))
                    c0['bfs_adj'] = torch.sparse_coo_tensor(ei_cpu, vals, (nnodes, nnodes)).coalesce()
                # (re)sample references + distances. F2: freeze_landmarks -> sample once,
                # never resample (static). F3: anchor_local -> cap BFS at k hops so the
                # distance signal is local-reach only (else full-reach = global).
                hop_cap = anchor_local_hops if anchor_local else None
                need_resample = (not freeze_landmarks) and \
                    (c0.get('anchor_epoch', -999) // anchor_resample != epoch // anchor_resample)
                if 'anchor_dist' not in c0 or need_resample:
                    lm = torch.randperm(nnodes)[:num_anchors]
                    c0['anchor_dist'] = _landmark_distances(
                        c0['bfs_adj'].to(device), lm.to(device), nnodes, device, max_hops=hop_cap)
                    c0['anchor_epoch'] = epoch
                adist = c0['anchor_dist'].to(device)        # [N, K] in [0,1]

                B = min(nnodes, 4096)
                if anchor_fixed:
                    # F1 fixed-input: per-node regression — predict node u's K distances from z_u.
                    u = torch.randint(0, nnodes, (B,), device=device)
                    target = adist[u]                        # [B, K]
                    pred = anchor_head(z_ssl[u])             # [B, K]
                    loss_anchor = F.mse_loss(pred, target)
                else:
                    # F1 varying-input: pairwise signed distance difference from concat(z_u, z_v).
                    u = torch.randint(0, nnodes, (B,), device=device)
                    v = torch.randint(0, nnodes, (B,), device=device)
                    a = torch.randint(0, num_anchors, (B,), device=device)   # which reference
                    target = adist[u, a] - adist[v, a]       # signed distance difference
                    pred = anchor_head(torch.cat([z_ssl[u], z_ssl[v]], dim=1)).squeeze(-1)
                    loss_anchor = F.mse_loss(pred, target)
                loss_ssl = loss_ssl + w_anchor * loss_anchor
                ssl_vals['anchor'] = loss_anchor.detach().item()

            if 'anchorcls' in ssl_set:
                # Per-node hop-bucket CLASSIFICATION of landmark distances. Fixed-input
                # (target = node's own distances), but hard + class-relevant: predict, for
                # each of K landmarks, which hop-bucket (0,1,2,3,4,>=5) the node falls in.
                # Cross-entropy can't collapse like the MSE-of-differences anchor does.
                nnodes = c['num_nodes']
                if 'bfs_adj' not in c0:
                    ei_cpu = edge_index.cpu()
                    vals = torch.ones(ei_cpu.size(1))
                    c0['bfs_adj'] = torch.sparse_coo_tensor(ei_cpu, vals, (nnodes, nnodes)).coalesce()
                need_resample = (not freeze_landmarks) and \
                    (c0.get('anchorcls_epoch', -999) // anchor_resample != epoch // anchor_resample)
                if 'anchorcls_buckets' not in c0 or need_resample:
                    lm = torch.randperm(nnodes)[:num_anchors]
                    raw = _landmark_distances(c0['bfs_adj'].to(device), lm.to(device),
                                              nnodes, device, raw_hops=True)     # [N, K] int hops
                    c0['anchorcls_buckets'] = raw.clamp(max=ANCHORCLS_BUCKETS - 1).long()  # bucket >=5
                    c0['anchorcls_epoch'] = epoch
                buckets = c0['anchorcls_buckets'].to(device)     # [N, K] in {0..5}

                B = min(nnodes, 4096)
                u = torch.randint(0, nnodes, (B,), device=device)
                logits = anchorcls_head(z_ssl[u]).view(B, num_anchors, ANCHORCLS_BUCKETS)
                loss_anchorcls = F.cross_entropy(
                    logits.reshape(B * num_anchors, ANCHORCLS_BUCKETS),
                    buckets[u].reshape(B * num_anchors))
                loss_ssl = loss_ssl + w_anchor * loss_anchorcls
                ssl_vals['anchorcls'] = loss_anchorcls.detach().item()

            # SSL warmup: for the first `ssl_warmup` epochs, classification is downweighted
            # (ramps linearly 0 -> 1) so the SSL objective shapes the encoder FIRST, before
            # classification dominates. After warmup, cls_scale = 1 (full w_cls).
            cls_scale = min(1.0, epoch / ssl_warmup) if ssl_warmup > 0 else 1.0
            loss_total = w_edge * loss_edge + cls_scale * w_cls * loss_cls + w_con * loss_con + loss_ssl

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            el['total'] += loss_total.detach().item()
            el['edge'] += loss_edge.detach().item()
            el['con'] += loss_con.detach().item()
            el['cls'] += loss_cls.detach().item()
            el['recon'] += ssl_vals['recon']
            el['hist'] += ssl_vals['hist']
            el['path'] += ssl_vals['path']
            el['triplet'] += ssl_vals['triplet']
            el['anchor'] += ssl_vals['anchor']
            el['anchorcls'] += ssl_vals['anchorcls']
            el['stats'] += ssl_vals['stats']
            el['centroid'] += ssl_vals['centroid']

        n = len(cluster_cache)
        epoch_time = time.time() - epoch_start
        log_entry = {
            'epoch': epoch,
            'loss_total': el['total'] / n, 'loss_edge': el['edge'] / n,
            'loss_con': el['con'] / n, 'loss_cls': el['cls'] / n,
            'loss_recon': el['recon'] / n, 'loss_hist': el['hist'] / n,
            'loss_path': el['path'] / n, 'loss_triplet': el['triplet'] / n,
            'loss_anchor': el['anchor'] / n, 'loss_anchorcls': el['anchorcls'] / n,
            'loss_stats': el['stats'] / n, 'loss_centroid': el['centroid'] / n,
            'epoch_time_s': round(epoch_time, 2),
        }

        # Loss line every epoch; validation (expensive) every 5 epochs. Only show
        # the SSL objectives that are active.
        ssl_str = ' '.join(f'{name.capitalize()}: {el[name]/n:.4f}'
                           for name in ('centroid', 'stats', 'recon', 'hist', 'path', 'triplet', 'anchor', 'anchorcls') if name in ssl_set)
        base = (f'  Epoch {epoch}/{epochs} | Total: {el["total"]/n:.3f} '
                f'Edge: {el["edge"]/n:.3f} Con: {el["con"]/n:.3f} '
                f'Cls: {el["cls"]/n:.3f}' + (f' {ssl_str}' if ssl_str else ''))

        if epoch % 5 == 0:
            val_f1, _ = _eval(model, cluster_cache, labels, vali_id, device)
            log_entry['val_f1'] = val_f1
            print(f'{base} | Val F1: {val_f1:.4f} | {epoch_time:.1f}s')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch}')
                    if log_path:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                    break
        else:
            print(f'{base} | {epoch_time:.1f}s')

        if log_path:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    total_train_time = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
    if weights_path is not None:
        torch.save(model.state_dict(), weights_path)

    test_f1, test_acc = _eval(model, cluster_cache, labels, test_id, device)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    print(f'  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)')
    return best_val_f1, test_f1, test_acc


def _eval(model, cluster_cache, labels, eval_id, device, cache_device='cpu'):
    from sklearn.metrics import f1_score
    model.eval()
    eval_id_set = set(eval_id.cpu().tolist())
    all_preds = {}
    with torch.no_grad():
        for c0 in cluster_cache:
            x = c0['x'].to(device); ei = c0['edge_index'].to(device)
            z = model.encoder(x, ei)
            preds = model.forward_classifier(z).argmax(dim=1)
            for local_i, g in enumerate(c0['global_indices'].tolist()):
                if g in eval_id_set:
                    all_preds[g] = preds[local_i].item()
    from sklearn.metrics import accuracy_score
    eval_list = [n for n in eval_id.cpu().tolist() if n in all_preds]
    y_pred = [all_preds[n] for n in eval_list]
    y_true = labels[torch.tensor(eval_list)].cpu().tolist()
    return f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)
