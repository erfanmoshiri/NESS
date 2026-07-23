"""
NESS — NEighborhood Statistics Self-supervision.

Our model for node classification under missing node features. Instead of
memorizing a per-node embedding for each missing node (as MATE does), NESS
predicts neighborhood feature statistics from graph structure via two SSL
objectives (neighborhood embedding spread + centroid residual), which
generalize under missingness.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from src.loss import *


def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "sage":
        # SAGEConv keeps a separate self-transform (W1·x_i + W2·mean(neigh)),
        # unlike GCN which blends self and neighbors — preserves per-node signal.
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gat":
        layer = GATConv(first_channels, second_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    else:
        raise ValueError("Unknown activation")

class GNNEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=2,
            dropout=0.5,
            bn=False,
            layer="gcn",
            activation="elu",
            use_node_feats=True,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats

        def heads_at(idx):  # GAT concatenates heads; non-final GAT layers use 4 heads
            return 1 if idx == num_layers - 1 or 'gat' not in layer else 4

        for i in range(num_layers):
            # input width must account for the previous GAT layer's head concatenation
            first_channels = in_channels if i == 0 else hidden_channels * heads_at(i - 1)
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = heads_at(i)

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels * heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x




class Con_Projector(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()

        self.proj = nn.Linear(in_channels, in_channels)
    def forward(self, x):
        x = self.proj(x)
        return x

class Projector(nn.Module):
    """Simple MLP Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.mlps[-1](x)
        x = self.activation(x)
        return x



def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges

class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, z_1, z_2, edge, sigmoid=True, reduction=False):
        x = z_1[edge[0]] * z_2[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x



def compute_neighborhood_embedding_stats(adj, embeddings, train_id):
    """
    Pre-compute neighborhood embedding statistics for SSL objective (adapted for embeddings).
    For each node, compute the SPREAD (std) of neighbor embeddings.

    EFFICIENT VERSION for large graphs: uses sparse matrix operations, no dense conversion.

    This captures local embedding diversity - analogous to IQR/density for raw features.
    The SSL objective will predict this spread, encouraging embeddings to encode
    neighborhood-level distributional information.

    Args:
        adj: Sparse adjacency matrix (FULL graph, before masking)
        embeddings: Node embeddings [N, D] (e.g., [N, 100] for OGBN-products)
        train_id: Indices of observable nodes

    Returns:
        stats: (N, D) tensor with std of neighbor embeddings per dimension
    """
    num_nodes = embeddings.size(0)
    device = embeddings.device

    print(f'  Computing neighborhood stats (efficient sparse version)...')

    # Create mask for observable nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_id] = True

    # Initialize stats tensor
    stats = torch.zeros_like(embeddings)

    # Work with coalesced sparse tensor
    adj = adj.coalesce()
    edge_index = adj.indices()  # [2, num_edges]

    # Batch process for efficiency
    batch_size = 10000
    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)

        for node_i in range(start_idx, end_idx):
            # Get neighbors from sparse edge_index
            neighbor_mask = edge_index[0] == node_i
            neighbors = edge_index[1, neighbor_mask]

            # Filter to observable neighbors
            obs_neighbors = neighbors[train_mask[neighbors]]

            if len(obs_neighbors) > 1:  # Need at least 2 for std
                # Standard deviation of neighbor embeddings per dimension
                neighbor_embs = embeddings[obs_neighbors].float()
                stats[node_i] = neighbor_embs.std(dim=0)
            elif len(obs_neighbors) == 1:
                # Only one neighbor: use small constant std
                stats[node_i] = torch.ones_like(embeddings[node_i]) * 0.1

        if (end_idx // batch_size) % 10 == 0:
            print(f'    Processed {end_idx:,} / {num_nodes:,} nodes...')

    return stats


def compute_neighborhood_centroid_residual(adj, embeddings, train_id):
    """
    Pre-compute residual (deviation from neighborhood centroid) for embeddings.
    For observable nodes: residual = node_embedding - mean(neighbor_embeddings)

    EFFICIENT VERSION for large graphs: uses sparse matrix operations.

    Same spirit as original residual loss, adapted for embedding space.
    Captures node-level deviations from local neighborhood patterns.

    Args:
        adj: Sparse adjacency matrix
        embeddings: Node embeddings [N, D] with masked nodes = 0
        train_id: Observable nodes

    Returns:
        residual: (N, D) tensor, non-zero only for observable nodes
    """
    num_nodes = embeddings.size(0)
    device = embeddings.device

    print(f'  Computing centroid residuals (efficient sparse version)...')

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_id] = True

    residual = torch.zeros_like(embeddings)

    # Work with coalesced sparse tensor
    adj = adj.coalesce()
    edge_index = adj.indices()  # [2, num_edges]

    # Only compute for observable nodes (much smaller set)
    for idx, node_i in enumerate(train_id):
        node_i = node_i.item() if torch.is_tensor(node_i) else node_i

        # Get neighbors from sparse edge_index
        neighbor_mask = edge_index[0] == node_i
        neighbors = edge_index[1, neighbor_mask]

        # Filter to observable neighbors
        obs_neighbors = neighbors[train_mask[neighbors]]

        if len(obs_neighbors) > 0:
            neighbor_mean = embeddings[obs_neighbors].float().mean(dim=0)
            residual[node_i] = embeddings[node_i].float() - neighbor_mean

        if idx % 50000 == 0 and idx > 0:
            print(f'    Processed {idx:,} / {len(train_id):,} observable nodes...')

    return residual


def compute_fixed_features(adj, true_features, train_id, vali_test_id):
    """
    Compute fixed features for missing nodes using neighbor averaging.

    EFFICIENT VERSION for large graphs: uses sparse matrix operations.

    Uses FULL adjacency but only averages features from OBSERVABLE nodes (train_id).
    This simulates the realistic scenario: structure is known, but features are missing.

    Args:
        adj: Sparse adjacency matrix (FULL graph)
        true_features: Features with missing nodes = 0
        train_id: Indices of observable nodes (40%)
        vali_test_id: Indices of missing nodes (60%)

    Returns:
        Fixed feature matrix (only missing nodes have non-zero values)
    """
    device = true_features.device
    num_nodes = true_features.size(0)

    print(f'  Computing fixed features (efficient sparse version)...')

    # Create mask for observable nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_id] = True

    # Global mean as fallback
    global_mean = true_features[train_id].mean(dim=0)

    # Initialize output
    fixed_features = torch.zeros_like(true_features)

    # Work with coalesced sparse tensor
    adj = adj.coalesce()
    edge_index = adj.indices()  # [2, num_edges]

    # Vectorized neighbor averaging for missing nodes
    vali_test_list = vali_test_id.tolist() if isinstance(vali_test_id, torch.Tensor) else vali_test_id

    for idx, node_idx in enumerate(vali_test_list):
        # Get all neighbors from sparse edge_index
        neighbor_mask = edge_index[0] == node_idx
        neighbors = edge_index[1, neighbor_mask]

        # Mask to only observable neighbors
        observable_mask = train_mask[neighbors]
        observable_neighbors = neighbors[observable_mask]

        if len(observable_neighbors) > 0:
            # Average observable neighbors' features
            fixed_features[node_idx] = true_features[observable_neighbors].mean(dim=0)
        else:
            # Fallback to global mean
            fixed_features[node_idx] = global_mean

        if idx % 100000 == 0 and idx > 0:
            print(f'    Processed {idx:,} / {len(vali_test_list):,} missing nodes...')

    return fixed_features


class Model(nn.Module):
    def __init__(
            self,
            encoder,
            edge_decoder,
            projector,
            con_projector,
            temp,
            pos_weight_tensor, neg_weight_tensor,
            mask=None,
            random_negative_sampling=False,
            loss="ce",
            feature_dim=None,  # For stats predictor (embedding dim)
            hidden_dim=None,
            num_classes=None,  # NEW: for classification head
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.projector = projector
        self.con_projector = con_projector
        self.mask = mask
        self.temp = temp
        self.pos_weight_tensor = pos_weight_tensor
        self.neg_weight_tensor = neg_weight_tensor

        # SSL Predictors (renamed for embeddings)
        if feature_dim is not None and hidden_dim is not None:
            # Stats predictor: predicts neighborhood embedding spread (std)
            self.stats_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
            # Residual predictor: predicts deviation from neighborhood centroid
            self.residual_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
        else:
            self.stats_predictor = None
            self.residual_predictor = None

        # NEW: Classification head for node classification (OGBN-products)
        if num_classes is not None and hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = None

        if loss == "ce":
            self.loss_edgefn = ce_loss
        else:
            raise ValueError(loss)
        self.contrastive_loss = calc_loss
        self.rec_loss = fts_rec_loss

        if random_negative_sampling:
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling


    def forward(self, data_1, data_2, norm_adj, fixed_features, train_fts_idx, vali_test_fts_idx):
        """
        Forward pass with FIXED features (no learnable parameters for features).

        Args:
            fixed_features: Pre-computed fixed features for missing nodes
        """
        x_1_, edge_index_1 = data_1.x, data_1.edge_index

        # Use fixed features instead of learnable
        zero_ = torch.zeros_like(fixed_features, device=fixed_features.device)
        zero = torch.zeros_like(fixed_features, device=fixed_features.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + fixed_features[vali_test_fts_idx]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index

        z_1 = self.encoder(x_1, edge_index_1)
        z_2 = self.encoder(x_2, edge_index_1)
        z = (z_1 + z_2) * 0.5  # Fuse views
        out = self.projector(z)
        return out


    def train_one_epoch(
            self, data_1, data_2, norm_adj, fixed_features, train_fts_idx, vali_test_fts_idx,
            target_stats=None, target_residual=None, labels=None,
            use_classification=False, batch_size=2 ** 16):
        """
        Train one epoch with FIXED features (adapted for embeddings).

        Args:
            target_stats: Neighborhood embedding spread (replaces target_stats)
            target_residual: Deviation from neighborhood centroid (same spirit)
            labels: Node labels for classification loss (for OGBN-products)
            use_classification: If True, add classification loss on masked nodes
        """
        x_1_, edge_index_1 = data_1.x, data_1.edge_index

        # Use fixed features instead of learnable
        zero_ = torch.zeros_like(fixed_features, device=fixed_features.device)
        zero = torch.zeros_like(fixed_features, device=fixed_features.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + fixed_features[vali_test_fts_idx]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index
        remaining_edges, masked_edges = self.mask(edge_index_1)

        aug_edge_index, _ = add_self_loops(edge_index_1)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data_1.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)


        for perm in DataLoader(
                range(masked_edges.size(1)), batch_size=batch_size, shuffle=True
        ):
            z_1 = self.encoder(x_1, remaining_edges)
            z_2 = self.encoder(x_2, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            # Cross-view edge prediction (both directions)
            pos_out_1 = self.edge_decoder(
                z_1, z_2, batch_masked_edges, sigmoid=False
            )
            neg_out_1 = self.edge_decoder(z_1, z_2, batch_neg_edges, sigmoid=False)

            pos_out_2 = self.edge_decoder(
                z_2, z_1, batch_masked_edges, sigmoid=False
            )
            neg_out_2 = self.edge_decoder(z_2, z_1, batch_neg_edges, sigmoid=False)

            loss_edge = (self.loss_edgefn(pos_out_1, neg_out_1) + self.loss_edgefn(pos_out_2, neg_out_2))

            # Contrastive loss between views
            z_1_p = z_1
            z_2_p = z_2
            loss_con = self.contrastive_loss(z_1_p, z_2_p, temperature=self.temp)

            z = (z_1 + z_2) * 0.5  # Fuse views for reconstruction


            x_recon = self.projector(z)
            loss_recon = self.rec_loss(x_recon[train_fts_idx], x_1_[train_fts_idx], self.pos_weight_tensor,
                                       self.neg_weight_tensor)

            # NEW: Classification loss on masked nodes (for OGBN-products)
            loss_classification = torch.tensor(0.0, device=z.device)
            if use_classification and self.classifier is not None and labels is not None:
                logits = self.classifier(z)
                loss_classification = F.cross_entropy(
                    logits[vali_test_fts_idx],
                    labels[vali_test_fts_idx]
                )

            # Stats SSL: predict neighborhood embedding spread from View 1
            loss_stats = torch.tensor(0.0, device=z.device)
            if self.stats_predictor is not None and target_stats is not None:
                pred_stats = self.stats_predictor(z_1)
                loss_stats = F.mse_loss(pred_stats, target_stats)

            # Residual SSL: predict deviation from centroid from View 2
            loss_residual = torch.tensor(0.0, device=z.device)
            if self.residual_predictor is not None and target_residual is not None:
                pred_residual = self.residual_predictor(z_2)
                # Only compute on observable nodes (where target is non-zero)
                mask = target_residual.abs().sum(dim=1) > 0
                if mask.sum() > 0:
                    loss_residual = F.mse_loss(pred_residual[mask], target_residual[mask])

            # Total loss (same weighting as original, plus classification)
            loss_total = loss_edge + loss_con + loss_recon + loss_classification + \
                        (100 * loss_stats) + (50 * loss_residual)


        return loss_total, loss_edge, loss_con, loss_recon, loss_classification, loss_stats, loss_residual

    def forward_classifier(self, z):
        """Get class logits from node embeddings (for OGBN-products evaluation)."""
        if self.classifier is not None:
            return self.classifier(z)
        return None

