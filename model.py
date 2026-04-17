# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from data_utils import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

class GATEncoder(nn.Module):
    """Graph Attention Network encoder."""
    def __init__(self, in_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM, hidden_dim=64, out_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=heads, concat=False, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, getattr(batch, "edge_attr", None)
        if getattr(self.gat1, "lin_edge", None) is None:
            edge_attr = None
        x = self.dropout(torch.relu(self.gat1(x, edge_index, edge_attr=edge_attr)))
        x = self.dropout(torch.relu(self.gat2(x, edge_index, edge_attr=edge_attr)))
        graph_emb = global_mean_pool(x, batch.batch)
        return x, graph_emb


class MetaboliteGenerator(nn.Module):
    """
    GAT encoder + transformer decoder to generate metabolite SMILES.
    Includes a transformation head and an optional enzyme head.
    """
    def __init__(
        self,
        vocab_size,
        hidden_dim=256,
        num_layers=4,
        encoder_hidden_dim=64,
        encoder_out_dim=128,
        encoder_heads=4,
        decoder_heads=8,
        num_transform_classes=20,
        num_coarse_transform_classes=8,
        num_enzyme_classes=50,
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        max_len=128,
        dropout=0.1,
        use_enzyme_head=False,
    ):
        super().__init__()

        self.encoder = GATEncoder(
            in_dim=atom_feature_dim,
            edge_dim=bond_feature_dim,
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_out_dim,
            heads=encoder_heads,
            dropout=dropout,
        )

        self.smiles_emb = nn.Embedding(vocab_size, hidden_dim)
        self.position_emb = nn.Embedding(max_len, hidden_dim)
        self.memory_proj = nn.Linear(encoder_out_dim, hidden_dim)
        self.coarse_transform_emb = nn.Embedding(max(1, num_coarse_transform_classes), hidden_dim)
        self.transform_emb = nn.Embedding(num_transform_classes, hidden_dim)

        layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=decoder_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        self.reaction_center_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Linear(encoder_out_dim, 1),
        )
        self.coarse_transform_head = nn.Linear(encoder_out_dim, max(1, num_coarse_transform_classes))
        self.transform_head = nn.Linear(encoder_out_dim, num_transform_classes)
        self.enzyme_head = None
        if use_enzyme_head and num_enzyme_classes > 0:
            self.enzyme_head = nn.Linear(encoder_out_dim, num_enzyme_classes)

    def forward(self, graph, tgt_tokens, transform_labels=None, coarse_transform_labels=None):
        """
        Inputs:
        - graph: precursor molecular graph
        - tgt_tokens: [batch, seq_len] metabolite SMILES tokens (teacher forcing)
        """
        node_enc, graph_enc = self.encoder(graph)
        pred_reaction_center = self.reaction_center_head(node_enc).squeeze(-1)
        pred_coarse_transform = self.coarse_transform_head(graph_enc)
        pred_transform = self.transform_head(graph_enc)

        if transform_labels is None:
            transform_labels = pred_transform.argmax(dim=-1)
        if coarse_transform_labels is None:
            coarse_transform_labels = pred_coarse_transform.argmax(dim=-1)

        positions = torch.arange(tgt_tokens.size(1), device=tgt_tokens.device).unsqueeze(0)
        tgt = self.smiles_emb(tgt_tokens) + self.position_emb(positions)
        tgt_key_padding_mask = tgt_tokens.eq(0)

        graph_context = self.memory_proj(graph_enc)
        coarse_transform_context = self.coarse_transform_emb(coarse_transform_labels)
        transform_context = self.transform_emb(transform_labels)
        memory = torch.stack([graph_context, coarse_transform_context, transform_context], dim=1)

        seq_len = tgt_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=tgt_tokens.device),
            diagonal=1,
        )

        out = self.decoder(tgt, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.output_layer(out)

        pred_enzyme = self.enzyme_head(graph_enc) if self.enzyme_head is not None else None

        return logits, pred_transform, pred_coarse_transform, pred_reaction_center, pred_enzyme