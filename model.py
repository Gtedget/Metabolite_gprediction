# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class GATEncoder(nn.Module):
    """Graph Attention Network encoder."""
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=128):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim*4, out_dim, heads=4, concat=False)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        graph_emb = global_mean_pool(x, batch.batch)
        return graph_emb  # (batch_size, out_dim)


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
        num_transform_classes=20,
        num_enzyme_classes=50,
        use_enzyme_head=False,
    ):
        super().__init__()

        # GAT encoder
        self.encoder = GATEncoder(in_dim=1)

        # SMILES embedding
        self.smiles_emb = nn.Embedding(vocab_size, hidden_dim)
        self.memory_proj = nn.Linear(128, hidden_dim)

        # Transformer decoder
        layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True
        )
        self.decoder = TransformerDecoder(layer, num_layers=num_layers)

        # Final output projection
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        # Multi-task heads
        self.transform_head = nn.Linear(128, num_transform_classes)
        self.enzyme_head = None
        if use_enzyme_head and num_enzyme_classes > 0:
            self.enzyme_head = nn.Linear(128, num_enzyme_classes)

    def forward(self, graph, tgt_tokens):
        """
        Inputs:
        - graph: precursor molecular graph
        - tgt_tokens: [batch, seq_len] metabolite SMILES tokens (teacher forcing)
        """
        enc = self.encoder(graph)  # [B, 128]

        # Transformer expects [B, S, D]
        tgt = self.smiles_emb(tgt_tokens)

        # Expand encoder embedding for each token
        memory = self.memory_proj(enc).unsqueeze(1)

        # Decode
        out = self.decoder(tgt, memory)
        logits = self.output_layer(out)

        # Multi-task predictions
        pred_transform = self.transform_head(enc)
        pred_enzyme = self.enzyme_head(enc) if self.enzyme_head is not None else None

        return logits, pred_transform, pred_enzyme