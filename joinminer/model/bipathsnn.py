"""BiPathsNN model for link prediction."""

from .tokens_encoder import TokensEncoder
from .pos_embed import PositionEmbedding
from .mlp import MLP

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BiPathsNN(nn.Module):
    """BiPaths Neural Network for link prediction.

    Architecture:
        token_embed -> token_seq_embed -> path_group_embed -> path_group_seq_embed -> pair_embed -> pair_pred

    Args:
        config: Model configuration dict with structure:
            {
                "dropout": float,
                "feat_proj": {"feat_len": {...}, "dim": int},
                "path_group_encoder": {"nhead": int, "num_layers": int},
                "pair_encoder": {"nhead": int, "num_layers": int},
            }
    """

    def __init__(self, config):
        super().__init__()

        # Dropout
        self.dropout = nn.Dropout(config["dropout"])

        # Feature projection config
        feat_proj_config = config["feat_proj"]
        self.feat_len = feat_proj_config["feat_len"]
        token_dim = feat_proj_config["dim"]

        # Special embedding: PAD, CLS_forward, CLS_backward, CLS_bipath
        self.special_token_embed = nn.Embedding(4, token_dim)

        # Feature projection for each element type
        self.feat_proj_dict = nn.ModuleDict({
            'node': nn.ModuleDict(),
            'edge': nn.ModuleDict()
        })
        for element_type in ['node', 'edge']:
            for type_name, feat_len in self.feat_len[element_type].items():
                if feat_len > 0:
                    proj = nn.Linear(feat_len, token_dim)
                else:
                    proj = nn.Embedding(1, token_dim)

                self.feat_proj_dict[element_type][type_name] = proj

        # Position embedding for tokens in path
        # (4 dims: path_pos, node_pos, u_node_pos, v_node_pos)
        self.token_pos_embed = PositionEmbedding(
            d_model=token_dim,
            num_dims=4,
            max_freq=100
        )

        # Path group encoder (token sequence -> path group embedding)
        self.path_group_encoder = TokensEncoder(
            d_model=token_dim,
            nhead=config["path_group_encoder"]["nhead"],
            num_layers=config["path_group_encoder"]["num_layers"],
            dropout=config["dropout"],
        )

        # Special embedding: PAD, CLS for path group
        self.special_path_group_embed = nn.Embedding(2, token_dim)

        # Pair encoder (path group sequence -> pair embedding)
        self.pair_encoder = TokensEncoder(
            d_model=token_dim,
            nhead=config["pair_encoder"]["nhead"],
            num_layers=config["pair_encoder"]["num_layers"],
            dropout=config["dropout"],
            final_norm=True,
        )

        # Output projection (MLP with one hidden layer)
        self.output_proj = MLP(
            input_dim=token_dim,
            output_dim=1,
            hidden_dims=[token_dim // 2],
            dropout=config["dropout"],
            activation='gelu',
            final_activation=None,
        )

    def forward(self, batch):
        # Project features for each element type
        token_embed_list = [self.special_token_embed.weight]
        for element_type in ['node', 'edge']:
            for type_name, feat_tensor in batch.features[element_type].items():
                feat_len = self.feat_len[element_type][type_name]
                if feat_len > 0:
                    proj = self.feat_proj_dict[element_type][type_name]
                    token_embed = proj(feat_tensor)
                    token_embed_list.append(token_embed)
                else:
                    token_embed = self.feat_proj_dict[element_type][type_name].weight
                    token_embed_list.append(token_embed)

        # Combine all token embeddings
        token_embed = torch.vstack(token_embed_list)

        # Build token sequence with position embedding
        token_seq_embed = (
            token_embed[batch.token_to_path_group['embed_id']]
            + self.token_pos_embed(batch.token_to_path_group['pos_id'])
        )
        token_seq_embed = self.dropout(token_seq_embed)

        # Create attention mask for padding
        token_seq_mask = batch.token_to_path_group['embed_id'] == 0

        # Encode token sequence -> path group embedding
        path_group_embed = self.path_group_encoder(token_seq_embed, token_seq_mask)
        path_group_embed = path_group_embed[:, 0, :]

        # Prepend special path group embeddings (PAD, CLS)
        path_group_embed = torch.vstack(
            [self.special_path_group_embed.weight, path_group_embed]
        )

        # Build path group sequence for each pair
        path_group_seq_embed = path_group_embed[batch.path_group_to_sample]
        path_group_seq_embed = self.dropout(path_group_seq_embed)

        # Create attention mask for padding
        path_group_seq_mask = batch.path_group_to_sample == 0

        # Encode path group sequence -> pair embedding
        pair_embed = self.pair_encoder(path_group_seq_embed, path_group_seq_mask)
        pair_embed = pair_embed[:, 0, :]

        # Project to output
        pair_pred = self.output_proj(pair_embed)

        return pair_pred
