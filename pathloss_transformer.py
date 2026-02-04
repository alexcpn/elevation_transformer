import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# CONSTANTS
# ============================================================

# Elevation normalization (applied in dataset)
ELEV_MEAN = 805.0
ELEV_STD = 736.0

# ============================================================
# MODEL COMPONENTS
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=768):
        super().__init__()

        # Learnable projection: elevation (1 dim) -> d_model dimensions
        self.elevation_projection = nn.Linear(1, d_model)

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        # Register as buffer (so it's automatically moved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len) - raw elevation values
        Returns: (batch_size, seq_len, d_model) - embedded elevation + positional encoding
        """
        seq_len = x.size(1)
        # Project elevation to d_model dimensions (learnable embedding)
        x = x.unsqueeze(-1)  # (B, seq_len, 1)
        x = self.elevation_projection(x)  # (B, seq_len, d_model) - learned representation
        # Add positional encoding
        return x + self.pe[:, :seq_len, :]  # (B, seq_len, d_model)

class PathLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        read_seq_length = 768
        d_model = 512  # embedding size
        final_size = 1
        num_heads = 8
        extra_feature_size = 4 # these are the 4 scalar features, frequency, receiver height, accesspoint height, and distance
        num_layers = 3 # Stack of 3 transformer layers
        dropout = 0.1

        self.pos_encoding = PositionalEncoding(d_model, read_seq_length)

        # Scalar feature embedding: Projects 4 scalars -> d_model
        self.extra_features_embedding = nn.Linear(extra_feature_size, d_model)

        # Transformer Encoder Stack (Processes Terrain)
        # batch_first=True means input is (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                 dim_feedforward=2048, dropout=dropout,
                                                 batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross Attention: Scalars (Query) -> Terrain (Key/Value)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Output Head
        # We take the transformed scalar token and predict path loss
        self.head = nn.Sequential(
            nn.Linear(d_model*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, final_size)
        )

    def forward(self, input_features, elevation_data, mask=None):
        """
        Args:
            input_features: (B, 4) scalar features (unnormalized)
            elevation_data: (B, 768) elevation profile (normalized)
            mask: (B, 768) bool tensor, True = padded position to ignore
        """
        # 1. Embed Scalars
        # input_features: (B, 4) -> (B, 1, 512)
        scalar_token = self.extra_features_embedding(input_features).unsqueeze(1)

        # 2. Embed Terrain
        # elevation_data: (B, 768) -> (B, 768, 512)
        elevation_tokens = self.pos_encoding(elevation_data)

        # 3. Process Terrain with Transformer (Self-Attention)
        # mask tells the transformer which positions are padding
        # (B, 768, 512)
        elevation_features = self.transformer(elevation_tokens, src_key_padding_mask=mask)

        # 4. Cross Attention: Scalars query the Terrain
        # Query: Scalar Token (B, 1, 512)
        # Key/Value: Elevation Features (B, 768, 512)
        # Output: (B, 1, 512)
        # This asks: "Given this Freq/Height, which terrain parts matter?"
        context, _ = self.cross_attention(query=scalar_token, key=elevation_features,
                                          value=elevation_features, key_padding_mask=mask)

        # 5. Fuse (Concatenate scalar + context)
        fused = torch.cat([scalar_token, context], dim=-1) # (B, 1, 1024)

        # 6. Predict
        logits = self.head(fused) # (B, 1, 1) -> (B, 1)
        return logits.squeeze(-1).squeeze(-1)


def create_model():
    return PathLossModel()


def load_weights(model, weights_path):
    """Load weights from file. Crashes if model architecture doesn't match."""
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model
