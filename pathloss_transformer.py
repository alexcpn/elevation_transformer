import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# CONSTANTS
# ============================================================

# Feature normalization constants
FEAT_MEAN = torch.tensor([135920.0, 6300.0, 41.0, 89.0])
FEAT_STD = torch.tensor([46380.0, 100.0, 150.0, 35.0])

# Elevation normalization
ELEV_MEAN = 805.0
ELEV_STD = 736.0

# Target normalization
TARGET_MEAN = 218.0
TARGET_STD = 31.0

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

    def forward(self, input_features, elevation_data):
        # 1. Embed Scalars
        # input_features: (B, 4) -> (B, 1, 512)
        scalar_token = self.extra_features_embedding(input_features).unsqueeze(1)
        
        # 2. Embed Terrain
        # elevation_data: (B, 768) -> (B, 768, 512)
        elevation_tokens = self.pos_encoding(elevation_data)
        
        # 3. Process Terrain with Transformer (Self-Attention)
        # Ideally, the terrain learns its own features (hills, valleys) first
        # (B, 768, 512)
        elevation_features = self.transformer(elevation_tokens)
        
        # 4. Cross Attention: Scalars query the Terrain
        # Query: Scalar Token (B, 1, 512)
        # Key/Value: Elevation Features (B, 768, 512)
        # Output: (B, 1, 512)
        # This asks: "Given this Freq/Height, which terrain parts matter?"
        context, _ = self.cross_attention(query=scalar_token, key=elevation_features, value=elevation_features)
        
        # 5. Fuse (Residual Connection)
        # Add the context back to the scalar token
        fused = torch.cat([scalar_token, context], dim=-1) # (B, 1, 1024)
        
        # 6. Predict
        logits = self.head(fused) # (B, 1, 1) -> (B, 1)
        return logits.squeeze(1)


def create_model():
    return PathLossModel()

# ============================================================
# DATA UTILS
# ============================================================

def process_batch(df, read_seq_length=768):
    """
    Convert batch data from Pandas DataFrame to PyTorch tensors.
    Handles variable-length elevation_data by padding/truncating.
    Normalizes features and target for stable training.
    """
    # Convert scalar float columns to tensors
    dEP_FSRx_m = torch.tensor(df['dEP_FSRx_m'].values, dtype=torch.float32)
    center_freq = torch.tensor(df['center_freq'].values, dtype=torch.float32)
    receiver_ht_m = torch.tensor(df['receiver_ht_m'].values, dtype=torch.float32)
    accesspoint_ht_m = torch.tensor(df['accesspoint_ht_m'].values, dtype=torch.float32)

    # Convert target labels to tensor and NORMALIZE
    path_loss = torch.tensor(df['path_loss'].values, dtype=torch.float32)
    path_loss = (path_loss - TARGET_MEAN) / (TARGET_STD + 1e-6)

    # Process elevation_data (handling variable sequence lengths)
    elevation_data_list = df['elevation_data'].tolist()  # Convert Series to list of lists

    # Ensure each sequence is exactly `read_seq_length`
    elevation_tensors = []
    for seq in elevation_data_list:
        seq = torch.tensor(seq, dtype=torch.float32)  # Convert to tensor
        if len(seq) > read_seq_length:
            seq = seq[:read_seq_length]  # Truncate if longer
        elif len(seq) < read_seq_length:
            seq = torch.cat([seq, torch.zeros(read_seq_length - len(seq))])  # Pad if shorter
        elevation_tensors.append(seq)

    # Stack to get (batch_size, read_seq_length)
    elevation_data = torch.stack(elevation_tensors)

    # Normalize elevation data
    elevation_data = (elevation_data - ELEV_MEAN) / (ELEV_STD + 1e-6)

    # Stack all other tensors into a single tensor for batch processing
    input_features = torch.stack([dEP_FSRx_m, center_freq, receiver_ht_m, accesspoint_ht_m], dim=1)

    # Normalize input features
    input_features = (input_features - FEAT_MEAN) / (FEAT_STD + 1e-6)

    return input_features, elevation_data, path_loss

def load_weights(model, weights_path):
    """Load weights from file. Crashes if model architecture doesn't match."""
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model
