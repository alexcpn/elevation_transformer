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
        vocab_size = 2000
        d_k = 64  # attention size
        read_seq_length = 768
        d_model = 512  # embediding size
        final_size = 1 # we just want one value
        num_heads = 8  
        extra_feature_size = 4
        
        self.pos_encoding = PositionalEncoding(d_model, read_seq_length)
        self.extra_features_embedding = nn.Linear(extra_feature_size, d_model)
        self.multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.prediction_layer1 = nn.Linear(d_model, vocab_size)
        self.layer_norm1 = nn.LayerNorm(vocab_size)
        self.prediction_layer2 = nn.Linear(vocab_size, final_size)
        self.layer_norm2 = nn.LayerNorm(final_size)

    def forward(self, input_features, elevation_data):
        # project to higher dim
        extra_features_tokens = self.extra_features_embedding(input_features)  # (B, 512)
        pos_embedded_tokens = self.pos_encoding(elevation_data)
        score, _ = self.multihead_attention(pos_embedded_tokens, pos_embedded_tokens, pos_embedded_tokens)
        hidden1 = score + pos_embedded_tokens
        pooled_score = hidden1.mean(dim=1)  # (B, 512)

        # Combine elevation features with extra features
        combined = pooled_score + extra_features_tokens  # (B, 512)

        hidden2 = self.prediction_layer1(combined)
        hidden2 = self.layer_norm1(hidden2)
        hidden2 = F.relu(hidden2)
        logits = self.prediction_layer2(hidden2)
        return logits

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
    state_dict = torch.load(weights_path)
    
    # Check if keys match the old ModuleList format (0.pe, 1.in_proj_weight, etc.)
    # and map them to the new PathLossModel format
    new_state_dict = {}
    
    # Mapping from index to attribute name
    key_map = {
        '0': 'pos_encoding',
        '1': 'multihead_attention',
        '2': 'extra_features_embedding',
        '3': 'layer_norm1',
        '4': 'layer_norm2',
        '5': 'prediction_layer1',
        '6': 'prediction_layer2'
    }
    
    keys_updated = False
    for key, value in state_dict.items():
        parts = key.split('.')
        if parts[0] in key_map:
            new_key = key_map[parts[0]] + '.' + '.'.join(parts[1:])
            new_state_dict[new_key] = value
            keys_updated = True
        else:
            new_state_dict[key] = value
            
    if keys_updated:
        print("Converted legacy ModuleList state_dict to PathLossModel format.")
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model
