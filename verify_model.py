import torch
from pathloss_transformer import create_model

def test_model():
    print("Initializing model...")
    model = create_model()
    print("Model initialized successfully.")
    
    batch_size = 4
    seq_len = 768
    num_scalars = 4
    
    print(f"Creating dummy input: Batch={batch_size}, Scalar={num_scalars}, Seq={seq_len}")
    input_features = torch.randn(batch_size, num_scalars)
    elevation_data = torch.randn(batch_size, seq_len)
    
    print("Running forward pass...")
    try:
        output = model(input_features, elevation_data)
        print(f"Output shape: {output.shape}")
        
        expected_shape = (batch_size, 1)
        if output.shape == expected_shape:
            print("SUCCESS: Output shape matches expectation.")
        else:
            print(f"FAILURE: Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"FAILURE: Forward pass crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
