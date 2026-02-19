import numpy as np
import matplotlib.pyplot as plt

# Load loss data from multiple .npz files (in order of training)
loss_log_files = [
    "logs/loss_log_20260204165247_.npy.npz",  # First training run
    "logs/loss_log_20260205140023_.npy.npz",  # Second training run (continuation)
]

def load_loss_file(filepath):
    """Load loss history from a .npz file."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return list(data["loss"])
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

# Combine losses from all files
combined_steps = []
combined_losses = []
step_offset = 0

for i, loss_file in enumerate(loss_log_files):
    loss_history = load_loss_file(loss_file)
    if not loss_history:
        continue

    steps = [entry[1] + step_offset for entry in loss_history]
    loss_values = [entry[2] for entry in loss_history]

    combined_steps.extend(steps)
    combined_losses.extend(loss_values)

    # Update offset for next file (continue from last step)
    if steps:
        step_offset = steps[-1]

    print(f"Loaded {loss_file}: {len(loss_history)} entries, steps {steps[0]}-{steps[-1]}")

print(f"\nTotal combined entries: {len(combined_steps)}")
print(f"Total steps: {combined_steps[-1] if combined_steps else 0}")

# Create the loss plot
plt.figure(figsize=(12, 6))
plt.plot(combined_steps, combined_losses, label="Training Loss", color="blue", alpha=0.7)
plt.scatter(combined_steps, combined_losses, color="red", s=10, alpha=0.5)

# Labels and Titles
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps (Combined Runs)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
