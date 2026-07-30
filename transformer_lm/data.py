import numpy as np 
import torch

# Load batches
def get_batch(dataset, batch_size, context_length, device):
    max_start = len(dataset) - context_length

    starts = np.random.randint(0, max_start, size = batch_size)    # Get all randoms in a batch size vector in a single call. 

    inputs = np.stack([dataset[i: i+context_length] for i in starts])
    targets = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)

    return inputs, targets


# Save model checkpoint
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoint, out)

# Load model checkpoint
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]



