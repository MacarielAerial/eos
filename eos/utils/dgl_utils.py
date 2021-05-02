import numpy as np
import torch


def sample_mask(idx, length):
    """Create mask."""
    mask = np.zeros(length)
    mask[idx] = 1
    return mask


def evaluate(model, graph, n_feats, e_feats, labels, mask):
    model.eval()
    with torch.no_grad():
        # Use logit to compress values into 0~1 range
        logits = model(graph, n_feats, e_feats)
        probs = torch.sigmoid(logits)
        probs = probs[mask]
        # Round for binary assignment and squeeze for shapeshift
        labels = labels[mask]
        _, indices = torch.max(probs, dim=1)
        correct = torch.sum(indices == labels)
        # Multiply by 1.0 to convert to float
        return correct.item() * 1.0 / len(labels)
