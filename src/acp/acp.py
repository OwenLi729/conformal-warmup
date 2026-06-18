from classic.model import MLP, load_data, train, evaluate
import torch
from torch import nn
import torch.nn.functional as F

# x are logits
def score(x, y):
    log_probs = F.log_softmax(x, dim=1)
    batch_indices = torch.arange(len(y), device=y.device)
    return -log_probs[batch_indices, y]

def soft_rank(total_score, n, x_test, y_test):
    test_score = score(x_test, y_test)
    return ((n + 1) * test_score) / (total_score + test_score)

@torch.no_grad()
def make_e_sets(model, x, alpha, total_score, n):
    model.eval()

    logits = model(x)

    csets = []

    for i in range(x.size(0)):
        cset = []

        for label in range(10):
            y = torch.tensor([label], device=x.device)

            e_value = soft_rank(
                total_score,
                n,
                logits[i:i+1],
                y
            )

            if e_value.item() <= (1 / alpha):
                cset.append(label)

        csets.append(cset)

    return csets

# calibration score sum does not change, this function improves efficiency
# and lets us cache the score for later

@torch.no_grad()
def get_cal_total_score(model, calloader, device):
    model.eval()
    total_score = 0.0
    n = 0

    for inputs, targets in calloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        batch_scores = score(logits, targets)

        total_score += batch_scores.sum()
        n += batch_scores.size(0)

    return total_score, n

@torch.no_grad()
def loo_average_size(model, calloader, alpha, device, num_classes=10):
    model.eval()
    all_logits = []
    all_labels = []

    for inputs, targets in calloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        all_logits.append(logits)
        all_labels.append(targets)

def main():
    pass