from model import MLP, load_data, train, evaluate 
import math
import torch
from torch import nn
import torch.nn.functional as F

# helpers 

# x should be logits
def score(x, y):
    probs = F.softmax(x, dim=1)
    true_probs = probs[torch.arange(len(y)), y]
    return 1 - true_probs
    
#c calibrate to find quantiles, then make predictive sets

@torch.no_grad()
def calibrate(model, loader, alpha, device):
    scores = []
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        scores.append(score(logits, targets))
    scores = torch.cat(scores)
    n = scores.size(0)
    # ⌈(1 − α)(n + 1)⌉/n 
    quantile = math.ceil((1 - alpha) * (n + 1)) / n
    quantile = min(quantile, 1.0) #clamp to 1.0 in case values are greater than 1
    qhat = torch.quantile(scores, quantile)
    return qhat

@torch.no_grad()
def make_sets(model, x, qhat):
    model.eval()

    logits = model(x)

    csets = []

    for i in range(x.size(0)):
        cset = []

        for label in range(10):
            y = torch.tensor([label], device=x.device)
            s = score(logits[i:i+1], y)
            if s.item() <= qhat:
                cset.append(label)
        csets.append(cset)
    return csets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP().to(device)

    trainloader, calloader, testloader = load_data(batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 5

    for epoch in range(epochs):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss = evaluate(model, testloader, criterion, device)

        print(f"epoch: {epoch + 1}")
        print(f"train loss: {train_loss}")
        print(f"test loss: {test_loss}")
    alphas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

    for alpha in alphas:
        qhat = calibrate(model, calloader, alpha, device)

        print(f"\nalpha: {alpha}")
        print("qhat:", qhat.item())

        covered = 0
        total = 0
        set_sizes = []
        examples_by_size = []

        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            csets = make_sets(model, inputs, qhat)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            for i, (cset, target) in enumerate(zip(csets, targets)):
                target = target.item()

                if target in cset:
                    covered += 1

                examples_by_size.append({
                    "set_size": len(cset),
                    "true_label": target,
                    "pred_label": probs[i].argmax().item(),
                    "top_prob": probs[i].max().item(),
                    "cset": cset,
                    "image": inputs[i].detach().cpu(),
                })

                set_sizes.append(len(cset))
                total += 1

        empirical_coverage = covered / total
        avg_set_size = sum(set_sizes) / len(set_sizes)
        size_dist = torch.bincount(torch.tensor(set_sizes), minlength=11)

        largest = sorted(examples_by_size, key=lambda d: d["set_size"], reverse=True)[:10]
        smallest = sorted(examples_by_size, key=lambda d: d["set_size"])[:10]

        print("largest examples:")
        for ex in largest:
            print({
                "set_size": ex["set_size"],
                "true_label": ex["true_label"],
                "pred_label": ex["pred_label"],
                "top_prob": ex["top_prob"],
                "cset": ex["cset"],
            })

        print("smallest examples:")
        for ex in smallest:
            print({
                "set_size": ex["set_size"],
                "true_label": ex["true_label"],
                "pred_label": ex["pred_label"],
                "top_prob": ex["top_prob"],
                "cset": ex["cset"],
            })

        print(f"empirical coverage: {empirical_coverage:.4f}")
        print(f"average set size: {avg_set_size:.4f}")
        print(f"set size distribution: {size_dist}")