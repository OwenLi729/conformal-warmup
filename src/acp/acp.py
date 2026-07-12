import argparse

import torch
import torch.nn.functional as F
from torch import nn

from classic.model import MLP, evaluate, load_data, train


class CoveragePolicy(nn.Module):
    """Map a calibration-score sum and candidate-label scores to alpha."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # prevents alpha collapse to exactly 0 or 1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.1972246)

    def forward(self, cal_sum, test_summary):
        if cal_sum.ndim == 0:
            cal_sum = cal_sum.expand(test_summary.size(0))
        cal_sum = cal_sum.reshape(-1, 1)

        features = torch.cat(
            [torch.log1p(cal_sum), torch.log1p(test_summary)], dim=1
        )
        raw = self.net(features).squeeze(-1)

        # Keeping alpha away from exactly zero prevents division by zero.
        eps = 1e-4
        return eps + (1 - 2 * eps) * torch.sigmoid(raw)


def score(logits, labels):
    """Cross-entropy conformity scores S(x, y) = -log p(y | x)."""
    log_probs = F.log_softmax(logits, dim=1)
    batch_indices = torch.arange(len(labels), device=labels.device)
    return -log_probs[batch_indices, labels]


def candidate_scores(logits):
    """Return S(x, y) for every row x and candidate label y."""
    return -F.log_softmax(logits, dim=1)


def soft_rank(total_score, n, test_scores):
    return (n + 1) * test_scores / (total_score + test_scores)


def smooth_size(logits, total_score, n, alpha, k=100.0):
    """Sigmoid approximation to classification set size (Equation 5)."""
    test_scores = candidate_scores(logits)
    total_score = torch.as_tensor(
        total_score, device=logits.device, dtype=logits.dtype
    ).reshape(-1, 1)
    alpha = torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype)
    e_values = soft_rank(total_score, n, test_scores)
    return torch.sigmoid(k * (alpha.reshape(-1, 1).reciprocal() - e_values)).sum(
        dim=1
    )


def policy_loss(smooth_sizes, alphas, lambda_reg):
    """Coverage-policy objective from Equation 9."""
    return (smooth_sizes + lambda_reg * alphas).mean()


@torch.no_grad()
def _calibration_outputs(model, calloader, device):
    model.eval()
    logits = []
    labels = []
    for inputs, targets in calloader:
        logits.append(model(inputs.to(device)))
        labels.append(targets.to(device))

    logits = torch.cat(logits)
    labels = torch.cat(labels)
    return logits, score(logits, labels)


def _train_policy_from_outputs(
    cal_logits,
    cal_scores,
    lambda_reg,
    *,
    epochs,
    lr,
    k,
    batch_size,
):
    n, num_classes = cal_logits.shape
    if n < 2:
        raise ValueError(
            "coverage-policy training requires at least two calibration points"
        )

    # Each row is one LOO pseudo calibration--test pair (Equation 7).
    loo_totals = cal_scores.sum() - cal_scores
    summaries = candidate_scores(cal_logits)

    policy = CoveragePolicy(num_classes=num_classes).to(cal_logits.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        permutation = torch.randperm(n, device=cal_logits.device)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            alphas = policy(loo_totals[indices], summaries[indices])
            sizes = smooth_size(
                cal_logits[indices],
                loo_totals[indices],
                n - 1,
                alphas,
                k=k,
            )
            loss = policy_loss(sizes, alphas, lambda_reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * indices.numel()

        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"policy epoch {epoch + 1}: loss={epoch_loss / n:.4f}")

    return policy


def train_policy(
    model,
    calloader,
    device,
    lambda_reg=1.0,
    epochs=2000,
    lr=1e-3,
    k=100.0,
    batch_size=64,
):
    """Train a coverage policy with Algorithm 1's LOO pseudo episodes."""
    cal_logits, cal_scores = _calibration_outputs(model, calloader, device)
    return _train_policy_from_outputs(
        cal_logits,
        cal_scores,
        lambda_reg,
        epochs=epochs,
        lr=lr,
        k=k,
        batch_size=batch_size,
    )


def _e_set_mask(test_scores, total_score, n, alpha):
    total_score = torch.as_tensor(
        total_score, device=test_scores.device, dtype=test_scores.dtype
    ).reshape(-1, 1)
    alpha = torch.as_tensor(
        alpha, device=test_scores.device, dtype=test_scores.dtype
    ).reshape(-1, 1)
    return soft_rank(total_score, n, test_scores) < alpha.reciprocal()


@torch.no_grad()
def _loo_policy_size_from_outputs(policy, cal_logits, cal_scores):
    policy.eval()
    n = cal_scores.numel()
    loo_totals = cal_scores.sum() - cal_scores
    summaries = candidate_scores(cal_logits)
    alphas = policy(loo_totals, summaries)
    mask = _e_set_mask(summaries, loo_totals, n - 1, alphas)
    return mask.sum(dim=1).float().mean().item()


@torch.no_grad()
def loo_policy_size(model, policy, calloader, device):
    """Compute the exact mean LOO set size used by Algorithm 2."""
    cal_logits, cal_scores = _calibration_outputs(model, calloader, device)
    return _loo_policy_size_from_outputs(policy, cal_logits, cal_scores)


def select_lambda(
    model,
    calloader,
    device,
    target_size,
    tolerance=0.1,
    initial_lambda=1.0,
    max_steps=8,
    *,
    epochs=2000,
    lr=1e-3,
    k=100.0,
    batch_size=64,
):
    """Select lambda by Algorithm 2's bracketing and bisection procedure."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if tolerance <= 0 or initial_lambda <= 0 or max_steps < 1:
        raise ValueError(
            "tolerance, initial_lambda, and max_steps must be positive"
        )

    # Cache black-box outputs once: every lambda trial uses the same LOO episodes.
    cal_logits, cal_scores = _calibration_outputs(model, calloader, device)
    if target_size > cal_logits.size(1):
        raise ValueError("target_size cannot exceed the number of classes")
    best = None

    def fit_and_measure(lam):
        nonlocal best
        policy = _train_policy_from_outputs(
            cal_logits,
            cal_scores,
            lam,
            epochs=epochs,
            lr=lr,
            k=k,
            batch_size=batch_size,
        )
        size = _loo_policy_size_from_outputs(policy, cal_logits, cal_scores)
        error = abs(size - target_size)
        if best is None or error < best[0]:
            best = (error, lam, policy, size)
        print(f"lambda={lam:.6g}, loo_size={size:.4f}")
        return size

    initial_size = fit_and_measure(initial_lambda)
    if abs(initial_size - target_size) <= tolerance:
        return best[1], best[2], best[3]

    # Proposition 2.7 motivates treating set size as non-decreasing in lambda.
    if initial_size < target_size:
        lam_low = initial_lambda
        lam_high = initial_lambda
        for _ in range(max_steps):
            lam_high *= 2
            size = fit_and_measure(lam_high)
            if size >= target_size:
                break
        else:
            return best[1], best[2], best[3]
    else:
        lam_low = initial_lambda
        lam_high = initial_lambda
        for _ in range(max_steps):
            lam_low /= 2
            size = fit_and_measure(lam_low)
            if size <= target_size:
                break
        else:
            return best[1], best[2], best[3]

    for _ in range(max_steps):
        lam_mid = (lam_low + lam_high) / 2
        size = fit_and_measure(lam_mid)
        if abs(size - target_size) <= tolerance:
            break
        if size < target_size:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    return best[1], best[2], best[3]


@torch.no_grad()
def make_e_sets(
    model,
    x,
    alpha,
    total_score,
    n,
    policy=None,
    return_alphas=False,
):
    """Construct fixed- or adaptive-alpha conformal e-prediction sets."""
    model.eval()
    logits = model(x)
    summaries = candidate_scores(logits)
    total_score = torch.as_tensor(
        total_score, device=logits.device, dtype=logits.dtype
    )

    if policy is None:
        if alpha is None or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1) when no policy is supplied")
        alphas = torch.full(
            (x.size(0),), alpha, device=logits.device, dtype=logits.dtype
        )
    else:
        policy.eval()
        alphas = policy(total_score, summaries)

    mask = _e_set_mask(summaries, total_score, n, alphas)
    csets = [row.nonzero(as_tuple=False).squeeze(1).tolist() for row in mask]
    if return_alphas:
        return csets, alphas.detach().cpu().tolist()
    return csets


@torch.no_grad()
def get_cal_total_score(model, calloader, device):
    """Cache the calibration-score sum required at test time."""
    model.eval()
    total_score = torch.zeros((), device=device)
    n = 0
    for inputs, targets in calloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_scores = score(model(inputs), targets)
        total_score += batch_scores.sum()
        n += batch_scores.numel()
    return total_score, n


@torch.no_grad()
def evaluate_policy(model, policy, testloader, total_score, n, device):
    """Evaluate test efficiency and the empirical post-hoc validity statistic."""
    total = 0
    covered = 0
    total_set_size = 0
    alphas = []
    posthoc_ratios = []

    for inputs, targets in testloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        csets, batch_alphas = make_e_sets(
            model,
            inputs,
            alpha=None,
            total_score=total_score,
            n=n,
            policy=policy,
            return_alphas=True,
        )
        for cset, target, alpha in zip(csets, targets.tolist(), batch_alphas):
            is_covered = target in cset
            total += 1
            covered += int(is_covered)
            total_set_size += len(cset)
            alphas.append(alpha)
            missed = float(not is_covered)
            posthoc_ratios.append(missed / alpha)

    alpha_tensor = torch.tensor(alphas)
    return {
        "coverage": covered / total,
        "average_set_size": total_set_size / total,
        "alpha_mean": alpha_tensor.mean().item(),
        "alpha_std": alpha_tensor.std().item(),
        "alpha_min": alpha_tensor.min().item(),
        "alpha_max": alpha_tensor.max().item(),
        "post_hoc_ratio": sum(posthoc_ratios) / len(posthoc_ratios),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an ACP policy")
    parser.add_argument("--target-size", type=float, default=3.0)
    parser.add_argument("--policy-epochs", type=int, default=2000)
    parser.add_argument("--classifier-epochs", type=int, default=5)
    parser.add_argument("--max-lambda-steps", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Match the paper: train on all 50k training images and calibrate on 100
    # randomly selected CIFAR-10 test images.
    trainloader, calloader, testloader = load_data(paper_acp_split=True)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("\nTraining base classifier...")
    for epoch in range(args.classifier_epochs):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss = evaluate(model, testloader, criterion, device)
        print(
            f"classifier epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}"
        )

    selected_lambda, policy, loo_size = select_lambda(
        model,
        calloader,
        device,
        target_size=args.target_size,
        max_steps=args.max_lambda_steps,
        epochs=args.policy_epochs,
    )
    print(
        f"\nselected lambda={selected_lambda:.6g}, "
        f"LOO average size={loo_size:.4f}"
    )

    total_score, n = get_cal_total_score(model, calloader, device)
    metrics = evaluate_policy(model, policy, testloader, total_score, n, device)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
