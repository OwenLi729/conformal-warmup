"""Self-contained Google Colab version of the ACP implementation.

Paste this file into one Colab cell (or upload and import it), run the cell,
then call ``run_colab()``. For a short smoke run, use::

    results = run_colab(
        classifier_epochs=1,
        policy_epochs=20,
        max_lambda_steps=2,
    )

The default policy settings match the paper and are substantially slower.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2


class MLP(nn.Module):
    """Small CIFAR-10 classifier used as the black-box predictor."""

    def __init__(self, input_dim=3072, hidden_dim=128, output_dim=10):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.out(F.relu(self.hidden(x)))


def load_data(batch_size=64, calibration_size=100, data_root="./data"):
    """Use the paper's 50k train / 100 calibration / 9.9k test split."""
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    full_testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    if not 0 < calibration_size < len(full_testset):
        raise ValueError("calibration_size must be between 1 and 9999")

    calset, testset = random_split(
        full_testset,
        [calibration_size, len(full_testset) - calibration_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader_args = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(trainset, shuffle=True, **loader_args),
        DataLoader(calset, shuffle=False, **loader_args),
        DataLoader(testset, shuffle=False, **loader_args),
    )


def train_classifier_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        running_loss += criterion(logits, targets).item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)


class CoveragePolicy(nn.Module):
    """Map a calibration-score sum and candidate-label scores to alpha."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
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
        eps = 1e-4
        return eps + (1 - 2 * eps) * torch.sigmoid(raw)


def score(logits, labels):
    """Cross-entropy conformity scores S(x, y) = -log p(y | x)."""
    return F.cross_entropy(logits, labels, reduction="none")


def candidate_scores(logits):
    """Return S(x, y) for every row x and candidate label y."""
    return -F.log_softmax(logits, dim=1)


def soft_rank(total_score, n, test_scores):
    """Compute the soft-rank e-value in Equation 2."""
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
    return (smooth_sizes + lambda_reg * alphas).mean()


@torch.no_grad()
def _calibration_outputs(model, calloader, device):
    model.eval()
    logits = []
    labels = []
    for inputs, targets in calloader:
        logits.append(model(inputs.to(device, non_blocking=True)))
        labels.append(targets.to(device, non_blocking=True))
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
        raise ValueError("policy training requires at least two calibration points")

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
    """Algorithm 2: bracket and bisect lambda for a target set size."""
    if target_size <= 0 or tolerance <= 0 or initial_lambda <= 0:
        raise ValueError("target_size, tolerance, and initial_lambda must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

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

    if initial_size < target_size:
        lam_low = initial_lambda
        lam_high = initial_lambda
        for _ in range(max_steps):
            lam_high *= 2
            if fit_and_measure(lam_high) >= target_size:
                break
        else:
            return best[1], best[2], best[3]
    else:
        lam_low = initial_lambda
        lam_high = initial_lambda
        for _ in range(max_steps):
            lam_low /= 2
            if fit_and_measure(lam_low) <= target_size:
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
    model.eval()
    logits = model(x)
    summaries = candidate_scores(logits)
    total_score = torch.as_tensor(
        total_score, device=logits.device, dtype=logits.dtype
    )

    if policy is None:
        if alpha is None or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1) without a policy")
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
    model.eval()
    total_score = torch.zeros((), device=device)
    n = 0
    for inputs, targets in calloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_scores = score(model(inputs), targets)
        total_score += batch_scores.sum()
        n += batch_scores.numel()
    return total_score, n


@torch.no_grad()
def evaluate_policy(model, policy, testloader, total_score, n, device):
    total = 0
    covered = 0
    total_set_size = 0
    alphas = []
    posthoc_ratios = []
    for inputs, targets in testloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
            posthoc_ratios.append(float(not is_covered) / alpha)

    alpha_tensor = torch.tensor(alphas)
    quantile_levels = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "coverage": covered / total,
        "average_set_size": total_set_size / total,
        "alpha_mean": alpha_tensor.mean().item(),
        "alpha_std": alpha_tensor.std().item(),
        "alpha_min": alpha_tensor.min().item(),
        "alpha_max": alpha_tensor.max().item(),
        "alpha_quantiles": torch.quantile(alpha_tensor, quantile_levels).tolist(),
        "post_hoc_ratio": sum(posthoc_ratios) / len(posthoc_ratios),
    }


def print_policy_metrics(metrics):
    """Print scalar metrics and the alpha-distribution quantiles."""
    for name, value in metrics.items():
        if name != "alpha_quantiles":
            print(f"{name}: {value:.4f}")
    print("alpha quantiles [0, .1, .25, .5, .75, .9, 1]:")
    print("[" + ", ".join(f"{value:.4f}" for value in metrics["alpha_quantiles"]) + "]")


def _has_nontrivial_alpha_variation(metrics, tolerance):
    return (
        metrics["alpha_std"] > tolerance
        and metrics["alpha_max"] - metrics["alpha_min"] > tolerance
    )


def verify_acp(
    model,
    calloader,
    testloader,
    total_score,
    n,
    device,
    *,
    selected_metrics,
    lambdas=(5.0, 10.0, 50.0),
    epochs=2000,
    lr=1e-3,
    k=100.0,
    batch_size=64,
    variation_tolerance=1e-3,
):
    """Run all three pre-experiment checks requested in Part 3."""
    if len(lambdas) < 2:
        raise ValueError("verification requires at least two lambda values")
    if variation_tolerance <= 0:
        raise ValueError("variation_tolerance must be positive")

    cal_logits, cal_scores = _calibration_outputs(model, calloader, device)
    reports = {}
    print("\n" + "=" * 60)
    print("ACP verification across lambda values")

    for lambda_reg in lambdas:
        print("\n" + "-" * 60)
        print(f"verification lambda={lambda_reg:g}")
        policy = _train_policy_from_outputs(
            cal_logits,
            cal_scores,
            lambda_reg,
            epochs=epochs,
            lr=lr,
            k=k,
            batch_size=batch_size,
        )
        metrics = evaluate_policy(
            model, policy, testloader, total_score, n, device
        )
        nontrivial = _has_nontrivial_alpha_variation(
            metrics, variation_tolerance
        )
        post_hoc_consistent = metrics["post_hoc_ratio"] <= 1.0
        reports[float(lambda_reg)] = {
            "policy": policy,
            "metrics": metrics,
            "nontrivial_alpha_variation": nontrivial,
            "post_hoc_consistent": post_hoc_consistent,
        }
        print_policy_metrics(metrics)
        print(
            "diagnostic alpha variation: "
            f"{'PASS' if nontrivial else 'DEGENERATE'}"
        )
        print(
            "empirical post-hoc ratio <= 1: "
            f"{'PASS' if post_hoc_consistent else 'FAIL'}"
        )

    quantiles = [
        reports[float(lambda_reg)]["metrics"]["alpha_quantiles"]
        for lambda_reg in lambdas
    ]
    maximum_quantile_shift = max(
        abs(a - b)
        for i, left in enumerate(quantiles)
        for right in quantiles[i + 1 :]
        for a, b in zip(left, right)
    )
    distribution_shift = maximum_quantile_shift > variation_tolerance
    selected_nontrivial = _has_nontrivial_alpha_variation(
        selected_metrics, variation_tolerance
    )
    selected_post_hoc_consistent = selected_metrics["post_hoc_ratio"] <= 1.0
    all_post_hoc_consistent = selected_post_hoc_consistent and all(
        report["post_hoc_consistent"] for report in reports.values()
    )
    degenerate_lambdas = [
        lambda_reg
        for lambda_reg, report in reports.items()
        if not report["nontrivial_alpha_variation"]
    ]
    checks = {
        "nontrivial_alpha_variation": selected_nontrivial,
        "alpha_distribution_shifts_with_lambda": distribution_shift,
        "post_hoc_consistent": all_post_hoc_consistent,
        "maximum_alpha_quantile_shift": maximum_quantile_shift,
    }

    print("\n" + "=" * 60)
    print("Verification summary")
    print(
        "1. Selected policy has non-trivial alpha variation: "
        f"{'PASS' if selected_nontrivial else 'FAIL'}"
    )
    print(
        "2. Alpha distribution shifts as lambda changes: "
        f"{'PASS' if distribution_shift else 'FAIL'} "
        f"(max quantile shift={maximum_quantile_shift:.6f})"
    )
    print(
        "3. Empirical post-hoc ratio is at most 1: "
        f"{'PASS' if all_post_hoc_consistent else 'FAIL'}"
    )
    if degenerate_lambdas:
        formatted = ", ".join(f"{value:g}" for value in degenerate_lambdas)
        print(
            "Warning: degenerate alpha distributions for diagnostic lambda(s): "
            f"{formatted}"
        )
    return {
        "selected_policy": {
            "metrics": selected_metrics,
            "nontrivial_alpha_variation": selected_nontrivial,
            "post_hoc_consistent": selected_post_hoc_consistent,
        },
        "reports": reports,
        "checks": checks,
        "warnings": {"degenerate_lambdas": degenerate_lambdas},
    }


def run_colab(
    target_size=3.0,
    classifier_epochs=5,
    policy_epochs=2000,
    max_lambda_steps=8,
    batch_size=64,
    seed=42,
    verification_lambdas=(5.0, 10.0, 50.0),
    verification_epochs=None,
    variation_tolerance=1e-3,
):
    """Run classifier training, lambda selection, and ACP test evaluation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("Warning: enable a GPU via Runtime > Change runtime type.")

    trainloader, calloader, testloader = load_data(batch_size=batch_size)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("\nTraining base classifier...")
    for epoch in range(classifier_epochs):
        train_loss = train_classifier_epoch(
            model, trainloader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate_classifier(
            model, testloader, criterion, device
        )
        print(
            f"classifier epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}"
        )

    selected_lambda, policy, loo_size = select_lambda(
        model,
        calloader,
        device,
        target_size=target_size,
        max_steps=max_lambda_steps,
        epochs=policy_epochs,
    )
    total_score, n = get_cal_total_score(model, calloader, device)
    metrics = evaluate_policy(model, policy, testloader, total_score, n, device)

    print(
        f"\nselected lambda={selected_lambda:.6g}, "
        f"LOO average size={loo_size:.4f}"
    )
    print_policy_metrics(metrics)

    verification = verify_acp(
        model,
        calloader,
        testloader,
        total_score,
        n,
        device,
        selected_metrics=metrics,
        lambdas=verification_lambdas,
        epochs=verification_epochs or policy_epochs,
        batch_size=batch_size,
        variation_tolerance=variation_tolerance,
    )

    return {
        "model": model,
        "policy": policy,
        "selected_lambda": selected_lambda,
        "loo_size": loo_size,
        "metrics": metrics,
        "verification": verification,
        "calibration_total_score": total_score,
        "calibration_size": n,
    }


# In Colab, run one of these in a new cell after loading this file:
#
# Quick smoke run:
# results = run_colab(classifier_epochs=1, policy_epochs=20, max_lambda_steps=2)
#
# Full default run:
# results = run_colab()
