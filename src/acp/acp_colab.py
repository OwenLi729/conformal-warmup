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

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
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


def aps_score(logits, labels):
    """APS score (Romano et al., 2020): Sort classes in decreasing order of predicted probability;
    S(x, y) is the cumulative predicted probability up to and including class y."""

    probabilities = F.softmax(logits, dim=1)
    sorted_probabilities, sorted_labels = probabilities.sort(
        dim=1, descending=True, stable=True
    )
    cumulative_probabilities = sorted_probabilities.cumsum(dim=1)
    label_ranks = sorted_labels.argsort(dim=1).gather(1, labels[:, None])
    return cumulative_probabilities.gather(1, label_ranks).squeeze(1)


SCORE_TYPES = ("cross_entropy", "aps")


def aps_candidate_scores(logits):
    """Return the APS score for every candidate label in every row."""
    probabilities = F.softmax(logits, dim=1)
    sorted_probabilities, sorted_labels = probabilities.sort(
        dim=1, descending=True, stable=True
    )
    cumulative_probabilities = sorted_probabilities.cumsum(dim=1)
    return torch.zeros_like(cumulative_probabilities).scatter(
        1, sorted_labels, cumulative_probabilities
    )


def candidate_scores(logits, score_type="cross_entropy"):
    """Return S(x, y) for every row x and candidate label y."""
    if score_type == "cross_entropy":
        return -F.log_softmax(logits, dim=1)
    if score_type == "aps":
        return aps_candidate_scores(logits)
    raise ValueError(
        f"unknown score_type {score_type!r}; expected one of {SCORE_TYPES}"
    )


@torch.no_grad()
def score_property_report(logits, score_type, tolerance=1e-7):
    """Check the score properties required by the soft-rank e-variable."""
    scores = candidate_scores(logits, score_type)
    probabilities = F.softmax(logits, dim=1)
    order = probabilities.argsort(dim=1, descending=True, stable=True)
    ordered_scores = scores.gather(1, order)
    finite = bool(torch.isfinite(scores).all().item())
    nonnegative = bool((scores >= -tolerance).all().item())
    negatively_oriented = bool(
        (ordered_scores[:, 1:] >= ordered_scores[:, :-1] - tolerance)
        .all()
        .item()
    )
    return {
        "finite": finite,
        "nonnegative": nonnegative,
        "negatively_oriented": negatively_oriented,
        "min_score": scores.min().item(),
        "max_score": scores.max().item(),
    }


def soft_rank(total_score, n, test_scores):
    """Compute the soft-rank e-value in Equation 2."""
    return (n + 1) * test_scores / (total_score + test_scores)


def smooth_size(
    logits, total_score, n, alpha, k=100.0, score_type="cross_entropy"
):
    """Sigmoid approximation to classification set size (Equation 5)."""
    test_scores = candidate_scores(logits, score_type)
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
def _calibration_outputs(model, calloader, device, score_type="cross_entropy"):
    model.eval()
    logits = []
    labels = []
    for inputs, targets in calloader:
        logits.append(model(inputs.to(device, non_blocking=True)))
        labels.append(targets.to(device, non_blocking=True))
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    if score_type == "cross_entropy":
        cal_scores = score(logits, labels)
    elif score_type == "aps":
        cal_scores = aps_score(logits, labels)
    else:
        raise ValueError(
            f"unknown score_type {score_type!r}; expected one of {SCORE_TYPES}"
        )
    return logits, cal_scores


def _train_policy_from_outputs(
    cal_logits,
    cal_scores,
    lambda_reg,
    *,
    epochs,
    lr,
    k,
    batch_size,
    score_type="cross_entropy",
):
    n, num_classes = cal_logits.shape
    if n < 2:
        raise ValueError("policy training requires at least two calibration points")

    loo_totals = cal_scores.sum() - cal_scores
    summaries = candidate_scores(cal_logits, score_type)
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
                score_type=score_type,
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
    score_type="cross_entropy",
):
    cal_logits, cal_scores = _calibration_outputs(
        model, calloader, device, score_type
    )
    return _train_policy_from_outputs(
        cal_logits,
        cal_scores,
        lambda_reg,
        epochs=epochs,
        lr=lr,
        k=k,
        batch_size=batch_size,
        score_type=score_type,
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
def _loo_policy_size_from_outputs(
    policy, cal_logits, cal_scores, score_type="cross_entropy"
):
    policy.eval()
    n = cal_scores.numel()
    loo_totals = cal_scores.sum() - cal_scores
    summaries = candidate_scores(cal_logits, score_type)
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
    return_history=False,
    score_type="cross_entropy",
):
    """Algorithm 2: bracket and bisect lambda for a target set size."""
    if target_size <= 0 or tolerance <= 0 or initial_lambda <= 0:
        raise ValueError("target_size, tolerance, and initial_lambda must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    cal_logits, cal_scores = _calibration_outputs(
        model, calloader, device, score_type
    )
    if target_size > cal_logits.size(1):
        raise ValueError("target_size cannot exceed the number of classes")
    best = None
    history = []

    def fit_and_measure(lam, phase):
        nonlocal best
        policy = _train_policy_from_outputs(
            cal_logits,
            cal_scores,
            lam,
            epochs=epochs,
            lr=lr,
            k=k,
            batch_size=batch_size,
            score_type=score_type,
        )
        size = _loo_policy_size_from_outputs(
            policy, cal_logits, cal_scores, score_type
        )
        error = abs(size - target_size)
        if best is None or error < best[0]:
            best = (error, lam, policy, size)
        history.append(
            {
                "iteration": len(history) + 1,
                "lambda": float(lam),
                "loo_size": size,
                "phase": phase,
            }
        )
        print(f"lambda={lam:.6g}, loo_size={size:.4f}")
        return size

    def selection_result():
        result = (best[1], best[2], best[3])
        return (*result, history) if return_history else result

    initial_size = fit_and_measure(initial_lambda, "bracketing")
    if abs(initial_size - target_size) <= tolerance:
        return selection_result()

    if initial_size < target_size:
        lam_low = initial_lambda
        lam_high = initial_lambda
        for _ in range(max_steps):
            lam_low = lam_high
            lam_high *= 2
            if fit_and_measure(lam_high, "bracketing") >= target_size:
                break
        else:
            return selection_result()
    else:
        lam_low = initial_lambda
        lam_high = initial_lambda
        for _ in range(max_steps):
            lam_high = lam_low
            lam_low /= 2
            if fit_and_measure(lam_low, "bracketing") <= target_size:
                break
        else:
            return selection_result()

    for _ in range(max_steps):
        lam_mid = (lam_low + lam_high) / 2
        size = fit_and_measure(lam_mid, "bisection")
        if abs(size - target_size) <= tolerance:
            break
        if size < target_size:
            lam_low = lam_mid
        else:
            lam_high = lam_mid
    return selection_result()


@torch.no_grad()
def make_e_sets(
    model,
    x,
    alpha,
    total_score,
    n,
    policy=None,
    return_alphas=False,
    score_type="cross_entropy",
):
    model.eval()
    logits = model(x)
    summaries = candidate_scores(logits, score_type)
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
def get_cal_total_score(
    model, calloader, device, score_type="cross_entropy"
):
    model.eval()
    total_score = torch.zeros((), device=device)
    n = 0
    for inputs, targets in calloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        if score_type == "cross_entropy":
            batch_scores = score(logits, targets)
        elif score_type == "aps":
            batch_scores = aps_score(logits, targets)
        else:
            raise ValueError(
                f"unknown score_type {score_type!r}; expected one of {SCORE_TYPES}"
            )
        total_score += batch_scores.sum()
        n += batch_scores.numel()
    return total_score, n


@torch.no_grad()
def evaluate_policy(
    model,
    policy,
    testloader,
    total_score,
    n,
    device,
    score_type="cross_entropy",
    return_alpha_values=False,
):
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
            score_type=score_type,
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
    metrics = {
        "coverage": covered / total,
        "average_set_size": total_set_size / total,
        "alpha_mean": alpha_tensor.mean().item(),
        "alpha_std": alpha_tensor.std().item(),
        "alpha_min": alpha_tensor.min().item(),
        "alpha_max": alpha_tensor.max().item(),
        "alpha_quantiles": torch.quantile(alpha_tensor, quantile_levels).tolist(),
        "post_hoc_ratio": sum(posthoc_ratios) / len(posthoc_ratios),
    }
    if return_alpha_values:
        metrics["alpha_values"] = alphas
    return metrics


@torch.no_grad()
def evaluate_fixed_alpha(
    model,
    testloader,
    total_score,
    n,
    device,
    alpha=0.1,
    score_type="cross_entropy",
):
    """Evaluate the fixed-alpha conformal e-predictor baseline."""
    total = 0
    covered = 0
    total_set_size = 0
    for inputs, targets in testloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        csets = make_e_sets(
            model,
            inputs,
            alpha=alpha,
            total_score=total_score,
            n=n,
            policy=None,
            score_type=score_type,
        )
        for cset, target in zip(csets, targets.tolist()):
            total += 1
            covered += int(target in cset)
            total_set_size += len(cset)
    coverage = covered / total
    return {
        "alpha": alpha,
        "coverage": coverage,
        "average_set_size": total_set_size / total,
        "miscoverage_ratio": (1.0 - coverage) / alpha,
    }


def print_policy_metrics(metrics):
    """Print scalar metrics and the alpha-distribution quantiles."""
    for name, value in metrics.items():
        if name not in {"alpha_quantiles", "alpha_values"}:
            print(f"{name}: {value:.4f}")
    print("alpha quantiles [0, .1, .25, .5, .75, .9, 1]:")
    print("[" + ", ".join(f"{value:.4f}" for value in metrics["alpha_quantiles"]) + "]")


def _mean_std(values):
    values = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
    }


def _aggregate_extension_runs(runs, lambdas):
    summary = {}
    for score_type in SCORE_TYPES:
        score_runs = [run["scores"][score_type] for run in runs]
        baseline = {
            metric: _mean_std(
                [run["baseline"][metric] for run in score_runs]
            )
            for metric in ("coverage", "average_set_size", "miscoverage_ratio")
        }
        lambda_summaries = {}
        for lambda_reg in lambdas:
            key = f"{lambda_reg:g}"
            policies = [run["policies"][key] for run in score_runs]
            lambda_summary = {
                metric: _mean_std([policy[metric] for policy in policies])
                for metric in (
                    "coverage",
                    "average_set_size",
                    "post_hoc_ratio",
                    "alpha_mean",
                    "alpha_std",
                    "absolute_efficiency_gain",
                    "relative_efficiency_gain",
                )
            }
            quantiles = torch.tensor(
                [policy["alpha_quantiles"] for policy in policies]
            )
            lambda_summary["alpha_quantiles_mean"] = quantiles.mean(dim=0).tolist()
            lambda_summary["alpha_quantiles_std"] = quantiles.std(
                dim=0, unbiased=False
            ).tolist()
            lambda_summary["post_hoc_consistent_all_seeds"] = all(
                policy["post_hoc_ratio"] <= 1.0 for policy in policies
            )
            lambda_summary["size_reduction_all_seeds"] = all(
                policy["absolute_efficiency_gain"] > 0.0 for policy in policies
            )
            lambda_summaries[key] = lambda_summary
        summary[score_type] = {
            "baseline": baseline,
            "lambdas": lambda_summaries,
        }

    candidates = []
    for score_type in SCORE_TYPES:
        for lambda_reg in lambdas:
            key = f"{lambda_reg:g}"
            metrics = summary[score_type]["lambdas"][key]
            if metrics["post_hoc_consistent_all_seeds"]:
                candidates.append(
                    {
                        "score_type": score_type,
                        "lambda": lambda_reg,
                        "absolute_gain": metrics["absolute_efficiency_gain"]["mean"],
                        "relative_gain": metrics["relative_efficiency_gain"]["mean"],
                    }
                )
    best = max(candidates, key=lambda item: item["relative_gain"], default=None)
    quantile_gaps = {}
    for lambda_reg in lambdas:
        key = f"{lambda_reg:g}"
        cross_entropy = summary["cross_entropy"]["lambdas"][key][
            "alpha_quantiles_mean"
        ]
        aps = summary["aps"]["lambdas"][key]["alpha_quantiles_mean"]
        quantile_gaps[key] = max(abs(left - right) for left, right in zip(cross_entropy, aps))
    summary["research_checks"] = {
        "reduction_consistent_for_both_scores": all(
            summary[score_type]["lambdas"][f"{lambda_reg:g}"][
                "size_reduction_all_seeds"
            ]
            for score_type in SCORE_TYPES
            for lambda_reg in lambdas
        ),
        "maximum_alpha_quantile_gap_by_lambda": quantile_gaps,
        "largest_valid_efficiency_gain": best,
    }
    return summary


def _plot_extension_efficiency(summary, lambdas, output_path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw

        width, height = 1200, 800
        left, right, top, bottom = 100, 50, 50, 100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        colors = {"cross_entropy": "#1f77b4", "aps": "#ff7f0e"}
        all_sizes = []
        for score_type in SCORE_TYPES:
            all_sizes.append(summary[score_type]["baseline"]["average_set_size"]["mean"])
            all_sizes.extend(
                summary[score_type]["lambdas"][f"{value:g}"]["average_set_size"]["mean"]
                for value in lambdas
            )
        y_max = max(1.0, max(all_sizes) * 1.1)

        def x_pixel(index):
            return left + index * (width - left - right) / max(1, len(lambdas) - 1)

        def y_pixel(value):
            return top + (y_max - value) * (height - top - bottom) / y_max

        for tick in range(6):
            value = tick * y_max / 5
            y_value = y_pixel(value)
            draw.line((left, y_value, width - right, y_value), fill="#dddddd")
            draw.text((20, y_value - 7), f"{value:.1f}", fill="black")
        for score_type in SCORE_TYPES:
            color = colors[score_type]
            means = [
                summary[score_type]["lambdas"][f"{value:g}"]["average_set_size"]["mean"]
                for value in lambdas
            ]
            points = [(x_pixel(index), y_pixel(value)) for index, value in enumerate(means)]
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            for point in points:
                draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
            baseline = summary[score_type]["baseline"]["average_set_size"]["mean"]
            baseline_y = y_pixel(baseline)
            for x_start in range(left, width - right, 24):
                draw.line((x_start, baseline_y, min(x_start + 12, width - right), baseline_y), fill=color, width=2)
        draw.line((left, top, left, height - bottom), fill="black", width=3)
        draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=3)
        for index, value in enumerate(lambdas):
            draw.text((x_pixel(index) - 8, height - bottom + 20), f"{value:g}", fill="black")
        draw.text((width // 2 - 25, height - 45), "Lambda", fill="black")
        draw.text((20, 20), "Mean set size: CE=blue, APS=orange; dashed=fixed alpha", fill="black")
        image.save(output_path)
        return

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    colors = {"cross_entropy": "#1f77b4", "aps": "#ff7f0e"}
    labels = {"cross_entropy": "Cross-entropy", "aps": "APS"}
    for score_type in SCORE_TYPES:
        means = [
            summary[score_type]["lambdas"][f"{value:g}"]["average_set_size"]["mean"]
            for value in lambdas
        ]
        stds = [
            summary[score_type]["lambdas"][f"{value:g}"]["average_set_size"]["std"]
            for value in lambdas
        ]
        axis.errorbar(
            lambdas,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            color=colors[score_type],
            label=f"{labels[score_type]} ACP",
        )
        baseline = summary[score_type]["baseline"]["average_set_size"]["mean"]
        axis.axhline(
            baseline,
            color=colors[score_type],
            linestyle="--",
            alpha=0.75,
            label=f"{labels[score_type]} fixed α=0.10",
        )
    axis.set_xlabel("λ")
    axis.set_ylabel("Average prediction-set size")
    axis.set_xticks(lambdas)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_extension_alpha_distributions(raw_alphas, lambdas, output_path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw

        width, height = 1500, 500
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        colors = {"cross_entropy": "#1f77b4", "aps": "#ff7f0e"}
        bins = 30
        panel_width = width // len(lambdas)
        for panel, lambda_reg in enumerate(lambdas):
            panel_left = panel * panel_width + 55
            panel_right = (panel + 1) * panel_width - 25
            top, bottom = 45, height - 65
            histograms = {}
            maximum = 1.0
            for score_type in SCORE_TYPES:
                values = torch.tensor(raw_alphas[score_type][f"{lambda_reg:g}"])
                histogram = torch.histc(values, bins=bins, min=0.0, max=1.0)
                histograms[score_type] = histogram
                maximum = max(maximum, histogram.max().item())
            for score_type in SCORE_TYPES:
                histogram = histograms[score_type]
                points = []
                for index, count in enumerate(histogram.tolist()):
                    x_value = panel_left + index * (panel_right - panel_left) / (bins - 1)
                    y_value = bottom - count * (bottom - top) / maximum
                    points.append((x_value, y_value))
                draw.line(points, fill=colors[score_type], width=3)
            draw.line((panel_left, top, panel_left, bottom), fill="black", width=2)
            draw.line((panel_left, bottom, panel_right, bottom), fill="black", width=2)
            title_x = panel_left + (panel_right - panel_left) // 2 - 30
            draw.text((title_x, 15), f"lambda={lambda_reg:g}", fill="black")
            draw.text((panel_left, bottom + 15), "0", fill="black")
            draw.text((panel_right - 8, bottom + 15), "1", fill="black")
        draw.text((width // 2 - 80, height - 25), "Adaptive alpha", fill="black")
        draw.text((width - 170, 15), "CE=blue, APS=orange", fill="black")
        image.save(output_path)
        return

    figure, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 4), sharey=True)
    if len(lambdas) == 1:
        axes = [axes]
    colors = {"cross_entropy": "#1f77b4", "aps": "#ff7f0e"}
    labels = {"cross_entropy": "Cross-entropy", "aps": "APS"}
    for axis, lambda_reg in zip(axes, lambdas):
        key = f"{lambda_reg:g}"
        for score_type in SCORE_TYPES:
            axis.hist(
                raw_alphas[score_type][key],
                bins=30,
                range=(0, 1),
                density=True,
                histtype="step",
                linewidth=2,
                color=colors[score_type],
                label=labels[score_type],
            )
        axis.set_title(f"λ={lambda_reg:g}")
        axis.set_xlabel(r"Adaptive $\tilde{\alpha}$")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _run_extension_for_model(
    model,
    calloader,
    testloader,
    device,
    *,
    seed,
    lambdas,
    fixed_alpha,
    policy_epochs,
    policy_lr,
    k,
    policy_batch_size,
):
    run = {"seed": seed, "scores": {}}
    raw_alphas = {
        score_type: {f"{lambda_reg:g}": [] for lambda_reg in lambdas}
        for score_type in SCORE_TYPES
    }
    for score_type in SCORE_TYPES:
        print(f"\nSeed {seed}: score={score_type}")
        cal_logits, cal_scores = _calibration_outputs(
            model, calloader, device, score_type
        )
        properties = score_property_report(cal_logits, score_type)
        required_checks = (
            properties["finite"],
            properties["nonnegative"],
            properties["negatively_oriented"],
        )
        if not all(required_checks):
            raise ValueError(
                f"{score_type} failed required score checks: {properties}"
            )
        print(f"score properties: {properties}")
        total_score = cal_scores.sum()
        n = cal_scores.numel()
        baseline = evaluate_fixed_alpha(
            model,
            testloader,
            total_score,
            n,
            device,
            alpha=fixed_alpha,
            score_type=score_type,
        )
        policies = {}
        for lambda_index, lambda_reg in enumerate(lambdas):
            policy_seed = seed * 1000 + lambda_index
            torch.manual_seed(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(policy_seed)
            print(f"\nscore={score_type}, lambda={lambda_reg:g}")
            policy = _train_policy_from_outputs(
                cal_logits,
                cal_scores,
                lambda_reg,
                epochs=policy_epochs,
                lr=policy_lr,
                k=k,
                batch_size=policy_batch_size,
                score_type=score_type,
            )
            metrics = evaluate_policy(
                model,
                policy,
                testloader,
                total_score,
                n,
                device,
                score_type=score_type,
                return_alpha_values=True,
            )
            key = f"{lambda_reg:g}"
            raw_alphas[score_type][key] = metrics.pop("alpha_values")
            absolute_gain = baseline["average_set_size"] - metrics["average_set_size"]
            relative_gain = absolute_gain / baseline["average_set_size"]
            policies[key] = {
                **metrics,
                "absolute_efficiency_gain": absolute_gain,
                "relative_efficiency_gain": relative_gain,
                "post_hoc_consistent": metrics["post_hoc_ratio"] <= 1.0,
                "policy_seed": policy_seed,
            }
        run["scores"][score_type] = {
            "score_properties": properties,
            "baseline": baseline,
            "policies": policies,
        }
    return run, raw_alphas


def _write_extension_analysis(summary, config, output_path):
    checks = summary["research_checks"]
    lines = [
        "# Conformity-Score Extension Study",
        "",
        "## Configuration",
        "",
        f"Seeds: `{config['seeds']}`; lambdas: `{config['lambdas']}`; "
        f"fixed baseline alpha: `{config['fixed_alpha']}`.",
        "",
        "## Efficiency summary",
        "",
        "| Score | Lambda | ACP mean size | Fixed mean size | Absolute gain | Relative gain | Post-hoc valid for all seeds |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for score_type in SCORE_TYPES:
        baseline_size = summary[score_type]["baseline"]["average_set_size"]["mean"]
        for lambda_reg in config["lambdas"]:
            metrics = summary[score_type]["lambdas"][f"{lambda_reg:g}"]
            lines.append(
                f"| {score_type} | {lambda_reg:g} | "
                f"{metrics['average_set_size']['mean']:.4f} ± {metrics['average_set_size']['std']:.4f} | "
                f"{baseline_size:.4f} | {metrics['absolute_efficiency_gain']['mean']:.4f} | "
                f"{100 * metrics['relative_efficiency_gain']['mean']:.2f}% | "
                f"{'yes' if metrics['post_hoc_consistent_all_seeds'] else 'no'} |"
            )
    best = checks["largest_valid_efficiency_gain"]
    best_text = (
        "No score/lambda combination satisfied the empirical post-hoc check for all seeds."
        if best is None
        else (
            f"The largest valid mean relative gain was produced by `{best['score_type']}` "
            f"at lambda `{best['lambda']:g}`: {100 * best['relative_gain']:.2f}% "
            f"({best['absolute_gain']:.4f} labels)."
        )
    )
    lines.extend(
        [
            "",
            "## Research questions",
            "",
            "1. **Does ACP consistently reduce set size?** "
            + (
                "Yes, for every score/lambda pair and every seed."
                if checks["reduction_consistent_for_both_scores"]
                else "Not universally; consult the table and per-seed JSON for exceptions."
            ),
            "2. **Are adaptive-alpha distributions similar?** The maximum matched-quantile gaps by lambda are "
            f"`{checks['maximum_alpha_quantile_gap_by_lambda']}`; interpret these with the distribution plot.",
            f"3. **Which score has the largest efficiency gain?** {best_text}",
            "4. **Are the conclusions score-robust?** Treat them as robust only where the direction of the efficiency effect and the post-hoc check agree across both scores and all seeds.",
            "5. **What score properties matter?** Non-negativity and negative orientation are required. Score scale, probability-mass concentration, and smoothness also matter in practice: cross-entropy is smooth and unbounded, while APS is bounded and rank-adaptive but piecewise smooth because its ordering changes at probability ties.",
            "",
            "The post-hoc statistic is an empirical estimate; a value above one in a finite run is reported as a failed diagnostic, not as a proof that the theorem is false.",
        ]
    )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_score_extension_colab(
    *,
    seeds=(42, 43, 44),
    lambdas=(5.0, 10.0, 50.0),
    fixed_alpha=0.1,
    classifier_epochs=5,
    policy_epochs=2000,
    batch_size=64,
    policy_lr=1e-3,
    k=100.0,
    output_dir="score_extension_outputs",
):
    """Run the paired cross-entropy/APS extension study in Colab."""
    if not seeds or not lambdas:
        raise ValueError("seeds and lambdas must be non-empty")
    if not 0 < fixed_alpha < 1:
        raise ValueError("fixed_alpha must be in (0, 1)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("Warning: enable a GPU via Runtime > Change runtime type.")
    runs = []
    raw_by_seed = {}
    combined_alphas = {
        score_type: {f"{lambda_reg:g}": [] for lambda_reg in lambdas}
        for score_type in SCORE_TYPES
    }
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        trainloader, calloader, testloader = load_data(batch_size=batch_size)
        model = MLP().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        print(f"\nTraining classifier for seed {seed}...")
        test_loss = None
        test_accuracy = None
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
        run, raw_alphas = _run_extension_for_model(
            model,
            calloader,
            testloader,
            device,
            seed=seed,
            lambdas=lambdas,
            fixed_alpha=fixed_alpha,
            policy_epochs=policy_epochs,
            policy_lr=policy_lr,
            k=k,
            policy_batch_size=batch_size,
        )
        run["classifier"] = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        runs.append(run)
        raw_by_seed[str(seed)] = raw_alphas
        for score_type in SCORE_TYPES:
            for lambda_reg in lambdas:
                key = f"{lambda_reg:g}"
                combined_alphas[score_type][key].extend(raw_alphas[score_type][key])

    config = {
        "seeds": list(seeds),
        "lambdas": list(lambdas),
        "fixed_alpha": fixed_alpha,
        "classifier_epochs": classifier_epochs,
        "policy_epochs": policy_epochs,
        "batch_size": batch_size,
        "policy_lr": policy_lr,
        "k": k,
    }
    summary = _aggregate_extension_runs(runs, lambdas)
    results = {"config": config, "runs": runs, "summary": summary}
    results_path = output_dir / "score_extension_results.json"
    raw_path = output_dir / "score_extension_alphas.pt"
    efficiency_path = output_dir / "score_extension_efficiency.png"
    alpha_path = output_dir / "score_extension_alpha_distributions.png"
    analysis_path = output_dir / "SCORE_EXTENSION_ANALYSIS.md"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(raw_by_seed, raw_path)
    _plot_extension_efficiency(summary, lambdas, efficiency_path)
    _plot_extension_alpha_distributions(combined_alphas, lambdas, alpha_path)
    _write_extension_analysis(summary, config, analysis_path)
    paths = {
        "results": str(results_path),
        "raw_alphas": str(raw_path),
        "efficiency_plot": str(efficiency_path),
        "alpha_distribution_plot": str(alpha_path),
        "analysis": str(analysis_path),
    }
    print("\nExtension-study outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    try:
        from IPython.display import Image, display

        display(Image(filename=str(efficiency_path)))
        display(Image(filename=str(alpha_path)))
    except ImportError:
        pass
    return {"results": results, "raw_alphas": raw_by_seed, "paths": paths}


def _sample_test_loader(testloader, sample_size=100, seed=42):
    if not 0 < sample_size <= len(testloader.dataset):
        raise ValueError("sample_size must be between 1 and the test-set size")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(testloader.dataset), generator=generator)[:sample_size]
    sample = Subset(testloader.dataset, indices.tolist())
    return DataLoader(
        sample,
        batch_size=min(testloader.batch_size or 64, sample_size),
        shuffle=False,
        num_workers=testloader.num_workers,
        pin_memory=testloader.pin_memory,
    )


def plot_figure_four(history, target_size, tolerance, output_path):
    """Plot Algorithm 2's mean LOO set size at every lambda trial."""
    if not history:
        raise ValueError("history must contain at least one record")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw, ImageFont

        width, height = 1200, 800
        left, right, top, bottom = 120, 45, 35, 110
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = small_font = annotation_font = None
        for font_path in (
            "DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/google-noto-vf/NotoSerif[wght].ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        ):
            try:
                font = ImageFont.truetype(font_path, 36)
                small_font = ImageFont.truetype(font_path, 24)
                annotation_font = ImageFont.truetype(font_path, 18)
                break
            except OSError:
                continue
        if font is None:
            font = small_font = annotation_font = ImageFont.load_default()
            lambda_prefix = "lambda"
        else:
            lambda_prefix = "λ"
        sizes = [record["loo_size"] for record in history]
        y_min = 0.0
        y_tick_step = 2.0
        largest_value = max(sizes + [target_size + tolerance])
        y_tick_max = max(y_tick_step, math.ceil(largest_value / y_tick_step) * y_tick_step)
        y_max = y_tick_max + 0.25 * y_tick_step

        def x_pixel(iteration):
            x_min = min(record["iteration"] for record in history) - 0.2
            x_max = max(record["iteration"] for record in history) + 0.2
            return left + (iteration - x_min) * (width - left - right) / (x_max - x_min)

        def y_pixel(value):
            return top + (y_max - value) * (height - top - bottom) / (y_max - y_min)

        band_top = y_pixel(target_size + tolerance)
        band_bottom = y_pixel(target_size - tolerance)
        draw.rectangle((left, band_top, width - right, band_bottom), fill="#ffcccc")
        for record in history:
            grid_x = x_pixel(record["iteration"])
            draw.line((grid_x, top, grid_x, height - bottom), fill="#b0b0b0", width=1)
        for tick in range(int(y_tick_max / y_tick_step) + 1):
            value = tick * y_tick_step
            tick_y = y_pixel(value)
            draw.line((left, tick_y, width - right, tick_y), fill="#b0b0b0", width=1)
            draw.text((left - 60, tick_y - 10), f"{value:g}", fill="black", font=small_font)
        target_y = y_pixel(target_size)
        for x_start in range(left, width - right, 24):
            draw.line((x_start, target_y, min(x_start + 12, width - right), target_y), fill="red", width=4)
        draw.line((left, top, left, height - bottom), fill="black", width=3)
        draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=3)

        points = [
            (x_pixel(record["iteration"]), y_pixel(record["loo_size"]))
            for record in history
        ]
        if len(points) > 1:
            draw.line(points, fill="#1f77b4", width=6)
        for record, (x_value, y_value) in zip(history, points):
            draw.ellipse(
                (x_value - 9, y_value - 9, x_value + 9, y_value + 9),
                fill="#1f77b4",
            )
            near_target = record["loo_size"] <= target_size + tolerance
            label_y = y_value + 13 if near_target and record["iteration"] % 2 else y_value - 27
            annotation = f"{lambda_prefix}={record['lambda']:.4g}"
            annotation_x = x_value + 8
            if record is history[-1]:
                annotation_width = draw.textbbox(
                    (0, 0), annotation, font=annotation_font
                )[2]
                annotation_x = x_value - annotation_width - 8
            draw.text(
                (annotation_x, label_y),
                annotation,
                fill="black",
                font=annotation_font,
            )
            draw.text((x_value - 5, height - bottom + 18), str(record["iteration"]), fill="black", font=small_font)
        draw.text((width // 2 - 40, height - 52), "Iteration", fill="black", font=font)
        y_label = Image.new("RGBA", (220, 60), (255, 255, 255, 0))
        ImageDraw.Draw(y_label).text((0, 5), "Mean Size", fill="black", font=font)
        rotated_label = y_label.rotate(90, expand=True)
        image.paste(
            rotated_label,
            (0, (height - rotated_label.height) // 2),
            rotated_label,
        )

        legend_left, legend_top = width - 315, 55
        draw.rectangle((legend_left, legend_top, width - 60, 175), fill="white", outline="#cccccc")
        draw.line((legend_left + 15, 82, legend_left + 65, 82), fill="#1f77b4", width=5)
        draw.ellipse((legend_left + 33, 74, legend_left + 49, 90), fill="#1f77b4")
        draw.text((legend_left + 75, 69), "Mean Size", fill="black", font=small_font)
        for x_start in range(legend_left + 15, legend_left + 65, 16):
            draw.line((x_start, 116, x_start + 9, 116), fill="red", width=4)
        draw.text((legend_left + 75, 103), f"Target M={target_size:g}", fill="black", font=small_font)
        draw.rectangle((legend_left + 15, 142, legend_left + 65, 158), fill="#ffcccc")
        draw.text((legend_left + 75, 136), f"Tolerance ±{tolerance:g}", fill="black", font=small_font)
        image.save(output_path)
        return image

    iterations = [record["iteration"] for record in history]
    sizes = [record["loo_size"] for record in history]
    largest_value = max(sizes + [target_size + tolerance])
    y_tick_step = 2.0
    y_tick_max = max(y_tick_step, math.ceil(largest_value / y_tick_step) * y_tick_step)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        iterations,
        sizes,
        color="#1f77b4",
        marker="o",
        linewidth=3,
        markersize=8,
        label="Mean Size",
    )
    axis.axhline(
        target_size,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=f"Target M={target_size:g}",
    )
    axis.axhspan(
        target_size - tolerance,
        target_size + tolerance,
        color="red",
        alpha=0.15,
        label=f"Tolerance ±{tolerance:g}",
    )
    for record in history:
        near_target = record["loo_size"] <= target_size + tolerance
        vertical_offset = -15 if near_target and record["iteration"] % 2 else 9
        is_last = record is history[-1]
        axis.annotate(
            rf"$\lambda$={record['lambda']:.4g}",
            (record["iteration"], record["loo_size"]),
            xytext=(-8 if is_last else 8, vertical_offset),
            textcoords="offset points",
            fontsize=9,
            horizontalalignment="right" if is_last else "left",
        )
    axis.set_xlabel("Iteration", fontsize=16)
    axis.set_ylabel("Mean Size", fontsize=16, labelpad=18)
    axis.set_xticks(iterations)
    axis.set_xlim(min(iterations) - 0.2, max(iterations) + 0.2)
    axis.set_ylim(0, y_tick_max + 0.25 * y_tick_step)
    axis.set_yticks(
        [tick * y_tick_step for tick in range(int(y_tick_max / y_tick_step) + 1)]
    )
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, color="#b0b0b0", alpha=0.7)
    axis.legend(loc="upper right", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def reproduce_figure_four(
    model,
    calloader,
    testloader,
    device,
    *,
    target_size=2.0,
    tolerance=0.1,
    initial_lambda=40.0,
    max_steps=8,
    policy_epochs=2000,
    policy_lr=1e-3,
    k=100.0,
    policy_batch_size=64,
    test_sample_size=100,
    test_seed=42,
    output_path="figure_4_reproduction.png",
    results_path=None,
    score_type="cross_entropy",
):
    """Reproduce Figure 4 using the paper's Algorithm 2 settings."""
    selected_lambda, policy, loo_size, history = select_lambda(
        model,
        calloader,
        device,
        target_size=target_size,
        tolerance=tolerance,
        initial_lambda=initial_lambda,
        max_steps=max_steps,
        epochs=policy_epochs,
        lr=policy_lr,
        k=k,
        batch_size=policy_batch_size,
        return_history=True,
        score_type=score_type,
    )
    total_score, n = get_cal_total_score(
        model, calloader, device, score_type
    )
    sampled_testloader = _sample_test_loader(
        testloader, sample_size=test_sample_size, seed=test_seed
    )
    test_metrics = evaluate_policy(
        model,
        policy,
        sampled_testloader,
        total_score,
        n,
        device,
        score_type,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_figure_four(history, target_size, tolerance, output_path)
    if results_path is None:
        results_path = Path("data") / f"{Path(output_path).stem}.json"
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_metrics = {
        name: value for name, value in test_metrics.items()
    }
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "target_size": target_size,
                "tolerance": tolerance,
                "initial_lambda": initial_lambda,
                "score_type": score_type,
                "selected_lambda": selected_lambda,
                "loo_size": loo_size,
                "test_sample_size": test_sample_size,
                "test_metrics": serializable_metrics,
                "history": history,
            },
            file,
            indent=2,
        )

    print(f"\nFigure 4 saved to {output_path}")
    print(f"Figure 4 data saved to {results_path}")
    print(f"selected lambda: {selected_lambda:.6g}")
    print(f"final mean LOO size: {loo_size:.4f}")
    print(
        f"mean test size over {test_sample_size} sampled points: "
        f"{test_metrics['average_set_size']:.4f}"
    )
    return {
        "figure": figure,
        "history": history,
        "selected_lambda": selected_lambda,
        "policy": policy,
        "loo_size": loo_size,
        "test_metrics": test_metrics,
        "output_path": str(output_path),
        "results_path": str(results_path),
    }


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
    score_type="cross_entropy",
):
    """Run all three pre-experiment checks requested in Part 3."""
    if len(lambdas) < 2:
        raise ValueError("verification requires at least two lambda values")
    if variation_tolerance <= 0:
        raise ValueError("variation_tolerance must be positive")

    cal_logits, cal_scores = _calibration_outputs(
        model, calloader, device, score_type
    )
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
            score_type=score_type,
        )
        metrics = evaluate_policy(
            model,
            policy,
            testloader,
            total_score,
            n,
            device,
            score_type,
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


def run_figure_four_colab(
    classifier_epochs=5,
    policy_epochs=2000,
    max_lambda_steps=8,
    batch_size=64,
    seed=42,
    output_path="figure_4_reproduction.png",
    results_path="data/figure_4_reproduction.json",
):
    """Train the current classifier and run the dedicated Figure 4 experiment."""
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

    reproduction = reproduce_figure_four(
        model,
        calloader,
        testloader,
        device,
        target_size=2.0,
        tolerance=0.1,
        initial_lambda=40.0,
        max_steps=max_lambda_steps,
        policy_epochs=policy_epochs,
        policy_batch_size=64,
        test_sample_size=100,
        test_seed=seed,
        output_path=output_path,
        results_path=results_path,
    )
    try:
        from IPython.display import display

        display(reproduction["figure"])
    except ImportError:
        pass
    reproduction["model"] = model
    return reproduction


# In Colab, run one of these in a new cell after loading this file:
#
# Quick smoke run:
# results = run_colab(classifier_epochs=1, policy_epochs=20, max_lambda_steps=2)
#
# Full default run:
# results = run_colab()
#
# Figure 4 reproduction:
# figure_four_results = run_figure_four_colab()
#
# Part 6 conformity-score extension study:
# extension_results = run_score_extension_colab()
