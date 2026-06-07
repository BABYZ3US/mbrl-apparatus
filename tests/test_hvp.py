"""The penalty math is the whole project — test it against analytic ground truth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from mbrl.regularization.hutchinson import hvp_penalty, laplacian_trace_penalty


def quadratic(H):
    """f(x) = 0.5 x^T H x per sample; exact Hessian H everywhere."""
    return lambda x: 0.5 * torch.einsum("bi,ij,bj->b", x, H, x)


@pytest.mark.parametrize("d", [3, 8])
def test_hutchinson_unbiased_frobenius(d):
    torch.manual_seed(0)
    H = torch.randn(d, d); H = (H + H.T) / 2
    exact = (H ** 2).sum().item()
    x = torch.randn(64, d)
    # average many probes -> converges to ||H||_F^2
    est = hvp_penalty(quadratic(H), x, n_probes=400, create_graph=False).item()
    assert abs(est - exact) / exact < 0.05


def test_two_probes_default_reasonable():
    torch.manual_seed(0)
    d = 6
    H = torch.randn(d, d); H = (H + H.T) / 2
    exact = (H ** 2).sum().item()
    x = torch.randn(256, d)
    ests = [hvp_penalty(quadratic(H), x, n_probes=2, create_graph=False).item()
            for _ in range(50)]
    mean = sum(ests) / len(ests)
    assert abs(mean - exact) / exact < 0.1  # unbiased; variance is the cost


def test_null_lagrangian_trace_matches_frobenius_EL():
    """R5: (Delta f)^2 unbiased estimator on a quadratic = tr(H)^2 exactly in mean."""
    torch.manual_seed(0)
    d = 5
    # diagonal-dominant H: sizable trace so the relative test is well-conditioned
    H = torch.diag(torch.arange(1.0, d + 1)) + 0.1 * torch.randn(d, d)
    H = (H + H.T) / 2
    exact_tr2 = H.trace().item() ** 2
    x = torch.randn(64, d)
    ests = [laplacian_trace_penalty(quadratic(H), x, n_probes=2,
                                    create_graph=False).item() for _ in range(400)]
    mean = sum(ests) / len(ests)
    assert abs(mean - exact_tr2) / exact_tr2 < 0.1


def test_trace_clamp_nonnegativity():
    """Report sec.3 (thermodynamic consistency): the clamped estimator must be
    non-negative ALWAYS; the unclamped product must go negative for a Hessian
    with strongly mixed-sign eigenvalues (which is what makes clamping bind)."""
    torch.manual_seed(0)
    d = 6
    # mixed-sign spectrum, ROTATED: for diagonal H, Rademacher probes give
    # v'Hv = tr(H) exactly (v_i^2 = 1) — the estimator would be constant.
    D = torch.diag(torch.tensor([3.0, -3.0, 2.0, -2.0, 1.0, -1.0]))  # tr = 0
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    H = Q.T @ D @ Q
    x = torch.randn(32, d)
    clamped, unclamped = [], []
    for i in range(200):
        g = torch.Generator().manual_seed(i)
        clamped.append(laplacian_trace_penalty(quadratic(H), x, 2, g,
                                               create_graph=False, clamp=True).item())
        g = torch.Generator().manual_seed(i)
        unclamped.append(laplacian_trace_penalty(quadratic(H), x, 2, g,
                                                 create_graph=False, clamp=False).item())
    assert min(clamped) >= 0
    assert min(unclamped) < 0  # sign-indefinite without the clamp
    # unclamped stays unbiased for tr(H)^2 = 0; clamped is intentionally biased up
    assert abs(sum(unclamped) / len(unclamped)) < sum(clamped) / len(clamped)


def test_one_probe_trace_rejected():
    with pytest.raises(ValueError):
        laplacian_trace_penalty(lambda x: x.pow(2).sum(-1), torch.randn(4, 3), n_probes=1)


def test_penalty_differentiable():
    """create_graph=True: the penalty must backprop into model params."""
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 1))
    x = torch.randn(16, 4)
    pen = hvp_penalty(lambda x: net(x).squeeze(-1), x, n_probes=2)
    pen.backward()
    # The output-layer bias cannot affect input-space curvature (additive const),
    # so its grad is legitimately absent; weights must all receive gradient.
    weight_grads = [p.grad for n, p in net.named_parameters() if "weight" in n]
    assert all(g is not None for g in weight_grads)
    assert sum(g.abs().sum() for g in weight_grads) > 0
