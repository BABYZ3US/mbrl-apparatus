"""GAE — analytic pins + the lambda-limit identities (battle-tested estimator)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.training.returns import gae_advantages, lambda_returns


def test_single_step_is_td_residual():
    r = torch.tensor([[1.0]])
    v = torch.tensor([[0.5], [2.0]])
    adv, ret = gae_advantages(r, v, gamma=0.9, lam=0.95)
    assert torch.allclose(adv, torch.tensor([[1.0 + 0.9 * 2.0 - 0.5]]))
    assert torch.allclose(ret, adv + 0.5)


def test_lam0_is_one_step_td():
    torch.manual_seed(0)
    r, v = torch.randn(5, 3), torch.randn(6, 3)
    adv, _ = gae_advantages(r, v, gamma=0.97, lam=0.0)
    delta = r + 0.97 * v[1:] - v[:-1]
    assert torch.allclose(adv, delta, atol=1e-6)


def test_lam1_is_discounted_mc_minus_baseline():
    torch.manual_seed(1)
    r, v = torch.randn(4, 2), torch.randn(5, 2)
    adv, _ = gae_advantages(r, v, gamma=0.9, lam=1.0)
    # MC return with bootstrap: G_t = sum gamma^l r_{t+l} + gamma^{H-t} v_H
    H = 4
    G = v[-1].clone()
    Gs = []
    for t in reversed(range(H)):
        G = r[t] + 0.9 * G
        Gs.append(G.clone())
    Gs = torch.stack(list(reversed(Gs)))
    assert torch.allclose(adv, Gs - v[:-1], atol=1e-5)


def test_gae_returns_equal_lambda_returns():
    # the value-regression target (adv + v) IS the lambda-return — the two
    # estimators are the same object viewed as advantage vs return
    torch.manual_seed(2)
    r, v = torch.randn(6, 4), torch.randn(7, 4)
    _, ret = gae_advantages(r, v, gamma=0.95, lam=0.9)
    assert torch.allclose(ret, lambda_returns(r, v, 0.95, 0.9), atol=1e-5)
