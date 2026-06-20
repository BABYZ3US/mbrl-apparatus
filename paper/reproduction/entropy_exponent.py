"""CAS reproduction — derivations.md Section 4.

log det A = sum_i log|lambda_i| is the entropy exponent (sum of discrete Lyapunov exponents);
det(A) > 0 keeps it finite and real; complex eigenvalues pair up so det is real; a rotation
block has det = 1 (orientation-preserving, the conservative op_p).

Run:  python entropy_exponent.py
"""
import sympy as sp


def log_det_is_sum_log_eig():
    l1, l2, l3 = sp.symbols("l1 l2 l3", positive=True)
    D = sp.diag(l1, l2, l3)
    assert sp.simplify(sp.log(D.det()) - (sp.log(l1) + sp.log(l2) + sp.log(l3))) == 0
    print("  log det diag(lambda) = sum_i log lambda_i   (entropy exponent)  OK")


def conjugate_pairs_keep_det_positive():
    re, im = sp.symbols("re im", real=True)
    lam = re + sp.I * im
    pair = sp.simplify(lam * sp.conjugate(lam))
    assert sp.simplify(pair - (re**2 + im**2)) == 0
    print(f"  lambda * conj(lambda) = |lambda|^2 = {pair} > 0  (det stays real & positive)  OK")


def rotation_block_det_one():
    th = sp.symbols("theta", real=True)
    Rot = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    assert sp.simplify(Rot.det() - 1) == 0
    print("  2D rotation det = 1 > 0  (op_p conservative / orientation-preserving)  OK")


def singular_blows_up_exponent():
    eps = sp.symbols("eps", positive=True)
    A = sp.diag(eps, 1)
    assert sp.limit(sp.log(A.det()), eps, 0, "+") == -sp.oo
    print("  det -> 0  =>  log det -> -oo  (entropy singularity; det>0 forbids it)  OK")


if __name__ == "__main__":
    print("=== Entropy exponent and det>0 (derivations Section 4) ===")
    log_det_is_sum_log_eig()
    conjugate_pairs_keep_det_positive()
    rotation_block_det_one()
    singular_blows_up_exponent()
    print("PASS")
