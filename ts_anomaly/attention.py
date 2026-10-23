"""A small Transformer encoder from scratch in pure Python.

Scaled dot-product self-attention, learned positional encodings,
a 2-layer encoder with residual connections, layer normalization,
and a feed-forward block, all with manual backpropagation.

Tensors are nested lists (rows x cols). Dimensions are kept small on
purpose: this is a teaching implementation that runs on CPU in pure
Python.
"""

from __future__ import annotations

import math
import random
from operator import mul


# --------------------------------------------------------------------------
# tiny matrix helpers (lists of lists of floats)
# --------------------------------------------------------------------------

def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    # Transpose once; operator.mul via map keeps the inner product loop
    # in C instead of per-element Python bytecode.
    cols = list(zip(*B))
    return [[sum(map(mul, row, col)) for col in cols] for row in A]


def transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*A)]


def add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def softmax_rows(X: list[list[float]]) -> list[list[float]]:
    out = []
    for row in X:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        out.append([e / s for e in exps])
    return out


def relu_rows(X: list[list[float]]) -> list[list[float]]:
    return [[v if v > 0.0 else 0.0 for v in row] for row in X]


# --------------------------------------------------------------------------
# parameters and modules
# --------------------------------------------------------------------------

class Parameter:
    """A trainable tensor with an accumulated gradient."""

    def __init__(self, value: list[list[float]]) -> None:
        self.value = value
        self.grad = _zeros(len(value), len(value[0]))

    def zero_grad(self) -> None:
        for row in self.grad:
            for j in range(len(row)):
                row[j] = 0.0


def _xavier(rng: random.Random, fan_in: int, fan_out: int) -> float:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit)


class Linear:
    """y = x @ W + b, with manual backward."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random) -> None:
        self.W = Parameter(
            [[_xavier(rng, in_dim, out_dim) for _ in range(out_dim)]
             for _ in range(in_dim)]
        )
        self.b = Parameter([[0.0] * out_dim])
        self._x: list[list[float]] = []

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        self._x = X
        Y = matmul(X, self.W.value)
        b = self.b.value[0]
        return [[v + bv for v, bv in zip(row, b)] for row in Y]

    def backward(self, G: list[list[float]]) -> list[list[float]]:
        X = self._x
        Xt = transpose(X)
        dW = matmul(Xt, G)
        Wg, bg = self.W.grad, self.b.grad[0]
        for i in range(len(dW)):
            for j in range(len(dW[0])):
                Wg[i][j] += dW[i][j]
        for j in range(len(bg)):
            s = 0.0
            for i in range(len(G)):
                s += G[i][j]
            bg[j] += s
        return matmul(G, transpose(self.W.value))

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]


class LayerNorm:
    """Per-row normalization with learnable scale/shift."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.gamma = Parameter([[1.0] * dim])
        self.beta = Parameter([[0.0] * dim])
        self.eps = eps
        self._cache: tuple = ()

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        d = len(X[0])
        xhats, invs, means = [], [], []
        Y = []
        for row in X:
            mu = sum(row) / d
            var = sum((v - mu) ** 2 for v in row) / d
            inv = 1.0 / math.sqrt(var + self.eps)
            xhat = [(v - mu) * inv for v in row]
            xhats.append(xhat)
            invs.append(inv)
            means.append(mu)
            g, b = self.gamma.value[0], self.beta.value[0]
            Y.append([g[j] * xhat[j] + b[j] for j in range(d)])
        self._cache = (X, xhats, invs, means)
        return Y

    def backward(self, G: list[list[float]]) -> list[list[float]]:
        X, xhats, invs, means = self._cache
        d = len(X[0])
        g_gamma, g_beta = self.gamma.grad[0], self.beta.grad[0]
        dX = []
        for gi, row, xhat, inv, mu in zip(G, X, xhats, invs, means):
            gamma = self.gamma.value[0]
            dxhat = [gi[j] * gamma[j] for j in range(d)]
            for j in range(d):
                g_gamma[j] += gi[j] * xhat[j]
                g_beta[j] += gi[j]
            diff = [v - mu for v in row]
            dvar = (
                sum(dxhat[j] * diff[j] for j in range(d))
                * (-0.5)
                * (inv ** 3)
            )
            dmu = sum(-dxhat[j] * inv for j in range(d)) + dvar * (
                -2.0 * sum(diff) / d
            )
            dX.append(
                [
                    dxhat[j] * inv + dvar * 2.0 * diff[j] / d + dmu / d
                    for j in range(d)
                ]
            )
        return dX

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]


class SelfAttention:
    """Single-head scaled dot-product self-attention."""

    def __init__(self, dim: int, rng: random.Random) -> None:
        self.dim = dim
        self.Wq = Linear(dim, dim, rng)
        self.Wk = Linear(dim, dim, rng)
        self.Wv = Linear(dim, dim, rng)
        self.Wo = Linear(dim, dim, rng)
        self.last_weights: list[list[float]] = []
        self._cache: tuple = ()

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        q = self.Wq.forward(X)
        k = self.Wk.forward(X)
        v = self.Wv.forward(X)
        scale = 1.0 / math.sqrt(self.dim)
        scores = matmul(q, transpose(k))
        scores = [[s * scale for s in row] for row in scores]
        w = softmax_rows(scores)
        self.last_weights = w
        a = matmul(w, v)
        self._cache = (q, k, v, w)
        return self.Wo.forward(a)

    def backward(self, G: list[list[float]]) -> list[list[float]]:
        q, k, v, w = self._cache
        scale = 1.0 / math.sqrt(self.dim)
        ga = self.Wo.backward(G)  # dL/da
        # a = w @ v
        gv = matmul(transpose(w), ga)
        gw = matmul(ga, transpose(v))
        # softmax backward per row: ds = w * (dg - sum(dg * w))
        gs = []
        for wi, gwi in zip(w, gw):
            dot = sum(a_ * b_ for a_, b_ in zip(wi, gwi))
            gs.append([wi[j] * (gwi[j] - dot) * scale for j in range(len(wi))])
        # scores = q @ k^T
        gq = matmul(gs, k)
        gk = matmul(transpose(gs), q)
        gx = self.Wq.backward(gq)
        gx = add(gx, self.Wk.backward(gk))
        gx = add(gx, self.Wv.backward(gv))
        return gx

    def parameters(self) -> list[Parameter]:
        return (
            self.Wq.parameters()
            + self.Wk.parameters()
            + self.Wv.parameters()
            + self.Wo.parameters()
        )


class FeedForward:
    """Two-layer MLP with ReLU, dim -> d_ff -> dim."""

    def __init__(self, dim: int, d_ff: int, rng: random.Random) -> None:
        self.l1 = Linear(dim, d_ff, rng)
        self.l2 = Linear(d_ff, dim, rng)
        self._z: list[list[float]] = []

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        self._z = self.l1.forward(X)
        return self.l2.forward(relu_rows(self._z))

    def backward(self, G: list[list[float]]) -> list[list[float]]:
        gh = self.l2.backward(G)
        gz = [
            [gh[i][j] if self._z[i][j] > 0.0 else 0.0 for j in range(len(gh[0]))]
            for i in range(len(gh))
        ]
        return self.l1.backward(gz)

    def parameters(self) -> list[Parameter]:
        return self.l1.parameters() + self.l2.parameters()


class EncoderBlock:
    """x + Attn(LN(x)) then y + FFN(LN(y))."""

    def __init__(self, dim: int, d_ff: int, rng: random.Random) -> None:
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, rng)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, d_ff, rng)

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        a = self.attn.forward(self.ln1.forward(X))
        y = add(X, a)
        f = self.ffn.forward(self.ln2.forward(y))
        return add(y, f)

    def backward(self, G: list[list[float]]) -> list[list[float]]:
        # out = y + f ; y = X + a
        gf = [row[:] for row in G]
        gy = [row[:] for row in G]
        gy_ln = self.ln2.backward(self.ffn.backward(gf))
        gy = add(gy, gy_ln)
        ga = [row[:] for row in gy]
        gx = [row[:] for row in gy]
        gx_ln = self.ln1.backward(self.attn.backward(ga))
        return add(gx, gx_ln)

    def parameters(self) -> list[Parameter]:
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.ffn.parameters()
        )


class LearnedPositionalEncoding:
    """A learnable embedding vector per position, added to the input."""

    def __init__(self, max_len: int, dim: int, rng: random.Random) -> None:
        self.P = Parameter(
            [[rng.uniform(-0.1, 0.1) for _ in range(dim)]
             for _ in range(max_len)]
        )
        self._t = 0

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        self._t = len(X)
        P = self.P.value
        return [[x + P[i][j] for j, x in enumerate(row)] for i, row in enumerate(X)]

    def backward(self, G: list[list[float]]) -> list[list[float]]:
        Pg = self.P.grad
        for i, grow in enumerate(G):
            for j, gv in enumerate(grow):
                Pg[i][j] += gv
        return [row[:] for row in G]

    def parameters(self) -> list[Parameter]:
        return [self.P]


class TransformerEncoder:
    """Embed scalar series -> 2 encoder blocks -> scalar reconstruction."""

    def __init__(
        self,
        window: int,
        d_model: int = 16,
        d_ff: int = 32,
        n_layers: int = 2,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        self.window = window
        self.d_model = d_model
        self.embed = Linear(1, d_model, rng)
        self.pos = LearnedPositionalEncoding(window, d_model, rng)
        self.blocks = [EncoderBlock(d_model, d_ff, rng) for _ in range(n_layers)]
        self.out = Linear(d_model, 1, rng)
        self._params = (
            self.embed.parameters()
            + self.pos.parameters()
            + [p for b in self.blocks for p in b.parameters()]
            + self.out.parameters()
        )

    def forward(self, window: list[float]) -> list[float]:
        X = [[v] for v in window]
        h = self.embed.forward(X)
        h = self.pos.forward(h)
        for block in self.blocks:
            h = block.forward(h)
        y = self.out.forward(h)
        return [row[0] for row in y]

    def backward(self, grad: list[float]) -> None:
        G = [[g] for g in grad]
        G = self.out.backward(G)
        for block in reversed(self.blocks):
            G = block.backward(G)
        G = self.pos.backward(G)
        self.embed.backward(G)

    def parameters(self) -> list[Parameter]:
        return self._params

    def zero_grad(self) -> None:
        for p in self._params:
            p.zero_grad()

    def sgd_step(self, lr: float, max_norm: float = 1.0) -> None:
        total = sum(
            g * g for p in self._params for row in p.grad for g in row
        )
        norm = math.sqrt(total)
        scale = min(1.0, max_norm / (norm + 1e-12))
        for p in self._params:
            for vr, gr in zip(p.value, p.grad):
                for j in range(len(vr)):
                    vr[j] -= lr * scale * gr[j]
