from fractions import Fraction

type vector = list[float|Fraction]
type real_vector = list[float]
type base = list[vector]

def vec_dot(u: vector, v: vector) -> float|Fraction:
    n = len(u)
    assert len(v) == n, "must have the same dimension"
    res = 0
    for i in range(n):
        res += u[i]*v[i]
    return res

def scalar(k: float|Fraction, v: vector) -> vector:
    return [k*u for u in v]

def norm(v: vector) -> float:
    res = 0
    for i in range(len(v)):
        res += v[i]**2
    return res**0.5

def vec_sum(*args:vector) -> vector:
    n = len(args[0])
    for i in range(len(args)):
        assert len(args[i]) == n, "must have the same dimension"
    w:vector = [0] * n
    for i in range(n):
        for j in range(len(args)):
            w[i] += args[j][i]
    return w

# we leave this bitch until tomorrow
# ei = (vi - sum(vec_dot(vi, ej)ej))/norm
def gram_schmidt(b: base) -> base:
    n = len(b)
    orthonormal_b: base = [[0]*n for _ in range(n)] # type: ignore
    for i in range(n):
        orthonormalized: vector = b[i]
        for j in range(i):
            orthonormalized = vec_sum(orthonormalized, scalar(-vec_dot(b[i], orthonormal_b[j]), orthonormal_b[j]))
        orthonormal_b[i] = scalar(1/norm(orthonormalized), orthonormalized)
    return orthonormal_b

b: base = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]
]
