from random import randint
from copy import deepcopy
from typing import cast
from fractions import Fraction
from vector import gram_schmidt, real_vector

type mat = list[list[float|Fraction]]
type real_mat = list[list[float]]

def gen_null_mat(n: int) -> mat:
    return [[0] * n for _ in range(n)]

def gen_null_mat_np(n: int, p: int) -> mat:
    return [[0] * p for _ in range(n)]

def gen_random_mat(n: int, max_val: int) -> mat:
    M = []
    for i in range(n):
        M.append([]) # type: ignore
        for _ in range(n):
            M[i].append(randint(0, max_val)) # type: ignore
    return M # type: ignore

# p(i, j) = In - Eii - Ejj + Eij + Eji
def permutation_mat(n: int, i: int, j: int) -> mat:
    P = gen_null_mat(n)
    for k in range(n):
        P[k][k] = 1
    P[i][i] = 0
    P[j][j] = 0
    P[i][j] = 1
    P[j][i] = 1
    return P

#D(i, a) = In + (a-1)Eii
def dilatation_mat(n: int, i: int, a: float) -> mat:
    D = gen_null_mat(n)
    for k in range(n):
        D[k][k] = 1
    D[i][i] = a
    return D

#T(i, j, a) = In + aEij
def transvection_mat(n: int, i: int, j: int, a: float|Fraction) -> mat:
    T = gen_null_mat(n)
    for k in range(n):
        T[k][k] = 1
    T[i][j] += a
    return T

def transpose(M: mat) -> mat:
    n = len(M)
    L = gen_null_mat(n)
    for i in range(n):
        for j in range(n):
            L[i][j] = M[j][i]
    return L

def get_mat_padding(M: mat) -> int:
    res: list[int] = []
    for i in range(len(M)):
        for j in range(len(M[0])):
            res.append(len(str(M[i][j])))
    return max(res)

def print_mat(M: mat, spacing: int = 2) -> None:
    n = len(M)
    padding = spacing*get_mat_padding(M)
    for i in range(n):
        for j in range(n):
            pad = " "*(padding-len(str(M[i][j])))
            print(M[i][j], end=pad)
        print()

def fractionize(M: mat) -> mat:
    n = len(M)
    for i in range(n):
        for j in range(n):
            M[i][j] = Fraction(str(M[i][j])).limit_denominator()
    return M

def permute(M: mat, i: int, j: int) -> None:
    temp = M[i]
    M[i] = M[j]
    M[j] = temp

def transvection(M: mat, i: int, j: int, a: float|Fraction) -> None:
    n = len(M)
    for k in range(n):
        M[i][k] += a*M[j][k]

def _echelon(M: mat, L_list: list[mat]=[]) -> int:
    n = len(M)
    permutations = 0
    starting_i = 0
    j = 0
    while j < n:
        for i in range(starting_i, n):
            if M[i][j] != 0:
                if i != starting_i:
                    permutations += 1
                    permute(M, starting_i, i)
                    # L_list.append(permutation_mat(n, starting_i, i))
                for ii in range(starting_i + 1, n):
                    a = Fraction(-M[ii][j]/M[starting_i][j])
                    transvection(M, ii, starting_i, a)
                    L_list.append(transvection_mat(n, ii, starting_i, -a))
                starting_i += 1
            else: continue
        j += 1 
    return permutations

def echelon(M: mat, L_list: list[mat]=[]) -> mat:
    L = deepcopy(M)
    _echelon(L, L_list)
    return L

def det_ech(M: mat, permutations: int = 0) -> float|Fraction:
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    res = 1
    for i in range(n):
        res *= M[i][i]
    return (-1)**permutations *res

def _det(M: mat) -> float|Fraction:
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    permutations = _echelon(M)
    res = 1
    for i in range(n):
        res *= M[i][i]
    return (-1)**permutations *res

def det(M: mat) -> float|Fraction:
    L = deepcopy(M)
    return _det(L)

def is_sym(M: mat) -> bool:
    L = transpose(M)
    n = len(M)
    for i in range(n):
        for j in range(n):
            if M[i][j] != L[i][j]: return False
    return True

def is_sym_pos_def(M: mat) -> bool:
    if not is_sym(M): return False
    n = len(M)
    for k in range(n+1):
        if det(M[0:k][0:k]) <= 0: return False
    return True

def mult(M: mat, N: mat) -> mat|None:
    n, p = len(M), len(M[0])
    q, r = len(N), len(N[0])
    if p != q: return None
    P = gen_null_mat_np(n, r)
    for i in range(n):
        for j in range(r):
            for k in range(p):
                P[i][j] += M[i][k]*N[k][j]
    return P

def mat_sum(M: mat, N: mat) -> mat|None:
    n, p = len(M), len(M[0])
    q, r = len(N), len(N[0])
    if not (n==q and p == r): return None
    S = gen_null_mat_np(n, p)
    for i in range(n):
        for j in range(p):
            S[i][j] = M[i][j] + N[i][j]
    return S

def LU(M: mat) -> tuple[mat|None, mat|None] :
    if det(M) == 0: return (None, None)
    n = len(M)
    L_list: list[mat] = []
    U = echelon(M, L_list)
    l = gen_null_mat(n)
    for k in range(n):
        l[k][k] = 1
    for i in range(len(L_list)):
        l = cast(mat, mult(l, L_list[i]))
    return (l, U)

# A = QR => R = t(Q)A
def QR(M: mat) -> tuple[mat|None, mat|None]:
    if det(M) == 0: return (None, None)
    cols = transpose(M)
    Q: mat = gram_schmidt(cols)
    R = mult(transpose(Q), M)
    return (Q, R)

def cholesky(M: mat) -> tuple[mat|None, mat|None]:
    if not is_sym_pos_def(M): return (None, None)
    n = len(M)
    L = gen_null_mat(n)
    for i in range(n):
        for j in range(i+1):
            s = 0
            for k in range(j):
                s += L[i][k]*L[j][k]
            if i == j:
                L[i][i] = (M[i][i] - s)**0.5
            else:
                L[i][j] = (M[i][j] - s)/L[j][j]
    return (L, transpose(L))

def solve_tri_sup(A: mat, b: real_vector) -> real_vector | None:
    if det_ech(A) == 0: return None
    n = len(A)
    sol: real_vector = [0] * n
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i+1, n):
            s += -1*A[i][j]*sol[j]
        sol[i] = s/A[i][i]
    return sol
    

# A = fractionize([
#     [3, 2, 1],
#     [1, 3, 2],
#     [2, 4, 6],
# ])

# B = fractionize([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 2]
# ])

# Q, R = QR(B)
# if Q!=None and R!=None:
#     print_mat(Q)
#     print()
#     print_mat(R)

# L, U = LU(A);

# if L != None and U != None:
#     print_mat(U);
#     print()
#     print_mat(L)


# C = fractionize([
#     [9, 6, 3],
#     [6, 8, 4],
#     [3, 4, 6]
# ])
# L, tL = cholesky(C)

# if L != None and tL != None:
#     print_mat(L)
#     print()
#     print_mat(tL)
#     print()
#     print_mat(cast(mat, mult(L, tL)))