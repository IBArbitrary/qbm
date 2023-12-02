#!/usr/bin/env python

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import matplotlib as mpl
from tqdm import tqdm

# abbreviations
exp = np.exp
pi = np.pi
sqrt = np.sqrt
mpower = np.linalg.matrix_power
inv = np.linalg.inv
eig = np.linalg.eig

class QBM:

    def __init__(self, N: int = 32):
        self.N = N
        self.ifUV = False
        self.UV = None
        self.H = None
        self.B = None
        self.psi0 = None
        self.harper_states = None

    def U_kkp(self, k: int, kp: int, N = None):
        U0 = 0j
        if N is None:
            N = self.N
        for _ in range(N):
            U0 += (1/N)*exp(1j*(2*pi/N)*(kp-k+1)*(_+0.5))
        return U0

    def V_kkp(self, k: int, kp: int, N = None):
        V0 = 0j
        if N is None:
            N = self.N
        if k == kp:
            return exp(1j*(2*pi/N)*(k+0.5))
        else:
            return V0

    # change of basis operator
    # for alpha values, 1 gives sareceno version, 0 gives balazs version
    def F_kn(self, k: int, n: int, N = None, alpha = 1):
        if N is None:
            N = self.N
        return (1/sqrt(N))*exp(-1j*(2*pi/N)*(k+alpha*0.5)*(n+alpha*0.5))

    def gen_mat(self, op: callable, N = None):
        if N is None:
            N = self.N
        mat = np.matrix(np.zeros((N, N), dtype=complex))
        for i in range(N):
            for j in range(N):
                mat[i, j] += op(i, j, N)
        return mat

    def gen_trans_ops(self, N = None):
        if N is None:
            N = self.N
        U = self.gen_mat(self.U_kkp, N)
        V = self.gen_mat(self.V_kkp, N)
        self.UV = {
            "N": N,
            "U": U,
            "V": V,
        }
        self.ifUV = True

    def harper(self, N = None):
        if N is None:
            N = self.N

        self.gen_trans_ops(N)
        U = self.UV["U"]
        V = self.UV["V"]

        I = np.identity(N, dtype=complex)
        H = 2*I - (U + U.H)/2 - (V + V.H)/2
        self.H = {
            "N": N,
            "H": H
        }
        return H

    def T_pq(self, p: int, q:int , N = None):
        if N is None:
            N = self.N
        if (self.UV is None) or (self.UV["N"] != N):
            self.gen_trans_ops()
        U = self.UV["U"]
        V = self.UV["V"]
        return exp((1j*pi/N))*(mpower(U, p) @ mpower(inv(V), q))

    def gen_cob_mat(self, N = None, alpha = 1):
        if N is None:
            N = self.N
        F_kn_ = lambda k, n, N: self.F_kn(k, n, N, alpha)
        F = self.gen_mat(F_kn_, N)
        return F

    def baker(self, N = None, alpha = 1):
        if N is None:
            N = self.N
        assert N % 2 == 0
        F_kn_ = lambda k, n, N: self.F_kn(k, n, N, alpha)
        F_n = self.gen_mat(F_kn_, N)
        F_n2 = self.gen_mat(F_kn_, N//2)
        B = inv(F_n) @ block_diag(F_n2, F_n2)
        self.B = {
            "N": N,
            "B": B
        }
        return B

    def gen_harper_states(self, N = None):
        if N is None:
            N = self.N
        if (self.H is None) or (self.H["N"] != N):
            self.harper(N)
        H = self.H["H"]
        evals, evecs_ = eig(H)
        idx = evals.argsort()
        evals = evals[idx]
        evecs_ = evecs_[:, idx]
        evecs = []
        for _ in range(len(evecs_)):
            evecs.append(evecs_[:, _])
        self.harper_states = {
            "N": N,
            "evals": evals,
            "evecs": evecs
        }
        psi0 = self.harper_states["evecs"][0]
        self.psi0 = {
            "N": N,
            "psi0": psi0
        }

    def gen_baker_states(self, N = None):
        if N is None:
            N = self.N
        if (self.B is None) or (self.B["N"] != N):
            self.baker(N)
        B = self.B["B"]
        evals, evecs_ = eig(B)
        idx = evals.argsort()
        evals = evals[idx]
        evecs_ = evecs_[:, idx]
        evecs = []
        for _ in range(len(evecs_)):
            evecs.append(evecs_[:, _])
        self.baker_states = {
            "N": N,
            "evals": evals,
            "evecs": evecs
        }

    def pq_state(self, p: int, q: int, N = None):
        if N is None:
            N = self.N
        if self.psi0 is None:
            self.gen_harper_states(N)
        return self.T_pq(p, q, N) @ self.psi0["psi0"]

    def W_pq(self, p: int, q: int, psi, N = None):
        if N is None:
            N = self.N
        return ((1/N)*(np.abs(self.pq_state(p, q, N).T @ psi)**2))[0,0]

    def R_sym(self, N = None):
        if N is None:
            N = self.N
        R = -1*mpower(self.gen_mat(self.F_kn, N), 2)
        self.R = {
            "N": N,
            "R": R
        }
        return R

    def autocorr(self, T: int, N = None, alpha = 1):
        if N is None:
            N = self.N
        if (self.ifUV is False) or (self.UV["N"] != N):
            self.gen_trans_ops()
        U = self.UV["U"]
        Vinv = inv(self.UV["V"])
        RT = np.zeros((N, N))
        if (self.psi0 is None) or (self.psi0["N"] != N):
            self.gen_harper_states()
        psi01 = self.psi0["psi0"]
        if (self.B is None) or (self.B["N"] != N):
            self.baker(N, alpha)
        B = self.B["B"]
        for _ in tqdm(range(N**2)):
            p = _ // N
            q =  _ % N
            pq = self.pq_state(p, q, N)
            RT[p, q] += (np.abs(pq.T @ mpower(B, T) @ pq)**2)[0, 0]
        self.RT = {
            "T": T,
            "N": N,
            "RT": RT
        }
        return RT

if __name__ == "__main__":
    pass