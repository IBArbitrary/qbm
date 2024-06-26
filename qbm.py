"""
Quantum Baker's Map module.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from tqdm import tqdm

# abbreviations
exp = np.exp
pi = np.pi
sqrt = np.sqrt
mpower = np.linalg.matrix_power
inv = np.linalg.inv
eig = np.linalg.eig

class QBM:

    def __init__(self, N: int = 32, alpha = 1):
        self.N = N
        self.alpha = alpha
        self.ifUV = False
        self.UV = None
        self.H = None
        self.B = None
        self.R = None
        self.psi0 = None
        self.harper_states = None
        self.baker_states = None
        self.RT = None

    # legacy
    def U_kkp(self, k: int, kp: int, N = None):
        U0 = 0j
        if N is None:
            N = self.N
        for n in range(N):
            U0 += (1/N)*exp(1j*(2*pi/N)*(kp-k+1)*(n+0.5))
        return U0

    # position basis
    def U_comp(self, n:int, n_: int, N = None, alpha = None):
        U0 = 0j
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if n == n_:
            return exp(1j*(2*pi/N)*(n + alpha*0.5))
        return U0

    # momentum basis
    def U_comp_(self, k:int, k_: int, N = None, alpha = None):
        U0 = 0j
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if k == (k_ + 1) % N:
            if k == 0:
                return (-1+0j)
            else:
                return (1+0j)
        return U0

    # legacy
    def V_kkp(self, k: int, kp: int, N = None):
        V0 = 0j
        if N is None:
            N = self.N
        if k == kp:
            return exp(1j*(2*pi/N)*(k+0.5))
        return V0

    # position basis
    def V_comp(self, n: int, n_: int, N = None, alpha = None):
        V0 = 0j
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if n == (n_ + 1) % N:
            return (1+0j)
        return V0

    # momentum basis
    def V_comp_(self, k: int, k_: int, N = None, alpha = None):
        V0 = 0j
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if k == k_:
            return exp(1j*(2*pi/N)*(k + alpha*0.5))
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

    # momentum basis
    def gen_trans_ops(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        U_ = lambda k, k_, N: self.U_comp_(k, k_, N, alpha)
        V_ = lambda k, k_, N: self.V_comp_(k, k_, N, alpha)
        U = self.gen_mat(U_, N)
        V = self.gen_mat(V_, N)
        self.UV = {
            "N": N,
            "U": U,
            "V": V,
        }
        self.ifUV = True
        return U, V

    # position basis
    def gen_trans_ops__(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        U_ = lambda n, n_, N: self.U_comp(n, n_, N, alpha)
        V_ = lambda n, n_, N: self.V_comp(n, n_, N, alpha)
        U = self.gen_mat(U_, N)
        V = self.gen_mat(V_, N)
        self.UV = {
            "N": N,
            "U": U,
            "V": V,
        }
        self.ifUV = True
        return U, V

    # legacy
    def gen_trans_ops_(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        U = self.gen_mat(self.U_kkp, N)
        V = self.gen_mat(self.V_kkp, N)
        self.UV = {
            "N": N,
            "U": U,
            "V": V,
        }
        self.ifUV = True
        return U, V

    def harper(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha

        self.gen_trans_ops(N, alpha)
        U = self.UV["U"]
        V = self.UV["V"]

        I = np.identity(N, dtype=complex)
        H = 2*I - (U + U.H)/2 - (V + V.H)/2
        self.H = {
            "N": N,
            "H": H
        }
        return H

    def T_pq(self, p: int, q:int , N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if (self.UV is None) or (self.UV["N"] != N):
            self.gen_trans_ops(N, alpha)
        U = self.UV["U"]
        V = self.UV["V"]
        if alpha == 1:
            V_ = inv(V)
        else:
            V_ = V
        return exp((1j*pi*(2 - alpha)/N)*(p*q))*(mpower(U, p) @ mpower(V_, q))

    def gen_cob_mat(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        F_kn_ = lambda k, n, N: self.F_kn(k, n, N, alpha)
        F = self.gen_mat(F_kn_, N)
        return F

    def baker(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        assert N % 2 == 0
        F_n = self.gen_cob_mat(N, alpha)
        F_n2 = self.gen_cob_mat(N//2, alpha)
        B = inv(F_n) @ block_diag(F_n2, F_n2)
        self.B = {
            "N": N,
            "B": B
        }
        return B

    def gen_harper_states(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if (self.H is None) or (self.H["N"] != N):
            self.harper(N, alpha)
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

    def gen_baker_states(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if (self.B is None) or (self.B["N"] != N):
            self.baker(N, alpha)
        B = self.B["B"]
        evals, evecs_ = eig(B)
        idx = evals.argsort()
        evals = evals[idx]
        evecs_ = evecs_[:, idx]
        evecs = []
        for _ in range(len(evecs_)):
            evecs.append(evecs_[:, _])

        qenergy = np.angle(evals) / (2*pi)

        R = self.R_sym()
        parities = np.array(
            [int(np.round(np.average(
                (R @ evecs[_]) / evecs[_]
                ).real, 1)) for _ in range(N)]
            )

        self.baker_states = {
            "N": N,
            "evals": evals,
            "evecs": evecs,
            "qenergy": qenergy,
            "parities": parities
        }

    def pq_state(self, p: int, q: int, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if self.psi0 is None:
            self.gen_harper_states(N, alpha)
        return self.T_pq(p, q, N) @ self.psi0["psi0"]

    def W_pq(self, p: int, q: int, psi, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        return ((1/N)*(np.abs(self.pq_state(p, q, N, alpha).H @ psi)**2))[0,0]

    def gen_W(self, state, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        W = np.zeros((N, N))
        for p in range(N):
            for q in range(N):
                W[q, p] = self.W_pq(q+1, p+1, state, alpha)
        return W

    def move_hs(self, i: int, crd: tuple, autocorr = True, N = None):
        if N is None:
            N = self.N
        if self.harper_states is None:
            self.gen_harper_states(N)
        hi = self.harper_states['evecs'][i]
        his = self.T_pq(*crd) @ hi
        if autocorr:
            his_pq = np.zeros((N, N))
            for p in range(N):
                for q in range(N):
                    his_pq[q, p] = self.W_pq(q+1, p+1, his)
            return his_pq
        return his

    def R_sym(self, N = None, alpha = None):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        R = -1*mpower(self.gen_cob_mat(N, alpha), 2)
        self.R = {
            "N": N,
            "R": R
        }
        return R

    def autocorr(self, T: int, N = None, alpha = 1):
        if N is None:
            N = self.N
        if alpha is None:
            alpha = self.alpha
        if (self.ifUV is False) or (self.UV["N"] != N):
            self.gen_trans_ops(N, alpha)
        RT = np.zeros((N, N))
        if (self.psi0 is None) or (self.psi0["N"] != N):
            self.gen_harper_states(N, alpha)
        if (self.B is None) or (self.B["N"] != N):
            self.baker(N, alpha)
        B = self.B["B"]
        for _ in tqdm(range(N**2)):
            p = _ // N
            q =  _ % N
            pq = self.pq_state(p, q, N, alpha)
            RT[p, q] += (np.abs(pq.H @ mpower(B, T) @ pq)**2)[0, 0]
        self.RT = {
            "T": T,
            "N": N,
            "RT": RT
        }
        return RT

class Duality:

    def __init__(self, s: QBM):
        self.s = s
        self.N = s.N
        self.alpha = None
        self.H = None
        self.h_st = None
        self.h_ev = None
        self.h_pqs = None
        self.sh_op = None
        self.shifted_harper_states = None

        self.init()

    def init(self):
        s = self.s
        N = self.N
        s.gen_harper_states()
        self.H = s.H["H"]
        self.h_st = s.harper_states["evecs"]
        self.h_ev = s.harper_states["evals"]
        self.sh_op = s.T_pq(N//2, N//2)
        self.alpha = s.alpha
        self.harper_states_pq()

    def harper_states_pq(self):
        s = self.s
        N = self.N
        h_st = self.h_st
        h_pqs = []
        for _ in tqdm(range(N)):
            atemp = np.zeros((N, N))
            for p in range(N):
                for q in range(N):
                    atemp[q, p] = s.W_pq(q+1, p+1, h_st[_])
            h_pqs.append(atemp)
        self.h_pqs = h_pqs

    def plot_compare_states(
        self, st1, st2, k, cmap = "binary", interp = "spline16"
        ):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].matshow(st1, cmap=cmap, interpolation=interp)
        axes[1].matshow(st2, cmap=cmap, interpolation=interp)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(aspect=1)
        plt.suptitle(f"Eigenstate (left) and shifted dual state (right) (k = {k})")
        plt.tight_layout()
        plt.show()

    def gen_shifted_states(
        self
        ):
        harper_states = self.s.harper_states
        sh_h_evecs = []
        sh_h_evals = []
        sh_hpqs = []
        N = self.N
        for st in harper_states['evecs']:
            sh_st = self.sh_op @ st
            sh_ev = np.mean((self.H @ sh_st) / sh_st)
            sh_h_evecs.append(sh_st)
            sh_h_evals.append(sh_ev)
        for _ in tqdm(range(N)):
            atemp = np.zeros((N, N))
            for p in range(N):
                for q in range(N):
                    atemp[q, p] = self.s.W_pq(q+1, p+1, sh_h_evecs[_])
            sh_hpqs.append(atemp)
        self.sh_hpqs = sh_hpqs
        self.shifted_harper_states = {
            "N": self.s.N,
            "evals": np.array(sh_h_evals),
            "evecs": sh_h_evecs
        }

    def get_shifted_phase(self, og, du, npc = False):
        og_st = self.h_st[og]
        du_st = self.h_st[du]
        op = self.sh_op
        npval = (og_st.H @ op @ du_st)[0, 0]
        if npc:
            return npval
        return np.angle(npval)

    def k_phase(self, k: int, npc = False):
        return self.get_shifted_phase(-1-k, k, npc)

    def plot_harper_states(
        self, sh = False, nc: int = 4, interp = 'spline16', cmap = 'hot'
        ):
        N = self.s.N
        if sh:
            self.gen_shifted_states()
            hpqs = self.sh_hpqs
            hvals = self.shifted_harper_states["evals"]
        else:
            hpqs = self.h_pqs
            hvals = self.h_ev
        if N % nc:
            nr = (N // nc) + 1
        else:
            nr = N // nc
        fig, axes = plt.subplots(nr, nc, figsize=(nc*2, nr*2))
        ind = 1
        for i in range(nr):
            for j in range(nc):
                if ind > N:
                    axes[i, j].axis('off')
                    continue
                axes[i, j].set_title(f"n = {ind} ({np.round(np.abs(hvals[ind-1]), 4)})", pad = 0.5, y = 2)
                axes[i, j].matshow(hpqs[ind-1], interpolation=interp, cmap=cmap)
                ind += 1
        for ax in axes.ravel():
            # ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
        if sh:
            plt.suptitle(f'Shifted Harper eigenstates in coherent-state representation (N = {N})')
        else:
            plt.suptitle(f'Harper eigenstates in coherent-state representation (N = {N})')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pass
