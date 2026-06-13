"""
================================================================================
一维边界扫描求解器: 三结果对称 POVM 的最大可证明功 W
================================================================================

【核心思想】
  对于三结果对称 (三角形) POVM, 概率向量 p 满足:
    1) sum_k p_k = 1  (POVM 完备性)  -> 实际只有 2 个自由度
    2) p_k 是宏观量期望 (g1, g2) = (Tr G1 rho, Tr G2 rho) 的仿射函数

  目标 D_KL(p||q) 是 p 的凸函数, p 又是 (g1,g2) 的仿射函数,
  所以目标是 (g1,g2) 的凸函数。凸函数在凸集上的最大值必在边界取得。

  可行的 (g1,g2) 集合 = 二维联合数值域 (joint numerical range):
    W = { (Tr G1 rho, Tr G2 rho) : rho >= 0, Tr rho = 1 }

  其边界由支撑函数给出:
    h(phi) = max_rho Tr[(cos phi G1 + sin phi G2) rho]
           = lambda_max( cos phi * G1 + sin phi * G2 )
  对应边界点 (g1(phi), g2(phi)) 由该算符的最大本征态给出。

  于是二维优化降为【一维角度扫描】:
    beta*W = max_{phi in [0,2pi)}  D_KL( p(g1(phi), g2(phi)) || q )
================================================================================
"""
import numpy as np

# ============================================================
# 基本算符与哈密顿量 (与主程序一致)
# ============================================================
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_list(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def single_site_op(N, site, op):
    ops = [I2] * N
    ops[site] = op
    return kron_list(ops)


def two_site_op(N, i, j, opi, opj):
    ops = [I2] * N
    ops[i] = opi
    ops[j] = opj
    return kron_list(ops)


def build_ising_hamiltonian(N, J=1.0, g=0.7, h=0.2):
    """横场 Ising: H = -J sum SZ SZ - g sum SX - h sum SZ"""
    d = 2 ** N
    H = np.zeros((d, d), dtype=complex)
    for j in range(N - 1):
        H += -J * two_site_op(N, j, j + 1, SZ, SZ)
    for j in range(N):
        H += -g * single_site_op(N, j, SX)
    for j in range(N):
        H += -h * single_site_op(N, j, SZ)
    return H


def diagonalize(H):
    evals, evecs = np.linalg.eigh(H)
    return evals - evals.min(), evecs


def gibbs_state_from_spec(evals, evecs, beta):
    w = np.exp(-beta * evals)
    Z = w.sum()
    return (evecs * w) @ evecs.conj().T / Z


def build_observables_G1G2(N):
    """G1 = (1/N) sum SZ_j,  G2 = (1/N) sum SX_j"""
    d = 2 ** N
    G1 = np.zeros((d, d), dtype=complex)
    G2 = np.zeros((d, d), dtype=complex)
    for j in range(N):
        G1 += single_site_op(N, j, SZ)
        G2 += single_site_op(N, j, SX)
    G1 /= N
    G2 /= N
    return G1, G2


# ============================================================
# 三结果三角形 POVM (与主程序一致)
# ============================================================
TRI_DIRS = np.array([
    [np.cos(0.0),       np.sin(0.0)],
    [np.cos(2*np.pi/3), np.sin(2*np.pi/3)],
    [np.cos(4*np.pi/3), np.sin(4*np.pi/3)],
], dtype=float)


def _max_scale(combs):
    """计算保证 POVM 半正定的最大缩放因子 c_max。"""
    cmax = np.inf
    for comb in combs:
        comb = (comb + comb.conj().T) / 2
        lam_min = np.linalg.eigvalsh(comb).min()
        if lam_min < -1e-12:
            cmax = min(cmax, -1.0 / lam_min)
    return cmax if np.isfinite(cmax) else 1.0


def build_triangle_povm(G1, G2, eta, c_max=None):
    """E_k = (1/3)[I + eta*c_max*(u_{k,1} G1 + u_{k,2} G2)], k=0,1,2"""
    d = G1.shape[0]
    Id = np.eye(d, dtype=complex)
    Gs = [G1, G2]
    combs = [sum(TRI_DIRS[k][m] * Gs[m] for m in range(2)) for k in range(3)]
    if c_max is None:
        c_max = _max_scale(combs)
    POVM = [(1/3) * (Id + eta * c_max * combs[k]) for k in range(3)]
    return POVM, c_max


# ============================================================
# 一维边界扫描求解器
# ============================================================
def _p_from_g(g1, g2, eta, c_max):
    """由宏观量期望 (g1,g2) 仿射映射到概率向量 p (3 维)。
       p_k = (1/3)[1 + eta*c_max*(cos(theta_k)*g1 + sin(theta_k)*g2)]
    """
    p = np.array([
        (1/3) * (1 + eta * c_max *
                 (TRI_DIRS[k][0] * g1 + TRI_DIRS[k][1] * g2))
        for k in range(3)
    ])
    return p


def _kl(p, q, clip=1e-16):
    """KL 散度 D_KL(p||q)。"""
    p = np.clip(p, clip, 1.0); p = p / p.sum()
    q = np.clip(q, clip, 1.0); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def max_work_via_boundary(G1, G2, eta, c_max, q_ref, n_phi=4000, refine=True):
    """
    一维边界扫描求最大可证明功 (返回 beta*W = max D_KL)。

    步骤:
      1) 对每个角度 phi, 求 A(phi)=cos(phi)G1+sin(phi)G2 的最大本征态;
         其期望 (g1,g2) 是联合数值域边界上的一点。
      2) 把 (g1,g2) 映射到 p, 计算 D_KL(p||q)。
      3) 在所有 phi 上取最大值。
      4) (可选) 在最优角度附近细化扫描, 提高精度。

    参数:
      n_phi  : 角度采样数
      refine : 是否在粗扫最优点附近做局部细化
    """
    if eta < 1e-12:
        return 0.0  # 无信息测量, p 恒等于 q, D_KL=0

    def kl_at_phi(phi):
        A = np.cos(phi) * G1 + np.sin(phi) * G2
        w, V = np.linalg.eigh(A)
        psi = V[:, -1]                          # 最大本征态 -> 边界点
        g1 = np.real(psi.conj() @ G1 @ psi)
        g2 = np.real(psi.conj() @ G2 @ psi)
        p = _p_from_g(g1, g2, eta, c_max)
        return _kl(p, q_ref)

    # --- 粗扫整个圆周 ---
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    vals = np.array([kl_at_phi(phi) for phi in phis])
    i_best = int(np.argmax(vals))
    best = float(vals[i_best])
    phi_best = phis[i_best]

    # --- 局部细化 (在最优角度邻域做更密的扫描) ---
    if refine:
        dphi = 2 * np.pi / n_phi
        for _ in range(3):                      # 多轮逐步收紧
            lo, hi = phi_best - dphi, phi_best + dphi
            fine = np.linspace(lo, hi, 201)
            fvals = np.array([kl_at_phi(phi) for phi in fine])
            j = int(np.argmax(fvals))
            if fvals[j] > best:
                best = float(fvals[j])
                phi_best = fine[j]
            dphi *= (2.0 / 200) * 2             # 缩小邻域

    return best


# ============================================================
# 交叉验证: 与外逼近方法比较 (若可用)
# ============================================================
def crossval_demo(N=5, J=1.0, g=0.7, h=0.2,
                  beta_list=(0.3, 1.0, 2.0, 4.0),
                  eta_list=(0.3, 0.6, 1.0)):
    """
    在若干 (beta, eta) 点上, 用一维边界扫描求 beta*W,
    若 pypoman 可用则同时调用外逼近方法对比。
    """
    H = build_ising_hamiltonian(N, J, g, h)
    evals, evecs = diagonalize(H)
    G1, G2 = build_observables_G1G2(N)
    _, c_max = build_triangle_povm(G1, G2, eta=1.0)  # 固定基准 c_max

    # 尝试导入外逼近求解器 (来自主程序文件)
    have_outer = False
    try:
        from W_heatmap_beta_eta import max_certifiable_work  
        have_outer = True
    except Exception:
        pass

    print(f"{'beta':>6} {'eta':>5} | {'boundary(1D)':>14}", end="")
    if have_outer:
        print(f" {'outer-approx':>14} {'|diff|':>10}")
    else:
        print()
    print("-" * (60 if have_outer else 32))

    for beta in beta_list:
        tau = gibbs_state_from_spec(evals, evecs, beta)
        for eta in eta_list:
            povm, _ = build_triangle_povm(G1, G2, eta, c_max=c_max)
            q_ref = np.array([np.real(np.trace(E @ tau)) for E in povm])
            q_ref = np.clip(q_ref, 1e-16, None); q_ref = q_ref / q_ref.sum()

            # 一维边界扫描
            bw_1d = max_work_via_boundary(G1, G2, eta, c_max, q_ref)

            line = f"{beta:6.2f} {eta:5.2f} | {bw_1d:14.8f}"
            if have_outer:
                bw_outer = max_certifiable_work(povm, q_ref, beta=beta,
                                                max_iter=60, tol=1e-9)
                line += f" {bw_outer:14.8f} {abs(bw_1d-bw_outer):10.2e}"
            print(line)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # ---- 单点演示 ----
    N = 3
    J, g, h = 1.0, 0.7, 0.5
    beta, eta = 2.0, 0.8

    H = build_ising_hamiltonian(N, J, g, h)
    evals, evecs = diagonalize(H)
    G1, G2 = build_observables_G1G2(N)
    _, c_max = build_triangle_povm(G1, G2, eta=1.0)

    tau = gibbs_state_from_spec(evals, evecs, beta)
    povm, _ = build_triangle_povm(G1, G2, eta, c_max=c_max)
    q_ref = np.array([np.real(np.trace(E @ tau)) for E in povm])
    q_ref = np.clip(q_ref, 1e-16, None); q_ref = q_ref / q_ref.sum()

    bw = max_work_via_boundary(G1, G2, eta, c_max, q_ref)
    print(f"[单点] N={N}, beta={beta}, eta={eta}")
    print(f"       beta*W (一维边界扫描) = {bw:.8f}")
    print(f"            W                = {bw/beta:.8f}\n")

    # ---- 多点交叉验证 ----
    print("=== 交叉验证表 ===")
    crossval_demo(N=N, J=J, g=g, h=h)
