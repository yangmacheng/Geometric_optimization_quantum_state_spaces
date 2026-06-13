"""
================================================================================
β–η 二维相图: 四结果 POVM 的最大可证明功 W
================================================================================

【测量设置】

  三个可观测量:
      G1 = Mz  = (1/N)   sum_j  sigma^z_j           (纵向磁化)
      G2 = Mx  = (1/N)   sum_j  sigma^x_j           (横向磁化)
      G3 = Mzz = (1/(N-1)) sum_j sigma^z_j sigma^z_{j+1}  (近邻 ZZ 相互作用)

  "可证明功" W 定义为: 测量得到的概率分布 p(rho) 相对参考分布 q (热态对应分布) 的最大 KL 散度:

    W = beta^-1 * max_{rho >= 0, Tr rho = 1}  D_KL( p(rho) || q )

  这是一个"在凸集上最大化凸函数"的问题 (凹优化), 全局最优一定在
  可行集的极点 (顶点) 上取得。

【优化方法: 外逼近 + 顶点枚举 (Outer Approximation)】
  - 可行的概率集合 P = { p(rho) } 是个凸集 (POVM 的像)。
  - 用线性不等式 A z <= b 构造一个 *包含* P 的外多面体 P_out。
  - 在 P_out 的所有顶点上求 max D_KL  ->  得到严格【上界】。
  - 用谱分解 (support function) 找到真实可行点  ->  得到【下界】。
  - 不断添加切平面 (cutting planes) 收紧 P_out, 直到上下界收敛到预设精度。
================================================================================
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.linalg import eigh, svd
from pypoman import compute_polytope_vertices  # 顶点枚举核心库


# ============================================================
# 第一部分: 基本算符与哈密顿量
# ============================================================
I2 = np.eye(2, dtype=complex)                       # 2x2 单位阵
SX = np.array([[0, 1], [1, 0]], dtype=complex)      # Pauli-X
SZ = np.array([[1, 0], [0, -1]], dtype=complex)     # Pauli-Z


def kron_list(ops):
    """对算符列表做连续张量积 (Kronecker product)。"""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def single_site_op(N, site, op):
    """构造作用在第 site 个格点上的单体算符 (其余格点为单位阵)。
       例如 single_site_op(N, 2, SZ) = I⊗I⊗SZ⊗I⊗...
    """
    ops = [I2] * N
    ops[site] = op
    return kron_list(ops)


def two_site_op(N, i, j, opi, opj):
    """构造作用在第 i, j 两个格点上的双体算符。"""
    ops = [I2] * N
    ops[i] = opi
    ops[j] = opj
    return kron_list(ops)


def build_ising_hamiltonian(N, J=1.0, g=0.7, h=0.2):
    """
    构造横场 Ising 模型哈密顿量 (开放边界 OBC):
        H = -J * sum_j SZ_j SZ_{j+1}   (近邻 ZZ 耦合)
            -g * sum_j SX_j            (横向场)
            -h * sum_j SZ_j            (纵向场)
    """
    d = 2 ** N
    H = np.zeros((d, d), dtype=complex)
    for j in range(N - 1):                          # 近邻耦合项
        H += -J * two_site_op(N, j, j + 1, SZ, SZ)
    for j in range(N):                              # 横向场项
        H += -g * single_site_op(N, j, SX)
    for j in range(N):                              # 纵向场项
        H += -h * single_site_op(N, j, SZ)
    return H


def diagonalize(H):
    """预对角化哈密顿量。返回 (本征值, 本征矢)。
       本征值减去最小值, 避免后续 exp(-beta*E) 数值溢出。
    """
    evals, evecs = np.linalg.eigh(H)
    return evals - evals.min(), evecs


def gibbs_state_from_spec(evals, evecs, beta):
    """由预对角化结果快速构造 Gibbs (热) 态:
           rho = exp(-beta*H) / Z,    Z = Tr[exp(-beta*H)]
       使用谱分解避免重复矩阵指数运算。
    """
    w = np.exp(-beta * evals)                       # 玻尔兹曼权重
    Z = w.sum()                                     # 配分函数
    return (evecs * w) @ evecs.conj().T / Z         # 谱分解重建密度矩阵


# ============================================================
# 第二部分: 构造三个可观测量 (G1=Mz, G2=Mx, G3=Mzz)
# ============================================================
def build_observables_G1G2G3(N):
    """
    构造三个宏观可观测量 (均做归一化, 量级在 [-1, 1]):
        G1 = (1/N)     sum_j SZ_j                (纵向磁化)
        G2 = (1/N)     sum_j SX_j                (横向磁化)
        G3 = (1/(N-1)) sum_j SZ_j SZ_{j+1}       (近邻 ZZ 相互作用, 对应哈密顿耦合项)
    """
    d = 2 ** N
    G1 = np.zeros((d, d), dtype=complex)
    G2 = np.zeros((d, d), dtype=complex)
    G3 = np.zeros((d, d), dtype=complex)
    for j in range(N):
        G1 += single_site_op(N, j, SZ)
        G2 += single_site_op(N, j, SX)
    for j in range(N - 1):
        G3 += two_site_op(N, j, j + 1, SZ, SZ)
    G1 /= N
    G2 /= N
    G3 /= (N - 1)
    return G1, G2, G3


# ============================================================
# 第三部分: 四结果空间四面体 POVM
# ============================================================
# 四个测量方向在三维 (G1,G2,G3) 空间内取正四面体顶点 (两两夹角 arccos(-1/3))
# 四面体顶点 (已归一化为单位向量), 满足 sum_k u_k = 0。
TET_DIRS = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
], dtype=float) / np.sqrt(3.0)


def _max_scale(combs):
    """
    计算最大允许缩放因子 c_max, 保证 POVM 元素半正定 (eta=1 时刚好饱和)。
    每个 POVM 元素形如 (1/4)(I + c * comb), 要求其最小本征值 >= 0,
    即 1 + c * lambda_min(comb) >= 0  =>  c <= -1/lambda_min  (当 lambda_min<0)。
    取所有方向中最严格的约束作为 c_max。
    """
    cmax = np.inf
    for comb in combs:
        comb = (comb + comb.conj().T) / 2           # 强制 Hermitian
        lam_min = np.linalg.eigvalsh(comb).min()
        if lam_min < -1e-12:
            cmax = min(cmax, -1.0 / lam_min)
    return cmax if np.isfinite(cmax) else 1.0


def build_tetrahedron_povm(G1, G2, G3, eta, c_max=None):
    """
    构造四结果四面体 POVM:
        E_k = (1/4) [ I + eta * c_max * (u_{k,1} G1 + u_{k,2} G2 + u_{k,3} G3) ],  k=0,1,2,3

    参数:
        eta   : 锐度参数 in [0,1]。eta=0 时四个元素都等于 I/4 (无信息测量);
                eta=1 时 POVM 元素半正定刚好饱和 (最尖锐)。
        c_max : 缩放因子, 若为 None 则自动计算。扫描时应预先固定 c_max,
                保证不同 eta 下使用同一基准。

    返回: (POVM 列表, c_max)
    """
    d = G1.shape[0]
    Id = np.eye(d, dtype=complex)
    Gs = [G1, G2, G3]
    # 每个方向的线性组合 comb_k = u_{k,1}*G1 + u_{k,2}*G2 + u_{k,3}*G3
    combs = [sum(TET_DIRS[k][m] * Gs[m] for m in range(3))
             for k in range(4)]
    if c_max is None:
        c_max = _max_scale(combs)
    POVM = [(1/4) * (Id + eta * c_max * combs[k]) for k in range(4)]
    return POVM, c_max


def verify_povm(POVM, name="", tol=1e-9):
    """检查 POVM 合法性: (1) sum_k E_k = I; (2) 每个 E_k >= 0 (半正定)。"""
    d = POVM[0].shape[0]
    S = sum(POVM)
    err = np.linalg.norm(S - np.eye(d))             # 完备性误差
    min_e = min(np.linalg.eigvalsh((E + E.conj().T) / 2).min() for E in POVM)
    print(f"[{name}] ||sum-I||={err:.2e}, min eig={min_e:.4f}, "
          f"{'OK' if (err < tol and min_e > -tol) else 'FAIL'}")


# ============================================================
# 第四部分: 外逼近 + 顶点枚举的工具函数
# ============================================================
def _normalize_povm(POVM, atol=1e-8):
    """
    若 POVM 不满足 sum E_k = I, 则做白化 (whitening) 归一化:
        S = sum E_k,  S = L L^H (Cholesky),  E_k -> L^{-H} E_k L^{-1}
    使归一化后满足完备性。
    """
    d = POVM[0].shape[0]
    S = sum(POVM)
    if np.allclose(S, np.eye(d), atol=atol):
        return POVM
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    return [Linv.conj().T @ E @ Linv for E in POVM]


def _vec_real(H):
    """把复 Hermitian 矩阵展平成实向量 (实部+虚部), 便于做实数线性代数。"""
    return np.hstack([np.real(H).ravel(), np.imag(H).ravel()])


def _affine_basis_from_povm(POVM, tol=1e-12):
    """
    构造概率向量 p 的【仿射参数化】:   p = s + Q z

    原理:
      p_i = Tr(E_i rho)。把 rho 写成 rho = I/d + (无迹部分),
      则 p_i = Tr(E_i)/d + Tr(E_i * 无迹部分)。
      - s_i = Tr(E_i)/d                  : 偏移向量 (rho=I/d 时的概率)
      - Q 的列张成 p 随 rho 变化的方向    : 由各 E_i 的"无迹化"算符的 SVD 得到

    同时把全 1 方向投影掉 (因为 sum p_i = 1 恒成立, 不是自由方向),
    使 z 只在真正自由的低维子空间中变化, 大幅降低顶点枚举维度。

    返回: s (偏移, 长度 n), Q (基矩阵, n x rdim)
    """
    d = POVM[0].shape[0]
    n = len(POVM)
    I = np.eye(d)
    trE = np.array([np.trace(E).real for E in POVM])
    s = trE / d                                     # 偏移向量

    # 把每个 E_i 的"无迹部分"展平成行向量, 堆叠成矩阵 M
    rows = []
    for i, E in enumerate(POVM):
        Ei0 = E - (trE[i] / d) * I                  # 减去迹的部分
        rows.append(_vec_real(Ei0))
    M = np.vstack(rows)

    # SVD 提取有效方向 (奇异值大于阈值的列)
    U, S, Vt = svd(M, full_matrices=False)
    r = int(np.sum(S > tol * (S[0] if S.size > 0 else 1.0)))
    Q = U[:, :r] if r > 0 else np.zeros((n, 0))

    # 投影掉全 1 方向 (因为 sum_i p_i = 1 约束, 该方向不自由)
    if r > 0:
        ones = np.ones(n)
        alpha_vec = (ones @ Q) / (ones @ ones)
        Q = Q - np.outer(ones, alpha_vec)
        Q, _ = np.linalg.qr(Q, mode='reduced')      # 重新正交化

    return s, Q


def _support_h(POVM, u):
    """
    支撑函数 h(u) = max_{rho} sum_i u_i * Tr(E_i rho) = lambda_max( sum_i u_i E_i )。
    几何意义: 可行集 P 在方向 u 上的最远支撑值。用于生成切平面。
    """
    R = sum(u[i] * POVM[i] for i in range(len(POVM)))
    w = np.linalg.eigvalsh(R)
    return float(w[-1])                             # 最大本征值


def _umax_all(POVM):
    """每个 E_i 的最大本征值, 用于构造单元素上界约束 p_i <= umax_i。"""
    return np.array([np.linalg.eigvalsh(E).max().real for E in POVM])


def _rand_pure_states(d, m, seed=0):
    """随机生成 m 个归一化的复纯态向量 (用于生成初始切平面)。"""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(d, m)) + 1j * rng.normal(size=(d, m))
    Z /= np.linalg.norm(Z, axis=0, keepdims=True)
    return [Z[:, k] for k in range(m)]


def _p_from_state(POVM, psi):
    """给定纯态 psi, 计算其测量概率分布 p_i = <psi|E_i|psi>。"""
    return np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])


# ---- 目标函数: 内部统一【最小化 F = -D_KL】, 等价于最大化 D_KL ----
def _neg_kl(p, q, clip=1e-16):
    """F(p) = -D_KL(p||q)。clip 防止 log(0)。"""
    p = np.clip(p, clip, 1.0)
    q = np.clip(q, clip, 1.0)
    return float(-np.sum(p * np.log(p / q)))


def _neg_kl_grad(p, q, clip=1e-16):
    """F 的梯度 (已归一化): grad F = -(log(p/q) + 1)。
       归一化只影响切平面方向, 不影响约束几何。
    """
    p = np.clip(p, clip, 1.0)
    q = np.clip(q, clip, 1.0)
    g = -(np.log(p / q) + 1.0)
    nrm = np.linalg.norm(g)
    if nrm > 0:
        g = g / nrm
    return g


def _build_initial_constraints_z(POVM, q_ref, s, Q,
                                 n_pair_cuts=3, n_rand_states=40,
                                 n_eigstate_cuts=3, seed=42):
    """
    在 z 空间 (仿射参数空间) 构造初始外多面体约束  A z <= b。
    约束来源有三类:
      1) 单元素 Box 约束:  0 <= p_i <= umax_i
      2) 对子上界约束:     p_i + p_j <= lambda_max(E_i + E_j)
      3) 初始切平面:       由若干纯态 (本征态 + 随机态) 处的目标梯度生成
    约束越多, 初始外多面体越紧, 迭代收敛越快。
    """
    n = len(POVM)
    rdim = Q.shape[1]
    if rdim == 0:                                   # 无自由维度 (p 唯一)
        return np.zeros((0, 0)), np.zeros(0)

    A, b = [], []

    # --- 1) 单元素 Box 约束: 0 <= p_i <= umax_i ---
    # p_i = s_i + Q[i,:] z
    #   下界 p_i >= 0     =>  -Q[i,:] z <= s_i
    #   上界 p_i <= umax  =>   Q[i,:] z <= umax_i - s_i
    umax = _umax_all(POVM)
    for i in range(n):
        A.append(-Q[i, :]); b.append(s[i])
        A.append(Q[i, :]);  b.append(umax[i] - s[i])

    # --- 2) 对子上界约束: p_i + p_j <= lambda_max(E_i + E_j) ---
    # 选取上界最紧 (lambda 最小) 的若干个对子, 加最有用的约束
    if n_pair_cuts and n > 1:
        rng = np.random.default_rng(seed)
        cand, seen = [], set()
        M_cand = min(5000, n * (n - 1) // 2)
        tries = 0
        while len(cand) < M_cand and tries < 10 * M_cand:
            i, j = rng.choice(n, 2, replace=False)
            if i > j: i, j = j, i
            if (i, j) in seen:
                tries += 1; continue
            seen.add((i, j))
            lam = np.linalg.eigvalsh(POVM[i] + POVM[j])[-1].real
            cand.append(((i, j), lam))
            tries += 1
        cand.sort(key=lambda x: x[1])               # lambda 越小约束越紧
        for (i, j), lam in cand[:min(n_pair_cuts, len(cand))]:
            A.append(Q[i, :] + Q[j, :])
            b.append(lam - (s[i] + s[j]))

    # --- 3) 初始切平面 (基于目标 F=-D_KL 的梯度) ---
    # 在某纯态对应的 p 处, 用支撑函数生成一个一定包含可行集的半空间。
    def add_cuts_from_states(states_list):
        local = []
        for psi in states_list:
            p_real = _p_from_state(POVM, psi)
            g = _neg_kl_grad(p_real, q_ref)         # 目标梯度
            hval = _support_h(POVM, -g)             # 支撑值
            Ai = -(g @ Q)                           # 切平面法向 (z 空间)
            bi = hval + float(g @ s)                # 切平面截距
            strength = -hval - np.min(g)            # 启发式: 约束"有用程度"
            local.append((Ai, bi, strength))
        return local

    cuts = []
    # (3a) 用最大本征值最大的几个 E_i 的最大本征态生成切平面
    if n_eigstate_cuts:
        idx_sorted = np.argsort(-umax)
        take = idx_sorted[:min(n_eigstate_cuts, n)]
        psis = []
        for i in take:
            _, V = eigh(POVM[i])
            psis.append(V[:, -1])                   # 最大本征态
        cuts.extend(add_cuts_from_states(psis))

    # (3b) 用随机纯态生成切平面
    if n_rand_states:
        psis = _rand_pure_states(POVM[0].shape[0], n_rand_states, seed=seed + 1)
        cuts.extend(add_cuts_from_states(psis))

    # 按"有用程度"排序后全部加入
    cuts.sort(key=lambda t: t[2], reverse=True)
    for Ai, bi, _ in cuts:
        A.append(Ai); b.append(bi)

    A = np.array(A) if len(A) > 0 else np.zeros((0, rdim))
    b = np.array(b) if len(b) > 0 else np.zeros(0)
    return A, b


# ============================================================
# 第五部分: 主优化器 (外逼近 + 顶点枚举)
# ============================================================
def max_certifiable_work(POVM, q_ref, beta=1.0,
                         max_iter=60, tol=1e-8, seed=42,
                         n_pair_cuts=3, n_rand_states=40,
                         n_eigstate_cuts=3):
    """
    用外逼近 + 顶点枚举求解:
        beta * W = max_{rho}  D_KL( p(rho) || q_ref )

    内部把问题转为【最小化 F = -D_KL】, 迭代维护两个界:
        f_minus = min_{外多面体顶点} F   ->   D_KL 的【上界】 = -f_minus
        f_plus  = min_{可行谱分解点} F   ->   D_KL 的【下界】 = -f_plus

    *** 返回 D_KL 的【上界】(certified upper bound) ***
        外逼近在更大的集合 (外多面体) 上取 max, 给出严格上界。

    迭代流程:
        A) 顶点枚举: 求当前外多面体所有顶点
        B) 在顶点上最小化 F  -> 更新上界 (f_minus)
        C) 在 B 给出的方向上做谱分解, 得到真实可行点 -> 更新下界 (f_plus)
        D) 收敛检查: 上下界差 < tol 则停止
        E) 添加切平面: 用当前最优方向收紧外多面体
    """
    POVM = _normalize_povm(POVM)                    # 确保完备性
    n = len(POVM)
    d = POVM[0].shape[0]

    # 仿射参数化 p = s + Q z
    s, Q = _affine_basis_from_povm(POVM)
    rdim = Q.shape[1]

    # 参考分布归一化
    q = np.clip(np.asarray(q_ref, dtype=float), 1e-16, 1.0)
    q = q / q.sum()

    # 退化情形: 没有自由维度 (p 被完全确定), 直接返回
    if rdim == 0:
        return -_neg_kl(s, q)

    # 构造初始外多面体约束 A z <= b
    A_z, b_z = _build_initial_constraints_z(
        POVM, q, s, Q,
        n_pair_cuts=n_pair_cuts, n_rand_states=n_rand_states,
        n_eigstate_cuts=n_eigstate_cuts, seed=seed)

    f_plus = float('inf')     # F 的最小上界 -> 对应 D_KL 下界
    f_minus = -float('inf')   # F 在外多面体上的最小值 -> 对应 D_KL 上界

    for k in range(1, max_iter + 1):
        # --- A) 顶点枚举: 列出当前外多面体所有顶点 ---
        try:
            verts = compute_polytope_vertices(A_z, b_z)
            verts = np.array(verts) if len(verts) > 0 else np.zeros((0, rdim))
        except Exception:
            break                                   # 枚举失败则退出
        if verts.shape[0] == 0:
            break

        # --- B) 在外多面体顶点上求 F 最小 (= D_KL 上界) ---
        vals = np.array([_neg_kl(s + Q @ z, q) for z in verts])
        best_idx = int(np.argmin(vals))
        f_minus = float(vals[best_idx])             # 更新上界对应值
        p_best = s + Q @ verts[best_idx]            # 最优顶点对应的概率

        # --- C) 谱分解给真实可行点 (= D_KL 下界) ---
        # 在最优方向上做支撑优化: 求 sum g_i E_i 的最小本征态,
        # 对应一个一定可行的纯态, 其 D_KL 给出下界。
        g = _neg_kl_grad(p_best, q)
        R_op = sum(g[i] * POVM[i] for i in range(n))
        evals_R, evecs_R = eigh(R_op)
        psi_min = evecs_R[:, 0]                      # 最小本征态
        p_real = _p_from_state(POVM, psi_min)        # 可行概率分布
        ub_val = _neg_kl(p_real, q)
        if ub_val < f_plus:
            f_plus = ub_val                          # 更新 F 上界 (D_KL 下界)

        # --- D) 收敛检查: 上下界夹逼 ---
        if f_plus - f_minus < tol:
            break

        # --- E) 添加切平面收紧外多面体 ---
        hval = _support_h(POVM, -g)
        Ai_new = -(g @ Q)
        bi_new = hval + float(g @ s)
        A_z = np.vstack([A_z, Ai_new.reshape(1, -1)])
        b_z = np.hstack([b_z, bi_new])

    # ====== 返回 D_KL 的【上界】(certified) ======
    DKL_ub = -f_minus
    return DKL_ub


# ============================================================
# 第六部分: 二维网格扫描 W(beta, eta)
# ============================================================
def scan_grid(N, J, g, h, beta_arr, eta_arr, dimensionless=True):
    """
    在 (beta, eta) 网格上计算最大可证明功。

    参数:
        dimensionless=True  -> 返回 beta*W = D_KL (无量纲)
        dimensionless=False -> 返回 W = D_KL / beta

    返回: W_grid, 形状 (len(eta_arr), len(beta_arr)), 行=eta, 列=beta
    """
    # 预对角化哈密顿量
    H = build_ising_hamiltonian(N, J, g, h)
    evals, evecs = diagonalize(H)
    G1, G2, G3 = build_observables_G1G2G3(N)

    # 预先固定 c_max (用 eta=1 计算), 保证整个扫描使用同一基准
    _, c_max = build_tetrahedron_povm(G1, G2, G3, eta=1.0)

    W_grid = np.zeros((len(eta_arr), len(beta_arr)))

    print(f"扫描网格 {len(eta_arr)} x {len(beta_arr)} (外逼近+顶点枚举, 返回上界) ...")
    for ie, eta in enumerate(eta_arr):
        for ib, beta in enumerate(beta_arr):
            # eta=0: POVM 无信息 (四元素全为 I/4), W 必为 0
            if eta < 1e-12:
                W_grid[ie, ib] = 0.0
                continue

            # 构造当前 (eta) 的 POVM (固定 c_max)
            povm, _ = build_tetrahedron_povm(G1, G2, G3, eta, c_max=c_max)

            # 当前 (beta) 的热态及其参考分布 q
            tau = gibbs_state_from_spec(evals, evecs, beta)
            q_ref = np.array([np.real(np.trace(E @ tau)) for E in povm])
            q_ref = np.clip(q_ref, 1e-16, None)
            q_ref = q_ref / q_ref.sum()

            # 求最大可证明功 (返回 D_KL 上界)
            bW = max_certifiable_work(povm, q_ref, beta=beta,
                                      max_iter=60, tol=1e-8,
                                      n_pair_cuts=3, n_rand_states=40,
                                      n_eigstate_cuts=3)
            W_grid[ie, ib] = bW if dimensionless else bW / beta
        print(f"  eta={eta:.3f} 行完成")
    return W_grid


# ============================================================
# 第七部分: 绘图样式与热图
# ============================================================
def set_pub_style():
    """设置出版级绘图样式 (优先用 LaTeX, 失败则退回普通 serif)。"""
    try:
        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]})
    except Exception:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif"})
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
        "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 2, "lines.markersize": 6,
        "axes.grid": False, "legend.frameon": True})


def plot_heatmap(beta_arr, eta_arr, W_grid,
                 dimensionless=True, savefig=None, n_contours=8):
    """绘制 β–η 二维热图 (pcolormesh) + 白色等高线。"""
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    BB, EE = np.meshgrid(beta_arr, eta_arr)
    norm = Normalize(vmin=W_grid.min(), vmax=W_grid.max())
    cmap = mpl.cm.get_cmap('viridis')

    pcm = ax.pcolormesh(BB, EE, W_grid, cmap=cmap, norm=norm,
                        shading='gouraud')

    # ---- 等高线: 排除与边界重合的 0 值层 ----
    # 手动生成 levels, 并剔除过于接近数据最小值(即 0)的层。
    wmin, wmax = W_grid.min(), W_grid.max()
    levels = MaxNLocator(nbins=n_contours).tick_values(wmin, wmax)
    eps = 1e-3 * (wmax - wmin)                  # 容差: 距最小值太近的层丢弃
    levels = levels[levels > wmin + eps]        # 剔除贴边的 0 等值线
    if len(levels) > 0:
        cs = ax.contour(BB, EE, W_grid, levels=levels,
                        colors='white', linewidths=0.8, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    if dimensionless:
        cbar.set_label(r'$\beta\, W_{\max}^{\mathrm{obs}} '
                       r'= \max_{\rho} D_{\mathrm{KL}}(\mathbf{p}\|\mathbf{q})$',
                       fontsize=14)
    else:
        cbar.set_label(r'$W_{\max}^{\mathrm{obs}}(\mathcal{E})$', fontsize=14)

    ax.set_xlabel(r'Inverse temperature $\beta$', fontsize=14)
    ax.set_ylabel(r'Sharpness parameter $\eta$', fontsize=14)
    ax.set_xlim(beta_arr.min(), beta_arr.max())
    ax.set_ylim(eta_arr.min(), eta_arr.max())

    ax.tick_params(direction='in', top=True, right=True,
                   color='white', length=5, width=1.0,
                   pad=8)                       # pad: 刻度标签离轴的距离
    ax.xaxis.labelpad = 8                       # x 轴标题离轴距离
    ax.yaxis.labelpad = 8                       # y 轴标题离轴距离

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # ---- 模型与扫描参数 ----
    N = 5                                # 自旋格点数
    J, g, h = 1.0, 0.7, 0.2              # Ising 模型参数
    n_beta, n_eta = 40, 40               # 网格分辨率
    beta_arr = np.linspace(0.05, 0.35, n_beta)  # 逆温度范围
    eta_arr = np.linspace(0.0, 1.0, n_eta)     # 锐度参数范围
    DIMLESS = False                      # True: 画 beta*W; False: 画 W

    # ---- POVM 合法性自检 ----
    G1, G2, G3 = build_observables_G1G2G3(N)
    povm_test, _ = build_tetrahedron_povm(G1, G2, G3, eta=1.0)
    verify_povm(povm_test, name="Tetrahedron (4-outcome)")

    # ---- 二维扫描 (外逼近+顶点枚举, 返回上界) ----
    W_grid = scan_grid(N, J, g, h, beta_arr, eta_arr,
                       dimensionless=DIMLESS)

    # ---- 绘制热图 ----
    plot_heatmap(beta_arr, eta_arr, W_grid,
                 dimensionless=DIMLESS,
                 savefig='W_heatmap_beta_eta.pdf',
                 n_contours=8)

    print("\n热图绘制完成。")
