"""
Entropic Uncertainty Relations (EUR) Bounds Visualization
=======================================================
此脚本用于计算并对比两个测量不同熵不确定性关系的下界（Bounds）。
主要包含以下界限的计算：
1. Maassen-Uffink (MU) Bound
2. Rudnicki-Puchała-Życzkowski (RPZ) Bound 
3. Coles-Piani (CP) Bound
4. RPZ Majorization Bound 
5. Optimal Bound 

作者: [yang]
日期: 2026-1-25
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# ==========================================
# 1. 模块导入与配置 (Imports & Config)
# ==========================================

# 尝试导入自定义优化模块
# 该模块用于计算基于 SDP/凸优化的紧致界限
try:
    import entropy_min_outerapproximate
    HAS_OPTIMIZER = True
    print("Success: 'entropy_min_outerapproximate' module loaded.")
except ImportError:
    HAS_OPTIMIZER = False
    print("Warning: 'entropy_min_outerapproximate' module not found.")
    print("         The 'q_Optimal' curve will be skipped/zeroed.")

# ==========================================
# 2. 基础数学工具 (Math Helpers)
# ==========================================

def bases_to_flat_projectors(bases_list):
    """
    将一组幺正矩阵（基底）转换为投影算符列表。
    
    Args:
        bases_list (list): 包含 L 个 d*d 幺正矩阵的列表。
                           每个矩阵的列向量代表该基底的一个态。
    
    Returns:
        list: 包含所有投影算符 P = |psi><psi| 的扁平列表。
              列表长度为 L * d。
    """
    all_projectors = []
    for U in bases_list:
        d = U.shape[1]
        for i in range(d):
            # 提取第 i 列作为基向量 |psi>
            psi = U[:, i]
            # 计算投影算符 P = |psi><psi|
            # np.outer(a, b) = a * b^T, 这里需要手动共轭第二个参数
            P = np.outer(psi, np.conj(psi))
            all_projectors.append(P)
    return all_projectors

def get_overlaps_sorted(basis_U, basis_V):
    """
    计算两组基底之间的重叠矩阵，并返回降序排列的模方值。
    
    Args:
        basis_U (ndarray): 第一组基底矩阵 (d x d)
        basis_V (ndarray): 第二组基底矩阵 (d x d)
        
    Returns:
        ndarray: 一维数组，包含所有 |<u_i | v_j>|^2 的值，按从大到小排序。
                 c = overlaps[0], c2 = overlaps[1], ...
    """
    # 计算内积矩阵: Inner[i, j] = <u_i | v_j>
    # 注意: basis_U.conj().T 是 U 的厄米共轭
    inner_products = basis_U.conj().T @ basis_V
    
    # 取模方得到重叠概率 c_ij
    overlaps = np.abs(inner_products)**2
    
    # 扁平化并降序排列
    return np.sort(overlaps.flatten())[::-1]

# ==========================================
# 3. 物理界限计算 (Bounds Calculation)
# ==========================================

def calc_bounds(basis_U, basis_V):
    """
    计算基于最大、次大重叠系数 c、 c2 的解析界限：MU, RPZ, CP。
    
    Args:
        basis_U, basis_V: 两组基底矩阵。
        
    Returns:
        tuple: (q_mu, q_rpz, q_cp)
    """
    # 获取排序后的重叠系数
    sorted_ov = get_overlaps_sorted(basis_U, basis_V)
    c = sorted_ov[0]   # 最大重叠 (Maassen-Uffink 常数)
    c2 = sorted_ov[1]  # 次大重叠
    
    # --- 1. Maassen-Uffink Bound (MU) ---
    # q_MU = -log2(c)
    q_mu = -np.log(c)
    
    # --- 2. RPZ Bound (Improved MU) ---
    # 参考文献: Rudnicki et al.,  PRA 89, 052115 (2014).
    b = (1.0 + np.sqrt(c)) / 2.0
    term_rpz = b**2 + (c2 / c) * (1.0 - b**2)
    
    # 防止数值误差导致 log(<=0)
    if term_rpz <= 1e-15: 
        term_rpz = 1e-15
        
    q_rpz = -np.log(c) - np.log(term_rpz)
    
    # --- 3. Coles-Piani Bound (CP) ---
    # 参考文献: Coles & Piani, PRA 89, 022112 (2014)
    # 利用次大重叠 c2 进行修正
    if c2 > 0:
        term_cp = 0.5 * (1.0 - np.sqrt(c)) * np.log(c / c2)
    else:
        term_cp = 0 # 避免除以零
        
    q_cp = -np.log(c) + term_cp
    
    return q_mu, q_rpz, q_cp

def RPZ_maj_bound(U_list, base=np.e):
    """
    基于 mjorization 技术的 RPZ 界限 (RPZ Majorization)。
    通过计算矩阵列向量组合的奇异值来构造 majorization 向量。
    
    Args:
        U_list (list): 基底矩阵列表 [U1, U2, ...]
        base (int): 对数的底数，默认为 2 (bits)。
        
    Returns:
        float: Shannon entropy bound calculated from the majorization vector.
    """
    # 1. 横向拼接所有基底矩阵 -> M (d x N), 其中 N = d * L
    M = np.hstack(U_list) 
    d, N = M.shape        
    
    # 2. 计算 S_k (k个列向量组合的最大奇异值平方和)
    # S[k] 代表选取 k 列所能构成的算符范数的最大可能值
    S = np.zeros(N + 1)
    S[0] = 1.0 # 定义初始值

    cols = range(N) # 列索引 [0, 1, ..., N-1]
    
    # 遍历所有可能的列数 k (从 1 到 N-1)
    # 注意：此步计算量随 d 和 L 组合数增长极快
    for k in range(1, N):
        m = k + 1 
        max_sigma2 = 0.0
        
        # 遍历所有可能的列组合
        for comb in combinations(cols, m):
            sub = M[:, comb]
            # 计算最大奇异值 (Spectra norm)
            sigma1 = np.linalg.svd(sub, compute_uv=False)[0]
            sigma2 = sigma1**2
            if sigma2 > max_sigma2:
                max_sigma2 = sigma2
        S[k] = max_sigma2

    # k=N 时，取全部列
    S[N] = np.linalg.svd(M, compute_uv=False)[0]**2

    # 3. 构造  majorization 向量 s
    # s 向量的元素和为 1
    s = np.zeros(N + 1)
    s[0] = 1.0 
    s[1:] = S[1:] - S[:-1]

    # 4. 计算该向量的香农熵
    # 过滤掉非正值以避免 log 报错
    mask = s > 1e-15
    valid_s = s[mask]
    
    if base == 2:
        H = -np.sum(valid_s * np.log2(valid_s))
    else:
        H = -np.sum(valid_s * np.log(valid_s) / np.log(base))

    return H

# ==========================================
# 4. 基底生成函数 (Basis Generators)
# ==========================================

def get_basis_3(a, phi):
    """
    生成一个参数化的 3x3 幺正矩阵 (用于测试)。
    Args:
        a (float): 幅度参数 [0, 1]
        phi (float): 相位参数
    Returns:
        ndarray: 3x3 幺正矩阵
    """
    a = np.clip(a, 0, 1)
    sqrt_a = np.sqrt(a)
    sqrt_1_a = np.sqrt(1 - a)
    
    # 列向量归一且正交
    return np.array([
        [sqrt_a,      np.exp(1j * phi) * sqrt_1_a, 0],
        [sqrt_1_a,   -np.exp(1j * phi) * sqrt_a,   0],
        [0,           0,                           1]
    ])

def get_M_theta(theta):
    """
    返回 M(theta) 旋转矩阵。
    物理意义：绕 X 轴在 2-3 子空间（索引1-2）进行的旋转。
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]
    ])

def get_O3():
    """
    返回 O3 矩阵。
    物理意义：特定的 3 维基底（类似于傅里叶基或 MUB 的变体）。
    """
    return (1/np.sqrt(6)) * np.array([
        [np.sqrt(2), np.sqrt(2),  np.sqrt(2)],
        [np.sqrt(3), 0,          -np.sqrt(3)],
        [1,          -2,          1]
    ])

# ==========================================
# 5. 绘图样式配置 (Plot Style)
# ==========================================

def set_pub_style():
    """
    配置 Matplotlib 以生成出版级质量的图表。
    尝试使用 LaTeX 字体，如果不可用则回退到标准衬线字体。
    """
    try:
        from matplotlib import rc
        # 检查是否安装了 LaTeX (这可能需要系统安装 texlive/miktex)
        # 如果报错，将 use_latex 设为 False
        use_latex = True 
        if use_latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
    except Exception as e:
        print(f"LaTeX rendering setup failed ({e}), using standard fonts.")
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
        })

    # 通用参数设置
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": ":",
        "legend.frameon": True,
    })

# ==========================================
# 6. 主程序入口 (Main Execution)
# ==========================================

if __name__ == "__main__":
    # --- A. 参数设定 ---
    x_value = np.linspace(0, np.pi/2, 500) # 扫描角度 theta
    
    # 结果容器
    results_mu = []
    results_rpz = []
    results_rpz_maj = []
    results_cp = []
    results_ours = []
    
    # --- B. 定义固定基底 ---
    # 基底 1: 单位阵 (计算基)
    O3 = get_O3()
    basis_1 = np.eye(3, dtype=complex)
    # basis_1 = O3 # O3.T
    
    print(f"Starting calculation for {len(x_value)} points...")
    print(f"Optimizer enabled: {HAS_OPTIMIZER}")
    
    # --- C. 循环计算 ---
    for i, theta in enumerate(x_value):
        if i % 50 == 0: print(f"Processing step {i}/{len(x_value)}...")

        # 1. 动态生成第二组基底
        # 使用旋转矩阵 M(theta) 对 O3 进行变换
        M = get_M_theta(theta)
        basis_2 = M @ O3 @ M.conj().T
        # basis_2 = M
        
        # 2. 计算解析界限 (MU, RPZ, CP)
        q_mu, q_rpz, q_cp = calc_bounds(basis_1, basis_2)
        
        # 3. 计算 majorization 界限 (RPZ Majorization)
        # 注意：这里传入的是基底列表
        q_rpz_maj = RPZ_maj_bound([basis_1, basis_2])
        
        # 存入结果
        results_mu.append(q_mu)
        results_rpz.append(q_rpz)
        results_rpz_maj.append(q_rpz_maj)
        results_cp.append(q_cp)
        
        # 4. 计算优化界限 (Custom/Optimal)
        if HAS_OPTIMIZER:
            # 参数: bases_to_flat_projectors 将两组基展平为投影算符集合
            c_minus, c_plus, _, _ = entropy_min_outerapproximate.tsallis_entropy_min(
                bases_to_flat_projectors([basis_1, basis_2]), 
                alpha=1, # Shannon entropy
                max_iter=60,      
                tol=1e-5,
                plot=False,       
                verbose=False     
            )
            # 根据物理定义将结果转换为图中的量纲
            # Bound = 2 * min_entropy - 2
            q_ours_val = 2 * c_minus - 2 * np.log(2)
            results_ours.append(q_ours_val)
        else:
            # 如果没有优化器，填 0 占位
            results_ours.append(0)
        
    # 转换为 numpy 数组以便绘图
    results_mu = np.array(results_mu)
    results_rpz = np.array(results_rpz)
    results_rpz_maj = np.array(results_rpz_maj)
    results_cp = np.array(results_cp)
    results_ours = np.array(results_ours)
    
    print("Calculation done. Generating plot...")

    # --- D. 绘图 (Visualization) ---
    set_pub_style()
    
    plt.figure(figsize=(7, 5))

    # 定义颜色方案 (Colorblind-friendly palette recommended)
    color_mu = '#1f77b4'       # Blue
    color_cp = '#2ca02c'       # Green
    color_rpz = '#ff7f0e'      # Orange
    color_rpz_maj = '#009E73'  # Bluish Green
    color_opt = '#d62728'      # Red
    
    # 绘制曲线
    plt.plot(x_value, results_mu, label=r'$q_{\mathrm{MU}}$', 
             color=color_mu, linestyle='-.', linewidth=2, alpha=0.9)
    
    plt.plot(x_value, results_cp, label=r'$q_{\mathrm{CP}}$', 
             color=color_cp, linestyle=':', linewidth=2, alpha=0.9)
    
    plt.plot(x_value, results_rpz, label=r'$q_{\mathrm{RPZ}}$', 
             color=color_rpz, linestyle='--', linewidth=2, alpha=0.9)
    
    plt.plot(x_value, results_rpz_maj, label=r'$q_{\mathrm{RPZ2}}$', 
             color=color_rpz_maj, linestyle='--', linewidth=2, alpha=0.9)
    
    # 仅当有优化器数据且不全为0时绘制 Optimal 线
    if HAS_OPTIMIZER and np.any(results_ours):
        plt.plot(x_value, results_ours, label=r'$q_{\mathrm{Optimal}}$', 
                 color=color_opt, linestyle='-', linewidth=2, zorder=10)
    
    # 图表标签与装饰
    # plt.title(r'Comparison of Entropic Uncertainty Bounds ($d=3$)', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('Bound', fontsize=14)
    plt.legend(fontsize=12, loc='lower right', handlelength=2.5)
    
    plt.xlim(0, np.pi/2)
    
    # 设置 X 轴刻度为 pi 的分数形式
    plt.xticks(
        [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2],
        ['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$']
    )
    
    print("Displaying plot window...")
    plt.tight_layout()
    plt.show()
