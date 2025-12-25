"""
机器学习高级数学补充
Advanced Mathematics Supplement for Machine Learning

补充内容：
1. 数列与级数（高中/本科）
2. 复数与复变函数（本科）
3. 常微分方程（本科）
4. 偏微分方程（本科/研究生）
5. 泛函分析基础（研究生）
6. 测度论基础（研究生）
7. 信息论（本科/研究生）
8. 图论基础（本科）
9. 数值分析（本科）
10. 组合数学（本科）

Java对应：需要使用Apache Commons Math等数学库
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special, integrate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx
import math

# 配置中文字体
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 第一部分：数列与级数 ====================

class SequencesAndSeries:
    """
    数列与级数
    
    涵盖：等差数列、等比数列、级数收敛、泰勒级数、傅里叶级数
    """
    
    @staticmethod
    def arithmetic_sequence():
        """
        等差数列
        
        通项公式: aₙ = a₁ + (n-1)d
        求和公式: Sₙ = n(a₁ + aₙ)/2 = na₁ + n(n-1)d/2
        """
        print("\n" + "="*60)
        print("【数列与级数】等差数列")
        print("="*60)
        
        a1, d, n = 1, 2, 10
        
        print(f"\n等差数列: a₁ = {a1}, 公差 d = {d}")
        print(f"前{n}项:")
        
        # 生成数列
        sequence = [a1 + (i-1)*d for i in range(1, n+1)]
        print(f"   {sequence}")
        
        # 通项公式
        an = a1 + (n-1)*d
        print(f"\n第{n}项: a_{n} = {a1} + ({n}-1)×{d} = {an}")
        
        # 求和公式
        Sn = n * (a1 + an) / 2
        print(f"前{n}项和: S_{n} = {n}×({a1}+{an})/2 = {Sn}")
        print(f"验证: {sum(sequence)}")
    
    @staticmethod
    def geometric_sequence():
        """
        等比数列
        
        通项公式: aₙ = a₁ · qⁿ⁻¹
        求和公式: Sₙ = a₁(1-qⁿ)/(1-q)  (q≠1)
        无穷级数: S∞ = a₁/(1-q)  (|q|<1)
        """
        print("\n" + "="*60)
        print("【数列与级数】等比数列")
        print("="*60)
        
        a1, q, n = 1, 2, 10
        
        print(f"\n等比数列: a₁ = {a1}, 公比 q = {q}")
        print(f"前{n}项:")
        
        # 生成数列
        sequence = [a1 * q**(i-1) for i in range(1, n+1)]
        print(f"   {sequence}")
        
        # 通项公式
        an = a1 * q**(n-1)
        print(f"\n第{n}项: a_{n} = {a1}×{q}^({n}-1) = {an}")
        
        # 求和公式
        if q != 1:
            Sn = a1 * (1 - q**n) / (1 - q)
            print(f"前{n}项和: S_{n} = {a1}×(1-{q}^{n})/(1-{q}) = {Sn}")
            print(f"验证: {sum(sequence)}")
        
        # 无穷级数（收敛条件）
        print(f"\n无穷级数收敛条件: |q| < 1")
        q_conv = 0.5
        S_inf = a1 / (1 - q_conv)
        print(f"当q = {q_conv}时, S∞ = {a1}/(1-{q_conv}) = {S_inf}")
    
    @staticmethod
    def power_series():
        """
        幂级数
        
        常见幂级数：
        1. e^x = Σ(x^n/n!)
        2. sin(x) = Σ((-1)^n·x^(2n+1)/(2n+1)!)
        3. cos(x) = Σ((-1)^n·x^(2n)/(2n)!)
        4. ln(1+x) = Σ((-1)^(n+1)·x^n/n)  (|x|<1)
        """
        print("\n" + "="*60)
        print("【数列与级数】幂级数")
        print("="*60)
        
        x = 0.5
        n_terms = 10
        
        # 1. e^x的泰勒级数
        print(f"\n1. e^x的泰勒级数 (x={x}):")
        exp_approx = sum([x**n / math.factorial(n) for n in range(n_terms)])
        exp_exact = np.exp(x)
        print(f"   级数近似({n_terms}项): {exp_approx:.10f}")
        print(f"   精确值: {exp_exact:.10f}")
        print(f"   误差: {abs(exp_approx - exp_exact):.2e}")
        
        # 2. sin(x)的泰勒级数
        print(f"\n2. sin(x)的泰勒级数 (x={x}):")
        sin_approx = sum([(-1)**n * x**(2*n+1) / math.factorial(2*n+1)
                         for n in range(n_terms)])
        sin_exact = np.sin(x)
        print(f"   级数近似({n_terms}项): {sin_approx:.10f}")
        print(f"   精确值: {sin_exact:.10f}")
        print(f"   误差: {abs(sin_approx - sin_exact):.2e}")
        
        # 3. cos(x)的泰勒级数
        print(f"\n3. cos(x)的泰勒级数 (x={x}):")
        cos_approx = sum([(-1)**n * x**(2*n) / math.factorial(2*n)
                         for n in range(n_terms)])
        cos_exact = np.cos(x)
        print(f"   级数近似({n_terms}项): {cos_approx:.10f}")
        print(f"   精确值: {cos_exact:.10f}")
        print(f"   误差: {abs(cos_approx - cos_exact):.2e}")


# ==================== 第二部分：信息论 ====================

class InformationTheory:
    """
    信息论基础
    
    涵盖：熵、互信息、KL散度、交叉熵
    """
    
    @staticmethod
    def entropy():
        """
        熵（Entropy）
        
        H(X) = -Σ p(x)·log₂(p(x))
        
        熵衡量随机变量的不确定性
        """
        print("\n" + "="*60)
        print("【信息论】熵")
        print("="*60)
        
        # 示例1：均匀分布（最大熵）
        print("\n1. 均匀分布（最大熵）:")
        p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        H_uniform = -np.sum(p_uniform * np.log2(p_uniform))
        print(f"   概率分布: {p_uniform}")
        print(f"   熵: H = {H_uniform:.4f} bits")
        print(f"   最大熵: log₂(4) = {np.log2(4):.4f} bits")
        
        # 示例2：不均匀分布
        print("\n2. 不均匀分布:")
        p_skewed = np.array([0.7, 0.2, 0.05, 0.05])
        H_skewed = -np.sum(p_skewed * np.log2(p_skewed + 1e-10))
        print(f"   概率分布: {p_skewed}")
        print(f"   熵: H = {H_skewed:.4f} bits")
        
        # 示例3：确定性分布（最小熵）
        print("\n3. 确定性分布（最小熵）:")
        p_certain = np.array([1.0, 0.0, 0.0, 0.0])
        H_certain = -np.sum(p_certain * np.log2(p_certain + 1e-10))
        print(f"   概率分布: {p_certain}")
        print(f"   熵: H = {H_certain:.4f} bits")
    
    @staticmethod
    def kl_divergence():
        """
        KL散度（Kullback-Leibler Divergence）
        
        D_KL(P||Q) = Σ p(x)·log(p(x)/q(x))
        
        衡量两个概率分布的差异（非对称）
        """
        print("\n" + "="*60)
        print("【信息论】KL散度")
        print("="*60)
        
        # 真实分布P
        P = np.array([0.4, 0.3, 0.2, 0.1])
        
        # 近似分布Q1（接近P）
        Q1 = np.array([0.35, 0.35, 0.2, 0.1])
        
        # 近似分布Q2（远离P）
        Q2 = np.array([0.1, 0.2, 0.3, 0.4])
        
        # 计算KL散度
        DKL_PQ1 = np.sum(P * np.log(P / Q1))
        DKL_PQ2 = np.sum(P * np.log(P / Q2))
        
        print(f"\n真实分布P: {P}")
        print(f"近似分布Q1: {Q1}")
        print(f"D_KL(P||Q1) = {DKL_PQ1:.4f}")
        
        print(f"\n近似分布Q2: {Q2}")
        print(f"D_KL(P||Q2) = {DKL_PQ2:.4f}")
        
        print(f"\nKL散度越小，两个分布越接近")
        print(f"注意：KL散度不对称，D_KL(P||Q) ≠ D_KL(Q||P)")
    
    @staticmethod
    def cross_entropy():
        """
        交叉熵（Cross Entropy）
        
        H(P,Q) = -Σ p(x)·log(q(x))
        
        关系：H(P,Q) = H(P) + D_KL(P||Q)
        """
        print("\n" + "="*60)
        print("【信息论】交叉熵")
        print("="*60)
        
        # 真实分布
        P = np.array([0.4, 0.3, 0.2, 0.1])
        
        # 预测分布
        Q = np.array([0.35, 0.35, 0.2, 0.1])
        
        # 计算熵和交叉熵
        H_P = -np.sum(P * np.log(P))
        H_PQ = -np.sum(P * np.log(Q))
        DKL_PQ = np.sum(P * np.log(P / Q))
        
        print(f"\n真实分布P: {P}")
        print(f"预测分布Q: {Q}")
        print(f"\n熵 H(P) = {H_P:.4f}")
        print(f"交叉熵 H(P,Q) = {H_PQ:.4f}")
        print(f"KL散度 D_KL(P||Q) = {DKL_PQ:.4f}")
        print(f"\n验证关系: H(P,Q) = H(P) + D_KL(P||Q)")
        print(f"   {H_PQ:.4f} = {H_P:.4f} + {DKL_PQ:.4f}")
    
    @staticmethod
    def mutual_information():
        """
        互信息（Mutual Information）
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        衡量两个随机变量之间的相互依赖程度
        """
        print("\n" + "="*60)
        print("【信息论】互信息")
        print("="*60)
        
        # 联合概率分布
        joint_prob = np.array([
            [0.1, 0.2, 0.1],
            [0.15, 0.25, 0.2]
        ])
        
        # 边缘概率
        p_x = joint_prob.sum(axis=1)
        p_y = joint_prob.sum(axis=0)
        
        # 计算熵
        H_X = -np.sum(p_x * np.log2(p_x + 1e-10))
        H_Y = -np.sum(p_y * np.log2(p_y + 1e-10))
        H_XY = -np.sum(joint_prob * np.log2(joint_prob + 1e-10))
        
        # 互信息
        I_XY = H_X + H_Y - H_XY
        
        print(f"\n联合概率分布 P(X,Y):")
        print(joint_prob)
        print(f"\nH(X) = {H_X:.4f} bits")
        print(f"H(Y) = {H_Y:.4f} bits")
        print(f"H(X,Y) = {H_XY:.4f} bits")
        print(f"\n互信息 I(X;Y) = {I_XY:.4f} bits")
        print(f"\n互信息衡量X和Y的相互依赖程度")
        print(f"I(X;Y) = 0 表示X和Y独立")


# ==================== 第三部分：图论基础 ====================

class GraphTheory:
    """
    图论基础
    
    涵盖：图的表示、最短路径、最小生成树、图的遍历
    """
    
    @staticmethod
    def graph_representation():
        """
        图的表示方法
        
        1. 邻接矩阵
        2. 邻接表
        3. 边列表
        """
        print("\n" + "="*60)
        print("【图论】图的表示")
        print("="*60)
        
        # 创建一个简单的图
        # 节点: 0, 1, 2, 3
        # 边: (0,1), (0,2), (1,2), (1,3), (2,3)
        
        print("\n图的节点: {0, 1, 2, 3}")
        print("图的边: {(0,1), (0,2), (1,2), (1,3), (2,3)}")
        
        # 1. 邻接矩阵
        adj_matrix = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        print("\n1. 邻接矩阵表示:")
        print(adj_matrix)
        
        # 2. 邻接表
        adj_list = {
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [1, 2]
        }
        
        print("\n2. 邻接表表示:")
        for node, neighbors in adj_list.items():
            print(f"   节点{node}: {neighbors}")
        
        # 3. 边列表
        edge_list = [(0,1), (0,2), (1,2), (1,3), (2,3)]
        
        print("\n3. 边列表表示:")
        print(f"   {edge_list}")
    
    @staticmethod
    def shortest_path_algorithms():
        """
        最短路径算法
        
        1. Dijkstra算法（单源最短路径）
        2. Floyd-Warshall算法（全源最短路径）
        """
        print("\n" + "="*60)
        print("【图论】最短路径算法")
        print("="*60)
        
        # 带权重的图（邻接矩阵）
        # 使用inf表示不连通
        graph = np.array([
            [0, 4, 0, 0, 0, 0, 0, 8, 0],
            [4, 0, 8, 0, 0, 0, 0, 11, 0],
            [0, 8, 0, 7, 0, 4, 0, 0, 2],
            [0, 0, 7, 0, 9, 14, 0, 0, 0],
            [0, 0, 0, 9, 0, 10, 0, 0, 0],
            [0, 0, 4, 14, 10, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 1, 6],
            [8, 11, 0, 0, 0, 0, 1, 0, 7],
            [0, 0, 2, 0, 0, 0, 6, 7, 0]
        ], dtype=float)
        
        # 将0替换为inf（除了对角线）
        graph[graph == 0] = np.inf
        np.fill_diagonal(graph, 0)
        
        print("\n图的邻接矩阵（权重）:")
        print("节点数: 9")
        
        # 使用Dijkstra算法
        dist_matrix = shortest_path(csr_matrix(graph), method='D', directed=False)
        
        source = 0
        target = 4
        
        print(f"\n从节点{source}到节点{target}的最短路径:")
        print(f"最短距离: {dist_matrix[source, target]:.0f}")
        
        print(f"\n从节点{source}到所有节点的最短距离:")
        for i in range(len(dist_matrix[source])):
            if dist_matrix[source, i] != np.inf:
                print(f"   到节点{i}: {dist_matrix[source, i]:.0f}")
    
    @staticmethod
    def graph_traversal():
        """
        图的遍历
        
        1. 深度优先搜索（DFS）
        2. 广度优先搜索（BFS）
        """
        print("\n" + "="*60)
        print("【图论】图的遍历")
        print("="*60)
        
        # 创建图
        G = nx.Graph()
        G.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,3), (3,4)])
        
        print("\n图的边: {(0,1), (0,2), (1,2), (1,3), (2,3), (3,4)}")
        
        # DFS
        start_node = 0
        # 使用 dfs_tree 替代已弃用的 dfs_preorder
        dfs_tree = nx.dfs_tree(G, source=start_node)
        dfs_order = list(dfs_tree.nodes())
        print(f"\n深度优先搜索（DFS）从节点{start_node}开始:")
        print(f"   遍历顺序: {dfs_order}")
        
        # BFS
        bfs_tree = nx.bfs_tree(G, source=start_node)
        bfs_order = list(bfs_tree.nodes())
        print(f"\n广度优先搜索（BFS）从节点{start_node}开始:")
        print(f"   遍历顺序: {bfs_order}")


# ==================== 第四部分：数值分析 ====================

class NumericalAnalysis:
    """
    数值分析
    
    涵盖：数值积分、数值微分、插值、方程求根
    """
    
    @staticmethod
    def numerical_integration():
        """
        数值积分方法
        
        1. 梯形法则
        2. 辛普森法则
        3. 高斯求积
        """
        print("\n" + "="*60)
        print("【数值分析】数值积分")
        print("="*60)
        
        # 被积函数
        f = lambda x: np.sin(x)
        a, b = 0, np.pi
        
        # 精确值
        exact = 2.0  # ∫₀^π sin(x)dx = 2
        
        print(f"\n计算 ∫₀^π sin(x)dx")
        print(f"精确值: {exact}")
        
        # 1. 梯形法则
        n = 100
        x = np.linspace(a, b, n)
        y = f(x)
        h = (b - a) / (n - 1)
        trapz_result = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)
        
        print(f"\n1. 梯形法则 (n={n}):")
        print(f"   结果: {trapz_result:.10f}")
        print(f"   误差: {abs(trapz_result - exact):.2e}")
        
        # 2. 辛普森法则
        simpson_result, simpson_error = integrate.quad(f, a, b)
        
        print(f"\n2. 自适应积分:")
        print(f"   结果: {simpson_result:.10f}")
        print(f"   估计误差: {simpson_error:.2e}")
    
    @staticmethod
    def interpolation():
        """
        插值方法
        
        1. 线性插值
        2. 多项式插值（拉格朗日）
        3. 样条插值
        """
        print("\n" + "="*60)
        print("【数值分析】插值")
        print("="*60)
        
        # 数据点
        x_data = np.array([0, 1, 2, 3, 4])
        y_data = np.array([0, 1, 4, 9, 16])  # y = x²
        
        print("\n已知数据点:")
        for i in range(len(x_data)):
            print(f"   ({x_data[i]}, {y_data[i]})")
        
        # 插值点
        x_interp = 2.5
        
        # 1. 线性插值
        y_linear = np.interp(x_interp, x_data, y_data)
        print(f"\n在x={x_interp}处:")
        print(f"线性插值: y ≈ {y_linear:.4f}")
        
        # 2. 多项式插值
        poly_coeffs = np.polyfit(x_data, y_data, deg=2)
        y_poly = np.polyval(poly_coeffs, x_interp)
        print(f"二次多项式插值: y ≈ {y_poly:.4f}")
        
        # 真实值
        y_true = x_interp**2
        print(f"真实值: y = {y_true:.4f}")
    
    @staticmethod
    def root_finding():
        """
        方程求根
        
        1. 二分法
        2. 牛顿法
        3. 割线法
        """
        print("\n" + "="*60)
        print("【数值分析】方程求根")
        print("="*60)
        
        # 方程: f(x) = x² - 2 = 0
        # 根: x = √2 ≈ 1.414213562
        f = lambda x: x**2 - 2
        f_prime = lambda x: 2*x
        
        print("\n求解方程: x² - 2 = 0")
        print(f"精确根: √2 = {np.sqrt(2):.10f}")
        
        # 1. 二分法
        def bisection(f, a, b, tol=1e-10):
            while (b - a) / 2 > tol:
                c = (a + b) / 2
                if f(c) == 0:
                    return c
                elif f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
            return (a + b) / 2
        
        root_bisection = bisection(f, 1, 2)
        print(f"\n1. 二分法:")
        print(f"   根: {root_bisection:.10f}")
        print(f"   误差: {abs(root_bisection - np.sqrt(2)):.2e}")
        
        # 2. 牛顿法
        def newton(f, f_prime, x0, tol=1e-10, max_iter=100):
            x = x0
            for _ in range(max_iter):
                x_new = x - f(x) / f_prime(x)
                if abs(x_new - x) < tol:
                    return x_new
                x = x_new
            return x
        
        root_newton = newton(f, f_prime, 1.5)
        print(f"\n2. 牛顿法:")
        print(f"   根: {root_newton:.10f}")
        print(f"   误差: {abs(root_newton - np.sqrt(2)):.2e}")


# ==================== 第五部分：常微分方程 ====================

class OrdinaryDifferentialEquations:
    """
    常微分方程（ODE）
    
    涵盖：一阶ODE、二阶ODE、数值解法
    """
    
    @staticmethod
    def first_order_ode():
        """
        一阶常微分方程
        
        示例: dy/dx = -2xy
        解析解: y = Ce^(-x²)
        """
        print("\n" + "="*60)
        print("【常微分方程】一阶ODE")
        print("="*60)
        
        print("\n方程: dy/dx = -2xy")
        print("初始条件: y(0) = 1")
        print("解析解: y = e^(-x²)")
        
        # 欧拉法数值求解
        def euler_method(f, y0, x_range, n):
            x = np.linspace(x_range[0], x_range[1], n)
            y = np.zeros(n)
            y[0] = y0
            h = (x_range[1] - x_range[0]) / (n - 1)
            
            for i in range(n - 1):
                y[i + 1] = y[i] + h * f(x[i], y[i])
            
            return x, y
        
        # 定义ODE
        f = lambda x, y: -2 * x * y
        
        # 数值解
        x_num, y_num = euler_method(f, 1.0, [0, 2], 100)
        
        # 解析解
        y_exact = np.exp(-x_num**2)
        
        print(f"\n在x=1处:")
        idx = np.argmin(np.abs(x_num - 1.0))
        print(f"数值解: y ≈ {y_num[idx]:.6f}")
        print(f"解析解: y = {y_exact[idx]:.6f}")
        print(f"误差: {abs(y_num[idx] - y_exact[idx]):.2e}")
    
    @staticmethod
    def second_order_ode():
        """
        二阶常微分方程
        
        示例: y'' + y = 0 (简谐振动)
        解析解: y = A·cos(x) + B·sin(x)
        """
        print("\n" + "="*60)
        print("【常微分方程】二阶ODE")
        print("="*60)
        
        print("\n方程: y'' + y = 0")
        print("初始条件: y(0) = 1, y'(0) = 0")
        print("解析解: y = cos(x)")
        
        # 转换为一阶方程组
        # y1 = y, y2 = y'
        # y1' = y2
        # y2' = -y1
        
        def system(t, Y):
            y1, y2 = Y
            return [y2, -y1]
        
        # 使用scipy求解
        from scipy.integrate import solve_ivp
        
        t_span = [0, 2*np.pi]
        y0 = [1, 0]
        
        sol = solve_ivp(system, t_span, y0, dense_output=True)
        
        t_eval = np.linspace(0, 2*np.pi, 100)
        y_num = sol.sol(t_eval)[0]
        y_exact = np.cos(t_eval)
        
        print(f"\n在t=π/2处:")
        idx = np.argmin(np.abs(t_eval - np.pi/2))
        print(f"数值解: y ≈ {y_num[idx]:.6f}")
        print(f"解析解: y = {y_exact[idx]:.6f}")
        print(f"误差: {abs(y_num[idx] - y_exact[idx]):.2e}")


# ==================== 第六部分：组合数学 ====================

class Combinatorics:
    """
    组合数学
    
    涵盖：排列、组合、二项式定理、生成函数
    """
    
    @staticmethod
    def permutations_and_combinations():
        """
        排列与组合
        
        排列: P(n,r) = n!/(n-r)!
        组合: C(n,r) = n!/(r!(n-r)!)
        """
        print("\n" + "="*60)
        print("【组合数学】排列与组合")
        print("="*60)
        
        n, r = 5, 3
        
        # 排列
        P_nr = math.factorial(n) // math.factorial(n - r)
        print(f"\n排列 P({n},{r}):")
        print(f"   从{n}个元素中选{r}个排列")
        print(f"   P({n},{r}) = {n}!/({n}-{r})! = {P_nr}")
        
        # 组合
        C_nr = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
        print(f"\n组合 C({n},{r}):")
        print(f"   从{n}个元素中选{r}个组合")
        print(f"   C({n},{r}) = {n}!/({r}!×({n}-{r})!) = {C_nr}")
        
        # 关系
        print(f"\n关系: P({n},{r}) = C({n},{r}) × {r}!")
        print(f"   {P_nr} = {C_nr} × {math.factorial(r)}")
    
    @staticmethod
    def binomial_theorem():
        """
        二项式定理
        
        (a+b)ⁿ = Σ C(n,k)·aⁿ⁻ᵏ·bᵏ
        """
        print("\n" + "="*60)
        print("【组合数学】二项式定理")
        print("="*60)
        
        a, b, n = 2, 3, 4
        
        print(f"\n计算 ({a}+{b})^{n}")
        
        # 直接计算
        direct = (a + b)**n
        print(f"直接计算: {direct}")
        
        # 使用二项式定理
        result = 0
        print(f"\n使用二项式定理:")
        for k in range(n + 1):
            C_nk = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            term = C_nk * (a**(n-k)) * (b**k)
            result += term
            print(f"   k={k}: C({n},{k})×{a}^{n-k}×{b}^{k} = {C_nk}×{a**(n-k)}×{b**k} = {term}")
        
        print(f"\n总和: {result}")
        print(f"验证: {result == direct}")
    
    @staticmethod
    def pascals_triangle():
        """
        杨辉三角（帕斯卡三角）
        
        每个数是它上方两个数的和
        第n行第k个数是C(n,k)
        """
        print("\n" + "="*60)
        print("【组合数学】杨辉三角")
        print("="*60)
        
        n_rows = 7
        
        print(f"\n杨辉三角前{n_rows}行:")
        
        for n in range(n_rows):
            row = []
            for k in range(n + 1):
                C_nk = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
                row.append(C_nk)
            
            # 居中打印
            spaces = " " * (n_rows - n) * 2
            print(f"{spaces}{' '.join(map(str, row))}")


def visualize_advanced_concepts():
    """可视化高级数学概念"""
    print("\n" + "="*60)
    print("生成高级数学概念可视化...")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 幂级数收敛
    ax1 = fig.add_subplot(3, 3, 1)
    x = np.linspace(-1, 1, 100)
    for n in [5, 10, 20, 50]:
        y = sum([x**k / math.factorial(k) for k in range(n)])
        ax1.plot(x, y, label=f'n={n}项')
    ax1.plot(x, np.exp(x), 'k--', linewidth=2, label='e^x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('e^x的泰勒级数收敛')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 熵与概率分布
    ax2 = fig.add_subplot(3, 3, 2)
    n_states = 4
    entropies = []
    p_values = np.linspace(0.01, 0.99, 50)
    for p in p_values:
        # 二元分布
        probs = np.array([p, 1-p])
        H = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(H)
    ax2.plot(p_values, entropies, 'b-', linewidth=2)
    ax2.axhline(y=1, color='r', linestyle='--', label='最大熵')
    ax2.set_xlabel('概率 p')
    ax2.set_ylabel('熵 H (bits)')
    ax2.set_title('二元分布的熵')
    ax2.legend()
    ax2.grid(True)
    
    # 3. KL散度可视化
    ax3 = fig.add_subplot(3, 3, 3)
    x = np.linspace(0, 1, 100)
    P = stats.beta.pdf(x, 2, 5)
    Q1 = stats.beta.pdf(x, 2.5, 5.5)
    Q2 = stats.beta.pdf(x, 5, 2)
    ax3.plot(x, P, 'b-', linewidth=2, label='P (真实分布)')
    ax3.plot(x, Q1, 'g--', linewidth=2, label='Q1 (接近P)')
    ax3.plot(x, Q2, 'r:', linewidth=2, label='Q2 (远离P)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('概率密度')
    ax3.set_title('KL散度：分布差异')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 图的可视化
    ax4 = fig.add_subplot(3, 3, 4)
    G = nx.Graph()
    G.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (3,5)])
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax4, with_labels=True, node_color='lightblue',
            node_size=500, font_size=12, font_weight='bold')
    ax4.set_title('无向图示例')
    
    # 5. 最短路径
    ax5 = fig.add_subplot(3, 3, 5)
    G_weighted = nx.Graph()
    edges = [(0,1,4), (0,2,2), (1,2,1), (1,3,5), (2,3,8), (2,4,10), (3,4,2)]
    G_weighted.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G_weighted, seed=42)
    nx.draw(G_weighted, pos, ax=ax5, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=12, font_weight='bold')
    labels = nx.get_edge_attributes(G_weighted, 'weight')
    nx.draw_networkx_edge_labels(G_weighted, pos, labels, ax=ax5)
    ax5.set_title('带权图与最短路径')
    
    # 6. 数值积分
    ax6 = fig.add_subplot(3, 3, 6)
    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)
    ax6.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax6.fill_between(x, 0, y, alpha=0.3)
    # 梯形近似
    x_trap = np.linspace(0, np.pi, 10)
    y_trap = np.sin(x_trap)
    for i in range(len(x_trap)-1):
        ax6.plot([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]], 'r-', linewidth=1)
        ax6.plot([x_trap[i], x_trap[i]], [0, y_trap[i]], 'r-', linewidth=1)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('数值积分（梯形法则）')
    ax6.legend()
    ax6.grid(True)
    
    # 7. 插值
    ax7 = fig.add_subplot(3, 3, 7)
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 4, 9, 16])
    x_fine = np.linspace(0, 4, 100)
    # 线性插值
    y_linear = np.interp(x_fine, x_data, y_data)
    # 多项式插值
    poly = np.polyfit(x_data, y_data, 2)
    y_poly = np.polyval(poly, x_fine)
    ax7.plot(x_data, y_data, 'ro', markersize=10, label='数据点')
    ax7.plot(x_fine, y_linear, 'b--', linewidth=2, label='线性插值')
    ax7.plot(x_fine, y_poly, 'g-', linewidth=2, label='多项式插值')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_title('插值方法比较')
    ax7.legend()
    ax7.grid(True)
    
    # 8. ODE数值解
    ax8 = fig.add_subplot(3, 3, 8)
    from scipy.integrate import solve_ivp
    def ode_system(t, y):
        return -2 * t * y
    t_span = [0, 2]
    y0 = [1]
    sol = solve_ivp(ode_system, t_span, y0, dense_output=True)
    t_plot = np.linspace(0, 2, 100)
    y_num = sol.sol(t_plot)[0]
    y_exact = np.exp(-t_plot**2)
    ax8.plot(t_plot, y_num, 'b-', linewidth=2, label='数值解')
    ax8.plot(t_plot, y_exact, 'r--', linewidth=2, label='解析解')
    ax8.set_xlabel('t')
    ax8.set_ylabel('y')
    ax8.set_title('ODE: dy/dt = -2ty')
    ax8.legend()
    ax8.grid(True)
    
    # 9. 杨辉三角
    ax9 = fig.add_subplot(3, 3, 9)
    n_rows = 10
    triangle = []
    for n in range(n_rows):
        row = []
        for k in range(n + 1):
            C_nk = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            row.append(C_nk)
        triangle.append(row)
    
    # 绘制热图
    max_val = max(max(row) for row in triangle)
    matrix = np.zeros((n_rows, n_rows))
    for i, row in enumerate(triangle):
        for j, val in enumerate(row):
            matrix[i, j] = val
    
    im = ax9.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax9.set_xlabel('k')
    ax9.set_ylabel('n')
    ax9.set_title('杨辉三角（热图）')
    plt.colorbar(im, ax=ax9)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


def main():
    """主函数：运行所有高级数学教程"""
    print("\n" + "="*70)
    print(" "*10 + "机器学习高级数学补充教程")
    print(" "*5 + "Advanced Mathematics Supplement for Machine Learning")
    print("="*70)
    
    # 数列与级数
    SequencesAndSeries.arithmetic_sequence()
    SequencesAndSeries.geometric_sequence()
    SequencesAndSeries.power_series()
    
    # 信息论
    InformationTheory.entropy()
    InformationTheory.kl_divergence()
    InformationTheory.cross_entropy()
    InformationTheory.mutual_information()
    
    # 图论
    GraphTheory.graph_representation()
    GraphTheory.shortest_path_algorithms()
    GraphTheory.graph_traversal()
    
    # 数值分析
    NumericalAnalysis.numerical_integration()
    NumericalAnalysis.interpolation()
    NumericalAnalysis.root_finding()
    
    # 常微分方程
    OrdinaryDifferentialEquations.first_order_ode()
    OrdinaryDifferentialEquations.second_order_ode()
    
    # 组合数学
    Combinatorics.permutations_and_combinations()
    Combinatorics.binomial_theorem()
    Combinatorics.pascals_triangle()
    
    # 可视化
    visualize_advanced_concepts()
    
    print("\n" + "="*70)
    print("高级数学教程完成！")
    print("="*70)


if __name__ == "__main__":
    main()