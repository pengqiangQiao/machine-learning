"""
机器学习数学基础详细教程
Comprehensive Mathematics Tutorial for Machine Learning

从高中数学到本科数学的系统讲解
Systematic explanation from high school to undergraduate mathematics

包含：
1. 高中数学基础（函数、三角、指数对数）
2. 微积分（极限、导数、积分、多元微积分）
3. 线性代数（向量、矩阵、特征值）
4. 概率论（概率、分布、期望）
5. 数理统计（估计、检验、回归）
6. 最优化理论（凸优化、拉格朗日乘数）

Java对应：需要使用Apache Commons Math等数学库
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from mpl_toolkits.mplot3d import Axes3D

# 配置中文字体
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 第一部分：高中数学基础 ====================

class HighSchoolMath:
    """
    高中数学基础
    
    涵盖：函数、三角函数、指数对数、数列
    """
    
    @staticmethod
    def function_basics():
        """
        函数基础
        
        1. 函数的定义：y = f(x)
        2. 函数的性质：单调性、奇偶性、周期性
        3. 常见函数：线性、二次、指数、对数
        """
        print("\n" + "="*60)
        print("【高中数学】函数基础")
        print("="*60)
        
        x = np.linspace(-5, 5, 100)
        
        # 1. 线性函数 y = ax + b
        y_linear = 2*x + 1
        print("\n1. 线性函数: y = 2x + 1")
        print("   特点: 斜率为2，截距为1")
        
        # 2. 二次函数 y = ax² + bx + c
        y_quadratic = x**2 - 2*x + 1
        print("\n2. 二次函数: y = x² - 2x + 1 = (x-1)²")
        print("   特点: 开口向上，顶点在(1, 0)")
        
        # 3. 指数函数 y = a^x
        y_exp = np.exp(x)
        print("\n3. 指数函数: y = e^x")
        print("   特点: 单调递增，恒为正")
        
        # 4. 对数函数 y = log(x)
        x_pos = x[x > 0]
        y_log = np.log(x_pos)
        print("\n4. 对数函数: y = ln(x)")
        print("   特点: 定义域x>0，单调递增")
        
        return x, y_linear, y_quadratic, y_exp, x_pos, y_log
    
    @staticmethod
    def trigonometric_functions():
        """
        三角函数
        
        1. 基本三角函数：sin, cos, tan
        2. 三角恒等式：sin²x + cos²x = 1
        3. 和差公式、倍角公式
        """
        print("\n" + "="*60)
        print("【高中数学】三角函数")
        print("="*60)
        
        x = np.linspace(0, 4*np.pi, 200)
        
        # 基本三角函数
        y_sin = np.sin(x)
        y_cos = np.cos(x)
        y_tan = np.tan(x)
        
        print("\n1. 正弦函数: y = sin(x)")
        print("   周期: 2π, 值域: [-1, 1]")
        
        print("\n2. 余弦函数: y = cos(x)")
        print("   周期: 2π, 值域: [-1, 1]")
        
        print("\n3. 正切函数: y = tan(x)")
        print("   周期: π, 值域: (-∞, +∞)")
        
        # 三角恒等式验证
        identity = y_sin**2 + y_cos**2
        print(f"\n4. 三角恒等式验证: sin²x + cos²x = {identity[0]:.10f} ≈ 1")
        
        return x, y_sin, y_cos, y_tan
    
    @staticmethod
    def exponential_and_logarithm():
        """
        指数与对数
        
        1. 指数运算：a^m · a^n = a^(m+n)
        2. 对数运算：log(ab) = log(a) + log(b)
        3. 换底公式：log_a(b) = ln(b)/ln(a)
        4. 自然对数e的重要性
        """
        print("\n" + "="*60)
        print("【高中数学】指数与对数")
        print("="*60)
        
        # 指数运算规则
        a, m, n = 2, 3, 4
        print(f"\n1. 指数运算规则:")
        print(f"   {a}^{m} × {a}^{n} = {a**(m+n)} = {a}^({m}+{n})")
        print(f"   验证: {a**m} × {a**n} = {a**m * a**n}")
        
        # 对数运算规则
        x, y = 10, 100
        print(f"\n2. 对数运算规则:")
        print(f"   log({x}×{y}) = log({x}) + log({y})")
        print(f"   验证: {np.log10(x*y):.4f} = {np.log10(x):.4f} + {np.log10(y):.4f}")
        
        # 自然对数e
        print(f"\n3. 自然对数的底e:")
        print(f"   e = {np.e:.10f}")
        print(f"   e的定义: lim(n→∞) (1 + 1/n)^n")
        
        # 换底公式
        base_a, base_b, x = 2, 10, 8
        log_direct = np.log(x) / np.log(base_a)
        print(f"\n4. 换底公式:")
        print(f"   log_{base_a}({x}) = ln({x})/ln({base_a}) = {log_direct:.4f}")
        print(f"   验证: {base_a}^{log_direct:.4f} = {base_a**log_direct:.4f} ≈ {x}")


# ==================== 第二部分：微积分 ====================

class Calculus:
    """
    微积分详解
    
    涵盖：极限、导数、积分、多元微积分
    """
    
    @staticmethod
    def limits():
        """
        极限理论
        
        1. 极限的定义：lim(x→a) f(x) = L
        2. 极限的性质：四则运算
        3. 重要极限：lim(x→0) sin(x)/x = 1
        4. 连续性：lim(x→a) f(x) = f(a)
        """
        print("\n" + "="*60)
        print("【微积分】极限理论")
        print("="*60)
        
        # 1. 极限的数值逼近
        print("\n1. 极限示例: lim(x→0) sin(x)/x")
        x_values = [1, 0.1, 0.01, 0.001, 0.0001]
        for x in x_values:
            result = np.sin(x) / x
            print(f"   x = {x:7.4f}, sin(x)/x = {result:.10f}")
        print("   极限值 = 1")
        
        # 2. 重要极限：e的定义
        print("\n2. 重要极限: lim(n→∞) (1 + 1/n)^n = e")
        n_values = [10, 100, 1000, 10000, 100000]
        for n in n_values:
            result = (1 + 1/n)**n
            print(f"   n = {n:6d}, (1 + 1/n)^n = {result:.10f}")
        print(f"   极限值 e = {np.e:.10f}")
        
        # 3. 连续性
        print("\n3. 函数的连续性")
        print("   函数f(x)在点a连续 ⟺ lim(x→a) f(x) = f(a)")
        print("   例：f(x) = x² 在所有点连续")
    
    @staticmethod
    def derivatives():
        """
        导数理论
        
        1. 导数的定义：f'(x) = lim(h→0) [f(x+h) - f(x)]/h
        2. 导数的几何意义：切线斜率
        3. 求导法则：和差积商、链式法则
        4. 常见函数的导数
        """
        print("\n" + "="*60)
        print("【微积分】导数理论")
        print("="*60)
        
        # 1. 导数的定义（数值逼近）
        print("\n1. 导数的定义: f'(x) = lim(h→0) [f(x+h) - f(x)]/h")
        f = lambda x: x**2
        x0 = 2
        print(f"   对于f(x) = x², 在x = {x0}处:")
        h_values = [1, 0.1, 0.01, 0.001, 0.0001]
        for h in h_values:
            derivative_approx = (f(x0 + h) - f(x0)) / h
            print(f"   h = {h:7.4f}, [f({x0}+h) - f({x0})]/h = {derivative_approx:.6f}")
        print(f"   理论值: f'({x0}) = 2×{x0} = {2*x0}")
        
        # 2. 常见函数的导数
        print("\n2. 常见函数的导数公式:")
        print("   (x^n)' = n·x^(n-1)")
        print("   (e^x)' = e^x")
        print("   (ln x)' = 1/x")
        print("   (sin x)' = cos x")
        print("   (cos x)' = -sin x")
        
        # 3. 求导法则
        print("\n3. 求导法则:")
        print("   和差法则: (f ± g)' = f' ± g'")
        print("   乘积法则: (f·g)' = f'·g + f·g'")
        print("   商法则: (f/g)' = (f'·g - f·g')/g²")
        print("   链式法则: (f(g(x)))' = f'(g(x))·g'(x)")
        
        # 4. 链式法则示例
        print("\n4. 链式法则示例: y = sin(x²)")
        print("   设u = x², 则y = sin(u)")
        print("   dy/dx = dy/du · du/dx = cos(u) · 2x = 2x·cos(x²)")
    
    @staticmethod
    def integrals():
        """
        积分理论
        
        1. 不定积分：∫f(x)dx = F(x) + C
        2. 定积分：∫[a,b] f(x)dx
        3. 微积分基本定理：∫[a,b] f(x)dx = F(b) - F(a)
        4. 积分的几何意义：面积
        """
        print("\n" + "="*60)
        print("【微积分】积分理论")
        print("="*60)
        
        # 1. 不定积分
        print("\n1. 不定积分（原函数）:")
        print("   ∫x^n dx = x^(n+1)/(n+1) + C  (n ≠ -1)")
        print("   ∫e^x dx = e^x + C")
        print("   ∫1/x dx = ln|x| + C")
        print("   ∫sin x dx = -cos x + C")
        print("   ∫cos x dx = sin x + C")
        
        # 2. 定积分计算
        print("\n2. 定积分示例: ∫[0,1] x² dx")
        a, b = 0, 1
        # 理论值
        theoretical = (b**3 - a**3) / 3
        print(f"   理论值: [x³/3]₀¹ = 1/3 = {theoretical:.6f}")
        
        # 数值积分（梯形法则）
        n = 1000
        x = np.linspace(a, b, n)
        y = x**2
        # 使用 trapezoid 替代已弃用的 trapz
        try:
            numerical = np.trapezoid(y, x)
        except AttributeError:
            # 兼容旧版本 NumPy
            numerical = np.trapz(y, x)
        print(f"   数值积分: {numerical:.6f}")
        
        # 3. 微积分基本定理
        print("\n3. 微积分基本定理:")
        print("   如果F'(x) = f(x), 则:")
        print("   ∫[a,b] f(x)dx = F(b) - F(a)")
        print("   这连接了导数和积分")
        
        # 4. 积分的几何意义
        print("\n4. 积分的几何意义:")
        print("   ∫[a,b] f(x)dx 表示曲线y=f(x)与x轴")
        print("   在区间[a,b]之间围成的面积")
    
    @staticmethod
    def multivariable_calculus():
        """
        多元微积分
        
        1. 偏导数：∂f/∂x, ∂f/∂y
        2. 梯度：∇f = (∂f/∂x, ∂f/∂y, ...)
        3. 方向导数
        4. 多重积分
        """
        print("\n" + "="*60)
        print("【微积分】多元微积分")
        print("="*60)
        
        # 1. 偏导数
        print("\n1. 偏导数:")
        print("   对于f(x,y) = x²y + xy²")
        print("   ∂f/∂x = 2xy + y²  (把y看作常数)")
        print("   ∂f/∂y = x² + 2xy  (把x看作常数)")
        
        # 数值验证
        f = lambda x, y: x**2 * y + x * y**2
        x0, y0 = 2, 3
        h = 0.0001
        
        # 对x的偏导数
        df_dx = (f(x0 + h, y0) - f(x0, y0)) / h
        theoretical_dx = 2*x0*y0 + y0**2
        print(f"\n   在点({x0}, {y0}):")
        print(f"   数值计算: ∂f/∂x ≈ {df_dx:.4f}")
        print(f"   理论值: 2×{x0}×{y0} + {y0}² = {theoretical_dx}")
        
        # 2. 梯度
        print("\n2. 梯度向量:")
        print("   ∇f = (∂f/∂x, ∂f/∂y)")
        print(f"   在点({x0}, {y0}): ∇f = ({theoretical_dx}, {x0**2 + 2*x0*y0})")
        print("   梯度指向函数增长最快的方向")
        
        # 3. 链式法则（多元）
        print("\n3. 多元链式法则:")
        print("   如果z = f(x,y), x = g(t), y = h(t)")
        print("   则 dz/dt = ∂f/∂x · dx/dt + ∂f/∂y · dy/dt")


# ==================== 第三部分：线性代数 ====================

class LinearAlgebra:
    """
    线性代数详解
    
    涵盖：向量、矩阵、行列式、特征值、线性变换
    """
    
    @staticmethod
    def vectors():
        """
        向量基础
        
        1. 向量的定义和表示
        2. 向量运算：加法、数乘
        3. 点积（内积）
        4. 向量的模（长度）
        5. 向量的夹角
        """
        print("\n" + "="*60)
        print("【线性代数】向量")
        print("="*60)
        
        # 1. 向量定义
        print("\n1. 向量的定义:")
        v1 = np.array([3, 4])
        v2 = np.array([1, 2])
        print(f"   v₁ = {v1}")
        print(f"   v₂ = {v2}")
        
        # 2. 向量加法
        v_sum = v1 + v2
        print(f"\n2. 向量加法:")
        print(f"   v₁ + v₂ = {v_sum}")
        
        # 3. 数乘
        scalar = 2
        v_scaled = scalar * v1
        print(f"\n3. 数乘:")
        print(f"   {scalar}v₁ = {v_scaled}")
        
        # 4. 点积（内积）
        dot_product = np.dot(v1, v2)
        print(f"\n4. 点积（内积）:")
        print(f"   v₁ · v₂ = {v1[0]}×{v2[0]} + {v1[1]}×{v2[1]} = {dot_product}")
        
        # 5. 向量的模
        norm_v1 = np.linalg.norm(v1)
        print(f"\n5. 向量的模（长度）:")
        print(f"   ||v₁|| = √({v1[0]}² + {v1[1]}²) = {norm_v1:.4f}")
        
        # 6. 向量夹角
        cos_theta = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(cos_theta)
        print(f"\n6. 向量夹角:")
        print(f"   cos θ = (v₁·v₂)/(||v₁||·||v₂||) = {cos_theta:.4f}")
        print(f"   θ = {np.degrees(theta):.2f}°")
    
    @staticmethod
    def matrices():
        """
        矩阵基础
        
        1. 矩阵的定义
        2. 矩阵运算：加法、乘法
        3. 转置矩阵
        4. 逆矩阵
        5. 行列式
        """
        print("\n" + "="*60)
        print("【线性代数】矩阵")
        print("="*60)
        
        # 1. 矩阵定义
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        print("\n1. 矩阵定义:")
        print(f"   A = \n{A}")
        print(f"   B = \n{B}")
        
        # 2. 矩阵加法
        C = A + B
        print(f"\n2. 矩阵加法:")
        print(f"   A + B = \n{C}")
        
        # 3. 矩阵乘法
        D = np.dot(A, B)
        print(f"\n3. 矩阵乘法:")
        print(f"   A × B = \n{D}")
        print("   注意: 矩阵乘法不满足交换律 (AB ≠ BA)")
        
        # 4. 转置矩阵
        A_T = A.T
        print(f"\n4. 转置矩阵:")
        print(f"   Aᵀ = \n{A_T}")
        
        # 5. 行列式
        det_A = np.linalg.det(A)
        print(f"\n5. 行列式:")
        print(f"   det(A) = {A[0,0]}×{A[1,1]} - {A[0,1]}×{A[1,0]} = {det_A:.1f}")
        
        # 6. 逆矩阵
        if det_A != 0:
            A_inv = np.linalg.inv(A)
            print(f"\n6. 逆矩阵:")
            print(f"   A⁻¹ = \n{A_inv}")
            print(f"   验证: A × A⁻¹ = \n{np.dot(A, A_inv)}")
    
    @staticmethod
    def eigenvalues_eigenvectors():
        """
        特征值和特征向量
        
        1. 定义：Av = λv
        2. 特征方程：det(A - λI) = 0
        3. 特征值的性质
        4. 对角化
        """
        print("\n" + "="*60)
        print("【线性代数】特征值和特征向量")
        print("="*60)
        
        # 1. 定义
        print("\n1. 特征值和特征向量的定义:")
        print("   如果存在非零向量v和标量λ，使得:")
        print("   Av = λv")
        print("   则λ是矩阵A的特征值，v是对应的特征向量")
        
        # 2. 计算示例
        A = np.array([[4, 1], [2, 3]])
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        print(f"\n2. 示例矩阵:")
        print(f"   A = \n{A}")
        
        print(f"\n3. 特征值:")
        for i, λ in enumerate(eigenvalues):
            print(f"   λ{i+1} = {λ:.4f}")
        
        print(f"\n4. 特征向量:")
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            print(f"   v{i+1} = {v}")
            
            # 验证 Av = λv
            Av = np.dot(A, v)
            λv = eigenvalues[i] * v
            print(f"   验证: Av = {Av}")
            print(f"         λv = {λv}")
            print(f"         误差: {np.linalg.norm(Av - λv):.10f}\n")
        
        # 5. 特征值的性质
        print("5. 特征值的重要性质:")
        print(f"   trace(A) = Σλᵢ = {np.trace(A)} = {np.sum(eigenvalues):.4f}")
        print(f"   det(A) = Πλᵢ = {np.linalg.det(A):.4f} = {np.prod(eigenvalues):.4f}")


# ==================== 第四部分：概率论 ====================

class ProbabilityTheory:
    """
    概率论详解
    
    涵盖：概率基础、随机变量、概率分布、期望方差
    """
    
    @staticmethod
    def probability_basics():
        """
        概率基础
        
        1. 概率的定义
        2. 概率的性质
        3. 条件概率
        4. 贝叶斯定理
        """
        print("\n" + "="*60)
        print("【概率论】概率基础")
        print("="*60)
        
        # 1. 概率的定义
        print("\n1. 概率的定义:")
        print("   P(A) = 事件A发生的次数 / 总次数")
        print("   0 ≤ P(A) ≤ 1")
        
        # 2. 概率的性质
        print("\n2. 概率的基本性质:")
        print("   P(Ω) = 1  (必然事件)")
        print("   P(∅) = 0  (不可能事件)")
        print("   P(A∪B) = P(A) + P(B) - P(A∩B)")
        print("   P(Ā) = 1 - P(A)  (对立事件)")
        
        # 3. 条件概率
        print("\n3. 条件概率:")
        print("   P(A|B) = P(A∩B) / P(B)")
        print("   表示在B发生的条件下A发生的概率")
        
        # 4. 贝叶斯定理
        print("\n4. 贝叶斯定理:")
        print("   P(A|B) = P(B|A)·P(A) / P(B)")
        
        # 实际例子
        print("\n5. 贝叶斯定理应用（医学诊断）:")
        P_disease = 0.01  # 患病率1%
        P_positive_given_disease = 0.95  # 灵敏度95%
        P_positive_given_healthy = 0.05  # 假阳性率5%
        
        # 计算P(阳性)
        P_positive = (P_positive_given_disease * P_disease + 
                     P_positive_given_healthy * (1 - P_disease))
        
        # 计算P(患病|阳性)
        P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive
        
        print(f"   患病率: {P_disease:.2%}")
        print(f"   检测灵敏度: {P_positive_given_disease:.2%}")
        print(f"   假阳性率: {P_positive_given_healthy:.2%}")
        print(f"   检测阳性后真正患病的概率: {P_disease_given_positive:.2%}")
    
    @staticmethod
    def random_variables():
        """
        随机变量
        
        1. 离散随机变量
        2. 连续随机变量
        3. 概率分布函数
        4. 概率密度函数
        """
        print("\n" + "="*60)
        print("【概率论】随机变量")
        print("="*60)
        
        # 1. 离散随机变量
        print("\n1. 离散随机变量示例（掷骰子）:")
        outcomes = np.arange(1, 7)
        probabilities = np.ones(6) / 6
        print(f"   可能取值: {outcomes}")
        print(f"   概率分布: {probabilities}")
        print(f"   概率和: {np.sum(probabilities)}")
        
        # 2. 连续随机变量
        print("\n2. 连续随机变量示例（正态分布）:")
        print("   X ~ N(μ, σ²)")
        print("   概率密度函数: f(x) = (1/√(2πσ²))·exp(-(x-μ)²/(2σ²))")
        
        # 3. 期望
        print("\n3. 期望（均值）:")
        print("   离散: E[X] = Σ xᵢ·P(X=xᵢ)")
        print("   连续: E[X] = ∫ x·f(x)dx")
        
        expected_value = np.sum(outcomes * probabilities)
        print(f"   骰子期望: E[X] = {expected_value}")
        
        # 4. 方差
        print("\n4. 方差:")
        print("   Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²")
        
        variance = np.sum((outcomes - expected_value)**2 * probabilities)
        print(f"   骰子方差: Var(X) = {variance:.4f}")
        print(f"   标准差: σ = {np.sqrt(variance):.4f}")
    
    @staticmethod
    def common_distributions():
        """
        常见概率分布
        
        1. 均匀分布
        2. 二项分布
        3. 泊松分布
        4. 正态分布
        5. 指数分布
        """
        print("\n" + "="*60)
        print("【概率论】常见概率分布")
        print("="*60)
        
        # 1. 均匀分布
        print("\n1. 均匀分布 U(a, b):")
        print("   PDF: f(x) = 1/(b-a), a ≤ x ≤ b")
        print("   E[X] = (a+b)/2")
        print("   Var(X) = (b-a)²/12")
        
        # 2. 二项分布
        print("\n2. 二项分布 B(n, p):")
        print("   n次独立试验，每次成功概率为p")
        print("   P(X=k) = C(n,k)·p^k·(1-p)^(n-k)")
        print("   E[X] = np")
        print("   Var(X) = np(1-p)")
        
        # 3. 泊松分布
        print("\n3. 泊松分布 P(λ):")
        print("   描述单位时间内随机事件发生的次数")
        print("   P(X=k) = (λ^k·e^(-λ))/k!")
        print("   E[X] = Var(X) = λ")
        
        # 4. 正态分布
        print("\n4. 正态分布 N(μ, σ²):")
        print("   最重要的连续分布")
        print("   PDF: f(x) = (1/√(2πσ²))·exp(-(x-μ)²/(2σ²))")
        print("   E[X] = μ")
        print("   Var(X) = σ²")
        print("   68-95-99.7规则:")
        print("   P(μ-σ < X < μ+σ) ≈ 68%")
        print("   P(μ-2σ < X < μ+2σ) ≈ 95%")
        print("   P(μ-3σ < X < μ+3σ) ≈ 99.7%")


# ==================== 第五部分：数理统计 ====================

class Statistics:
    """
    数理统计详解
    
    涵盖：参数估计、假设检验、回归分析
    """
    
    @staticmethod
    def parameter_estimation():
        """
        参数估计
        
        1. 点估计：样本均值、样本方差
        2. 最大似然估计（MLE）
        3. 区间估计：置信区间
        """
        print("\n" + "="*60)
        print("【数理统计】参数估计")
        print("="*60)
        
        # 生成样本数据
        np.random.seed(42)
        true_mu, true_sigma = 5, 2
        sample_size = 100
        data = np.random.normal(true_mu, true_sigma, sample_size)
        
        # 1. 点估计
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)  # 无偏估计
        sample_std = np.std(data, ddof=1)
        
        print("\n1. 点估计:")
        print(f"   真实参数: μ = {true_mu}, σ = {true_sigma}")
        print(f"   样本均值: x̄ = {sample_mean:.4f}")
        print(f"   样本标准差: s = {sample_std:.4f}")
        
        # 2. 置信区间
        confidence_level = 0.95
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin_of_error = z_score * sample_std / np.sqrt(sample_size)
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        print(f"\n2. 95%置信区间:")
        print(f"   [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   真实μ={true_mu}在置信区间内: {ci_lower <= true_mu <= ci_upper}")
    
    @staticmethod
    def hypothesis_testing():
        """
        假设检验
        
        1. 原假设和备择假设
        2. 显著性水平
        3. p值
        4. t检验
        """
        print("\n" + "="*60)
        print("【数理统计】假设检验")
        print("="*60)
        
        print("\n1. 假设检验的步骤:")
        print("   ① 建立假设: H₀(原假设) vs H₁(备择假设)")
        print("   ② 选择显著性水平α (通常0.05)")
        print("   ③ 计算检验统计量")
        print("   ④ 计算p值")
        print("   ⑤ 做出决策: p < α则拒绝H₀")
        
        # 示例：单样本t检验
        np.random.seed(42)
        data = np.random.normal(5.5, 2, 30)
        hypothesized_mean = 5.0
        
        t_statistic, p_value = stats.ttest_1samp(data, hypothesized_mean)
        
        print(f"\n2. 单样本t检验示例:")
        print(f"   H₀: μ = {hypothesized_mean}")
        print(f"   H₁: μ ≠ {hypothesized_mean}")
        print(f"   样本均值: {np.mean(data):.4f}")
        print(f"   t统计量: {t_statistic:.4f}")
        print(f"   p值: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"   结论: p < 0.05, 拒绝H₀")
        else:
            print(f"   结论: p ≥ 0.05, 不能拒绝H₀")


# ==================== 第六部分：最优化理论 ====================

class OptimizationTheory:
    """
    最优化理论
    
    涵盖：凸函数、拉格朗日乘数法、KKT条件
    """
    
    @staticmethod
    def convex_functions():
        """
        凸函数
        
        1. 凸函数的定义
        2. 凸函数的性质
        3. 凸优化问题
        """
        print("\n" + "="*60)
        print("【最优化】凸函数")
        print("="*60)
        
        print("\n1. 凸函数的定义:")
        print("   函数f是凸函数，当且仅当对于任意x, y和λ∈[0,1]:")
        print("   f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)")
        
        print("\n2. 凸函数的性质:")
        print("   ① 局部最小值就是全局最小值")
        print("   ② 凸函数的和仍是凸函数")
        print("   ③ 凸函数的非负加权和是凸函数")
        
        print("\n3. 常见的凸函数:")
        print("   ① 线性函数: f(x) = ax + b")
        print("   ② 二次函数: f(x) = x²")
        print("   ③ 指数函数: f(x) = e^x")
        print("   ④ 负熵: f(x) = x·log(x)")
        
        print("\n4. 凸优化问题:")
        print("   minimize f(x)")
        print("   subject to gᵢ(x) ≤ 0  (凸约束)")
        print("              hⱼ(x) = 0  (仿射约束)")
    
    @staticmethod
    def lagrange_multipliers():
        """
        拉格朗日乘数法
        
        1. 无约束优化
        2. 等式约束优化
        3. 不等式约束优化（KKT条件）
        """
        print("\n" + "="*60)
        print("【最优化】拉格朗日乘数法")
        print("="*60)
        
        print("\n1. 无约束优化:")
        print("   minimize f(x)")
        print("   解法: ∇f(x) = 0")
        
        print("\n2. 等式约束优化:")
        print("   minimize f(x)")
        print("   subject to h(x) = 0")
        print("   ")
        print("   拉格朗日函数: L(x, λ) = f(x) + λh(x)")
        print("   最优性条件:")
        print("   ∇ₓL = ∇f(x) + λ∇h(x) = 0")
        print("   h(x) = 0")
        
        print("\n3. 不等式约束优化（KKT条件）:")
        print("   minimize f(x)")
        print("   subject to g(x) ≤ 0")
        print("   ")
        print("   拉格朗日函数: L(x, μ) = f(x) + μg(x)")
        print("   KKT条件:")
        print("   ① ∇ₓL = ∇f(x) + μ∇g(x) = 0  (稳定性)")
        print("   ② g(x) ≤ 0  (原始可行性)")
        print("   ③ μ ≥ 0  (对偶可行性)")
        print("   ④ μg(x) = 0  (互补松弛性)")


def visualize_all():
    """综合可视化所有数学概念"""
    print("\n" + "="*60)
    print("生成数学概念可视化...")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 函数图像
    ax1 = fig.add_subplot(3, 4, 1)
    x = np.linspace(-5, 5, 100)
    ax1.plot(x, x**2, 'b-', label='y = x²', linewidth=2)
    ax1.plot(x, 2*x + 1, 'r-', label='y = 2x + 1', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('基本函数')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 三角函数
    ax2 = fig.add_subplot(3, 4, 2)
    x = np.linspace(0, 4*np.pi, 200)
    ax2.plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
    ax2.plot(x, np.cos(x), 'r-', label='cos(x)', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('三角函数')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 导数的几何意义
    ax3 = fig.add_subplot(3, 4, 3)
    x = np.linspace(-2, 4, 100)
    y = x**2
    x0 = 1
    y0 = x0**2
    slope = 2*x0
    tangent_y = slope * (x - x0) + y0
    ax3.plot(x, y, 'b-', linewidth=2, label='y = x²')
    ax3.plot(x, tangent_y, 'r--', linewidth=2, label=f'切线(斜率={slope})')
    ax3.plot(x0, y0, 'ro', markersize=10)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('导数的几何意义')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 积分的几何意义
    ax4 = fig.add_subplot(3, 4, 4)
    x = np.linspace(0, 2, 100)
    y = x**2
    ax4.plot(x, y, 'b-', linewidth=2)
    ax4.fill_between(x, 0, y, alpha=0.3)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('积分的几何意义（面积）')
    ax4.grid(True)
    
    # 5. 向量
    ax5 = fig.add_subplot(3, 4, 5)
    v1 = np.array([3, 4])
    v2 = np.array([1, 2])
    v_sum = v1 + v2
    ax5.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.01, label='v₁')
    ax5.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.01, label='v₂')
    ax5.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='g', width=0.01, label='v₁+v₂')
    ax5.set_xlim(-1, 6)
    ax5.set_ylim(-1, 7)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('向量加法')
    ax5.legend()
    ax5.grid(True)
    ax5.set_aspect('equal')
    
    # 6. 正态分布
    ax6 = fig.add_subplot(3, 4, 6)
    x = np.linspace(-4, 4, 200)
    y = stats.norm.pdf(x, 0, 1)
    ax6.plot(x, y, 'b-', linewidth=2)
    ax6.fill_between(x, 0, y, alpha=0.3)
    ax6.axvline(0, color='r', linestyle='--', label='μ=0')
    ax6.set_xlabel('x')
    ax6.set_ylabel('概率密度')
    ax6.set_title('标准正态分布 N(0,1)')
    ax6.legend()
    ax6.grid(True)
    
    # 7. 二项分布
    ax7 = fig.add_subplot(3, 4, 7)
    n, p = 20, 0.5
    x = np.arange(0, n+1)
    y = stats.binom.pmf(x, n, p)
    ax7.bar(x, y, alpha=0.7, color='blue')
    ax7.set_xlabel('k')
    ax7.set_ylabel('概率')
    ax7.set_title(f'二项分布 B({n}, {p})')
    ax7.grid(True, axis='y')
    
    # 8. 梯度下降
    ax8 = fig.add_subplot(3, 4, 8)
    x = np.linspace(-2, 2, 100)
    y = x**2
    # 模拟梯度下降路径
    x_path = [1.5]
    learning_rate = 0.1
    for _ in range(10):
        gradient = 2 * x_path[-1]
        x_new = x_path[-1] - learning_rate * gradient
        x_path.append(x_new)
    y_path = [xi**2 for xi in x_path]
    ax8.plot(x, y, 'b-', linewidth=2)
    ax8.plot(x_path, y_path, 'ro-', markersize=6, linewidth=1.5, label='梯度下降路径')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y = x²')
    ax8.set_title('梯度下降优化')
    ax8.legend()
    ax8.grid(True)
    
    # 9. 凸函数
    ax9 = fig.add_subplot(3, 4, 9)
    x = np.linspace(-2, 2, 100)
    y_convex = x**2
    y_concave = -x**2
    ax9.plot(x, y_convex, 'b-', linewidth=2, label='凸函数 y=x²')
    ax9.plot(x, y_concave, 'r-', linewidth=2, label='凹函数 y=-x²')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    ax9.set_title('凸函数 vs 凹函数')
    ax9.legend()
    ax9.grid(True)
    
    # 10. 特征值可视化
    ax10 = fig.add_subplot(3, 4, 10)
    A = np.array([[2, 1], [1, 2]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # 绘制特征向量
    for i in range(2):
        v = eigenvectors[:, i] * eigenvalues[i]
        ax10.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                   width=0.01, label=f'λ{i+1}={eigenvalues[i]:.2f}')
    
    ax10.set_xlim(-1, 4)
    ax10.set_ylim(-1, 4)
    ax10.set_xlabel('x')
    ax10.set_ylabel('y')
    ax10.set_title('特征向量')
    ax10.legend()
    ax10.grid(True)
    ax10.set_aspect('equal')
    
    # 11. 3D曲面
    ax11 = fig.add_subplot(3, 4, 11, projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax11.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax11.set_xlabel('x')
    ax11.set_ylabel('y')
    ax11.set_zlabel('z')
    ax11.set_title('3D曲面 z = x² + y²')
    
    # 12. 置信区间
    ax12 = fig.add_subplot(3, 4, 12)
    np.random.seed(42)
    data = np.random.normal(5, 2, 100)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ci = 1.96 * std / np.sqrt(len(data))
    
    ax12.hist(data, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
    x = np.linspace(data.min(), data.max(), 100)
    ax12.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='拟合正态分布')
    ax12.axvline(mean, color='g', linestyle='--', linewidth=2, label=f'均值={mean:.2f}')
    ax12.axvline(mean - ci, color='orange', linestyle=':', linewidth=2)
    ax12.axvline(mean + ci, color='orange', linestyle=':', linewidth=2, label='95%置信区间')
    ax12.set_xlabel('值')
    ax12.set_ylabel('密度')
    ax12.set_title('样本分布与置信区间')
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


def main():
    """主函数：运行所有教程"""
    print("\n" + "="*70)
    print(" "*15 + "机器学习数学基础完整教程")
    print(" "*10 + "从高中数学到本科数学的系统讲解")
    print("="*70)
    
    # 高中数学
    HighSchoolMath.function_basics()
    HighSchoolMath.trigonometric_functions()
    HighSchoolMath.exponential_and_logarithm()
    
    # 微积分
    Calculus.limits()
    Calculus.derivatives()
    Calculus.integrals()
    Calculus.multivariable_calculus()
    
    # 线性代数
    LinearAlgebra.vectors()
    LinearAlgebra.matrices()
    LinearAlgebra.eigenvalues_eigenvectors()
    
    # 概率论
    ProbabilityTheory.probability_basics()
    ProbabilityTheory.random_variables()
    ProbabilityTheory.common_distributions()
    
    # 数理统计
    Statistics.parameter_estimation()
    Statistics.hypothesis_testing()
    
    # 最优化理论
    OptimizationTheory.convex_functions()
    OptimizationTheory.lagrange_multipliers()
    
    # 可视化
    visualize_all()
    
    print("\n" + "="*70)
    print("教程完成！您已经系统学习了机器学习所需的所有数学基础。")
    print("="*70)


if __name__ == "__main__":
    main()