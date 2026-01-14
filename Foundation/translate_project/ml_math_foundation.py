import math
import numpy as np
import scipy.linalg as la
from scipy.stats import norm
import sympy as sp
import warnings

# 过滤字体警告
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')

from ml_font_config import setup_chinese_font
setup_chinese_font()

# ========================
# 高中数学部分（保持不变）
# ========================

def linear_function(x, a=2, b=1):
    return a * x + b


def quadratic_function(x, a=1, b=-2, c=1):
    return a * x ** 2 + b * x + c


def exponential_function(x):
    return math.exp(x)


def logarithm_function(x):
    if x <= 0:
        raise ValueError("ln(x) 定义域为 x > 0")
    return math.log(x)


def trigonometric_identities():
    x_vals = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]
    print("【三角恒等式验证】sin²x + cos²x = 1")
    for x in x_vals:
        lhs = math.sin(x) ** 2 + math.cos(x) ** 2
        print(f"  x = {x:6.3f} → sin²+cos² = {lhs:.12f} ≈ 1")


def exponent_rules():
    base = 2
    m, n = 3, 4
    left = (base ** m) * (base ** n)
    right = base ** (m + n)
    print(f"【指数规则】{base}^{m} × {base}^{n} = {left} = {base}^({m}+{n}) = {right}")
    print(f"  验证: {base ** m} × {base ** n} = {left}")


def logarithm_rules():
    a, b = 10, 100
    left = math.log10(a * b)
    right = math.log10(a) + math.log10(b)
    print(f"【对数规则】log₁₀({a}×{b}) = {left:.4f} = log₁₀({a}) + log₁₀({b}) = {right:.4f}")


def natural_e_definition(n=1000000):
    approx_e = (1 + 1 / n) ** n
    true_e = math.e
    print(f"【自然常数 e 的定义】(1 + 1/{n})^{n} ≈ {approx_e:.10f}")
    print(f"  math.e = {true_e:.10f}")
    print(f"  误差 = {abs(true_e - approx_e):.2e}")


def change_of_base_formula():
    a, base = 8, 2
    log_by_ln = math.log(a) / math.log(base)
    print(f"【换底公式】log_{base}({a}) = ln({a})/ln({base}) = {log_by_ln:.4f}")
    print(f"  验证: {base}^{log_by_ln:.4f} = {base ** log_by_ln:.4f} ≈ {a}")


# ========================
# 本科数学部分（新增）
# ========================

def calculus_demo():
    """微积分：符号求导、积分 + 数值验证"""
    print("\n【本科数学】微积分")

    # 符号定义
    x = sp.symbols('x')
    f = x ** 3 + sp.sin(x)

    # 求导
    df_dx = sp.diff(f, x)
    print(f"函数 f(x) = {f}")
    print(f"导数 f'(x) = {df_dx}")

    # 不定积分
    integral_f = sp.integrate(f, x)
    print(f"不定积分 ∫f(x)dx = {integral_f}")

    # 定积分数值验证
    f_np = sp.lambdify(x, f, 'numpy')
    x_vals = np.linspace(0, np.pi, 1000)
    y_vals = f_np(x_vals)
    from scipy.integrate import simpson
    numeric_integral = simpson(y_vals, x_vals)
    symbolic_integral = float(sp.integrate(f, (x, 0, sp.pi)))
    print(f"定积分 ∫₀^π f(x)dx ≈ {numeric_integral:.6f} (数值) vs {symbolic_integral:.6f} (符号)")


def linear_algebra_demo():
    """线性代数：矩阵运算、解方程"""
    print("\n【本科数学】线性代gebra")

    # 定义矩阵
    A = np.array([[2, 1],
                  [1, 3]], dtype=float)
    b = np.array([5, 10], dtype=float)

    print(f"矩阵 A =\n{A}")
    print(f"向量 b = {b}")

    # 行列式
    det_A = la.det(A)
    print(f"det(A) = {det_A:.4f}")

    # 特征值
    eigvals, eigvecs = la.eig(A)
    print(f"特征值: {eigvals}")

    # 解线性方程 Ax = b
    x_sol = la.solve(A, b)
    print(f"解 Ax = b → x = {x_sol}")
    print(f"验证 A @ x = {A @ x_sol}")


def probability_demo():
    """概率基础：均值、方差、正态分布"""
    print("\n【本科数学】概率与统计")

    # 生成样本
    np.random.seed(42)
    data = np.random.normal(loc=2.0, scale=1.5, size=1000)

    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)  # 无偏估计
    print(f"样本均值 = {sample_mean:.4f} (理论=2.0)")
    print(f"样本方差 = {sample_var:.4f} (理论=2.25)")

    # 正态分布 PDF 验证
    x0 = 2.0
    pdf_val = norm.pdf(x0, loc=2.0, scale=1.5)
    theoretical_pdf = 1 / (1.5 * np.sqrt(2 * np.pi))  # 因为 x0 = 均值
    print(f"正态分布 N(2, 1.5²) 在 x=2 处的 PDF = {pdf_val:.6f}")
    print(f"理论值 1/(σ√(2π)) = {theoretical_pdf:.6f}")


# ========================
# 进阶本科数学（新增）
# ========================

def multivariable_calculus_demo():
    """多元微积分：梯度、Hessian"""
    print("\n【进阶数学】多元函数与梯度")

    # 符号定义
    x, y = sp.symbols('x y')
    f = x ** 2 + 2 * y ** 2 - x * y

    # 梯度 ∇f = [∂f/∂x, ∂f/∂y]
    grad_f = [sp.diff(f, var) for var in (x, y)]
    print(f"函数 f(x,y) = {f}")
    print(f"梯度 ∇f = [{grad_f[0]}, {grad_f[1]}]")

    # Hessian 矩阵
    hessian = sp.hessian(f, (x, y))
    print(f"Hessian 矩阵 =\n{hessian}")

    # 数值验证：在点 (1, 1) 处
    grad_np = [sp.lambdify((x, y), g, 'numpy') for g in grad_f]
    grad_at_11 = [g(1, 1) for g in grad_np]
    print(f"∇f 在 (1,1) 处 = {grad_at_11}")

    # 可视化梯度场（可选）
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，避免字体警告
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        U = 2 * X - Y  # ∂f/∂x = 2x - y
        V = 4 * Y - X  # ∂f/∂y = 4y - x
        plt.figure(figsize=(6, 5))
        plt.quiver(X, Y, U, V, color='teal')
        plt.title('梯度场 (箭头方向 = 最速上升)')  # 移除 ∇ 符号避免警告
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        # 保存图片而不是显示，避免字体警告
        plt.savefig('gradient_field.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  梯度场图已保存为 gradient_field.png")
    except ImportError:
        print("  (跳过绘图：未安装 matplotlib)")


def svd_demo():
    """奇异值分解 SVD"""
    print("\n【进阶数学】奇异值分解 (SVD)")

    A = np.array([[3, 1],
                  [1, 3],
                  [1, 1]], dtype=float)
    print(f"矩阵 A =\n{A}")

    # SVD: A = U Σ V^T
    U, s, VT = la.svd(A)
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, s)

    print(f"奇异值 σ = {s}")
    print(f"U =\n{U}")
    print(f"V^T =\n{VT}")

    # 重构验证
    A_recon = U @ Sigma @ VT
    print(f"重构误差 ||A - UΣVᵀ||_F = {la.norm(A - A_recon):.2e}")


def bayes_theorem_demo():
    """贝叶斯公式：疾病检测经典案例"""
    print("\n【进阶数学】贝叶斯定理")

    # 假设：
    # P(病) = 1% → prior
    # P(阳性|病) = 99% → sensitivity
    # P(阳性|健康) = 5% → false positive rate

    prior = 0.01
    sensitivity = 0.99
    false_positive = 0.05

    # 贝叶斯公式：
    # P(病|阳性) = P(阳性|病) * P(病) / P(阳性)
    p_positive = sensitivity * prior + false_positive * (1 - prior)
    posterior = sensitivity * prior / p_positive

    print(f"先验概率 P(患病) = {prior:.2%}")
    print(f"检测灵敏度 P(阳性|患病) = {sensitivity:.2%}")
    print(f"假阳性率 P(阳性|健康) = {false_positive:.2%}")
    print(f"后验概率 P(患病|阳性) = {posterior:.2%}")
    print("→ 即使检测阳性，真实患病概率仅约 16.7%！")


def lagrange_multiplier_demo():
    """拉格朗日乘数法：带约束的优化"""
    print("\n【进阶数学】拉格朗日乘数法")

    # 问题：最大化 f(x,y) = x*y，约束 g(x,y) = x + y - 10 = 0
    x, y, λ = sp.symbols('x y λ')
    f = x * y
    g = x + y - 10
    L = f - λ * g  # 拉格朗日函数

    # 求偏导并解方程组
    eqs = [
        sp.diff(L, x),
        sp.diff(L, y),
        sp.diff(L, λ)
    ]
    sol = sp.solve(eqs, (x, y, λ))
    print(f"优化问题：maximize xy, s.t. x + y = 10")
    print(f"拉格朗日函数 L = {L}")
    print(f"解得：x = {sol[x]}, y = {sol[y]} → 最大值 = {f.subs(sol)}")
    print("  (理论：当 x=y=5 时，xy=25 最大)")


# ========================
# 更新主函数（替换原 main）
# ========================

def main():
    print("=" * 70)
    print("           机器学习数学基础完整教程 - Python 验证版")
    print("=" * 70)

    # ===== 高中数学 =====
    print("\n【高中数学】函数基础")
    x_test = 3.0
    print(f"线性函数 y = 2x + 1, x={x_test} → y = {linear_function(x_test)}")
    print(f"二次函数 y = x² - 2x + 1, x={x_test} → y = {quadratic_function(x_test)}")
    print(f"指数函数 y = e^x, x={x_test} → y = {exponential_function(x_test):.6f}")
    print(f"对数函数 y = ln(x), x={x_test} → y = {logarithm_function(x_test):.6f}")

    print("\n【高中数学】三角函数")
    trigonometric_identities()

    print("\n【高中数学】指数与对数")
    exponent_rules()
    logarithm_rules()
    natural_e_definition(n=1_000_000)
    change_of_base_formula()

    # ===== 本科数学 =====
    calculus_demo()
    linear_algebra_demo()
    probability_demo()

    # ===== 进阶本科数学（新增）=====
    multivariable_calculus_demo()
    svd_demo()
    bayes_theorem_demo()
    lagrange_multiplier_demo()

    print("\n" + "=" * 70)
    print("✅ 从高中 → 本科 → 机器学习核心数学，全部验证完成！")
    print("=" * 70)


# 注意：保留原有的 if __name__ == '__main__': 不变
if __name__ == '__main__':
    main()