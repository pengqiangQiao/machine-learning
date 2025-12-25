"""
机器学习数学基础
Mathematical Foundations for Machine Learning

包含：微积分、线性代数、概率论与数理统计
Including: Calculus, Linear Algebra, Probability and Statistics

Java对应实现：可以使用Apache Commons Math库
Java equivalent: Use Apache Commons Math library
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import eig, svd

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 微积分 Calculus ====================

class Calculus:
    """
    微积分相关函数
    
    Java对应：
    public class Calculus {
        // 数值微分
        public static double numericalDerivative(Function<Double, Double> f, double x, double h);
        // 数值积分
        public static double numericalIntegral(Function<Double, Double> f, double a, double b, int n);
    }
    """
    
    @staticmethod
    def numerical_derivative(f, x, h=1e-5):
        """
        数值微分（导数近似）
        使用中心差分法: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        
        Java对应：
        public static double numericalDerivative(Function<Double, Double> f, double x, double h) {
            return (f.apply(x + h) - f.apply(x - h)) / (2 * h);
        }
        
        Args:
            f: 函数
            x: 求导点
            h: 步长
        Returns:
            导数近似值
        """
        return (f(x + h) - f(x - h)) / (2 * h)
    
    @staticmethod
    def gradient(f, x, h=1e-5):
        """
        梯度计算（多元函数的偏导数向量）
        ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
        
        Java对应：
        public static double[] gradient(Function<double[], Double> f, double[] x, double h) {
            double[] grad = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                double[] xPlusH = x.clone();
                double[] xMinusH = x.clone();
                xPlusH[i] += h;
                xMinusH[i] -= h;
                grad[i] = (f.apply(xPlusH) - f.apply(xMinusH)) / (2 * h);
            }
            return grad;
        }
        
        Args:
            f: 多元函数
            x: 点的坐标（向量）
            h: 步长
        Returns:
            梯度向量
        """
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        
        return grad
    
    @staticmethod
    def numerical_integral(f, a, b, n=1000):
        """
        数值积分（梯形法则）
        ∫ₐᵇ f(x)dx ≈ h/2 * [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]
        
        Java对应：
        public static double numericalIntegral(Function<Double, Double> f, 
                                              double a, double b, int n) {
            double h = (b - a) / n;
            double sum = (f.apply(a) + f.apply(b)) / 2.0;
            
            for (int i = 1; i < n; i++) {
                double x = a + i * h;
                sum += f.apply(x);
            }
            
            return sum * h;
        }
        
        Args:
            f: 被积函数
            a: 下限
            b: 上限
            n: 分割数
        Returns:
            积分近似值
        """
        x = np.linspace(a, b, n)
        y = f(x)
        h = (b - a) / (n - 1)
        
        # 梯形法则
        integral = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)
        return integral
    
    @staticmethod
    def taylor_series(f, x0, x, n=5):
        """
        泰勒级数展开
        f(x) ≈ f(x₀) + f'(x₀)(x-x₀) + f''(x₀)(x-x₀)²/2! + ...
        
        Java对应：
        public static double taylorSeries(Function<Double, Double> f, 
                                         double x0, double x, int n) {
            double result = f.apply(x0);
            double term = 1.0;
            double factorial = 1.0;
            
            for (int i = 1; i <= n; i++) {
                term *= (x - x0);
                factorial *= i;
                double derivative = numericalDerivativeN(f, x0, i);
                result += derivative * term / factorial;
            }
            
            return result;
        }
        
        Args:
            f: 函数
            x0: 展开点
            x: 计算点
            n: 展开阶数
        Returns:
            泰勒级数近似值
        """
        result = f(x0)
        term = 1.0
        
        for i in range(1, n + 1):
            term *= (x - x0) / i
            # 计算第i阶导数（数值方法）
            derivative = Calculus._nth_derivative(f, x0, i)
            result += derivative * term
        
        return result
    
    @staticmethod
    def _nth_derivative(f, x, n, h=1e-3):
        """计算第n阶导数（辅助函数）"""
        if n == 0:
            return f(x)
        elif n == 1:
            return Calculus.numerical_derivative(f, x, h)
        else:
            # 递归计算高阶导数
            def f_prime(x):
                return Calculus.numerical_derivative(f, x, h)
            return Calculus._nth_derivative(f_prime, x, n-1, h)


# ==================== 线性代数 Linear Algebra ====================

class LinearAlgebra:
    """
    线性代数相关函数
    
    Java对应：
    import org.apache.commons.math3.linear.*;
    
    public class LinearAlgebra {
        public static RealMatrix matrixMultiply(RealMatrix A, RealMatrix B);
        public static EigenDecomposition eigenDecomposition(RealMatrix A);
        public static SingularValueDecomposition svd(RealMatrix A);
    }
    """
    
    @staticmethod
    def matrix_multiply(A, B):
        """
        矩阵乘法: C = A × B
        
        Java对应：
        public static double[][] matrixMultiply(double[][] A, double[][] B) {
            int m = A.length;
            int n = B[0].length;
            int p = A[0].length;
            
            double[][] C = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < p; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return C;
        }
        """
        return np.dot(A, B)
    
    @staticmethod
    def matrix_inverse(A):
        """
        矩阵求逆: A⁻¹
        
        Java对应：
        public static RealMatrix inverse(RealMatrix A) {
            LUDecomposition lu = new LUDecomposition(A);
            return lu.getSolver().getInverse();
        }
        """
        return np.linalg.inv(A)
    
    @staticmethod
    def eigenvalue_decomposition(A):
        """
        特征值分解: A = QΛQ⁻¹
        其中Λ是特征值对角矩阵，Q是特征向量矩阵
        
        Java对应：
        public static EigenDecomposition eigenDecomposition(RealMatrix A) {
            return new EigenDecomposition(A);
        }
        
        Returns:
            eigenvalues: 特征值
            eigenvectors: 特征向量
        """
        eigenvalues, eigenvectors = eig(A)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def singular_value_decomposition(A):
        """
        奇异值分解: A = UΣVᵀ
        
        Java对应：
        public static SingularValueDecomposition svd(RealMatrix A) {
            return new SingularValueDecomposition(A);
        }
        
        Returns:
            U: 左奇异向量矩阵
            S: 奇异值
            Vt: 右奇异向量矩阵的转置
        """
        U, S, Vt = svd(A)
        return U, S, Vt
    
    @staticmethod
    def matrix_rank(A, tol=1e-10):
        """
        矩阵的秩
        
        Java对应：
        public static int matrixRank(RealMatrix A, double tol) {
            SingularValueDecomposition svd = new SingularValueDecomposition(A);
            double[] singularValues = svd.getSingularValues();
            
            int rank = 0;
            for (double s : singularValues) {
                if (s > tol) rank++;
            }
            return rank;
        }
        """
        return np.linalg.matrix_rank(A, tol=tol)
    
    @staticmethod
    def qr_decomposition(A):
        """
        QR分解: A = QR
        Q是正交矩阵，R是上三角矩阵
        
        Java对应：
        public static QRDecomposition qrDecomposition(RealMatrix A) {
            return new QRDecomposition(A);
        }
        """
        Q, R = np.linalg.qr(A)
        return Q, R
    
    @staticmethod
    def gram_schmidt(vectors):
        """
        Gram-Schmidt正交化
        
        Java对应：
        public static double[][] gramSchmidt(double[][] vectors) {
            int n = vectors.length;
            int m = vectors[0].length;
            double[][] orthogonal = new double[n][m];
            
            for (int i = 0; i < n; i++) {
                orthogonal[i] = vectors[i].clone();
                
                for (int j = 0; j < i; j++) {
                    double proj = dotProduct(vectors[i], orthogonal[j]) / 
                                 dotProduct(orthogonal[j], orthogonal[j]);
                    for (int k = 0; k < m; k++) {
                        orthogonal[i][k] -= proj * orthogonal[j][k];
                    }
                }
                
                // 归一化
                double norm = vectorNorm(orthogonal[i]);
                for (int k = 0; k < m; k++) {
                    orthogonal[i][k] /= norm;
                }
            }
            
            return orthogonal;
        }
        """
        vectors = np.array(vectors, dtype=float)
        n = len(vectors)
        orthogonal = np.zeros_like(vectors)
        
        for i in range(n):
            orthogonal[i] = vectors[i]
            for j in range(i):
                # 投影并减去
                proj = np.dot(vectors[i], orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
                orthogonal[i] -= proj * orthogonal[j]
            # 归一化
            orthogonal[i] /= np.linalg.norm(orthogonal[i])
        
        return orthogonal


# ==================== 概率论与统计 Probability and Statistics ====================

class ProbabilityStatistics:
    """
    概率论与数理统计
    
    Java对应：
    import org.apache.commons.math3.distribution.*;
    import org.apache.commons.math3.stat.descriptive.*;
    
    public class ProbabilityStatistics {
        public static double mean(double[] data);
        public static double variance(double[] data);
        public static double normalPDF(double x, double mu, double sigma);
    }
    """
    
    @staticmethod
    def mean(data):
        """
        均值: μ = (1/n)Σxᵢ
        
        Java对应：
        public static double mean(double[] data) {
            return Arrays.stream(data).average().getAsDouble();
        }
        """
        return np.mean(data)
    
    @staticmethod
    def variance(data, ddof=0):
        """
        方差: σ² = (1/n)Σ(xᵢ - μ)²
        
        Java对应：
        public static double variance(double[] data) {
            double mean = mean(data);
            double sum = 0.0;
            for (double x : data) {
                sum += Math.pow(x - mean, 2);
            }
            return sum / data.length;
        }
        
        Args:
            data: 数据
            ddof: 自由度修正（0为总体方差，1为样本方差）
        """
        return np.var(data, ddof=ddof)
    
    @staticmethod
    def standard_deviation(data, ddof=0):
        """
        标准差: σ = √σ²
        
        Java对应：
        public static double standardDeviation(double[] data) {
            return Math.sqrt(variance(data));
        }
        """
        return np.std(data, ddof=ddof)
    
    @staticmethod
    def covariance(x, y):
        """
        协方差: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
        
        Java对应：
        public static double covariance(double[] x, double[] y) {
            double meanX = mean(x);
            double meanY = mean(y);
            double sum = 0.0;
            
            for (int i = 0; i < x.length; i++) {
                sum += (x[i] - meanX) * (y[i] - meanY);
            }
            
            return sum / x.length;
        }
        """
        return np.cov(x, y)[0, 1]
    
    @staticmethod
    def correlation(x, y):
        """
        相关系数: ρ = Cov(X,Y) / (σₓσᵧ)
        
        Java对应：
        public static double correlation(double[] x, double[] y) {
            return covariance(x, y) / (standardDeviation(x) * standardDeviation(y));
        }
        """
        return np.corrcoef(x, y)[0, 1]
    
    @staticmethod
    def normal_pdf(x, mu=0, sigma=1):
        """
        正态分布概率密度函数
        f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        
        Java对应：
        public static double normalPDF(double x, double mu, double sigma) {
            NormalDistribution normal = new NormalDistribution(mu, sigma);
            return normal.density(x);
        }
        """
        return stats.norm.pdf(x, mu, sigma)
    
    @staticmethod
    def normal_cdf(x, mu=0, sigma=1):
        """
        正态分布累积分布函数
        
        Java对应：
        public static double normalCDF(double x, double mu, double sigma) {
            NormalDistribution normal = new NormalDistribution(mu, sigma);
            return normal.cumulativeProbability(x);
        }
        """
        return stats.norm.cdf(x, mu, sigma)
    
    @staticmethod
    def maximum_likelihood_estimation_normal(data):
        """
        正态分布的最大似然估计
        
        Java对应：
        public static double[] mleNormal(double[] data) {
            double mu = mean(data);
            double sigma = standardDeviation(data);
            return new double[]{mu, sigma};
        }
        
        Returns:
            mu: 均值估计
            sigma: 标准差估计
        """
        mu = np.mean(data)
        sigma = np.std(data, ddof=0)
        return mu, sigma
    
    @staticmethod
    def bayes_theorem(prior, likelihood, evidence):
        """
        贝叶斯定理: P(A|B) = P(B|A)P(A) / P(B)
        
        Java对应：
        public static double bayesTheorem(double prior, double likelihood, double evidence) {
            return (likelihood * prior) / evidence;
        }
        
        Args:
            prior: 先验概率 P(A)
            likelihood: 似然 P(B|A)
            evidence: 证据 P(B)
        Returns:
            posterior: 后验概率 P(A|B)
        """
        return (likelihood * prior) / evidence


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("机器学习数学基础示例")
    print("=" * 60)
    
    # ========== 微积分示例 ==========
    print("\n【1. 微积分】")
    print("-" * 60)
    
    # 求导示例
    f = lambda x: x**2 + 2*x + 1
    x = 2.0
    derivative = Calculus.numerical_derivative(f, x)
    print(f"f(x) = x² + 2x + 1")
    print(f"f'({x}) ≈ {derivative:.6f} (理论值: {2*x + 2})")
    
    # 梯度示例
    g = lambda x: x[0]**2 + x[1]**2
    point = np.array([1.0, 2.0])
    grad = Calculus.gradient(g, point)
    print(f"\ng(x,y) = x² + y²")
    print(f"∇g({point}) = {grad} (理论值: [2, 4])")
    
    # 积分示例
    h = lambda x: x**2
    integral = Calculus.numerical_integral(h, 0, 1)
    print(f"\n∫₀¹ x² dx ≈ {integral:.6f} (理论值: 1/3 = 0.333333)")
    
    # ========== 线性代数示例 ==========
    print("\n【2. 线性代数】")
    print("-" * 60)
    
    # 矩阵乘法
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = LinearAlgebra.matrix_multiply(A, B)
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"A × B = \n{C}")
    
    # 特征值分解
    eigenvalues, eigenvectors = LinearAlgebra.eigenvalue_decomposition(A)
    print(f"\nA的特征值: {eigenvalues.real}")
    print(f"A的特征向量:\n{eigenvectors.real}")
    
    # SVD分解
    U, S, Vt = LinearAlgebra.singular_value_decomposition(A)
    print(f"\nSVD分解:")
    print(f"奇异值: {S}")
    
    # ========== 概率统计示例 ==========
    print("\n【3. 概率论与统计】")
    print("-" * 60)
    
    # 生成正态分布数据
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000)
    
    mean = ProbabilityStatistics.mean(data)
    var = ProbabilityStatistics.variance(data, ddof=1)
    std = ProbabilityStatistics.standard_deviation(data, ddof=1)
    
    print(f"样本均值: {mean:.4f}")
    print(f"样本方差: {var:.4f}")
    print(f"样本标准差: {std:.4f}")
    
    # 最大似然估计
    mu_mle, sigma_mle = ProbabilityStatistics.maximum_likelihood_estimation_normal(data)
    print(f"\n正态分布MLE估计:")
    print(f"μ = {mu_mle:.4f}")
    print(f"σ = {sigma_mle:.4f}")
    
    # 贝叶斯定理示例
    prior = 0.01  # 患病率1%
    sensitivity = 0.95  # 灵敏度95%
    specificity = 0.90  # 特异度90%
    
    # P(阳性) = P(阳性|患病)P(患病) + P(阳性|健康)P(健康)
    evidence = sensitivity * prior + (1 - specificity) * (1 - prior)
    posterior = ProbabilityStatistics.bayes_theorem(prior, sensitivity, evidence)
    
    print(f"\n贝叶斯定理应用（医学诊断）:")
    print(f"先验概率（患病率）: {prior:.2%}")
    print(f"检测阳性后的后验概率: {posterior:.2%}")
    
    # 可视化
    print("\n绘制正态分布...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    x = np.linspace(-5, 5, 100)
    y = ProbabilityStatistics.normal_pdf(x, 0, 1)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.fill_between(x, y, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('标准正态分布 N(0,1)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
    x_range = np.linspace(data.min(), data.max(), 100)
    plt.plot(x_range, ProbabilityStatistics.normal_pdf(x_range, mu_mle, sigma_mle),
             'r-', linewidth=2, label='拟合的正态分布')
    plt.xlabel('值')
    plt.ylabel('频率密度')
    plt.title('样本数据分布')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # 绘制函数及其导数
    x = np.linspace(-3, 3, 100)
    f_vals = x**2
    f_prime_vals = 2*x
    plt.plot(x, f_vals, 'b-', linewidth=2, label='f(x) = x²')
    plt.plot(x, f_prime_vals, 'r--', linewidth=2, label="f'(x) = 2x")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('函数及其导数')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    example_usage()