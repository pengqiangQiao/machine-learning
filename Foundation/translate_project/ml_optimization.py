"""
优化算法实现
Optimization Algorithms Implementation

包含：梯度下降、拟牛顿法、凸优化
Including: Gradient Descent, Quasi-Newton Methods, Convex Optimization

Java对应实现：可以使用Apache Commons Math的optimization包
Java equivalent: Use Apache Commons Math optimization package
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, line_search

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 梯度下降 Gradient Descent ====================

class GradientDescent:
    """
    梯度下降优化器
    
    Java对应：
    public class GradientDescent {
        private double learningRate;
        private int maxIterations;
        private double tolerance;
        
        public double[] optimize(Function<double[], Double> f,
                                Function<double[], double[]> gradient,
                                double[] x0);
    }
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        初始化梯度下降优化器
        
        Java对应：
        public GradientDescent(double learningRate, int maxIterations, double tolerance) {
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
        }
        
        Args:
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
    
    def optimize(self, f, gradient, x0):
        """
        使用梯度下降优化函数
        
        Java对应：
        public double[] optimize(Function<double[], Double> f,
                                Function<double[], double[]> gradient,
                                double[] x0) {
            double[] x = x0.clone();
            
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] grad = gradient.apply(x);
                double[] xNew = new double[x.length];
                
                // x_new = x - learning_rate * gradient
                for (int i = 0; i < x.length; i++) {
                    xNew[i] = x[i] - learningRate * grad[i];
                }
                
                // 检查收敛
                if (vectorNorm(subtract(xNew, x)) < tolerance) {
                    return xNew;
                }
                
                x = xNew;
            }
            
            return x;
        }
        
        Args:
            f: 目标函数
            gradient: 梯度函数
            x0: 初始点
        Returns:
            最优解
        """
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]
        
        for iteration in range(self.max_iterations):
            # 计算梯度
            grad = gradient(x)
            
            # 更新: x = x - α∇f(x)
            x_new = x - self.learning_rate * grad
            
            # 记录历史
            self.history.append(x_new.copy())
            
            # 检查收敛
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"在第 {iteration} 次迭代时收敛")
                return x_new
            
            x = x_new
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, f(x) = {f(x):.6f}")
        
        print("达到最大迭代次数")
        return x


class MomentumGradientDescent:
    """
    动量梯度下降
    
    Java对应：
    public class MomentumGradientDescent extends GradientDescent {
        private double momentum;
        private double[] velocity;
    }
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9, max_iterations=1000, tolerance=1e-6):
        """
        初始化动量梯度下降
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
    
    def optimize(self, f, gradient, x0):
        """
        使用动量梯度下降优化
        v = β*v + ∇f(x)
        x = x - α*v
        
        Java对应：
        public double[] optimize(Function<double[], Double> f,
                                Function<double[], double[]> gradient,
                                double[] x0) {
            double[] x = x0.clone();
            double[] velocity = new double[x.length];
            
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] grad = gradient.apply(x);
                
                // 更新速度: v = momentum * v + grad
                for (int i = 0; i < x.length; i++) {
                    velocity[i] = momentum * velocity[i] + grad[i];
                    x[i] -= learningRate * velocity[i];
                }
                
                if (checkConvergence(x, xOld)) break;
            }
            
            return x;
        }
        """
        x = np.array(x0, dtype=float)
        velocity = np.zeros_like(x)
        self.history = [x.copy()]
        
        for iteration in range(self.max_iterations):
            grad = gradient(x)
            
            # 更新速度和位置
            velocity = self.momentum * velocity + grad
            x_new = x - self.learning_rate * velocity
            
            self.history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"在第 {iteration} 次迭代时收敛")
                return x_new
            
            x = x_new
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, f(x) = {f(x):.6f}")
        
        return x


class AdamOptimizer:
    """
    Adam优化器（自适应矩估计）
    
    Java对应：
    public class AdamOptimizer {
        private double learningRate;
        private double beta1;
        private double beta2;
        private double epsilon;
    }
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 max_iterations=1000, tolerance=1e-6):
        """
        初始化Adam优化器
        
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 数值稳定性常数
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
    
    def optimize(self, f, gradient, x0):
        """
        使用Adam优化
        
        m = β₁*m + (1-β₁)*∇f(x)
        v = β₂*v + (1-β₂)*(∇f(x))²
        m̂ = m/(1-β₁ᵗ)
        v̂ = v/(1-β₂ᵗ)
        x = x - α*m̂/(√v̂ + ε)
        """
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)  # 一阶矩估计
        v = np.zeros_like(x)  # 二阶矩估计
        self.history = [x.copy()]
        
        for t in range(1, self.max_iterations + 1):
            grad = gradient(x)
            
            # 更新偏差修正的一阶和二阶矩估计
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # 更新参数
            x_new = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            self.history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"在第 {t} 次迭代时收敛")
                return x_new
            
            x = x_new
            
            if t % 100 == 0:
                print(f"迭代 {t}, f(x) = {f(x):.6f}")
        
        return x


# ==================== 拟牛顿法 Quasi-Newton Methods ====================

class BFGS:
    """
    BFGS拟牛顿法（Broyden-Fletcher-Goldfarb-Shanno）
    
    Java对应：
    import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;
    
    public class BFGS {
        public double[] optimize(MultivariateFunction f,
                                MultivariateVectorFunction gradient,
                                double[] x0);
    }
    """
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        """
        初始化BFGS优化器
        
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
    
    def optimize(self, f, gradient, x0):
        """
        使用BFGS方法优化
        
        BFGS更新公式：
        H_{k+1} = H_k + (1 + y_k^T H_k y_k / s_k^T y_k) * (s_k s_k^T / s_k^T y_k)
                  - (s_k y_k^T H_k + H_k y_k s_k^T) / s_k^T y_k
        
        其中：
        s_k = x_{k+1} - x_k
        y_k = ∇f(x_{k+1}) - ∇f(x_k)
        
        Java对应：
        public double[] optimize(MultivariateFunction f,
                                MultivariateVectorFunction gradient,
                                double[] x0) {
            int n = x0.length;
            double[] x = x0.clone();
            double[][] H = identityMatrix(n);  // 初始Hessian逆矩阵近似
            
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] grad = gradient.value(x);
                
                // 搜索方向: p = -H * grad
                double[] p = matrixVectorMultiply(H, negate(grad));
                
                // 线搜索找步长
                double alpha = lineSearch(f, gradient, x, p);
                
                // 更新x
                double[] xNew = add(x, multiply(alpha, p));
                double[] gradNew = gradient.value(xNew);
                
                // 计算s和y
                double[] s = subtract(xNew, x);
                double[] y = subtract(gradNew, grad);
                
                // 更新H（BFGS公式）
                H = updateHessian(H, s, y);
                
                if (vectorNorm(subtract(xNew, x)) < tolerance) break;
                
                x = xNew;
            }
            
            return x;
        }
        
        Args:
            f: 目标函数
            gradient: 梯度函数
            x0: 初始点
        Returns:
            最优解
        """
        n = len(x0)
        x = np.array(x0, dtype=float)
        H = np.eye(n)  # 初始Hessian逆矩阵近似为单位矩阵
        self.history = [x.copy()]
        
        grad = gradient(x)
        
        for iteration in range(self.max_iterations):
            # 搜索方向: p = -H·∇f(x)
            p = -np.dot(H, grad)
            
            # 线搜索确定步长
            alpha = self._line_search(f, gradient, x, p, grad)
            
            # 更新x
            x_new = x + alpha * p
            grad_new = gradient(x_new)
            
            self.history.append(x_new.copy())
            
            # 检查收敛
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"在第 {iteration} 次迭代时收敛")
                return x_new
            
            # 计算s和y
            s = x_new - x
            y = grad_new - grad
            
            # BFGS更新Hessian逆矩阵近似
            rho = 1.0 / np.dot(y, s)
            if rho > 0:  # 确保正定性
                I = np.eye(n)
                H = np.dot(I - rho * np.outer(s, y), np.dot(H, I - rho * np.outer(y, s))) + rho * np.outer(s, s)
            
            x = x_new
            grad = grad_new
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, f(x) = {f(x):.6f}")
        
        return x
    
    def _line_search(self, f, gradient, x, p, grad, alpha_init=1.0):
        """
        简单的回溯线搜索
        
        Args:
            f: 目标函数
            gradient: 梯度函数
            x: 当前点
            p: 搜索方向
            grad: 当前梯度
            alpha_init: 初始步长
        Returns:
            步长alpha
        """
        alpha = alpha_init
        c = 0.5  # Armijo条件参数
        rho = 0.5  # 步长缩减因子
        
        f_x = f(x)
        grad_dot_p = np.dot(grad, p)
        
        # Armijo条件: f(x + α*p) ≤ f(x) + c*α*∇f(x)^T*p
        while f(x + alpha * p) > f_x + c * alpha * grad_dot_p:
            alpha *= rho
            if alpha < 1e-10:
                break
        
        return alpha


# ==================== 凸优化 Convex Optimization ====================

class ConvexOptimization:
    """
    凸优化相关方法
    
    Java对应：
    import org.apache.commons.math3.optim.linear.*;
    
    public class ConvexOptimization {
        public double[] linearProgramming(double[] c, double[][] A, double[] b);
        public double[] quadraticProgramming(double[][] Q, double[] c, 
                                            double[][] A, double[] b);
    }
    """
    
    @staticmethod
    def is_convex_function(f, x_samples):
        """
        检查函数是否为凸函数（数值方法）
        凸函数满足: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y), ∀λ∈[0,1]
        
        Java对应：
        public static boolean isConvexFunction(Function<double[], Double> f,
                                              double[][] xSamples) {
            // 随机采样检查凸性条件
            for (int i = 0; i < xSamples.length; i++) {
                for (int j = i + 1; j < xSamples.length; j++) {
                    for (double lambda = 0.1; lambda < 1.0; lambda += 0.1) {
                        double[] midpoint = linearCombination(lambda, xSamples[i],
                                                             1 - lambda, xSamples[j]);
                        double fMid = f.apply(midpoint);
                        double fLinear = lambda * f.apply(xSamples[i]) +
                                       (1 - lambda) * f.apply(xSamples[j]);
                        
                        if (fMid > fLinear + 1e-6) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        
        Args:
            f: 函数
            x_samples: 采样点
        Returns:
            是否为凸函数
        """
        n_samples = len(x_samples)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # 检查线段上的点
                for lam in np.linspace(0, 1, 10):
                    x_mid = lam * x_samples[i] + (1 - lam) * x_samples[j]
                    f_mid = f(x_mid)
                    f_linear = lam * f(x_samples[i]) + (1 - lam) * f(x_samples[j])
                    
                    if f_mid > f_linear + 1e-6:
                        return False
        
        return True
    
    @staticmethod
    def projected_gradient_descent(f, gradient, x0, project_fn, 
                                   learning_rate=0.01, max_iterations=1000):
        """
        投影梯度下降（用于约束优化）
        
        Java对应：
        public static double[] projectedGradientDescent(
                Function<double[], Double> f,
                Function<double[], double[]> gradient,
                double[] x0,
                Function<double[], double[]> projectFn,
                double learningRate,
                int maxIterations) {
            
            double[] x = x0.clone();
            
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] grad = gradient.apply(x);
                
                // 梯度下降步
                double[] xNew = new double[x.length];
                for (int i = 0; i < x.length; i++) {
                    xNew[i] = x[i] - learningRate * grad[i];
                }
                
                // 投影到可行域
                xNew = projectFn.apply(xNew);
                
                if (vectorNorm(subtract(xNew, x)) < 1e-6) break;
                
                x = xNew;
            }
            
            return x;
        }
        
        Args:
            f: 目标函数
            gradient: 梯度函数
            x0: 初始点
            project_fn: 投影函数（将点投影到可行域）
            learning_rate: 学习率
            max_iterations: 最大迭代次数
        Returns:
            最优解
        """
        x = np.array(x0, dtype=float)
        
        for iteration in range(max_iterations):
            grad = gradient(x)
            
            # 梯度下降步
            x_new = x - learning_rate * grad
            
            # 投影到可行域
            x_new = project_fn(x_new)
            
            if np.linalg.norm(x_new - x) < 1e-6:
                print(f"在第 {iteration} 次迭代时收敛")
                return x_new
            
            x = x_new
        
        return x


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("优化算法示例")
    print("=" * 60)
    
    # 定义测试函数：Rosenbrock函数
    # f(x,y) = (1-x)² + 100(y-x²)²
    # 最小值在(1,1)处，f(1,1) = 0
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_gradient(x):
        dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dfdy = 200 * (x[1] - x[0]**2)
        return np.array([dfdx, dfdy])
    
    x0 = np.array([-1.0, 1.0])
    
    # ========== 1. 梯度下降 ==========
    print("\n【1. 标准梯度下降】")
    print("-" * 60)
    gd = GradientDescent(learning_rate=0.001, max_iterations=10000)
    x_gd = gd.optimize(rosenbrock, rosenbrock_gradient, x0)
    print(f"最优解: {x_gd}")
    print(f"函数值: {rosenbrock(x_gd):.10f}")
    
    # ========== 2. 动量梯度下降 ==========
    print("\n【2. 动量梯度下降】")
    print("-" * 60)
    mgd = MomentumGradientDescent(learning_rate=0.001, momentum=0.9, max_iterations=10000)
    x_mgd = mgd.optimize(rosenbrock, rosenbrock_gradient, x0)
    print(f"最优解: {x_mgd}")
    print(f"函数值: {rosenbrock(x_mgd):.10f}")
    
    # ========== 3. Adam优化器 ==========
    print("\n【3. Adam优化器】")
    print("-" * 60)
    adam = AdamOptimizer(learning_rate=0.01, max_iterations=10000)
    x_adam = adam.optimize(rosenbrock, rosenbrock_gradient, x0)
    print(f"最优解: {x_adam}")
    print(f"函数值: {rosenbrock(x_adam):.10f}")
    
    # ========== 4. BFGS拟牛顿法 ==========
    print("\n【4. BFGS拟牛顿法】")
    print("-" * 60)
    bfgs = BFGS(max_iterations=100)
    x_bfgs = bfgs.optimize(rosenbrock, rosenbrock_gradient, x0)
    print(f"最优解: {x_bfgs}")
    print(f"函数值: {rosenbrock(x_bfgs):.10f}")
    
    # ========== 5. 凸性检查 ==========
    print("\n【5. 凸函数检查】")
    print("-" * 60)
    
    # 凸函数示例: f(x) = x²
    convex_f = lambda x: np.sum(x**2)
    samples_convex = [np.array([i, j]) for i in range(-2, 3) for j in range(-2, 3)]
    is_convex = ConvexOptimization.is_convex_function(convex_f, samples_convex)
    print(f"f(x) = x² 是凸函数: {is_convex}")
    
    # 非凸函数示例: f(x) = x⁴ - x²
    nonconvex_f = lambda x: np.sum(x**4 - x**2)
    is_convex_2 = ConvexOptimization.is_convex_function(nonconvex_f, samples_convex)
    print(f"f(x) = x⁴ - x² 是凸函数: {is_convex_2}")
    
    # ========== 可视化 ==========
    print("\n绘制优化路径...")
    plt.figure(figsize=(15, 5))
    
    # 绘制Rosenbrock函数等高线
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    # 子图1: 梯度下降
    plt.subplot(1, 3, 1)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    history = np.array(gd.history)
    plt.plot(history[:, 0], history[:, 1], 'r.-', linewidth=2, markersize=4, label='优化路径')
    plt.plot(1, 1, 'g*', markersize=15, label='最优点')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('梯度下降')
    plt.legend()
    plt.grid(True)
    
    # 子图2: Adam
    plt.subplot(1, 3, 2)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    history = np.array(adam.history)
    plt.plot(history[:, 0], history[:, 1], 'b.-', linewidth=2, markersize=4, label='优化路径')
    plt.plot(1, 1, 'g*', markersize=15, label='最优点')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam优化器')
    plt.legend()
    plt.grid(True)
    
    # 子图3: BFGS
    plt.subplot(1, 3, 3)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    history = np.array(bfgs.history)
    plt.plot(history[:, 0], history[:, 1], 'm.-', linewidth=2, markersize=4, label='优化路径')
    plt.plot(1, 1, 'g*', markersize=15, label='最优点')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('BFGS拟牛顿法')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    example_usage()