"""
线性回归模型实现
Linear Regression Implementation

Java对应实现：可以使用Apache Commons Math的SimpleRegression类
Java equivalent: Use Apache Commons Math's SimpleRegression class
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()

# Java对应：import org.apache.commons.math3.stat.regression.SimpleRegression;


class LinearRegressionModel:
    """
    线性回归模型类
    
    Java对应实现：
    public class LinearRegressionModel {
        private double[] weights;  // 权重
        private double bias;       // 偏置
        private double learningRate;
        private int iterations;
    }
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        初始化线性回归模型
        
        Java对应：
        public LinearRegressionModel(double learningRate, int iterations) {
            this.learningRate = learningRate;
            this.iterations = iterations;
            this.weights = null;
            this.bias = 0.0;
        }
        
        Args:
            learning_rate: 学习率
            iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        训练模型（梯度下降法）
        
        Java对应：
        public void fit(double[][] X, double[] y) {
            int n_samples = X.length;
            int n_features = X[0].length;
            
            // 初始化权重和偏置
            this.weights = new double[n_features];
            this.bias = 0.0;
            
            // 梯度下降
            for (int iter = 0; iter < iterations; iter++) {
                // 预测值: y_pred = X * weights + bias
                double[] y_pred = predict(X);
                
                // 计算梯度
                double[] dw = new double[n_features];
                double db = 0.0;
                
                for (int i = 0; i < n_samples; i++) {
                    double error = y_pred[i] - y[i];
                    db += error;
                    for (int j = 0; j < n_features; j++) {
                        dw[j] += error * X[i][j];
                    }
                }
                
                // 更新参数
                for (int j = 0; j < n_features; j++) {
                    weights[j] -= learningRate * dw[j] / n_samples;
                }
                bias -= learningRate * db / n_samples;
                
                // 计算损失
                double cost = calculateCost(y, y_pred);
                if (iter % 100 == 0) {
                    System.out.println("Iteration " + iter + ", Cost: " + cost);
                }
            }
        }
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        # Java: this.weights = new double[n_features];
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.iterations):
            # 预测: y_pred = X·w + b
            # Java: y_pred = matrixMultiply(X, weights) + bias
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算梯度
            # dw = (1/n) * X^T · (y_pred - y)
            # Java: dw = matrixTransposeMultiply(X, subtract(y_pred, y)) / n_samples
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            # w = w - α * dw
            # Java: weights = subtract(weights, multiply(learningRate, dw))
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 记录损失
            # MSE = (1/n) * Σ(y_pred - y)²
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"迭代 {i}, 损失: {cost:.4f}")
    
    def predict(self, X):
        """
        预测
        
        Java对应：
        public double[] predict(double[][] X) {
            int n_samples = X.length;
            double[] predictions = new double[n_samples];
            
            for (int i = 0; i < n_samples; i++) {
                predictions[i] = bias;
                for (int j = 0; j < weights.length; j++) {
                    predictions[i] += X[i][j] * weights[j];
                }
            }
            return predictions;
        }
        
        Args:
            X: 特征矩阵
        Returns:
            预测值
        """
        # y = X·w + b
        # Java: return matrixMultiply(X, weights) + bias
        return np.dot(X, self.weights) + self.bias
    
    def plot_cost_history(self):
        """
        绘制损失函数历史
        
        Java对应：
        public void plotCostHistory() {
            // 使用JFreeChart库绘图
            XYSeries series = new XYSeries("Cost");
            for (int i = 0; i < costHistory.size(); i++) {
                series.add(i, costHistory.get(i));
            }
            
            XYSeriesCollection dataset = new XYSeriesCollection(series);
            JFreeChart chart = ChartFactory.createXYLineChart(
                "Cost History", "Iteration", "Cost", dataset);
            
            ChartPanel chartPanel = new ChartPanel(chart);
            JFrame frame = new JFrame("Cost History");
            frame.setContentPane(chartPanel);
            frame.pack();
            frame.setVisible(true);
        }
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值 (MSE)')
        plt.title('训练过程中的损失变化')
        plt.grid(True)
        plt.show()


class SklearnLinearRegression:
    """
    使用sklearn的线性回归（封装类）
    
    Java对应：使用Weka的LinearRegression
    import weka.classifiers.functions.LinearRegression;
    
    public class SklearnLinearRegression {
        private LinearRegression model;
    }
    """
    
    def __init__(self):
        """
        Java对应：
        public SklearnLinearRegression() {
            this.model = new LinearRegression();
        }
        """
        self.model = LinearRegression()
    
    def fit(self, X, y):
        """
        训练模型
        
        Java对应：
        public void fit(Instances data) throws Exception {
            model.buildClassifier(data);
        }
        """
        self.model.fit(X, y)
        print(f"权重: {self.model.coef_}")
        print(f"截距: {self.model.intercept_}")
    
    def predict(self, X):
        """
        预测
        
        Java对应：
        public double predict(Instance instance) throws Exception {
            return model.classifyInstance(instance);
        }
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Java对应：
        public void evaluate(Instances testData) throws Exception {
            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel(model, testData);
            
            double rmse = eval.rootMeanSquaredError();
            double r2 = eval.correlationCoefficient();
            
            System.out.println("RMSE: " + rmse);
            System.out.println("R²: " + r2);
        }
        
        Args:
            X: 测试特征
            y: 真实标签
        """
        y_pred = self.predict(X)
        
        # 均方误差 MSE = (1/n) * Σ(y_true - y_pred)²
        # Java: mse = calculateMSE(y_true, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        # 均方根误差 RMSE = √MSE
        # Java: rmse = Math.sqrt(mse)
        rmse = np.sqrt(mse)
        
        # R² 决定系数
        # Java: r2 = calculateR2(y_true, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"R² 分数: {r2:.4f}")
        
        return mse, rmse, r2


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) {
        // 生成示例数据
        double[][] X = generateData(100, 1);
        double[] y = new double[100];
        
        for (int i = 0; i < 100; i++) {
            y[i] = 2 * X[i][0] + 3 + random.nextGaussian();
        }
        
        // 训练模型
        LinearRegressionModel model = new LinearRegressionModel(0.01, 1000);
        model.fit(X, y);
        
        // 预测
        double[] predictions = model.predict(X);
    }
    """
    print("=" * 50)
    print("线性回归示例")
    print("=" * 50)
    
    # 生成示例数据: y = 2x + 3 + noise
    # Java: X = new double[100][1]; y = new double[100];
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # 划分训练集和测试集
    # Java: 手动划分或使用Weka的RemovePercentage
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("\n1. 使用自定义梯度下降实现")
    print("-" * 50)
    # Java: LinearRegressionModel customModel = new LinearRegressionModel(0.1, 1000);
    custom_model = LinearRegressionModel(learning_rate=0.1, iterations=1000)
    custom_model.fit(X_train, y_train)
    
    # 预测
    # Java: double[] predictions = customModel.predict(X_test);
    y_pred_custom = custom_model.predict(X_test)
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    print(f"\n测试集 MSE: {mse_custom:.4f}")
    print(f"学到的权重: {custom_model.weights}")
    print(f"学到的偏置: {custom_model.bias:.4f}")
    
    print("\n2. 使用sklearn实现")
    print("-" * 50)
    # Java: SklearnLinearRegression sklearnModel = new SklearnLinearRegression();
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_model.evaluate(X_test, y_test)
    
    # 可视化结果
    # Java: 使用JFreeChart绘制散点图和回归线
    print("\n绘制预测结果和损失曲线...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, y_test, color='blue', label='真实值')
    plt.plot(X_test, y_pred_custom, color='red', linewidth=2, label='预测值')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('自定义模型预测结果')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(custom_model.cost_history)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('训练损失变化')
    plt.grid(True)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    """
    Java对应：
    public static void main(String[] args) {
        exampleUsage();
    }
    """
    example_usage()