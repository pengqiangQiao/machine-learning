"""
逻辑回归模型实现
Logistic Regression Implementation

Java对应实现：可以使用Weka的Logistic类或自己实现
Java equivalent: Use Weka's Logistic class or implement from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()

# Java对应：import weka.classifiers.functions.Logistic;


class LogisticRegressionModel:
    """
    逻辑回归模型类（用于二分类）
    
    Java对应实现：
    public class LogisticRegressionModel {
        private double[] weights;
        private double bias;
        private double learningRate;
        private int iterations;
        
        // Sigmoid函数
        private double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
    }
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        初始化逻辑回归模型
        
        Java对应：
        public LogisticRegressionModel(double learningRate, int iterations) {
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
    
    def sigmoid(self, z):
        """
        Sigmoid激活函数: σ(z) = 1 / (1 + e^(-z))
        
        Java对应：
        private double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
        
        // 对于数组
        private double[] sigmoid(double[] z) {
            double[] result = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                result[i] = 1.0 / (1.0 + Math.exp(-z[i]));
            }
            return result;
        }
        
        Args:
            z: 输入值
        Returns:
            sigmoid(z)
        """
        # 防止溢出
        # Java: 需要处理Math.exp()的溢出情况
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        训练模型
        
        Java对应：
        public void fit(double[][] X, int[] y) {
            int n_samples = X.length;
            int n_features = X[0].length;
            
            // 初始化参数
            this.weights = new double[n_features];
            this.bias = 0.0;
            
            // 梯度下降
            for (int iter = 0; iter < iterations; iter++) {
                // 前向传播
                double[] z = new double[n_samples];
                for (int i = 0; i < n_samples; i++) {
                    z[i] = bias;
                    for (int j = 0; j < n_features; j++) {
                        z[i] += X[i][j] * weights[j];
                    }
                }
                double[] y_pred = sigmoid(z);
                
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
                
                // 计算损失（交叉熵）
                double cost = calculateCrossEntropy(y, y_pred);
                if (iter % 100 == 0) {
                    System.out.println("Iteration " + iter + ", Cost: " + cost);
                }
            }
        }
        
        Args:
            X: 特征矩阵
            y: 标签 (0或1)
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        # Java: this.weights = new double[n_features];
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.iterations):
            # 线性组合: z = X·w + b
            # Java: z = matrixMultiply(X, weights) + bias
            linear_model = np.dot(X, self.weights) + self.bias
            
            # 应用sigmoid函数
            # Java: y_pred = sigmoid(z)
            y_pred = self.sigmoid(linear_model)
            
            # 计算梯度
            # dw = (1/n) * X^T · (y_pred - y)
            # Java: dw = matrixTransposeMultiply(X, subtract(y_pred, y)) / n_samples
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            # Java: weights = subtract(weights, multiply(learningRate, dw))
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 计算交叉熵损失
            # Cost = -(1/n) * Σ[y*log(y_pred) + (1-y)*log(1-y_pred)]
            # Java: cost = calculateCrossEntropy(y, y_pred)
            epsilon = 1e-15  # 防止log(0)
            cost = -np.mean(y * np.log(y_pred + epsilon) + 
                           (1 - y) * np.log(1 - y_pred + epsilon))
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"迭代 {i}, 损失: {cost:.4f}")
    
    def predict_proba(self, X):
        """
        预测概率
        
        Java对应：
        public double[] predictProba(double[][] X) {
            int n_samples = X.length;
            double[] probabilities = new double[n_samples];
            
            for (int i = 0; i < n_samples; i++) {
                double z = bias;
                for (int j = 0; j < weights.length; j++) {
                    z += X[i][j] * weights[j];
                }
                probabilities[i] = sigmoid(z);
            }
            return probabilities;
        }
        
        Args:
            X: 特征矩阵
        Returns:
            预测为正类的概率
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        
        Java对应：
        public int[] predict(double[][] X, double threshold) {
            double[] probabilities = predictProba(X);
            int[] predictions = new int[probabilities.length];
            
            for (int i = 0; i < probabilities.length; i++) {
                predictions[i] = probabilities[i] >= threshold ? 1 : 0;
            }
            return predictions;
        }
        
        Args:
            X: 特征矩阵
            threshold: 分类阈值
        Returns:
            预测的类别 (0或1)
        """
        y_pred_proba = self.predict_proba(X)
        # 概率 >= 阈值 -> 类别1，否则 -> 类别0
        # Java: predictions[i] = (probabilities[i] >= threshold) ? 1 : 0
        return (y_pred_proba >= threshold).astype(int)
    
    def plot_decision_boundary(self, X, y):
        """
        绘制决策边界（仅适用于2D特征）
        
        Java对应：
        public void plotDecisionBoundary(double[][] X, int[] y) {
            // 使用JFreeChart绘制
            XYSeriesCollection dataset = new XYSeriesCollection();
            
            // 绘制数据点
            XYSeries class0 = new XYSeries("Class 0");
            XYSeries class1 = new XYSeries("Class 1");
            
            for (int i = 0; i < X.length; i++) {
                if (y[i] == 0) {
                    class0.add(X[i][0], X[i][1]);
                } else {
                    class1.add(X[i][0], X[i][1]);
                }
            }
            
            // 绘制决策边界
            // w1*x1 + w2*x2 + b = 0
            // x2 = -(w1*x1 + b) / w2
        }
        """
        if X.shape[1] != 2:
            print("只能绘制2D特征的决策边界")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 绘制数据点
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 
                   color='blue', label='类别 0', alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
                   color='red', label='类别 1', alpha=0.6)
        
        # 绘制决策边界
        # 决策边界: w1*x1 + w2*x2 + b = 0
        # 即: x2 = -(w1*x1 + b) / w2
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_line = np.linspace(x1_min, x1_max, 100)
        x2_line = -(self.weights[0] * x1_line + self.bias) / self.weights[1]
        
        plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='决策边界')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title('逻辑回归决策边界')
        plt.legend()
        plt.grid(True)
        plt.show()


class SklearnLogisticRegression:
    """
    使用sklearn的逻辑回归
    
    Java对应：使用Weka的Logistic
    import weka.classifiers.functions.Logistic;
    
    public class SklearnLogisticRegression {
        private Logistic model;
    }
    """
    
    def __init__(self, max_iter=1000):
        """
        Java对应：
        public SklearnLogisticRegression(int maxIter) {
            this.model = new Logistic();
            this.model.setMaxIts(maxIter);
        }
        """
        self.model = LogisticRegression(max_iter=max_iter)
    
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
        预测类别
        
        Java对应：
        public double predict(Instance instance) throws Exception {
            return model.classifyInstance(instance);
        }
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测概率
        
        Java对应：
        public double[] predictProba(Instance instance) throws Exception {
            return model.distributionForInstance(instance);
        }
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Java对应：
        public void evaluate(Instances testData) throws Exception {
            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel(model, testData);
            
            double accuracy = eval.pctCorrect() / 100.0;
            double[][] confusionMatrix = eval.confusionMatrix();
            
            System.out.println("Accuracy: " + accuracy);
            System.out.println("Confusion Matrix:");
            for (double[] row : confusionMatrix) {
                System.out.println(Arrays.toString(row));
            }
        }
        
        Args:
            X: 测试特征
            y: 真实标签
        """
        y_pred = self.predict(X)
        
        # 准确率
        # Java: accuracy = correctPredictions / totalPredictions
        accuracy = accuracy_score(y, y_pred)
        
        # 混淆矩阵
        # Java: confusionMatrix = calculateConfusionMatrix(y_true, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        # 分类报告
        # Java: 需要手动计算precision, recall, f1-score
        report = classification_report(y, y_pred)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"\n混淆矩阵:\n{cm}")
        print(f"\n分类报告:\n{report}")
        
        return accuracy, cm


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) {
        // 生成示例数据
        Random random = new Random(42);
        double[][] X = new double[200][2];
        int[] y = new int[200];
        
        for (int i = 0; i < 200; i++) {
            X[i][0] = random.nextGaussian();
            X[i][1] = random.nextGaussian();
            y[i] = (X[i][0] + X[i][1] > 0) ? 1 : 0;
        }
        
        // 训练模型
        LogisticRegressionModel model = new LogisticRegressionModel(0.1, 1000);
        model.fit(X, y);
    }
    """
    print("=" * 50)
    print("逻辑回归示例")
    print("=" * 50)
    
    # 生成示例数据（线性可分）
    # Java: 使用Random类生成数据
    np.random.seed(42)
    n_samples = 200
    
    # 类别0: 均值在(-2, -2)附近
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    # 类别1: 均值在(2, 2)附近
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # 打乱数据
    # Java: shuffleData(X, y)
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    # 划分训练集和测试集
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("\n1. 使用自定义梯度下降实现")
    print("-" * 50)
    # Java: LogisticRegressionModel customModel = new LogisticRegressionModel(0.1, 1000);
    custom_model = LogisticRegressionModel(learning_rate=0.1, iterations=1000)
    custom_model.fit(X_train, y_train)
    
    # 预测
    y_pred_custom = custom_model.predict(X_test)
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    print(f"\n测试集准确率: {accuracy_custom:.4f}")
    
    # 绘制决策边界
    print("\n绘制决策边界...")
    custom_model.plot_decision_boundary(X_train, y_train)
    print("决策边界图形已显示")
    
    print("\n2. 使用sklearn实现")
    print("-" * 50)
    # Java: SklearnLogisticRegression sklearnModel = new SklearnLogisticRegression(1000);
    sklearn_model = SklearnLogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    sklearn_model.evaluate(X_test, y_test)
    
    # 绘制损失历史
    print("\n绘制损失历史曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(custom_model.cost_history)
    plt.xlabel('迭代次数')
    plt.ylabel('交叉熵损失')
    plt.title('训练过程中的损失变化')
    plt.grid(True)
    print("正在显示图形...")
    plt.show()
    print("损失历史图形已显示")


if __name__ == "__main__":
    """
    Java对应：
    public static void main(String[] args) {
        exampleUsage();
    }
    """
    example_usage()