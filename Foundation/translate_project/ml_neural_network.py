"""
神经网络基础实现
Neural Network Implementation

Java对应实现：可以使用DL4J (DeepLearning4J)库
Java equivalent: Use DL4J (DeepLearning4J) library
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()

# Java对应：import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
# Java对应：import org.nd4j.linalg.api.ndarray.INDArray;


class NeuralNetwork:
    """
    简单的前馈神经网络（多层感知器）
    
    Java对应实现：
    public class NeuralNetwork {
        private double[][] weightsInputHidden;   // 输入层到隐藏层的权重
        private double[][] weightsHiddenOutput;  // 隐藏层到输出层的权重
        private double[] biasHidden;             // 隐藏层偏置
        private double[] biasOutput;             // 输出层偏置
        private double learningRate;
        private int epochs;
        
        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, 
                           double learningRate, int epochs) {
            this.learningRate = learningRate;
            this.epochs = epochs;
            
            // 初始化权重和偏置
            Random random = new Random();
            weightsInputHidden = new double[inputSize][hiddenSize];
            weightsHiddenOutput = new double[hiddenSize][outputSize];
            biasHidden = new double[hiddenSize];
            biasOutput = new double[outputSize];
            
            // 随机初始化
            initializeWeights(weightsInputHidden, random);
            initializeWeights(weightsHiddenOutput, random);
        }
    }
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        """
        初始化神经网络
        
        Java对应：
        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize,
                           double learningRate, int epochs) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;
            this.epochs = epochs;
            
            // 初始化权重
            initializeWeights();
        }
        
        Args:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
            learning_rate: 学习率
            epochs: 训练轮数
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # 初始化权重和偏置（Xavier初始化）
        # Java: weightsInputHidden = initializeXavier(inputSize, hiddenSize)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid激活函数
        
        Java对应：
        private double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
        
        private double[] sigmoid(double[] z) {
            double[] result = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                result[i] = 1.0 / (1.0 + Math.exp(-z[i]));
            }
            return result;
        }
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """
        Sigmoid导数
        
        Java对应：
        private double sigmoidDerivative(double z) {
            double sig = sigmoid(z);
            return sig * (1 - sig);
        }
        """
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def relu(self, z):
        """
        ReLU激活函数
        
        Java对应：
        private double relu(double z) {
            return Math.max(0, z);
        }
        
        private double[] relu(double[] z) {
            double[] result = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                result[i] = Math.max(0, z[i]);
            }
            return result;
        }
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        ReLU导数
        
        Java对应：
        private double reluDerivative(double z) {
            return z > 0 ? 1.0 : 0.0;
        }
        """
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """
        Softmax函数（用于多分类）
        
        Java对应：
        private double[] softmax(double[] z) {
            double max = Arrays.stream(z).max().getAsDouble();
            double[] expZ = new double[z.length];
            double sum = 0.0;
            
            for (int i = 0; i < z.length; i++) {
                expZ[i] = Math.exp(z[i] - max);
                sum += expZ[i];
            }
            
            for (int i = 0; i < expZ.length; i++) {
                expZ[i] /= sum;
            }
            
            return expZ;
        }
        """
        # 减去最大值以提高数值稳定性
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        Java对应：
        private ForwardResult forward(double[][] X) {
            // 隐藏层
            double[][] Z1 = matrixMultiply(X, weightsInputHidden);
            Z1 = addBias(Z1, biasHidden);
            double[][] A1 = relu(Z1);
            
            // 输出层
            double[][] Z2 = matrixMultiply(A1, weightsHiddenOutput);
            Z2 = addBias(Z2, biasOutput);
            double[][] A2 = sigmoid(Z2);
            
            return new ForwardResult(Z1, A1, Z2, A2);
        }
        
        Args:
            X: 输入数据
        Returns:
            各层的激活值
        """
        # 隐藏层
        # Z1 = X·W1 + b1
        # Java: Z1 = matrixMultiply(X, W1) + b1
        self.Z1 = np.dot(X, self.W1) + self.b1
        # A1 = ReLU(Z1)
        # Java: A1 = relu(Z1)
        self.A1 = self.relu(self.Z1)
        
        # 输出层
        # Z2 = A1·W2 + b2
        # Java: Z2 = matrixMultiply(A1, W2) + b2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # A2 = Sigmoid(Z2) 或 Softmax(Z2)
        # Java: A2 = sigmoid(Z2)
        if self.output_size == 1:
            self.A2 = self.sigmoid(self.Z2)
        else:
            self.A2 = self.softmax(self.Z2)
        
        return self.A2
    
    def backward(self, X, y, output):
        """
        反向传播
        
        Java对应：
        private void backward(double[][] X, double[][] y, double[][] output) {
            int m = X.length;
            
            // 输出层梯度
            double[][] dZ2 = subtract(output, y);
            double[][] dW2 = matrixTransposeMultiply(A1, dZ2);
            dW2 = divide(dW2, m);
            double[] db2 = sumColumns(dZ2);
            db2 = divide(db2, m);
            
            // 隐藏层梯度
            double[][] dA1 = matrixMultiply(dZ2, transpose(weightsHiddenOutput));
            double[][] dZ1 = multiply(dA1, reluDerivative(Z1));
            double[][] dW1 = matrixTransposeMultiply(X, dZ1);
            dW1 = divide(dW1, m);
            double[] db1 = sumColumns(dZ1);
            db1 = divide(db1, m);
            
            // 更新权重
            weightsHiddenOutput = subtract(weightsHiddenOutput, multiply(learningRate, dW2));
            biasOutput = subtract(biasOutput, multiply(learningRate, db2));
            weightsInputHidden = subtract(weightsInputHidden, multiply(learningRate, dW1));
            biasHidden = subtract(biasHidden, multiply(learningRate, db1));
        }
        
        Args:
            X: 输入数据
            y: 真实标签
            output: 预测输出
        """
        m = X.shape[0]
        
        # 输出层梯度
        # dZ2 = A2 - y
        # Java: dZ2 = subtract(output, y)
        dZ2 = output - y
        # dW2 = (1/m) * A1^T · dZ2
        # Java: dW2 = matrixTransposeMultiply(A1, dZ2) / m
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # 隐藏层梯度
        # dA1 = dZ2 · W2^T
        # Java: dA1 = matrixMultiply(dZ2, transpose(W2))
        dA1 = np.dot(dZ2, self.W2.T)
        # dZ1 = dA1 ⊙ ReLU'(Z1)
        # Java: dZ1 = multiply(dA1, reluDerivative(Z1))
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        # dW1 = (1/m) * X^T · dZ1
        # Java: dW1 = matrixTransposeMultiply(X, dZ1) / m
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # 更新权重（梯度下降）
        # Java: W2 = subtract(W2, multiply(learningRate, dW2))
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def fit(self, X, y):
        """
        训练神经网络
        
        Java对应：
        public void fit(double[][] X, double[][] y) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                // 前向传播
                ForwardResult forward = forward(X);
                
                // 计算损失
                double loss = calculateLoss(y, forward.output);
                
                // 反向传播
                backward(X, y, forward.output);
                
                if (epoch % 100 == 0) {
                    System.out.println("Epoch " + epoch + ", Loss: " + loss);
                }
            }
        }
        
        Args:
            X: 训练数据
            y: 标签
        """
        # 如果y是一维的，转换为二维
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        for epoch in range(self.epochs):
            # 前向传播
            # Java: output = forward(X)
            output = self.forward(X)
            
            # 计算损失（交叉熵）
            # Java: loss = calculateCrossEntropy(y, output)
            epsilon = 1e-15
            if self.output_size == 1:
                loss = -np.mean(y * np.log(output + epsilon) + 
                              (1 - y) * np.log(1 - output + epsilon))
            else:
                loss = -np.mean(np.sum(y * np.log(output + epsilon), axis=1))
            
            self.loss_history.append(loss)
            
            # 反向传播
            # Java: backward(X, y, output)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"轮次 {epoch}, 损失: {loss:.4f}")
    
    def predict(self, X):
        """
        预测
        
        Java对应：
        public int[] predict(double[][] X) {
            double[][] output = forward(X).output;
            int[] predictions = new int[output.length];
            
            for (int i = 0; i < output.length; i++) {
                predictions[i] = output[i][0] >= 0.5 ? 1 : 0;
            }
            
            return predictions;
        }
        
        Args:
            X: 输入数据
        Returns:
            预测结果
        """
        output = self.forward(X)
        if self.output_size == 1:
            return (output >= 0.5).astype(int).flatten()
        else:
            return np.argmax(output, axis=1)
    
    def plot_loss(self):
        """
        绘制损失曲线
        
        Java对应：
        public void plotLoss() {
            XYSeries series = new XYSeries("Loss");
            for (int i = 0; i < lossHistory.size(); i++) {
                series.add(i, lossHistory.get(i));
            }
            
            XYSeriesCollection dataset = new XYSeriesCollection(series);
            JFreeChart chart = ChartFactory.createXYLineChart(
                "Training Loss", "Epoch", "Loss", dataset);
        }
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('训练过程中的损失变化')
        plt.grid(True)
        plt.show()


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) {
        // 生成XOR问题数据
        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] y = {{0}, {1}, {1}, {0}};
        
        // 创建神经网络
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1, 0.1, 5000);
        
        // 训练
        nn.fit(X, y);
        
        // 预测
        int[] predictions = nn.predict(X);
        
        System.out.println("Predictions: " + Arrays.toString(predictions));
    }
    """
    print("=" * 50)
    print("神经网络示例 - XOR问题")
    print("=" * 50)
    
    # XOR问题（非线性问题）
    # Java: double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("\n1. 使用自定义神经网络")
    print("-" * 50)
    # Java: NeuralNetwork nn = new NeuralNetwork(2, 4, 1, 0.1, 5000)
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, 
                      learning_rate=0.1, epochs=5000)
    nn.fit(X, y)
    
    # 预测
    predictions = nn.predict(X)
    print(f"\n输入:\n{X}")
    print(f"真实标签: {y.flatten()}")
    print(f"预测结果: {predictions}")
    
    # 绘制损失曲线
    print("\n绘制损失曲线...")
    nn.plot_loss()
    print("损失曲线已显示")
    
    print("\n2. 使用sklearn的MLPClassifier")
    print("-" * 50)
    # Java: 使用DL4J的MultiLayerNetwork
    sklearn_nn = MLPClassifier(hidden_layer_sizes=(4,), 
                              activation='relu',
                              solver='sgd',
                              learning_rate_init=0.1,
                              max_iter=5000,
                              random_state=42)
    sklearn_nn.fit(X, y.ravel())
    
    predictions_sklearn = sklearn_nn.predict(X)
    accuracy = accuracy_score(y.ravel(), predictions_sklearn)
    print(f"准确率: {accuracy:.4f}")
    print(f"预测结果: {predictions_sklearn}")
    
    # 更复杂的示例：分类问题
    print("\n" + "=" * 50)
    print("神经网络示例 - 分类问题")
    print("=" * 50)
    
    # 生成螺旋数据
    np.random.seed(42)
    n_samples = 300
    
    # 生成两个螺旋
    theta = np.linspace(0, 4 * np.pi, n_samples // 2)
    r = np.linspace(0, 1, n_samples // 2)
    
    X_spiral = np.zeros((n_samples, 2))
    y_spiral = np.zeros(n_samples)
    
    # 第一个螺旋
    X_spiral[:n_samples // 2, 0] = r * np.cos(theta)
    X_spiral[:n_samples // 2, 1] = r * np.sin(theta)
    y_spiral[:n_samples // 2] = 0
    
    # 第二个螺旋
    X_spiral[n_samples // 2:, 0] = r * np.cos(theta + np.pi)
    X_spiral[n_samples // 2:, 1] = r * np.sin(theta + np.pi)
    y_spiral[n_samples // 2:] = 1
    
    # 添加噪声
    X_spiral += np.random.randn(n_samples, 2) * 0.1
    
    # 训练神经网络
    nn_spiral = NeuralNetwork(input_size=2, hidden_size=10, output_size=1,
                             learning_rate=0.1, epochs=2000)
    nn_spiral.fit(X_spiral, y_spiral)
    
    predictions_spiral = nn_spiral.predict(X_spiral)
    accuracy_spiral = accuracy_score(y_spiral, predictions_spiral)
    print(f"\n螺旋数据准确率: {accuracy_spiral:.4f}")
    
    # 可视化结果
    print("\n绘制螺旋数据分类结果...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_spiral[y_spiral == 0][:, 0], X_spiral[y_spiral == 0][:, 1],
               color='blue', label='类别 0', alpha=0.6)
    plt.scatter(X_spiral[y_spiral == 1][:, 0], X_spiral[y_spiral == 1][:, 1],
               color='red', label='类别 1', alpha=0.6)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('螺旋数据分布')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(nn_spiral.loss_history)
    plt.xlabel('训练轮次')
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