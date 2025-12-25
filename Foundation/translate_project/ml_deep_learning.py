"""
深度学习算法实现
Deep Learning Algorithms

包含：卷积神经网络CNN、循环神经网络RNN
Including: Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN)

Java对应实现：可以使用DL4J (DeepLearning4J)库
Java equivalent: Use DL4J (DeepLearning4J) library
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 卷积神经网络 CNN ====================

class ConvLayer:
    """
    卷积层
    
    Java对应：
    import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
    
    public class ConvLayer {
        private double[][][][] filters;  // [num_filters, channels, height, width]
        private double[] biases;
        private int stride;
        private int padding;
    }
    """
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        """
        初始化卷积层
        
        Java对应：
        public ConvLayer(int numFilters, int filterSize, int stride, int padding) {
            this.numFilters = numFilters;
            this.filterSize = filterSize;
            this.stride = stride;
            this.padding = padding;
        }
        
        Args:
            num_filters: 卷积核数量
            filter_size: 卷积核大小
            stride: 步长
            padding: 填充
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.biases = None
    
    def initialize(self, input_channels):
        """
        初始化权重
        
        Java对应：
        public void initialize(int inputChannels) {
            // He初始化
            double std = Math.sqrt(2.0 / (inputChannels * filterSize * filterSize));
            filters = new double[numFilters][inputChannels][filterSize][filterSize];
            
            Random random = new Random();
            for (int i = 0; i < numFilters; i++) {
                for (int j = 0; j < inputChannels; j++) {
                    for (int k = 0; k < filterSize; k++) {
                        for (int l = 0; l < filterSize; l++) {
                            filters[i][j][k][l] = random.nextGaussian() * std;
                        }
                    }
                }
            }
            
            biases = new double[numFilters];
        }
        """
        # He初始化
        std = np.sqrt(2.0 / (input_channels * self.filter_size * self.filter_size))
        self.filters = np.random.randn(self.num_filters, input_channels,
                                      self.filter_size, self.filter_size) * std
        self.biases = np.zeros(self.num_filters)
    
    def forward(self, input_data):
        """
        前向传播
        
        卷积操作: output[i,j] = Σ(input * filter) + bias
        
        Java对应：
        public double[][][] forward(double[][][] input) {
            int batchSize = input.length;
            int inputHeight = input[0].length;
            int inputWidth = input[0][0].length;
            
            // 计算输出尺寸
            int outputHeight = (inputHeight + 2 * padding - filterSize) / stride + 1;
            int outputWidth = (inputWidth + 2 * padding - filterSize) / stride + 1;
            
            double[][][] output = new double[numFilters][outputHeight][outputWidth];
            
            // 执行卷积
            for (int f = 0; f < numFilters; f++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        double sum = biases[f];
                        
                        for (int c = 0; c < inputChannels; c++) {
                            for (int ki = 0; ki < filterSize; ki++) {
                                for (int kj = 0; kj < filterSize; kj++) {
                                    int ii = i * stride + ki - padding;
                                    int jj = j * stride + kj - padding;
                                    
                                    if (ii >= 0 && ii < inputHeight && 
                                        jj >= 0 && jj < inputWidth) {
                                        sum += input[c][ii][jj] * filters[f][c][ki][kj];
                                    }
                                }
                            }
                        }
                        
                        output[f][i][j] = sum;
                    }
                }
            }
            
            return output;
        }
        
        Args:
            input_data: 输入数据 [batch, channels, height, width]
        Returns:
            输出特征图
        """
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        
        # 执行卷积
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size
                        
                        # 确保不越界
                        if h_end <= height and w_end <= width:
                            # 提取感受野
                            receptive_field = input_data[b, :, h_start:h_end, w_start:w_end]
                            
                            # 卷积操作
                            output[b, f, i, j] = np.sum(receptive_field * self.filters[f]) + self.biases[f]
        
        return output


class MaxPooling:
    """
    最大池化层
    
    Java对应：
    import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
    
    public class MaxPooling {
        private int poolSize;
        private int stride;
        
        public double[][][] forward(double[][][] input);
    }
    """
    
    def __init__(self, pool_size=2, stride=2):
        """
        初始化池化层
        
        Java对应：
        public MaxPooling(int poolSize, int stride) {
            this.poolSize = poolSize;
            this.stride = stride;
        }
        
        Args:
            pool_size: 池化窗口大小
            stride: 步长
        """
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, input_data):
        """
        前向传播
        
        最大池化: output[i,j] = max(input[region])
        
        Java对应：
        public double[][][] forward(double[][][] input) {
            int channels = input.length;
            int inputHeight = input[0].length;
            int inputWidth = input[0][0].length;
            
            int outputHeight = (inputHeight - poolSize) / stride + 1;
            int outputWidth = (inputWidth - poolSize) / stride + 1;
            
            double[][][] output = new double[channels][outputHeight][outputWidth];
            
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        double max = Double.NEGATIVE_INFINITY;
                        
                        for (int pi = 0; pi < poolSize; pi++) {
                            for (int pj = 0; pj < poolSize; pj++) {
                                int ii = i * stride + pi;
                                int jj = j * stride + pj;
                                max = Math.max(max, input[c][ii][jj]);
                            }
                        }
                        
                        output[c][i][j] = max;
                    }
                }
            }
            
            return output;
        }
        
        Args:
            input_data: 输入数据
        Returns:
            池化后的输出
        """
        batch_size, channels, height, width = input_data.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        # 最大池化
                        output[b, c, i, j] = np.max(input_data[b, c, h_start:h_end, w_start:w_end])
        
        return output


class SimpleCNN:
    """
    简单的卷积神经网络
    
    Java对应：
    import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
    import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
    
    public class SimpleCNN {
        private ConvLayer conv1;
        private MaxPooling pool1;
        private ConvLayer conv2;
        private MaxPooling pool2;
        private double[][] fcWeights;
        private double[] fcBias;
    }
    """
    
    def __init__(self, input_shape, num_classes):
        """
        初始化CNN
        
        Args:
            input_shape: 输入形状 (channels, height, width)
            num_classes: 类别数
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 构建网络层
        self.conv1 = ConvLayer(num_filters=16, filter_size=3, stride=1, padding=1)
        self.pool1 = MaxPooling(pool_size=2, stride=2)
        self.conv2 = ConvLayer(num_filters=32, filter_size=3, stride=1, padding=1)
        self.pool2 = MaxPooling(pool_size=2, stride=2)
        
        # 初始化卷积层
        self.conv1.initialize(input_shape[0])
        self.conv2.initialize(16)
        
        print("CNN架构:")
        print(f"输入: {input_shape}")
        print("Conv1: 16个3x3卷积核 -> ReLU -> MaxPool(2x2)")
        print("Conv2: 32个3x3卷积核 -> ReLU -> MaxPool(2x2)")
        print(f"全连接层 -> {num_classes}类")
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        Args:
            X: 输入数据 [batch, channels, height, width]
        Returns:
            输出概率
        """
        # Conv1 + ReLU + Pool1
        out = self.conv1.forward(X)
        out = self.relu(out)
        out = self.pool1.forward(out)
        
        # Conv2 + ReLU + Pool2
        out = self.conv2.forward(out)
        out = self.relu(out)
        out = self.pool2.forward(out)
        
        # 展平
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        
        # 全连接层（简化版：使用随机权重）
        if not hasattr(self, 'fc_weights'):
            self.fc_weights = np.random.randn(out.shape[1], self.num_classes) * 0.01
            self.fc_bias = np.zeros(self.num_classes)
        
        out = np.dot(out, self.fc_weights) + self.fc_bias
        out = self.softmax(out)
        
        return out


# ==================== 循环神经网络 RNN ====================

class SimpleRNN:
    """
    简单的循环神经网络
    
    Java对应：
    import org.deeplearning4j.nn.conf.layers.LSTM;
    
    public class SimpleRNN {
        private double[][] Wxh;  // 输入到隐藏层权重
        private double[][] Whh;  // 隐藏层到隐藏层权重
        private double[][] Why;  // 隐藏层到输出权重
        private double[] bh;     // 隐藏层偏置
        private double[] by;     // 输出层偏置
        private int hiddenSize;
    }
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化RNN
        
        Java对应：
        public SimpleRNN(int inputSize, int hiddenSize, int outputSize, double learningRate) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;
            
            // Xavier初始化
            Random random = new Random();
            Wxh = initializeWeights(inputSize, hiddenSize, random);
            Whh = initializeWeights(hiddenSize, hiddenSize, random);
            Why = initializeWeights(hiddenSize, outputSize, random);
            bh = new double[hiddenSize];
            by = new double[outputSize];
        }
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Xavier初始化
        self.Wxh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)
        
        print(f"RNN架构: 输入({input_size}) -> 隐藏层({hidden_size}) -> 输出({output_size})")
    
    def tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, inputs, h_prev):
        """
        前向传播
        
        RNN公式:
        h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
        y_t = W_hy·h_t + b_y
        
        Java对应：
        public ForwardResult forward(double[][] inputs, double[] hPrev) {
            int timeSteps = inputs.length;
            double[][] hiddenStates = new double[timeSteps + 1][];
            double[][] outputs = new double[timeSteps][];
            
            hiddenStates[0] = hPrev.clone();
            
            for (int t = 0; t < timeSteps; t++) {
                // h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
                double[] h = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    h[i] = bh[i];
                    for (int j = 0; j < inputSize; j++) {
                        h[i] += Wxh[j][i] * inputs[t][j];
                    }
                    for (int j = 0; j < hiddenSize; j++) {
                        h[i] += Whh[j][i] * hiddenStates[t][j];
                    }
                    h[i] = Math.tanh(h[i]);
                }
                hiddenStates[t + 1] = h;
                
                // y_t = Why * h_t + by
                double[] y = new double[outputSize];
                for (int i = 0; i < outputSize; i++) {
                    y[i] = by[i];
                    for (int j = 0; j < hiddenSize; j++) {
                        y[i] += Why[j][i] * h[j];
                    }
                }
                outputs[t] = softmax(y);
            }
            
            return new ForwardResult(hiddenStates, outputs);
        }
        
        Args:
            inputs: 输入序列 [time_steps, input_size]
            h_prev: 前一个隐藏状态
        Returns:
            outputs: 输出序列
            h_last: 最后的隐藏状态
        """
        time_steps = len(inputs)
        h = h_prev
        outputs = []
        
        for t in range(time_steps):
            # h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
            h = self.tanh(np.dot(inputs[t], self.Wxh) + np.dot(h, self.Whh) + self.bh)
            
            # y_t = W_hy·h_t + b_y
            y = np.dot(h, self.Why) + self.by
            y = self.softmax(y)
            
            outputs.append(y)
        
        return outputs, h
    
    def predict(self, inputs):
        """
        预测
        
        Args:
            inputs: 输入序列
        Returns:
            预测结果
        """
        h_prev = np.zeros(self.hidden_size)
        outputs, _ = self.forward(inputs, h_prev)
        return np.array(outputs)


class LSTM:
    """
    长短期记忆网络
    
    Java对应：
    import org.deeplearning4j.nn.conf.layers.LSTM;
    
    public class LSTM {
        // 遗忘门
        private double[][] Wf;
        // 输入门
        private double[][] Wi;
        // 输出门
        private double[][] Wo;
        // 候选记忆
        private double[][] Wc;
    }
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化LSTM
        
        LSTM包含三个门：
        - 遗忘门 (forget gate): 决定丢弃哪些信息
        - 输入门 (input gate): 决定更新哪些信息
        - 输出门 (output gate): 决定输出哪些信息
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重（简化版）
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # 遗忘门
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bf = np.zeros(hidden_size)
        
        # 输入门
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bi = np.zeros(hidden_size)
        
        # 候选记忆
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bc = np.zeros(hidden_size)
        
        # 输出门
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bo = np.zeros(hidden_size)
        
        # 输出层
        self.Why = np.random.randn(hidden_size, output_size) * scale
        self.by = np.zeros(output_size)
        
        print(f"LSTM架构: 输入({input_size}) -> LSTM({hidden_size}) -> 输出({output_size})")
    
    def sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, h_prev, c_prev):
        """
        LSTM前向传播
        
        LSTM公式:
        f_t = σ(W_f·[h_{t-1}, x_t] + b_f)  # 遗忘门
        i_t = σ(W_i·[h_{t-1}, x_t] + b_i)  # 输入门
        c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)  # 候选记忆
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t  # 更新记忆
        o_t = σ(W_o·[h_{t-1}, x_t] + b_o)  # 输出门
        h_t = o_t ⊙ tanh(c_t)  # 更新隐藏状态
        
        Args:
            inputs: 输入序列
            h_prev: 前一个隐藏状态
            c_prev: 前一个记忆单元
        Returns:
            outputs, h_last, c_last
        """
        time_steps = len(inputs)
        h = h_prev
        c = c_prev
        outputs = []
        
        for t in range(time_steps):
            # 拼接输入和隐藏状态
            combined = np.concatenate([h, inputs[t]])
            
            # 遗忘门
            f = self.sigmoid(np.dot(combined, self.Wf) + self.bf)
            
            # 输入门
            i = self.sigmoid(np.dot(combined, self.Wi) + self.bi)
            
            # 候选记忆
            c_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc)
            
            # 更新记忆单元
            c = f * c + i * c_tilde
            
            # 输出门
            o = self.sigmoid(np.dot(combined, self.Wo) + self.bo)
            
            # 更新隐藏状态
            h = o * np.tanh(c)
            
            # 输出
            y = np.dot(h, self.Why) + self.by
            outputs.append(y)
        
        return outputs, h, c


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("深度学习算法示例")
    print("=" * 60)
    
    # ========== 1. CNN示例 ==========
    print("\n【1. 卷积神经网络 CNN】")
    print("-" * 60)
    
    # 加载手写数字数据集
    digits = load_digits()
    X = digits.data.reshape(-1, 1, 8, 8) / 16.0  # 归一化
    y = digits.target
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建CNN
    cnn = SimpleCNN(input_shape=(1, 8, 8), num_classes=10)
    
    # 前向传播（演示）
    print("\n执行前向传播...")
    sample_batch = X_train[:10]
    output = cnn.forward(sample_batch)
    predictions = np.argmax(output, axis=1)
    print(f"预测结果: {predictions}")
    print(f"真实标签: {y_train[:10]}")
    
    # ========== 2. RNN示例 ==========
    print("\n【2. 循环神经网络 RNN】")
    print("-" * 60)
    
    # 创建简单的序列数据
    # 任务：学习序列模式 [0,1,2] -> 3
    print("创建序列预测任务...")
    
    # 创建RNN
    rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=4)
    
    # 示例序列
    sequence = np.array([[0], [1], [2]])
    print(f"输入序列: {sequence.flatten()}")
    
    # 预测
    outputs = rnn.predict(sequence)
    predicted_next = np.argmax(outputs[-1])
    print(f"预测下一个数字: {predicted_next}")
    
    # ========== 3. LSTM示例 ==========
    print("\n【3. 长短期记忆网络 LSTM】")
    print("-" * 60)
    
    # 创建LSTM
    lstm = LSTM(input_size=1, hidden_size=10, output_size=4)
    
    # 初始状态
    h_prev = np.zeros(10)
    c_prev = np.zeros(10)
    
    # 前向传播
    outputs, h_last, c_last = lstm.forward(sequence, h_prev, c_prev)
    predicted_next_lstm = np.argmax(outputs[-1])
    print(f"LSTM预测下一个数字: {predicted_next_lstm}")
    
    # ========== 可视化 ==========
    print("\n绘制CNN特征图和数字样本...")
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    sample_images = X_train[:16].reshape(16, 8, 8)
    grid = np.zeros((4*8, 4*8))
    for i in range(4):
        for j in range(4):
            grid[i*8:(i+1)*8, j*8:(j+1)*8] = sample_images[i*4+j]
    plt.imshow(grid, cmap='gray')
    plt.title('手写数字样本')
    plt.axis('off')
    
    # 显示卷积核
    plt.subplot(1, 3, 2)
    filters = cnn.conv1.filters[0, 0]  # 第一个卷积核
    plt.imshow(filters, cmap='viridis')
    plt.title('第一层卷积核')
    plt.colorbar()
    
    # 显示RNN/LSTM架构
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.7, 'RNN架构', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.5, f'输入层: {rnn.input_size}维', ha='center', fontsize=10)
    plt.text(0.5, 0.4, f'隐藏层: {rnn.hidden_size}维', ha='center', fontsize=10)
    plt.text(0.5, 0.3, f'输出层: {rnn.output_size}维', ha='center', fontsize=10)
    plt.text(0.5, 0.1, '循环连接: h_t = f(x_t, h_{t-1})', ha='center', fontsize=9, style='italic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('RNN结构')
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    example_usage()