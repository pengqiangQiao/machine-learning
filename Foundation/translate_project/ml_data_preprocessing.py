"""
机器学习数据预处理模块
Data Preprocessing for Machine Learning

Java对应实现：可以使用Apache Commons Math、Weka或DL4J库
Java equivalent: Use Apache Commons Math, Weka, or DL4J libraries
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Java对应：import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
# Java对应：import weka.core.Instances;


class DataPreprocessor:
    """
    数据预处理类
    
    Java对应实现：
    public class DataPreprocessor {
        private double[][] data;
        private String[] labels;
    }
    """
    
    def __init__(self):
        """
        初始化预处理器
        
        Java对应：
        public DataPreprocessor() {
            this.scaler = new StandardScaler();
        }
        """
        self.scaler = None
        self.label_encoder = None
    
    def load_data(self, filepath):
        """
        加载数据
        
        Java对应：
        public DataFrame loadData(String filepath) throws IOException {
            return new CsvReader().read(new File(filepath));
        }
        
        Args:
            filepath: 文件路径
        Returns:
            DataFrame: pandas数据框
        """
        # Python使用pandas读取CSV
        data = pd.read_csv(filepath)
        print(f"数据形状: {data.shape}")
        print(f"数据列: {data.columns.tolist()}")
        return data
    
    def handle_missing_values(self, data, strategy='mean'):
        """
        处理缺失值
        
        Java对应：
        public double[][] handleMissingValues(double[][] data, String strategy) {
            for (int col = 0; col < data[0].length; col++) {
                double mean = calculateMean(data, col);
                for (int row = 0; row < data.length; row++) {
                    if (Double.isNaN(data[row][col])) {
                        data[row][col] = mean;
                    }
                }
            }
            return data;
        }
        
        Args:
            data: 数据框
            strategy: 填充策略 ('mean', 'median', 'mode')
        Returns:
            处理后的数据
        """
        if strategy == 'mean':
            # 用均值填充（只对数值列）
            # Java: Arrays.stream(column).average().getAsDouble()
            return data.fillna(data.mean(numeric_only=True))
        elif strategy == 'median':
            # 用中位数填充（只对数值列）
            # Java: Arrays.stream(column).sorted().skip(n/2).findFirst()
            return data.fillna(data.median(numeric_only=True))
        elif strategy == 'mode':
            # 用众数填充
            return data.fillna(data.mode().iloc[0])
        else:
            # 删除缺失值
            # Java: data = removeNullRows(data)
            return data.dropna()
    
    def normalize_data(self, data, method='standard'):
        """
        数据标准化/归一化
        
        Java对应：
        public double[][] normalizeData(double[][] data, String method) {
            if (method.equals("standard")) {
                // 标准化: (x - mean) / std
                double mean = calculateMean(data);
                double std = calculateStd(data);
                for (int i = 0; i < data.length; i++) {
                    for (int j = 0; j < data[i].length; j++) {
                        data[i][j] = (data[i][j] - mean) / std;
                    }
                }
            }
            return data;
        }
        
        Args:
            data: 输入数据
            method: 标准化方法 ('standard' 或 'minmax')
        Returns:
            标准化后的数据
        """
        if method == 'standard':
            # 标准化: (x - μ) / σ
            # Java: 使用Apache Commons Math的StandardDeviation
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(data)
        elif method == 'minmax':
            # 归一化到[0,1]: (x - min) / (max - min)
            # Java: normalized = (value - min) / (max - min)
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(data)
    
    def encode_categorical(self, data, column):
        """
        编码分类变量
        
        Java对应：
        public int[] encodeCategorical(String[] categories) {
            Map<String, Integer> encoder = new HashMap<>();
            int[] encoded = new int[categories.length];
            int index = 0;
            
            for (int i = 0; i < categories.length; i++) {
                if (!encoder.containsKey(categories[i])) {
                    encoder.put(categories[i], index++);
                }
                encoded[i] = encoder.get(categories[i]);
            }
            return encoded;
        }
        
        Args:
            data: 数据框
            column: 要编码的列名
        Returns:
            编码后的数据
        """
        self.label_encoder = LabelEncoder()
        # 将类别转换为数字
        # 例如: ['red', 'blue', 'green'] -> [0, 1, 2]
        data[column] = self.label_encoder.fit_transform(data[column])
        return data
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        划分训练集和测试集
        
        Java对应：
        public DataSplit splitData(double[][] X, double[] y, double testSize) {
            int trainSize = (int)(X.length * (1 - testSize));
            double[][] X_train = Arrays.copyOfRange(X, 0, trainSize);
            double[][] X_test = Arrays.copyOfRange(X, trainSize, X.length);
            double[] y_train = Arrays.copyOfRange(y, 0, trainSize);
            double[] y_test = Arrays.copyOfRange(y, trainSize, y.length);
            
            return new DataSplit(X_train, X_test, y_train, y_test);
        }
        
        Args:
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            random_state: 随机种子
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Python使用sklearn的train_test_split
        # Java需要手动实现或使用Weka的RemovePercentage
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def create_feature_matrix(self, data, feature_columns):
        """
        创建特征矩阵
        
        Java对应：
        public double[][] createFeatureMatrix(DataFrame data, String[] featureColumns) {
            double[][] features = new double[data.getRowCount()][featureColumns.length];
            for (int i = 0; i < data.getRowCount(); i++) {
                for (int j = 0; j < featureColumns.length; j++) {
                    features[i][j] = data.getValue(i, featureColumns[j]);
                }
            }
            return features;
        }
        
        Args:
            data: 数据框
            feature_columns: 特征列名列表
        Returns:
            numpy数组形式的特征矩阵
        """
        # 选择指定列作为特征
        # Java: 需要遍历DataFrame并提取指定列
        return data[feature_columns].values


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) {
        DataPreprocessor preprocessor = new DataPreprocessor();
        // ... 使用预处理器
    }
    """
    # 创建示例数据
    # Java: double[][] data = new double[100][3];
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # 初始化预处理器
    # Java: DataPreprocessor preprocessor = new DataPreprocessor();
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    # Java: data = preprocessor.handleMissingValues(data, "mean");
    data = preprocessor.handle_missing_values(data)
    
    # 编码分类变量
    # Java: data = preprocessor.encodeCategorical(data, "category");
    data = preprocessor.encode_categorical(data, 'category')
    
    # 准备特征和标签
    X = data[['feature1', 'feature2', 'category']].values
    y = data['target'].values
    
    # 标准化特征
    # Java: X = preprocessor.normalizeData(X, "standard");
    X = preprocessor.normalize_data(X, method='standard')
    
    # 划分数据集
    # Java: DataSplit split = preprocessor.splitData(X, y, 0.2);
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")


if __name__ == "__main__":
    """
    Java对应：
    public static void main(String[] args) {
        exampleUsage();
    }
    """
    example_usage()