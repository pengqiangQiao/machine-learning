"""
决策树模型实现
Decision Tree Implementation

Java对应实现：可以使用Weka的J48(C4.5)或ID3算法
Java equivalent: Use Weka's J48 (C4.5) or implement ID3 algorithm
"""

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()

# Java对应：import weka.classifiers.trees.J48;


class Node:
    """
    决策树节点
    
    Java对应实现：
    public class Node {
        private int feature;        // 分裂特征索引
        private double threshold;   // 分裂阈值
        private Node left;          // 左子树
        private Node right;         // 右子树
        private Integer value;      // 叶节点的预测值
        
        public Node(int feature, double threshold) {
            this.feature = feature;
            this.threshold = threshold;
        }
        
        public boolean isLeaf() {
            return value != null;
        }
    }
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Java对应：
        public Node(Integer feature, Double threshold, Node left, Node right, Integer value) {
            this.feature = feature;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.value = value;
        }
        """
        self.feature = feature      # 分裂特征
        self.threshold = threshold  # 分裂阈值
        self.left = left           # 左子树
        self.right = right         # 右子树
        self.value = value         # 叶节点值
    
    def is_leaf(self):
        """
        判断是否为叶节点
        
        Java对应：
        public boolean isLeaf() {
            return this.value != null;
        }
        """
        return self.value is not None


class DecisionTree:
    """
    决策树分类器（CART算法）
    
    Java对应实现：
    public class DecisionTree {
        private Node root;
        private int maxDepth;
        private int minSamplesSplit;
        
        public DecisionTree(int maxDepth, int minSamplesSplit) {
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
        }
        
        // 计算基尼不纯度
        private double calculateGini(int[] labels) {
            Map<Integer, Integer> counts = new HashMap<>();
            for (int label : labels) {
                counts.put(label, counts.getOrDefault(label, 0) + 1);
            }
            
            double gini = 1.0;
            int total = labels.length;
            for (int count : counts.values()) {
                double prob = (double) count / total;
                gini -= prob * prob;
            }
            return gini;
        }
    }
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        初始化决策树
        
        Java对应：
        public DecisionTree(int maxDepth, int minSamplesSplit) {
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.root = null;
        }
        
        Args:
            max_depth: 最大深度
            min_samples_split: 分裂所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def gini_impurity(self, y):
        """
        计算基尼不纯度
        Gini = 1 - Σ(p_i)²
        
        Java对应：
        private double giniImpurity(int[] y) {
            Map<Integer, Integer> counts = new HashMap<>();
            for (int label : y) {
                counts.put(label, counts.getOrDefault(label, 0) + 1);
            }
            
            double gini = 1.0;
            int total = y.length;
            
            for (int count : counts.values()) {
                double prob = (double) count / total;
                gini -= prob * prob;
            }
            
            return gini;
        }
        
        Args:
            y: 标签数组
        Returns:
            基尼不纯度
        """
        # 统计每个类别的数量
        # Java: Map<Integer, Integer> counts = countLabels(y)
        counter = Counter(y)
        
        # 计算基尼不纯度
        # Java: gini = 1.0 - sum(prob * prob)
        gini = 1.0
        total = len(y)
        for count in counter.values():
            prob = count / total
            gini -= prob ** 2
        
        return gini
    
    def split_data(self, X, y, feature, threshold):
        """
        根据特征和阈值分裂数据
        
        Java对应：
        private DataSplit splitData(double[][] X, int[] y, int feature, double threshold) {
            List<Integer> leftIndices = new ArrayList<>();
            List<Integer> rightIndices = new ArrayList<>();
            
            for (int i = 0; i < X.length; i++) {
                if (X[i][feature] <= threshold) {
                    leftIndices.add(i);
                } else {
                    rightIndices.add(i);
                }
            }
            
            return new DataSplit(
                getSubset(X, leftIndices), getSubset(y, leftIndices),
                getSubset(X, rightIndices), getSubset(y, rightIndices)
            );
        }
        
        Args:
            X: 特征矩阵
            y: 标签
            feature: 分裂特征索引
            threshold: 分裂阈值
        Returns:
            左右子集
        """
        # 根据阈值分裂
        # Java: leftMask = X[:, feature] <= threshold
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
    
    def find_best_split(self, X, y):
        """
        找到最佳分裂点
        
        Java对应：
        private BestSplit findBestSplit(double[][] X, int[] y) {
            int bestFeature = -1;
            double bestThreshold = 0.0;
            double bestGain = -1.0;
            
            double currentGini = giniImpurity(y);
            
            // 遍历所有特征
            for (int feature = 0; feature < X[0].length; feature++) {
                // 获取该特征的所有唯一值作为候选阈值
                Set<Double> thresholds = getUniqueValues(X, feature);
                
                for (double threshold : thresholds) {
                    // 分裂数据
                    DataSplit split = splitData(X, y, feature, threshold);
                    
                    // 计算信息增益
                    double leftGini = giniImpurity(split.yLeft);
                    double rightGini = giniImpurity(split.yRight);
                    
                    int n = y.length;
                    int nLeft = split.yLeft.length;
                    int nRight = split.yRight.length;
                    
                    double weightedGini = (nLeft * leftGini + nRight * rightGini) / n;
                    double gain = currentGini - weightedGini;
                    
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = threshold;
                    }
                }
            }
            
            return new BestSplit(bestFeature, bestThreshold, bestGain);
        }
        
        Args:
            X: 特征矩阵
            y: 标签
        Returns:
            最佳特征索引、阈值和信息增益
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_gini = self.gini_impurity(y)
        n_features = X.shape[1]
        
        # 遍历所有特征
        # Java: for (int feature = 0; feature < n_features; feature++)
        for feature in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            # Java: Set<Double> thresholds = getUniqueValues(X, feature)
            thresholds = np.unique(X[:, feature])
            
            # 遍历所有阈值
            for threshold in thresholds:
                # 分裂数据
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # 计算加权基尼不纯度
                n = len(y)
                n_left, n_right = len(y_left), len(y_right)
                weighted_gini = (n_left * self.gini_impurity(y_left) + 
                               n_right * self.gini_impurity(y_right)) / n
                
                # 计算信息增益
                gain = current_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        
        Java对应：
        private Node buildTree(double[][] X, int[] y, int depth) {
            // 停止条件
            if (depth >= maxDepth || y.length < minSamplesSplit || isPure(y)) {
                int leafValue = getMostCommonLabel(y);
                return new Node(null, null, null, null, leafValue);
            }
            
            // 找到最佳分裂点
            BestSplit split = findBestSplit(X, y);
            
            if (split.gain <= 0) {
                int leafValue = getMostCommonLabel(y);
                return new Node(null, null, null, null, leafValue);
            }
            
            // 分裂数据
            DataSplit dataSplit = splitData(X, y, split.feature, split.threshold);
            
            // 递归构建左右子树
            Node left = buildTree(dataSplit.XLeft, dataSplit.yLeft, depth + 1);
            Node right = buildTree(dataSplit.XRight, dataSplit.yRight, depth + 1);
            
            return new Node(split.feature, split.threshold, left, right, null);
        }
        
        Args:
            X: 特征矩阵
            y: 标签
            depth: 当前深度
        Returns:
            节点
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        # Java: if (depth >= maxDepth || n_samples < minSamplesSplit || n_labels == 1)
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_labels == 1):
            # 创建叶节点，值为最常见的类别
            # Java: leafValue = getMostCommonLabel(y)
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # 找到最佳分裂点
        # Java: BestSplit split = findBestSplit(X, y)
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        # 如果没有信息增益，创建叶节点
        if best_gain <= 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # 分裂数据
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_feature, best_threshold)
        
        # 递归构建左右子树
        # Java: Node left = buildTree(X_left, y_left, depth + 1)
        left = self.build_tree(X_left, y_left, depth + 1)
        right = self.build_tree(X_right, y_right, depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def fit(self, X, y):
        """
        训练决策树
        
        Java对应：
        public void fit(double[][] X, int[] y) {
            this.root = buildTree(X, y, 0);
        }
        
        Args:
            X: 特征矩阵
            y: 标签
        """
        self.root = self.build_tree(X, y)
    
    def predict_sample(self, x, node):
        """
        预测单个样本
        
        Java对应：
        private int predictSample(double[] x, Node node) {
            if (node.isLeaf()) {
                return node.getValue();
            }
            
            if (x[node.getFeature()] <= node.getThreshold()) {
                return predictSample(x, node.getLeft());
            } else {
                return predictSample(x, node.getRight());
            }
        }
        
        Args:
            x: 单个样本
            node: 当前节点
        Returns:
            预测值
        """
        # 如果是叶节点，返回值
        # Java: if (node.isLeaf()) return node.getValue()
        if node.is_leaf():
            return node.value
        
        # 根据特征值决定走左子树还是右子树
        # Java: if (x[node.feature] <= node.threshold)
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """
        预测多个样本
        
        Java对应：
        public int[] predict(double[][] X) {
            int[] predictions = new int[X.length];
            for (int i = 0; i < X.length; i++) {
                predictions[i] = predictSample(X[i], root);
            }
            return predictions;
        }
        
        Args:
            X: 特征矩阵
        Returns:
            预测值数组
        """
        # Java: return Arrays.stream(X).map(x -> predictSample(x, root)).toArray()
        return np.array([self.predict_sample(x, self.root) for x in X])


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) throws Exception {
        // 生成示例数据
        double[][] X = generateData(200, 2);
        int[] y = new int[200];
        
        for (int i = 0; i < 200; i++) {
            y[i] = (X[i][0] * X[i][0] + X[i][1] * X[i][1] > 1) ? 1 : 0;
        }
        
        // 训练决策树
        DecisionTree tree = new DecisionTree(5, 2);
        tree.fit(X, y);
        
        // 预测
        int[] predictions = tree.predict(X);
        
        // 评估
        double accuracy = calculateAccuracy(y, predictions);
        System.out.println("Accuracy: " + accuracy);
    }
    """
    print("=" * 50)
    print("决策树示例")
    print("=" * 50)
    
    # 生成示例数据（非线性可分）
    # Java: 使用Random类生成数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成圆形分布的数据
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    
    # 划分训练集和测试集
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("\n1. 使用自定义决策树实现")
    print("-" * 50)
    # Java: DecisionTree customTree = new DecisionTree(5, 2);
    custom_tree = DecisionTree(max_depth=5, min_samples_split=2)
    custom_tree.fit(X_train, y_train)
    
    # 预测
    y_pred_custom = custom_tree.predict(X_test)
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    print(f"测试集准确率: {accuracy_custom:.4f}")
    
    print("\n2. 使用sklearn决策树")
    print("-" * 50)
    # Java: J48 sklearnTree = new J48();
    sklearn_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    sklearn_tree.fit(X_train, y_train)
    
    y_pred_sklearn = sklearn_tree.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"测试集准确率: {accuracy_sklearn:.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred_sklearn)}")
    
    # 可视化决策边界
    print("\n绘制决策树分类结果...")
    plt.figure(figsize=(12, 5))
    
    # 自定义决策树
    plt.subplot(1, 2, 1)
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
               color='blue', label='类别 0', alpha=0.6)
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
               color='red', label='类别 1', alpha=0.6)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(f'自定义决策树 (准确率: {accuracy_custom:.2f})')
    plt.legend()
    plt.grid(True)
    
    # sklearn决策树
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
               color='blue', label='类别 0', alpha=0.6)
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
               color='red', label='类别 1', alpha=0.6)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(f'Sklearn决策树 (准确率: {accuracy_sklearn:.2f})')
    plt.legend()
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