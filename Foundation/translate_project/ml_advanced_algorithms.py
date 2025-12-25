"""
高级机器学习算法实现
Advanced Machine Learning Algorithms

包含：SVM、随机森林、提升算法、最大熵模型、贝叶斯网络
Including: SVM, Random Forest, Boosting, Maximum Entropy, Bayesian Networks

Java对应实现：可以使用Weka或自己实现
Java equivalent: Use Weka or implement from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 支持向量机 SVM ====================

class SimpleSVM:
    """
    简化的SVM实现（使用梯度下降）
    
    Java对应：
    import weka.classifiers.functions.SMO;
    
    public class SimpleSVM {
        private double[] weights;
        private double bias;
        private double C;  // 正则化参数
        
        public void fit(double[][] X, int[] y);
        public int[] predict(double[][] X);
    }
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, epochs=1000):
        """
        初始化SVM
        
        Java对应：
        public SimpleSVM(double C, double learningRate, int epochs) {
            this.C = C;
            this.learningRate = learningRate;
            this.epochs = epochs;
        }
        
        Args:
            C: 正则化参数
            learning_rate: 学习率
            epochs: 训练轮数
        """
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        训练SVM（使用Hinge损失）
        
        损失函数: L = (1/2)||w||² + C·Σmax(0, 1 - yᵢ(w·xᵢ + b))
        
        Java对应：
        public void fit(double[][] X, int[] y) {
            int n_samples = X.length;
            int n_features = X[0].length;
            
            // 初始化权重
            weights = new double[n_features];
            bias = 0.0;
            
            // 梯度下降
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < n_samples; i++) {
                    double decision = dotProduct(weights, X[i]) + bias;
                    
                    if (y[i] * decision < 1) {
                        // 更新权重（违反边界）
                        for (int j = 0; j < n_features; j++) {
                            weights[j] -= learningRate * (weights[j] - C * y[i] * X[i][j]);
                        }
                        bias -= learningRate * (-C * y[i]);
                    } else {
                        // 只更新正则化项
                        for (int j = 0; j < n_features; j++) {
                            weights[j] -= learningRate * weights[j];
                        }
                    }
                }
            }
        }
        
        Args:
            X: 训练数据
            y: 标签（必须是+1或-1）
        """
        n_samples, n_features = X.shape
        
        # 确保标签是+1和-1
        y_ = np.where(y <= 0, -1, 1)
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # 计算决策函数值
                decision = np.dot(X[i], self.weights) + self.bias
                
                # Hinge损失的次梯度
                if y_[i] * decision < 1:
                    # 违反边界，更新权重
                    self.weights -= self.learning_rate * (self.weights - self.C * y_[i] * X[i])
                    self.bias -= self.learning_rate * (-self.C * y_[i])
                else:
                    # 满足边界，只更新正则化项
                    self.weights -= self.learning_rate * self.weights
            
            if epoch % 100 == 0:
                loss = self._compute_loss(X, y_)
                print(f"轮次 {epoch}, 损失: {loss:.4f}")
    
    def _compute_loss(self, X, y):
        """计算Hinge损失"""
        decisions = np.dot(X, self.weights) + self.bias
        hinge_loss = np.maximum(0, 1 - y * decisions)
        return 0.5 * np.dot(self.weights, self.weights) + self.C * np.sum(hinge_loss)
    
    def predict(self, X):
        """
        预测
        
        Java对应：
        public int[] predict(double[][] X) {
            int[] predictions = new int[X.length];
            for (int i = 0; i < X.length; i++) {
                double decision = dotProduct(weights, X[i]) + bias;
                predictions[i] = decision >= 0 ? 1 : -1;
            }
            return predictions;
        }
        
        Args:
            X: 测试数据
        Returns:
            预测标签
        """
        decisions = np.dot(X, self.weights) + self.bias
        return np.where(decisions >= 0, 1, -1)
    
    def decision_function(self, X):
        """计算决策函数值"""
        return np.dot(X, self.weights) + self.bias


# ==================== 随机森林 Random Forest ====================

class SimpleRandomForest:
    """
    简化的随机森林实现
    
    Java对应：
    import weka.classifiers.trees.RandomForest;
    
    public class SimpleRandomForest {
        private DecisionTree[] trees;
        private int nTrees;
        private int maxDepth;
        private int maxFeatures;
        
        public void fit(double[][] X, int[] y);
        public int[] predict(double[][] X);
    }
    """
    
    def __init__(self, n_trees=10, max_depth=10, max_features=None, random_state=42):
        """
        初始化随机森林
        
        Java对应：
        public SimpleRandomForest(int nTrees, int maxDepth, 
                                 Integer maxFeatures, int randomState) {
            this.nTrees = nTrees;
            this.maxDepth = maxDepth;
            this.maxFeatures = maxFeatures;
            this.random = new Random(randomState);
            this.trees = new DecisionTree[nTrees];
        }
        
        Args:
            n_trees: 树的数量
            max_depth: 最大深度
            max_features: 每次分裂考虑的最大特征数
            random_state: 随机种子
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        """
        Bootstrap采样
        
        Java对应：
        private DataSample bootstrapSample(double[][] X, int[] y) {
            int n = X.length;
            double[][] X_sample = new double[n][];
            int[] y_sample = new int[n];
            
            for (int i = 0; i < n; i++) {
                int idx = random.nextInt(n);
                X_sample[i] = X[idx].clone();
                y_sample[i] = y[idx];
            }
            
            return new DataSample(X_sample, y_sample);
        }
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        训练随机森林
        
        Java对应：
        public void fit(double[][] X, int[] y) {
            for (int i = 0; i < nTrees; i++) {
                // Bootstrap采样
                DataSample sample = bootstrapSample(X, y);
                
                // 训练决策树
                DecisionTree tree = new DecisionTree(maxDepth, 2, maxFeatures);
                tree.fit(sample.X, sample.y);
                trees[i] = tree;
            }
        }
        
        Args:
            X: 训练数据
            y: 标签
        """
        np.random.seed(self.random_state)
        
        # 如果未指定max_features，使用sqrt(n_features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        
        print(f"训练 {self.n_trees} 棵决策树...")
        for i in range(self.n_trees):
            # Bootstrap采样
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 创建并训练决策树（这里使用简化版本）
            tree = self._create_simple_tree(X_sample, y_sample)
            self.trees.append(tree)
            
            if (i + 1) % 10 == 0:
                print(f"已训练 {i + 1}/{self.n_trees} 棵树")
    
    def _create_simple_tree(self, X, y):
        """创建简单的决策树（存储训练数据用于预测）"""
        return {'X': X, 'y': y}
    
    def predict(self, X):
        """
        预测（多数投票）
        
        Java对应：
        public int[] predict(double[][] X) {
            int[] predictions = new int[X.length];
            
            for (int i = 0; i < X.length; i++) {
                Map<Integer, Integer> votes = new HashMap<>();
                
                // 收集所有树的投票
                for (DecisionTree tree : trees) {
                    int prediction = tree.predict(X[i]);
                    votes.put(prediction, votes.getOrDefault(prediction, 0) + 1);
                }
                
                // 多数投票
                predictions[i] = Collections.max(votes.entrySet(),
                                                Map.Entry.comparingByValue()).getKey();
            }
            
            return predictions;
        }
        
        Args:
            X: 测试数据
        Returns:
            预测标签
        """
        predictions = []
        
        for x in X:
            # 收集所有树的预测
            tree_predictions = []
            for tree in self.trees:
                # 简化版：使用最近邻
                distances = np.sum((tree['X'] - x) ** 2, axis=1)
                nearest_idx = np.argmin(distances)
                tree_predictions.append(tree['y'][nearest_idx])
            
            # 多数投票
            prediction = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)


# ==================== 提升算法 Boosting ====================

class AdaBoost:
    """
    AdaBoost算法实现
    
    Java对应：
    import weka.classifiers.meta.AdaBoostM1;
    
    public class AdaBoost {
        private WeakClassifier[] classifiers;
        private double[] alphas;
        private int nEstimators;
        
        public void fit(double[][] X, int[] y);
        public int[] predict(double[][] X);
    }
    """
    
    def __init__(self, n_estimators=50):
        """
        初始化AdaBoost
        
        Java对应：
        public AdaBoost(int nEstimators) {
            this.nEstimators = nEstimators;
            this.classifiers = new WeakClassifier[nEstimators];
            this.alphas = new double[nEstimators];
        }
        
        Args:
            n_estimators: 弱分类器数量
        """
        self.n_estimators = n_estimators
        self.classifiers = []
        self.alphas = []
    
    def fit(self, X, y):
        """
        训练AdaBoost
        
        算法流程：
        1. 初始化样本权重 w = 1/n
        2. 对于每个弱分类器：
           a. 使用权重训练分类器
           b. 计算加权错误率 ε
           c. 计算分类器权重 α = 0.5 * ln((1-ε)/ε)
           d. 更新样本权重
        
        Java对应：
        public void fit(double[][] X, int[] y) {
            int n = X.length;
            double[] weights = new double[n];
            Arrays.fill(weights, 1.0 / n);
            
            for (int t = 0; t < nEstimators; t++) {
                // 训练弱分类器
                WeakClassifier clf = new DecisionStump();
                clf.fit(X, y, weights);
                
                // 计算加权错误率
                int[] predictions = clf.predict(X);
                double error = 0.0;
                for (int i = 0; i < n; i++) {
                    if (predictions[i] != y[i]) {
                        error += weights[i];
                    }
                }
                
                // 计算分类器权重
                double alpha = 0.5 * Math.log((1 - error) / (error + 1e-10));
                
                // 更新样本权重
                for (int i = 0; i < n; i++) {
                    weights[i] *= Math.exp(-alpha * y[i] * predictions[i]);
                }
                
                // 归一化权重
                double sum = Arrays.stream(weights).sum();
                for (int i = 0; i < n; i++) {
                    weights[i] /= sum;
                }
                
                classifiers[t] = clf;
                alphas[t] = alpha;
            }
        }
        
        Args:
            X: 训练数据
            y: 标签（必须是+1或-1）
        """
        n_samples = X.shape[0]
        
        # 确保标签是+1和-1
        y_ = np.where(y <= 0, -1, 1)
        
        # 初始化样本权重
        weights = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            # 训练弱分类器（这里使用决策树桩）
            clf = self._train_weak_classifier(X, y_, weights)
            
            # 预测
            predictions = clf['predict'](X)
            
            # 计算加权错误率
            incorrect = (predictions != y_)
            error = np.sum(weights[incorrect])
            
            # 避免除零
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # 计算分类器权重
            alpha = 0.5 * np.log((1 - error) / error)
            
            # 更新样本权重
            weights *= np.exp(-alpha * y_ * predictions)
            weights /= np.sum(weights)  # 归一化
            
            self.classifiers.append(clf)
            self.alphas.append(alpha)
            
            if (t + 1) % 10 == 0:
                print(f"训练第 {t + 1}/{self.n_estimators} 个弱分类器, 错误率: {error:.4f}")
    
    def _train_weak_classifier(self, X, y, weights):
        """训练弱分类器（决策树桩）"""
        n_samples, n_features = X.shape
        best_error = float('inf')
        best_feature = None
        best_threshold = None
        best_polarity = 1
        
        # 遍历所有特征
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                for polarity in [1, -1]:
                    # 预测
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1
                    
                    # 计算加权错误
                    error = np.sum(weights[predictions != y])
                    
                    if error < best_error:
                        best_error = error
                        best_feature = feature
                        best_threshold = threshold
                        best_polarity = polarity
        
        # 创建预测函数
        def predict(X):
            predictions = np.ones(len(X))
            if best_polarity == 1:
                predictions[X[:, best_feature] < best_threshold] = -1
            else:
                predictions[X[:, best_feature] >= best_threshold] = -1
            return predictions
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'polarity': best_polarity,
            'predict': predict
        }
    
    def predict(self, X):
        """
        预测
        
        Java对应：
        public int[] predict(double[][] X) {
            int[] predictions = new int[X.length];
            
            for (int i = 0; i < X.length; i++) {
                double sum = 0.0;
                for (int t = 0; t < nEstimators; t++) {
                    sum += alphas[t] * classifiers[t].predict(X[i]);
                }
                predictions[i] = sum >= 0 ? 1 : -1;
            }
            
            return predictions;
        }
        
        Args:
            X: 测试数据
        Returns:
            预测标签
        """
        # 加权投票
        predictions = np.zeros(len(X))
        for clf, alpha in zip(self.classifiers, self.alphas):
            predictions += alpha * clf['predict'](X)
        
        return np.where(predictions >= 0, 1, -1)


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("高级机器学习算法示例")
    print("=" * 60)
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成线性可分数据
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # 划分训练集和测试集
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # ========== 1. SVM ==========
    print("\n【1. 支持向量机 SVM】")
    print("-" * 60)
    
    # 自定义SVM
    print("训练自定义SVM...")
    custom_svm = SimpleSVM(C=1.0, learning_rate=0.001, epochs=1000)
    custom_svm.fit(X_train, y_train)
    
    y_pred_svm = custom_svm.predict(X_test)
    y_pred_svm = np.where(y_pred_svm == 1, 1, 0)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"自定义SVM准确率: {accuracy_svm:.4f}")
    
    # Sklearn SVM
    print("\n使用sklearn SVM...")
    sklearn_svm = SVC(kernel='linear', C=1.0)
    sklearn_svm.fit(X_train, y_train)
    y_pred_sklearn = sklearn_svm.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn SVM准确率: {accuracy_sklearn:.4f}")
    
    # ========== 2. 随机森林 ==========
    print("\n【2. 随机森林】")
    print("-" * 60)
    
    # Sklearn随机森林
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林准确率: {accuracy_rf:.4f}")
    print(f"特征重要性: {rf.feature_importances_}")
    
    # ========== 3. AdaBoost ==========
    print("\n【3. AdaBoost提升算法】")
    print("-" * 60)
    
    # 自定义AdaBoost
    print("训练AdaBoost...")
    adaboost = AdaBoost(n_estimators=50)
    adaboost.fit(X_train, y_train)
    
    y_pred_ada = adaboost.predict(X_test)
    y_pred_ada = np.where(y_pred_ada == 1, 1, 0)
    accuracy_ada = accuracy_score(y_test, y_pred_ada)
    print(f"AdaBoost准确率: {accuracy_ada:.4f}")
    
    # ========== 4. Gradient Boosting ==========
    print("\n【4. 梯度提升】")
    print("-" * 60)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print(f"梯度提升准确率: {accuracy_gb:.4f}")
    
    # ========== 可视化 ==========
    print("\n绘制决策边界...")
    plt.figure(figsize=(15, 5))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # SVM决策边界
    plt.subplot(1, 3, 1)
    Z = sklearn_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
               c='blue', label='类别 0', edgecolors='k')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
               c='red', label='类别 1', edgecolors='k')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(f'SVM (准确率: {accuracy_sklearn:.2f})')
    plt.legend()
    plt.grid(True)
    
    # 随机森林决策边界
    plt.subplot(1, 3, 2)
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
               c='blue', label='类别 0', edgecolors='k')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
               c='red', label='类别 1', edgecolors='k')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(f'随机森林 (准确率: {accuracy_rf:.2f})')
    plt.legend()
    plt.grid(True)
    
    # 梯度提升决策边界
    plt.subplot(1, 3, 3)
    Z = gb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
               c='blue', label='类别 0', edgecolors='k')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
               c='red', label='类别 1', edgecolors='k')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(f'梯度提升 (准确率: {accuracy_gb:.2f})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    example_usage()