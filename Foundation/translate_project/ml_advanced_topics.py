"""
机器学习高级主题
Advanced Machine Learning Topics

包含：EM算法、主题模型(LDA)、推荐系统、采样和变分推断、最大熵模型
Including: EM Algorithm, Topic Models (LDA), Recommender Systems, 
          Sampling and Variational Inference, Maximum Entropy Model

Java对应实现：可以使用Mallet、Mahout等库
Java equivalent: Use Mallet, Mahout, or implement from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
from scipy.stats import multivariate_normal

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== EM算法 Expectation-Maximization ====================

class GaussianMixtureEM:
    """
    高斯混合模型（使用EM算法）
    
    Java对应：
    public class GaussianMixtureEM {
        private int numComponents;
        private double[] weights;        // 混合权重
        private double[][] means;        // 均值
        private double[][][] covariances; // 协方差矩阵
        
        public void fit(double[][] X, int maxIterations);
        public int[] predict(double[][] X);
        public double[][] predictProba(double[][] X);
    }
    """
    
    def __init__(self, n_components=3, max_iterations=100, tolerance=1e-4):
        """
        初始化GMM
        
        Java对应：
        public GaussianMixtureEM(int numComponents, int maxIterations, double tolerance) {
            this.numComponents = numComponents;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
        }
        
        Args:
            n_components: 高斯分量数
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.means = None
        self.covariances = None
        
        print(f"GMM初始化: {n_components}个高斯分量")
    
    def _initialize_parameters(self, X):
        """
        初始化参数
        
        Java对应：
        private void initializeParameters(double[][] X) {
            int n = X.length;
            int d = X[0].length;
            
            // 随机初始化
            weights = new double[numComponents];
            Arrays.fill(weights, 1.0 / numComponents);
            
            means = new double[numComponents][d];
            covariances = new double[numComponents][d][d];
            
            // 随机选择样本作为初始均值
            Random random = new Random();
            for (int k = 0; k < numComponents; k++) {
                int idx = random.nextInt(n);
                means[k] = X[idx].clone();
                
                // 初始化为单位矩阵
                for (int i = 0; i < d; i++) {
                    covariances[k][i][i] = 1.0;
                }
            }
        }
        """
        n_samples, n_features = X.shape
        
        # 初始化权重（均匀分布）
        self.weights = np.ones(self.n_components) / self.n_components
        
        # 随机选择样本作为初始均值
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        
        # 初始化协方差矩阵为单位矩阵
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def _e_step(self, X):
        """
        E步：计算后验概率（责任）
        
        γ(z_nk) = π_k * N(x_n | μ_k, Σ_k) / Σ_j π_j * N(x_n | μ_j, Σ_j)
        
        Java对应：
        private double[][] eStep(double[][] X) {
            int n = X.length;
            double[][] responsibilities = new double[n][numComponents];
            
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                
                // 计算每个分量的概率
                for (int k = 0; k < numComponents; k++) {
                    double prob = weights[k] * gaussianPDF(X[i], means[k], covariances[k]);
                    responsibilities[i][k] = prob;
                    sum += prob;
                }
                
                // 归一化
                for (int k = 0; k < numComponents; k++) {
                    responsibilities[i][k] /= sum;
                }
            }
            
            return responsibilities;
        }
        
        Args:
            X: 数据
        Returns:
            responsibilities: 后验概率矩阵
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算高斯概率密度
            responsibilities[:, k] = self.weights[k] * \
                multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        
        # 归一化
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        M步：更新参数
        
        π_k = N_k / N
        μ_k = (1/N_k) * Σ_n γ(z_nk) * x_n
        Σ_k = (1/N_k) * Σ_n γ(z_nk) * (x_n - μ_k)(x_n - μ_k)^T
        
        Java对应：
        private void mStep(double[][] X, double[][] responsibilities) {
            int n = X.length;
            int d = X[0].length;
            
            for (int k = 0; k < numComponents; k++) {
                // 计算N_k
                double Nk = 0.0;
                for (int i = 0; i < n; i++) {
                    Nk += responsibilities[i][k];
                }
                
                // 更新权重
                weights[k] = Nk / n;
                
                // 更新均值
                double[] newMean = new double[d];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < d; j++) {
                        newMean[j] += responsibilities[i][k] * X[i][j];
                    }
                }
                for (int j = 0; j < d; j++) {
                    newMean[j] /= Nk;
                }
                means[k] = newMean;
                
                // 更新协方差
                double[][] newCov = new double[d][d];
                for (int i = 0; i < n; i++) {
                    double[] diff = subtract(X[i], means[k]);
                    for (int j1 = 0; j1 < d; j1++) {
                        for (int j2 = 0; j2 < d; j2++) {
                            newCov[j1][j2] += responsibilities[i][k] * diff[j1] * diff[j2];
                        }
                    }
                }
                for (int j1 = 0; j1 < d; j1++) {
                    for (int j2 = 0; j2 < d; j2++) {
                        newCov[j1][j2] /= Nk;
                    }
                }
                covariances[k] = newCov;
            }
        }
        
        Args:
            X: 数据
            responsibilities: 后验概率矩阵
        """
        n_samples, n_features = X.shape
        
        # 计算N_k
        Nk = responsibilities.sum(axis=0)
        
        # 更新权重
        self.weights = Nk / n_samples
        
        # 更新均值
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            # 添加正则化项防止奇异
            self.covariances[k] += np.eye(n_features) * 1e-6
    
    def fit(self, X):
        """
        训练GMM
        
        Java对应：
        public void fit(double[][] X) {
            initializeParameters(X);
            
            double prevLogLikelihood = Double.NEGATIVE_INFINITY;
            
            for (int iter = 0; iter < maxIterations; iter++) {
                // E步
                double[][] responsibilities = eStep(X);
                
                // M步
                mStep(X, responsibilities);
                
                // 计算对数似然
                double logLikelihood = computeLogLikelihood(X);
                
                if (Math.abs(logLikelihood - prevLogLikelihood) < tolerance) {
                    System.out.println("Converged at iteration " + iter);
                    break;
                }
                
                prevLogLikelihood = logLikelihood;
            }
        }
        
        Args:
            X: 训练数据
        """
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # E步
            responsibilities = self._e_step(X)
            
            # M步
            self._m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 对数似然: {log_likelihood:.4f}")
            
            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"在第 {iteration} 次迭代时收敛")
                break
            
            prev_log_likelihood = log_likelihood
    
    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            prob = 0
            for k in range(self.n_components):
                prob += self.weights[k] * \
                    multivariate_normal.pdf(X[i], self.means[k], self.covariances[k])
            log_likelihood += np.log(prob + 1e-10)
        
        return log_likelihood
    
    def predict(self, X):
        """
        预测聚类标签
        
        Args:
            X: 测试数据
        Returns:
            聚类标签
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)


# ==================== 主题模型 LDA ====================

class LatentDirichletAllocation:
    """
    隐狄利克雷分配（LDA）主题模型
    
    Java对应：
    import cc.mallet.topics.ParallelTopicModel;
    
    public class LatentDirichletAllocation {
        private int numTopics;
        private int numWords;
        private double alpha;  // 文档-主题Dirichlet先验
        private double beta;   // 主题-词Dirichlet先验
        private double[][] phi;    // 主题-词分布
        private double[][] theta;  // 文档-主题分布
        
        public void fit(int[][] documents, int maxIterations);
        public int[] transform(int[] document);
    }
    """
    
    def __init__(self, n_topics=10, alpha=0.1, beta=0.01, max_iterations=100):
        """
        初始化LDA
        
        Java对应：
        public LatentDirichletAllocation(int numTopics, double alpha, 
                                        double beta, int maxIterations) {
            this.numTopics = numTopics;
            this.alpha = alpha;
            this.beta = beta;
            this.maxIterations = maxIterations;
        }
        
        Args:
            n_topics: 主题数
            alpha: 文档-主题Dirichlet先验
            beta: 主题-词Dirichlet先验
            max_iterations: 最大迭代次数
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        
        print(f"LDA初始化: {n_topics}个主题")
    
    def fit(self, documents, vocabulary_size):
        """
        训练LDA（使用Gibbs采样）
        
        Java对应：
        public void fit(int[][] documents, int vocabularySize) {
            this.numWords = vocabularySize;
            
            // 初始化
            initializeTopicAssignments(documents);
            
            // Gibbs采样
            for (int iter = 0; iter < maxIterations; iter++) {
                for (int d = 0; d < documents.length; d++) {
                    for (int n = 0; n < documents[d].length; n++) {
                        // 移除当前词的主题分配
                        int word = documents[d][n];
                        int topic = topicAssignments[d][n];
                        decrementCounts(d, word, topic);
                        
                        // 重新采样主题
                        int newTopic = sampleTopic(d, word);
                        topicAssignments[d][n] = newTopic;
                        incrementCounts(d, word, newTopic);
                    }
                }
                
                if (iter % 10 == 0) {
                    double perplexity = computePerplexity(documents);
                    System.out.println("Iteration " + iter + ", Perplexity: " + perplexity);
                }
            }
            
            // 计算最终的主题-词和文档-主题分布
            computeDistributions();
        }
        
        Args:
            documents: 文档集合（词ID列表的列表）
            vocabulary_size: 词汇表大小
        """
        self.vocabulary_size = vocabulary_size
        n_documents = len(documents)
        
        # 初始化计数矩阵
        self.doc_topic_count = np.zeros((n_documents, self.n_topics))
        self.topic_word_count = np.zeros((self.n_topics, vocabulary_size))
        self.topic_count = np.zeros(self.n_topics)
        
        # 随机初始化主题分配
        self.topic_assignments = []
        for d, doc in enumerate(documents):
            doc_topics = []
            for word in doc:
                topic = np.random.randint(0, self.n_topics)
                doc_topics.append(topic)
                
                # 更新计数
                self.doc_topic_count[d, topic] += 1
                self.topic_word_count[topic, word] += 1
                self.topic_count[topic] += 1
            
            self.topic_assignments.append(doc_topics)
        
        # Gibbs采样
        print("开始Gibbs采样...")
        for iteration in range(self.max_iterations):
            for d, doc in enumerate(documents):
                for n, word in enumerate(doc):
                    # 移除当前词的主题分配
                    old_topic = self.topic_assignments[d][n]
                    self.doc_topic_count[d, old_topic] -= 1
                    self.topic_word_count[old_topic, word] -= 1
                    self.topic_count[old_topic] -= 1
                    
                    # 计算每个主题的概率
                    p_topic = np.zeros(self.n_topics)
                    for k in range(self.n_topics):
                        p_topic[k] = (self.doc_topic_count[d, k] + self.alpha) * \
                                    (self.topic_word_count[k, word] + self.beta) / \
                                    (self.topic_count[k] + vocabulary_size * self.beta)
                    
                    # 归一化并采样
                    p_topic /= p_topic.sum()
                    new_topic = np.random.choice(self.n_topics, p=p_topic)
                    
                    # 更新主题分配和计数
                    self.topic_assignments[d][n] = new_topic
                    self.doc_topic_count[d, new_topic] += 1
                    self.topic_word_count[new_topic, word] += 1
                    self.topic_count[new_topic] += 1
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}")
        
        # 计算最终分布
        self.phi = (self.topic_word_count + self.beta) / \
                   (self.topic_count[:, np.newaxis] + vocabulary_size * self.beta)
        self.theta = (self.doc_topic_count + self.alpha) / \
                     (self.doc_topic_count.sum(axis=1, keepdims=True) + self.n_topics * self.alpha)
        
        print("LDA训练完成")
    
    def get_top_words(self, n_words=10):
        """
        获取每个主题的高频词
        
        Args:
            n_words: 每个主题返回的词数
        Returns:
            每个主题的top词索引
        """
        top_words = []
        for k in range(self.n_topics):
            top_indices = np.argsort(self.phi[k])[-n_words:][::-1]
            top_words.append(top_indices)
        return top_words


# ==================== 推荐系统 Recommender Systems ====================

class CollaborativeFiltering:
    """
    协同过滤推荐系统
    
    Java对应：
    import org.apache.mahout.cf.taste.recommender.Recommender;
    
    public class CollaborativeFiltering {
        private double[][] userItemMatrix;
        private double[][] userFactors;
        private double[][] itemFactors;
        private int numFactors;
        
        public void fit(double[][] ratings, int maxIterations);
        public double predict(int userId, int itemId);
        public int[] recommend(int userId, int topN);
    }
    """
    
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.01, max_iterations=100):
        """
        初始化协同过滤（矩阵分解）
        
        Java对应：
        public CollaborativeFiltering(int numFactors, double learningRate,
                                     double regularization, int maxIterations) {
            this.numFactors = numFactors;
            this.learningRate = learningRate;
            this.regularization = regularization;
            this.maxIterations = maxIterations;
        }
        
        Args:
            n_factors: 隐因子数
            learning_rate: 学习率
            regularization: 正则化系数
            max_iterations: 最大迭代次数
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iterations = max_iterations
        
        print(f"协同过滤初始化: {n_factors}个隐因子")
    
    def fit(self, ratings):
        """
        训练推荐模型（矩阵分解）
        
        目标：最小化 ||R - UV^T||² + λ(||U||² + ||V||²)
        
        Java对应：
        public void fit(double[][] ratings) {
            int numUsers = ratings.length;
            int numItems = ratings[0].length;
            
            // 初始化用户和物品因子矩阵
            userFactors = initializeMatrix(numUsers, numFactors);
            itemFactors = initializeMatrix(numItems, numFactors);
            
            // 随机梯度下降
            for (int iter = 0; iter < maxIterations; iter++) {
                double totalError = 0.0;
                
                for (int i = 0; i < numUsers; i++) {
                    for (int j = 0; j < numItems; j++) {
                        if (ratings[i][j] > 0) {  // 只考虑已评分项
                            // 预测评分
                            double prediction = dotProduct(userFactors[i], itemFactors[j]);
                            double error = ratings[i][j] - prediction;
                            totalError += error * error;
                            
                            // 更新因子
                            for (int k = 0; k < numFactors; k++) {
                                double userGrad = -2 * error * itemFactors[j][k] + 
                                                 2 * regularization * userFactors[i][k];
                                double itemGrad = -2 * error * userFactors[i][k] + 
                                                 2 * regularization * itemFactors[j][k];
                                
                                userFactors[i][k] -= learningRate * userGrad;
                                itemFactors[j][k] -= learningRate * itemGrad;
                            }
                        }
                    }
                }
                
                if (iter % 10 == 0) {
                    double rmse = Math.sqrt(totalError / countNonZero(ratings));
                    System.out.println("Iteration " + iter + ", RMSE: " + rmse);
                }
            }
        }
        
        Args:
            ratings: 用户-物品评分矩阵
        """
        n_users, n_items = ratings.shape
        
        # 初始化因子矩阵
        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.01
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.01
        
        # 找出非零评分的位置
        nonzero_mask = ratings > 0
        
        print("开始训练推荐模型...")
        for iteration in range(self.max_iterations):
            total_error = 0
            count = 0
            
            # 随机梯度下降
            for i in range(n_users):
                for j in range(n_items):
                    if nonzero_mask[i, j]:
                        # 预测评分
                        prediction = np.dot(self.user_factors[i], self.item_factors[j])
                        error = ratings[i, j] - prediction
                        total_error += error ** 2
                        count += 1
                        
                        # 更新因子
                        user_grad = -2 * error * self.item_factors[j] + \
                                   2 * self.regularization * self.user_factors[i]
                        item_grad = -2 * error * self.user_factors[i] + \
                                   2 * self.regularization * self.item_factors[j]
                        
                        self.user_factors[i] -= self.learning_rate * user_grad
                        self.item_factors[j] -= self.learning_rate * item_grad
            
            rmse = np.sqrt(total_error / count)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, RMSE: {rmse:.4f}")
        
        print("推荐模型训练完成")
    
    def predict(self, user_id, item_id):
        """
        预测评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
        Returns:
            预测评分
        """
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def recommend(self, user_id, n_recommendations=5):
        """
        为用户推荐物品
        
        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
        Returns:
            推荐物品ID列表
        """
        # 计算该用户对所有物品的预测评分
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        # 返回评分最高的N个物品
        top_items = np.argsort(scores)[-n_recommendations:][::-1]
        return top_items


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("机器学习高级主题示例")
    print("=" * 60)
    
    # ========== 1. EM算法（GMM） ==========
    print("\n【1. EM算法 - 高斯混合模型】")
    print("-" * 60)
    
    # 生成混合高斯数据
    np.random.seed(42)
    n_samples = 300
    
    # 三个高斯分布
    X1 = np.random.randn(n_samples // 3, 2) + np.array([0, 0])
    X2 = np.random.randn(n_samples // 3, 2) + np.array([5, 5])
    X3 = np.random.randn(n_samples // 3, 2) + np.array([0, 5])
    X = np.vstack([X1, X2, X3])
    
    # 训练GMM
    gmm = GaussianMixtureEM(n_components=3, max_iterations=50)
    gmm.fit(X)
    
    # 预测聚类
    labels = gmm.predict(X)
    print(f"\n聚类结果: {np.bincount(labels)}")
    
    # ========== 2. LDA主题模型 ==========
    print("\n【2. LDA主题模型】")
    print("-" * 60)
    
    # 生成示例文档（词ID序列）
    vocabulary_size = 20
    documents = [
        [0, 1, 2, 0, 1, 3, 4],      # 文档1
        [5, 6, 7, 5, 6, 8],         # 文档2
        [0, 1, 2, 3, 0, 1],         # 文档3
        [5, 6, 7, 8, 9, 5],         # 文档4
        [10, 11, 12, 10, 11, 13],   # 文档5
    ]
    
    # 训练LDA
    lda = LatentDirichletAllocation(n_topics=3, max_iterations=50)
    lda.fit(documents, vocabulary_size)
    
    # 获取主题的高频词
    top_words = lda.get_top_words(n_words=5)
    print("\n每个主题的高频词:")
    for k, words in enumerate(top_words):
        print(f"主题 {k}: {words}")
    
    # ========== 3. 推荐系统 ==========
    print("\n【3. 协同过滤推荐系统】")
    print("-" * 60)
    
    # 创建用户-物品评分矩阵（0表示未评分）
    ratings = np.array([
        [5, 3, 0, 1, 0],
        [4, 0, 0, 1, 0],
        [1, 1, 0, 5, 0],
        [1, 0, 0, 4, 0],
        [0, 1, 5, 4, 0],
    ], dtype=float)
    
    print("原始评分矩阵:")
    print(ratings)
    
    # 训练推荐模型
    cf = CollaborativeFiltering(n_factors=5, max_iterations=100)
    cf.fit(ratings)
    
    # 为用户0推荐物品
    user_id = 0
    recommendations = cf.recommend(user_id, n_recommendations=3)
    print(f"\n为用户 {user_id} 推荐的物品: {recommendations}")
    
    # 预测评分
    print(f"预测用户 {user_id} 对物品 2 的评分: {cf.predict(user_id, 2):.2f}")
    
    # ========== 可视化 ==========
    print("\n绘制结果...")
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1：GMM聚类结果
    ax1 = fig.add_subplot(131)
    colors = ['red', 'blue', 'green']
    for k in range(3):
        cluster_points = X[labels == k]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=colors[k], label=f'簇 {k}', alpha=0.6)
    
    # 绘制高斯分布的均值
    for k in range(3):
        ax1.plot(gmm.means[k, 0], gmm.means[k, 1], 'k*', markersize=15)
    
    ax1.set_xlabel('特征 1')
    ax1.set_ylabel('特征 2')
    ax1.set_title('GMM聚类结果')
    ax1.legend()
    ax1.grid(True)
    
    # 子图2：LDA文档-主题分布
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(lda.theta, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('主题')
    ax2.set_ylabel('文档')
    ax2.set_title('LDA文档-主题分布')
    plt.colorbar(im, ax=ax2)
    
    # 子图3：推荐系统评分矩阵
    ax3 = fig.add_subplot(133)
    
    # 重构完整的评分矩阵
    predicted_ratings = np.dot(cf.user_factors, cf.item_factors.T)
    
    im = ax3.imshow(predicted_ratings, cmap='RdYlGn', aspect='auto')
    ax3.set_xlabel('物品')
    ax3.set_ylabel('用户')
    ax3.set_title('预测评分矩阵')
    plt.colorbar(im, ax=ax3)
    
    # 标记原始评分
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[1]):
            if ratings[i, j] > 0:
                ax3.text(j, i, f'{ratings[i,j]:.0f}',
                        ha="center", va="center", color="black", fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    example_usage()