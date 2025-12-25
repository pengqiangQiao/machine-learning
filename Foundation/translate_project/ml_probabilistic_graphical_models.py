"""
概率图模型实现
Probabilistic Graphical Models

包含：隐马尔可夫模型HMM、条件随机场CRF
Including: Hidden Markov Model (HMM), Conditional Random Field (CRF)

Java对应实现：可以使用自己实现或第三方库
Java equivalent: Implement from scratch or use third-party libraries
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()


# ==================== 隐马尔可夫模型 HMM ====================

class HiddenMarkovModel:
    """
    隐马尔可夫模型
    
    HMM包含三个基本问题：
    1. 评估问题：给定模型和观测序列，计算观测序列出现的概率（Forward算法）
    2. 解码问题：给定模型和观测序列，找出最可能的隐藏状态序列（Viterbi算法）
    3. 学习问题：给定观测序列，学习模型参数（Baum-Welch算法）
    
    Java对应：
    public class HiddenMarkovModel {
        private double[][] transitionProb;  // 状态转移概率矩阵 A
        private double[][] emissionProb;    // 发射概率矩阵 B
        private double[] initialProb;       // 初始状态概率 π
        private int numStates;              // 隐藏状态数
        private int numObservations;        // 观测符号数
        
        public HiddenMarkovModel(int numStates, int numObservations);
        public double forward(int[] observations);
        public int[] viterbi(int[] observations);
        public void baumWelch(int[] observations, int maxIterations);
    }
    """
    
    def __init__(self, n_states, n_observations):
        """
        初始化HMM
        
        Java对应：
        public HiddenMarkovModel(int numStates, int numObservations) {
            this.numStates = numStates;
            this.numObservations = numObservations;
            
            // 随机初始化参数
            this.transitionProb = initializeMatrix(numStates, numStates);
            this.emissionProb = initializeMatrix(numStates, numObservations);
            this.initialProb = initializeArray(numStates);
            
            // 归一化
            normalizeRows(transitionProb);
            normalizeRows(emissionProb);
            normalizeArray(initialProb);
        }
        
        Args:
            n_states: 隐藏状态数
            n_observations: 观测符号数
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # 随机初始化参数
        self.transition_prob = np.random.rand(n_states, n_states)
        self.emission_prob = np.random.rand(n_states, n_observations)
        self.initial_prob = np.random.rand(n_states)
        
        # 归一化
        self.transition_prob /= self.transition_prob.sum(axis=1, keepdims=True)
        self.emission_prob /= self.emission_prob.sum(axis=1, keepdims=True)
        self.initial_prob /= self.initial_prob.sum()
        
        print(f"HMM初始化: {n_states}个隐藏状态, {n_observations}个观测符号")
    
    def forward(self, observations):
        """
        Forward算法：计算观测序列的概率
        
        前向概率: α_t(i) = P(o_1,...,o_t, q_t=i | λ)
        递推公式: α_t(j) = [Σ_i α_{t-1}(i) * a_{ij}] * b_j(o_t)
        
        Java对应：
        public double forward(int[] observations) {
            int T = observations.length;
            double[][] alpha = new double[T][numStates];
            
            // 初始化
            for (int i = 0; i < numStates; i++) {
                alpha[0][i] = initialProb[i] * emissionProb[i][observations[0]];
            }
            
            // 递推
            for (int t = 1; t < T; t++) {
                for (int j = 0; j < numStates; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < numStates; i++) {
                        sum += alpha[t-1][i] * transitionProb[i][j];
                    }
                    alpha[t][j] = sum * emissionProb[j][observations[t]];
                }
            }
            
            // 计算总概率
            double prob = 0.0;
            for (int i = 0; i < numStates; i++) {
                prob += alpha[T-1][i];
            }
            
            return prob;
        }
        
        Args:
            observations: 观测序列
        Returns:
            观测序列的概率
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # 初始化
        alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_prob[:, j]) * \
                             self.emission_prob[j, observations[t]]
        
        # 返回总概率
        return np.sum(alpha[-1])
    
    def viterbi(self, observations):
        """
        Viterbi算法：找出最可能的隐藏状态序列
        
        维特比变量: δ_t(i) = max P(q_1,...,q_{t-1}, q_t=i, o_1,...,o_t | λ)
        递推公式: δ_t(j) = max_i [δ_{t-1}(i) * a_{ij}] * b_j(o_t)
        
        Java对应：
        public int[] viterbi(int[] observations) {
            int T = observations.length;
            double[][] delta = new double[T][numStates];
            int[][] psi = new int[T][numStates];
            
            // 初始化
            for (int i = 0; i < numStates; i++) {
                delta[0][i] = initialProb[i] * emissionProb[i][observations[0]];
                psi[0][i] = 0;
            }
            
            // 递推
            for (int t = 1; t < T; t++) {
                for (int j = 0; j < numStates; j++) {
                    double maxProb = Double.NEGATIVE_INFINITY;
                    int maxState = 0;
                    
                    for (int i = 0; i < numStates; i++) {
                        double prob = delta[t-1][i] * transitionProb[i][j];
                        if (prob > maxProb) {
                            maxProb = prob;
                            maxState = i;
                        }
                    }
                    
                    delta[t][j] = maxProb * emissionProb[j][observations[t]];
                    psi[t][j] = maxState;
                }
            }
            
            // 回溯
            int[] path = new int[T];
            path[T-1] = argmax(delta[T-1]);
            
            for (int t = T-2; t >= 0; t--) {
                path[t] = psi[t+1][path[t+1]];
            }
            
            return path;
        }
        
        Args:
            observations: 观测序列
        Returns:
            最可能的隐藏状态序列
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化
        delta[0] = self.initial_prob * self.emission_prob[:, observations[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                # 找到最大概率和对应的前一状态
                prob = delta[t-1] * self.transition_prob[:, j]
                psi[t, j] = np.argmax(prob)
                delta[t, j] = prob[psi[t, j]] * self.emission_prob[j, observations[t]]
        
        # 回溯最优路径
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path
    
    def baum_welch(self, observations, max_iterations=100, tolerance=1e-6):
        """
        Baum-Welch算法（EM算法）：学习HMM参数
        
        E步：计算前向和后向概率
        M步：更新参数
        
        Java对应：
        public void baumWelch(int[] observations, int maxIterations, double tolerance) {
            for (int iter = 0; iter < maxIterations; iter++) {
                // E步：计算前向和后向概率
                double[][] alpha = computeForward(observations);
                double[][] beta = computeBackward(observations);
                
                // 计算gamma和xi
                double[][] gamma = computeGamma(alpha, beta);
                double[][][] xi = computeXi(observations, alpha, beta);
                
                // M步：更新参数
                updateParameters(observations, gamma, xi);
                
                // 检查收敛
                double logLikelihood = computeLogLikelihood(observations);
                if (Math.abs(logLikelihood - prevLogLikelihood) < tolerance) {
                    break;
                }
                prevLogLikelihood = logLikelihood;
            }
        }
        
        Args:
            observations: 观测序列
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        T = len(observations)
        
        for iteration in range(max_iterations):
            # E步：前向-后向算法
            alpha = self._forward_prob(observations)
            beta = self._backward_prob(observations)
            
            # 计算gamma和xi
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(observations, alpha, beta)
            
            # M步：更新参数
            # 更新初始概率
            self.initial_prob = gamma[0]
            
            # 更新转移概率
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.transition_prob[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
            
            # 更新发射概率
            for j in range(self.n_states):
                for k in range(self.n_observations):
                    mask = (observations == k)
                    self.emission_prob[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])
            
            # 计算对数似然
            log_likelihood = np.log(self.forward(observations) + 1e-10)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 对数似然: {log_likelihood:.4f}")
            
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < tolerance:
                print(f"在第 {iteration} 次迭代时收敛")
                break
            
            prev_log_likelihood = log_likelihood
    
    def _forward_prob(self, observations):
        """计算前向概率"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_prob[:, j]) * \
                             self.emission_prob[j, observations[t]]
        
        return alpha
    
    def _backward_prob(self, observations):
        """计算后向概率"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        beta[-1] = 1
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_prob[i] * 
                                   self.emission_prob[:, observations[t+1]] * 
                                   beta[t+1])
        
        return beta
    
    def _compute_gamma(self, alpha, beta):
        """计算gamma（状态后验概率）"""
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
    
    def _compute_xi(self, observations, alpha, beta):
        """计算xi（状态转移后验概率）"""
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = alpha[t, i] * self.transition_prob[i, j] * \
                                 self.emission_prob[j, observations[t+1]] * beta[t+1, j]
            
            xi[t] /= xi[t].sum()
        
        return xi


# ==================== 条件随机场 CRF ====================

class LinearChainCRF:
    """
    线性链条件随机场
    
    CRF是判别式模型，直接建模条件概率 P(y|x)
    
    Java对应：
    public class LinearChainCRF {
        private double[][] transitionWeights;  // 转移特征权重
        private double[][] emissionWeights;    // 发射特征权重
        private int numStates;
        private int numFeatures;
        
        public void train(int[][] observations, int[][] labels, int maxIterations);
        public int[] predict(int[] observations);
    }
    """
    
    def __init__(self, n_states, n_features):
        """
        初始化CRF
        
        Java对应：
        public LinearChainCRF(int numStates, int numFeatures) {
            this.numStates = numStates;
            this.numFeatures = numFeatures;
            
            // 初始化权重
            this.transitionWeights = new double[numStates][numStates];
            this.emissionWeights = new double[numStates][numFeatures];
        }
        
        Args:
            n_states: 状态数
            n_features: 特征数
        """
        self.n_states = n_states
        self.n_features = n_features
        
        # 初始化权重
        self.transition_weights = np.random.randn(n_states, n_states) * 0.01
        self.emission_weights = np.random.randn(n_states, n_features) * 0.01
        
        print(f"CRF初始化: {n_states}个状态, {n_features}个特征")
    
    def _compute_scores(self, observations):
        """
        计算特征得分
        
        Args:
            observations: 观测序列
        Returns:
            emission_scores, transition_scores
        """
        T = len(observations)
        
        # 发射得分
        emission_scores = np.zeros((T, self.n_states))
        for t in range(T):
            emission_scores[t] = self.emission_weights[:, observations[t]]
        
        return emission_scores, self.transition_weights
    
    def forward_backward(self, observations):
        """
        前向-后向算法（用于CRF）
        
        Args:
            observations: 观测序列
        Returns:
            alpha, beta, Z (配分函数)
        """
        T = len(observations)
        emission_scores, transition_scores = self._compute_scores(observations)
        
        # 前向
        alpha = np.zeros((T, self.n_states))
        alpha[0] = emission_scores[0]
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = logsumexp(alpha[t-1] + transition_scores[:, j]) + emission_scores[t, j]
        
        # 配分函数
        Z = logsumexp(alpha[-1])
        
        # 后向
        beta = np.zeros((T, self.n_states))
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = logsumexp(transition_scores[i] + emission_scores[t+1] + beta[t+1])
        
        return alpha, beta, Z
    
    def viterbi(self, observations):
        """
        Viterbi解码
        
        Java对应：
        public int[] viterbi(int[] observations) {
            int T = observations.length;
            double[][] delta = new double[T][numStates];
            int[][] psi = new int[T][numStates];
            
            // 初始化
            for (int i = 0; i < numStates; i++) {
                delta[0][i] = emissionWeights[i][observations[0]];
            }
            
            // 递推
            for (int t = 1; t < T; t++) {
                for (int j = 0; j < numStates; j++) {
                    double maxScore = Double.NEGATIVE_INFINITY;
                    int maxState = 0;
                    
                    for (int i = 0; i < numStates; i++) {
                        double score = delta[t-1][i] + transitionWeights[i][j];
                        if (score > maxScore) {
                            maxScore = score;
                            maxState = i;
                        }
                    }
                    
                    delta[t][j] = maxScore + emissionWeights[j][observations[t]];
                    psi[t][j] = maxState;
                }
            }
            
            // 回溯
            int[] path = new int[T];
            path[T-1] = argmax(delta[T-1]);
            
            for (int t = T-2; t >= 0; t--) {
                path[t] = psi[t+1][path[t+1]];
            }
            
            return path;
        }
        
        Args:
            observations: 观测序列
        Returns:
            最优标签序列
        """
        T = len(observations)
        emission_scores, transition_scores = self._compute_scores(observations)
        
        # 初始化
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = emission_scores[0]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                scores = delta[t-1] + transition_scores[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + emission_scores[t, j]
        
        # 回溯
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("概率图模型示例")
    print("=" * 60)
    
    # ========== 1. HMM示例 ==========
    print("\n【1. 隐马尔可夫模型 HMM】")
    print("-" * 60)
    
    # 创建HMM（天气预测示例）
    # 隐藏状态：晴天(0)、雨天(1)
    # 观测：散步(0)、购物(1)、打扫(2)
    hmm = HiddenMarkovModel(n_states=2, n_observations=3)
    
    # 设置参数（示例）
    hmm.initial_prob = np.array([0.6, 0.4])  # 初始概率
    hmm.transition_prob = np.array([
        [0.7, 0.3],  # 晴天->晴天, 晴天->雨天
        [0.4, 0.6]   # 雨天->晴天, 雨天->雨天
    ])
    hmm.emission_prob = np.array([
        [0.6, 0.3, 0.1],  # 晴天：散步、购物、打扫
        [0.1, 0.4, 0.5]   # 雨天：散步、购物、打扫
    ])
    
    # 观测序列
    observations = np.array([0, 1, 2, 1, 0])  # 散步、购物、打扫、购物、散步
    print(f"观测序列: {observations}")
    
    # Forward算法：计算观测序列概率
    prob = hmm.forward(observations)
    print(f"观测序列概率: {prob:.6f}")
    
    # Viterbi算法：解码最可能的隐藏状态序列
    states = hmm.viterbi(observations)
    state_names = ['晴天', '雨天']
    print(f"最可能的天气序列: {[state_names[s] for s in states]}")
    
    # Baum-Welch算法：学习参数
    print("\n使用Baum-Welch算法学习参数...")
    # 生成训练数据
    np.random.seed(42)
    train_observations = np.random.randint(0, 3, 50)
    
    hmm_learned = HiddenMarkovModel(n_states=2, n_observations=3)
    hmm_learned.baum_welch(train_observations, max_iterations=50)
    
    print("\n学习到的转移概率:")
    print(hmm_learned.transition_prob)
    print("\n学习到的发射概率:")
    print(hmm_learned.emission_prob)
    
    # ========== 2. CRF示例 ==========
    print("\n【2. 条件随机场 CRF】")
    print("-" * 60)
    
    # 创建CRF（词性标注示例）
    # 状态：名词(0)、动词(1)、形容词(2)
    # 特征：词的特征表示
    crf = LinearChainCRF(n_states=3, n_features=5)
    
    # 示例观测序列
    observations_crf = np.array([0, 1, 2, 1, 3])
    print(f"观测序列: {observations_crf}")
    
    # Viterbi解码
    labels = crf.viterbi(observations_crf)
    label_names = ['名词', '动词', '形容词']
    print(f"预测标签: {[label_names[l] for l in labels]}")
    
    # ========== 可视化 ==========
    print("\n绘制HMM状态转移图和观测序列...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1：HMM转移概率矩阵
    ax1 = axes[0, 0]
    im1 = ax1.imshow(hmm.transition_prob, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['晴天', '雨天'])
    ax1.set_yticklabels(['晴天', '雨天'])
    ax1.set_xlabel('下一状态')
    ax1.set_ylabel('当前状态')
    ax1.set_title('HMM状态转移概率')
    plt.colorbar(im1, ax=ax1)
    
    # 在每个格子中显示数值
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{hmm.transition_prob[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=12)
    
    # 子图2：HMM发射概率矩阵
    ax2 = axes[0, 1]
    im2 = ax2.imshow(hmm.emission_prob, cmap='YlGnBu', aspect='auto')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['散步', '购物', '打扫'])
    ax2.set_yticklabels(['晴天', '雨天'])
    ax2.set_xlabel('观测')
    ax2.set_ylabel('隐藏状态')
    ax2.set_title('HMM发射概率')
    plt.colorbar(im2, ax=ax2)
    
    for i in range(2):
        for j in range(3):
            text = ax2.text(j, i, f'{hmm.emission_prob[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # 子图3：观测序列和预测状态
    ax3 = axes[1, 0]
    time_steps = np.arange(len(observations))
    ax3.plot(time_steps, observations, 'bo-', linewidth=2, markersize=8, label='观测序列')
    ax3.plot(time_steps, states, 'rs--', linewidth=2, markersize=8, label='预测状态')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('值')
    ax3.set_title('HMM观测序列与预测状态')
    ax3.legend()
    ax3.grid(True)
    
    # 子图4：CRF转移权重
    ax4 = axes[1, 1]
    im4 = ax4.imshow(crf.transition_weights, cmap='RdBu_r', aspect='auto')
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xticklabels(['名词', '动词', '形容词'])
    ax4.set_yticklabels(['名词', '动词', '形容词'])
    ax4.set_xlabel('下一标签')
    ax4.set_ylabel('当前标签')
    ax4.set_title('CRF转移特征权重')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


if __name__ == "__main__":
    example_usage()