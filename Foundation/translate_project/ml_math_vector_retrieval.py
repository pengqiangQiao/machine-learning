"""
向量检索与相似度计算数学基础
Mathematical Foundations for Vector Retrieval and Similarity

补充内容：
1. 向量相似度度量（余弦、欧氏、内积、曼哈顿）
2. 向量归一化（L1、L2）
3. 局部敏感哈希（LSH）
4. 量化方法的数学基础
5. 蒙特卡洛树搜索（MCTS）

Java对应：需要使用Apache Commons Math等数学库
"""

import numpy as np
import matplotlib.pyplot as plt
from ml_font_config import setup_chinese_font

# 配置中文字体
setup_chinese_font()

# ==================== 第一部分：向量相似度度量 ====================

class VectorSimilarity:
    """
    向量相似度度量
    
    涵盖：余弦相似度、欧氏距离、内积、曼哈顿距离、闵可夫斯基距离
    """
    
    @staticmethod
    def cosine_similarity(u, v):
        """
        余弦相似度
        
        公式: sim(u,v) = u·v / (||u||·||v||)
        
        取值范围: [-1, 1]
        - 1: 完全相同方向
        - 0: 正交（无关）
        - -1: 完全相反方向
        
        Args:
            u: 向量1
            v: 向量2
        Returns:
            余弦相似度
        """
        u = np.asarray(u)
        v = np.asarray(v)
        
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0
        
        return dot_product / (norm_u * norm_v)
    
    @staticmethod
    def euclidean_distance(u, v):
        """
        欧氏距离（L2距离）
        
        公式: d(u,v) = ||u - v|| = √(Σ(uᵢ - vᵢ)²)
        
        取值范围: [0, +∞)
        - 0: 完全相同
        - 越大越不相似
        
        Args:
            u: 向量1
            v: 向量2
        Returns:
            欧氏距离
        """
        u = np.asarray(u)
        v = np.asarray(v)
        return np.linalg.norm(u - v)
    
    @staticmethod
    def inner_product(u, v):
        """
        内积（点积）
        
        公式: u·v = Σ(uᵢ·vᵢ)
        
        取值范围: (-∞, +∞)
        - 正值: 方向相似
        - 0: 正交
        - 负值: 方向相反
        
        Args:
            u: 向量1
            v: 向量2
        Returns:
            内积
        """
        u = np.asarray(u)
        v = np.asarray(v)
        return np.dot(u, v)
    
    @staticmethod
    def manhattan_distance(u, v):
        """
        曼哈顿距离（L1距离）
        
        公式: d(u,v) = Σ|uᵢ - vᵢ|
        
        取值范围: [0, +∞)
        
        Args:
            u: 向量1
            v: 向量2
        Returns:
            曼哈顿距离
        """
        u = np.asarray(u)
        v = np.asarray(v)
        return np.sum(np.abs(u - v))
    
    @staticmethod
    def minkowski_distance(u, v, p=2):
        """
        闵可夫斯基距离
        
        公式: d(u,v) = (Σ|uᵢ - vᵢ|ᵖ)^(1/p)
        
        特殊情况：
        - p=1: 曼哈顿距离
        - p=2: 欧氏距离
        - p=∞: 切比雪夫距离
        
        Args:
            u: 向量1
            v: 向量2
            p: 参数p
        Returns:
            闵可夫斯基距离
        """
        u = np.asarray(u)
        v = np.asarray(v)
        return np.power(np.sum(np.power(np.abs(u - v), p)), 1/p)
    
    @staticmethod
    def compare_all_metrics():
        """比较所有相似度度量"""
        print("\n" + "="*60)
        print("【向量相似度度量】比较")
        print("="*60)
        
        # 测试向量
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        
        print(f"\n向量u: {u}")
        print(f"向量v: {v}")
        
        print(f"\n1. 余弦相似度: {VectorSimilarity.cosine_similarity(u, v):.6f}")
        print(f"   (取值范围: [-1, 1], 1表示完全相同方向)")
        
        print(f"\n2. 欧氏距离: {VectorSimilarity.euclidean_distance(u, v):.6f}")
        print(f"   (取值范围: [0, +∞), 0表示完全相同)")
        
        print(f"\n3. 内积: {VectorSimilarity.inner_product(u, v):.6f}")
        print(f"   (取值范围: (-∞, +∞), 正值表示方向相似)")
        
        print(f"\n4. 曼哈顿距离: {VectorSimilarity.manhattan_distance(u, v):.6f}")
        print(f"   (L1距离)")
        
        print(f"\n5. 闵可夫斯基距离:")
        print(f"   p=1 (曼哈顿): {VectorSimilarity.minkowski_distance(u, v, 1):.6f}")
        print(f"   p=2 (欧氏): {VectorSimilarity.minkowski_distance(u, v, 2):.6f}")
        print(f"   p=3: {VectorSimilarity.minkowski_distance(u, v, 3):.6f}")


# ==================== 第二部分：向量归一化 ====================

class VectorNormalization:
    """
    向量归一化
    
    涵盖：L1归一化、L2归一化、最大值归一化、标准化
    """
    
    @staticmethod
    def l1_normalize(v):
        """
        L1归一化
        
        公式: v_norm = v / ||v||₁ = v / Σ|vᵢ|
        
        结果：所有元素绝对值之和为1
        
        Args:
            v: 输入向量
        Returns:
            L1归一化后的向量
        """
        v = np.asarray(v, dtype=float)
        l1_norm = np.sum(np.abs(v))
        
        if l1_norm == 0:
            return v
        
        return v / l1_norm
    
    @staticmethod
    def l2_normalize(v):
        """
        L2归一化（单位向量化）
        
        公式: v_norm = v / ||v||₂ = v / √(Σvᵢ²)
        
        结果：向量长度为1
        
        Args:
            v: 输入向量
        Returns:
            L2归一化后的向量
        """
        v = np.asarray(v, dtype=float)
        l2_norm = np.linalg.norm(v)
        
        if l2_norm == 0:
            return v
        
        return v / l2_norm
    
    @staticmethod
    def max_normalize(v):
        """
        最大值归一化
        
        公式: v_norm = v / max(|v|)
        
        结果：最大绝对值为1
        
        Args:
            v: 输入向量
        Returns:
            最大值归一化后的向量
        """
        v = np.asarray(v, dtype=float)
        max_val = np.max(np.abs(v))
        
        if max_val == 0:
            return v
        
        return v / max_val
    
    @staticmethod
    def standardize(v):
        """
        标准化（Z-score归一化）
        
        公式: v_norm = (v - μ) / σ
        
        结果：均值为0，标准差为1
        
        Args:
            v: 输入向量
        Returns:
            标准化后的向量
        """
        v = np.asarray(v, dtype=float)
        mean = np.mean(v)
        std = np.std(v)
        
        if std == 0:
            return v - mean
        
        return (v - mean) / std
    
    @staticmethod
    def compare_normalizations():
        """比较不同归一化方法"""
        print("\n" + "="*60)
        print("【向量归一化】比较")
        print("="*60)
        
        v = np.array([1, 2, 3, 4, 5])
        
        print(f"\n原始向量: {v}")
        
        v_l1 = VectorNormalization.l1_normalize(v)
        print(f"\nL1归一化: {v_l1}")
        print(f"  验证 Σ|v| = {np.sum(np.abs(v_l1)):.6f}")
        
        v_l2 = VectorNormalization.l2_normalize(v)
        print(f"\nL2归一化: {v_l2}")
        print(f"  验证 ||v|| = {np.linalg.norm(v_l2):.6f}")
        
        v_max = VectorNormalization.max_normalize(v)
        print(f"\n最大值归一化: {v_max}")
        print(f"  验证 max(|v|) = {np.max(np.abs(v_max)):.6f}")
        
        v_std = VectorNormalization.standardize(v)
        print(f"\n标准化: {v_std}")
        print(f"  验证 均值 = {np.mean(v_std):.6f}, 标准差 = {np.std(v_std):.6f}")


# ==================== 第三部分：局部敏感哈希（LSH） ====================

class LocalitySensitiveHashing:
    """
    局部敏感哈希（LSH）
    
    原理：相似的向量映射到相同的哈希桶的概率高
    
    应用：近似最近邻搜索（ANN）
    """
    
    def __init__(self, input_dim, num_hash_functions=10, num_hash_tables=5):
        """
        初始化LSH
        
        Args:
            input_dim: 输入向量维度
            num_hash_functions: 每个哈希表的哈希函数数量
            num_hash_tables: 哈希表数量
        """
        self.input_dim = input_dim
        self.num_hash_functions = num_hash_functions
        self.num_hash_tables = num_hash_tables
        
        # 生成随机投影向量
        np.random.seed(42)
        self.random_projections = []
        for _ in range(num_hash_tables):
            projections = np.random.randn(num_hash_functions, input_dim)
            self.random_projections.append(projections)
    
    def hash_vector(self, v, table_idx):
        """
        对向量进行哈希
        
        原理：使用随机超平面投影
        h(v) = sign(w·v)
        
        Args:
            v: 输入向量
            table_idx: 哈希表索引
        Returns:
            哈希值（二进制数组）
        """
        v = np.asarray(v)
        projections = self.random_projections[table_idx]
        
        # 计算投影
        dot_products = np.dot(projections, v)
        
        # 二值化：>0为1，<=0为0
        hash_values = (dot_products >= 0).astype(int)
        
        return tuple(hash_values)
    
    def hash_to_bucket(self, hash_values):
        """
        将哈希值转换为桶索引
        
        Args:
            hash_values: 哈希值（二进制数组）
        Returns:
            桶索引（整数）
        """
        # 将二进制数组转换为整数
        bucket = 0
        for i, bit in enumerate(hash_values):
            bucket += bit * (2 ** i)
        return bucket
    
    @staticmethod
    def demonstrate_lsh():
        """演示LSH原理"""
        print("\n" + "="*60)
        print("【局部敏感哈希（LSH）】原理演示")
        print("="*60)
        
        # 创建LSH实例
        lsh = LocalitySensitiveHashing(input_dim=10, num_hash_functions=5, num_hash_tables=3)
        
        # 创建相似向量
        np.random.seed(42)
        v1 = np.random.randn(10)
        v2 = v1 + np.random.randn(10) * 0.1  # 与v1相似
        v3 = np.random.randn(10)  # 与v1不相似
        
        print(f"\n向量v1与v2的余弦相似度: {VectorSimilarity.cosine_similarity(v1, v2):.4f}")
        print(f"向量v1与v3的余弦相似度: {VectorSimilarity.cosine_similarity(v1, v3):.4f}")
        
        print("\n哈希结果：")
        same_bucket_count_v2 = 0
        same_bucket_count_v3 = 0
        
        for table_idx in range(lsh.num_hash_tables):
            hash1 = lsh.hash_vector(v1, table_idx)
            hash2 = lsh.hash_vector(v2, table_idx)
            hash3 = lsh.hash_vector(v3, table_idx)
            
            bucket1 = lsh.hash_to_bucket(hash1)
            bucket2 = lsh.hash_to_bucket(hash2)
            bucket3 = lsh.hash_to_bucket(hash3)
            
            if bucket1 == bucket2:
                same_bucket_count_v2 += 1
            if bucket1 == bucket3:
                same_bucket_count_v3 += 1
            
            print(f"\n哈希表 {table_idx}:")
            print(f"  v1 -> 桶 {bucket1}")
            print(f"  v2 -> 桶 {bucket2} {'[相同]' if bucket1 == bucket2 else '[不同]'}")
            print(f"  v3 -> 桶 {bucket3} {'[相同]' if bucket1 == bucket3 else '[不同]'}")
        
        print(f"\n总结：")
        print(f"  相似向量(v1, v2)碰撞次数: {same_bucket_count_v2}/{lsh.num_hash_tables}")
        print(f"  不相似向量(v1, v3)碰撞次数: {same_bucket_count_v3}/{lsh.num_hash_tables}")


# ==================== 第四部分：量化数学基础 ====================

class QuantizationMath:
    """
    量化的数学基础
    
    涵盖：均匀量化、量化误差分析
    """
    
    @staticmethod
    def uniform_quantization(x, num_bits=8):
        """
        均匀量化
        
        将浮点数映射到固定数量的离散值
        
        公式：
        x_min, x_max = min(x), max(x)
        scale = (x_max - x_min) / (2^num_bits - 1)
        x_quantized = round((x - x_min) / scale)
        x_dequantized = x_quantized * scale + x_min
        
        Args:
            x: 输入数据
            num_bits: 量化位数
        Returns:
            量化后的数据, 反量化后的数据, 量化误差
        """
        x = np.asarray(x, dtype=float)
        
        # 计算量化范围
        x_min = np.min(x)
        x_max = np.max(x)
        
        # 量化级数
        num_levels = 2 ** num_bits
        
        # 量化步长
        scale = (x_max - x_min) / (num_levels - 1) if x_max != x_min else 1.0
        
        # 量化
        x_quantized = np.round((x - x_min) / scale).astype(int)
        
        # 反量化
        x_dequantized = x_quantized * scale + x_min
        
        # 量化误差
        quantization_error = x - x_dequantized
        
        return x_quantized, x_dequantized, quantization_error
    
    @staticmethod
    def quantization_error_analysis(x, num_bits_list=[4, 8, 16]):
        """
        量化误差分析
        
        分析不同量化位数下的误差
        """
        print("\n" + "="*60)
        print("【量化误差分析】")
        print("="*60)
        
        x = np.asarray(x, dtype=float)
        
        print(f"\n原始数据统计:")
        print(f"  均值: {np.mean(x):.6f}")
        print(f"  标准差: {np.std(x):.6f}")
        print(f"  范围: [{np.min(x):.6f}, {np.max(x):.6f}]")
        
        for num_bits in num_bits_list:
            x_q, x_dq, error = QuantizationMath.uniform_quantization(x, num_bits)
            
            mse = np.mean(error ** 2)
            mae = np.mean(np.abs(error))
            max_error = np.max(np.abs(error))
            
            print(f"\n{num_bits}-bit 量化:")
            print(f"  量化级数: {2**num_bits}")
            print(f"  均方误差(MSE): {mse:.6e}")
            print(f"  平均绝对误差(MAE): {mae:.6e}")
            print(f"  最大误差: {max_error:.6e}")
            
            if np.var(x) > 0 and mse > 0:
                snr = 10 * np.log10(np.var(x) / mse)
                print(f"  信噪比(SNR): {snr:.2f} dB")


# ==================== 第五部分：蒙特卡洛树搜索（MCTS） ====================

class MonteCarloTreeSearch:
    """
    蒙特卡洛树搜索（MCTS）的数学基础
    
    核心思想：通过随机模拟评估决策的价值
    
    四个步骤：
    1. Selection（选择）：使用UCB公式选择节点
    2. Expansion（扩展）：添加新节点
    3. Simulation（模拟）：随机模拟到终局
    4. Backpropagation（回传）：更新路径上的节点
    """
    
    class Node:
        """MCTS节点"""
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
        
        def ucb_score(self, exploration_weight=1.414):
            """
            UCB（Upper Confidence Bound）公式
            
            UCB = Q(s,a) / N(s,a) + c * √(ln(N(s)) / N(s,a))
            
            其中：
            - Q(s,a): 节点累计价值
            - N(s,a): 节点访问次数
            - N(s): 父节点访问次数
            - c: 探索权重（通常为√2）
            
            Args:
                exploration_weight: 探索权重
            Returns:
                UCB分数
            """
            if self.visits == 0:
                return float('inf')
            
            # 利用项：平均价值
            exploitation = self.value / self.visits
            
            # 探索项：鼓励访问次数少的节点
            exploration = exploration_weight * np.sqrt(
                np.log(self.parent.visits) / self.visits
            )
            
            return exploitation + exploration
    
    @staticmethod
    def demonstrate_ucb():
        """演示UCB公式"""
        print("\n" + "="*60)
        print("【蒙特卡洛树搜索（MCTS）】UCB公式演示")
        print("="*60)
        
        # 创建父节点
        parent = MonteCarloTreeSearch.Node(state="parent")
        parent.visits = 100
        
        # 创建子节点
        child1 = MonteCarloTreeSearch.Node(state="child1", parent=parent)
        child1.visits = 30
        child1.value = 20  # 胜率 20/30 = 0.67
        
        child2 = MonteCarloTreeSearch.Node(state="child2", parent=parent)
        child2.visits = 10
        child2.value = 8   # 胜率 8/10 = 0.80
        
        child3 = MonteCarloTreeSearch.Node(state="child3", parent=parent)
        child3.visits = 5
        child3.value = 2   # 胜率 2/5 = 0.40
        
        print("\n节点统计:")
        print(f"父节点访问次数: {parent.visits}")
        
        for i, child in enumerate([child1, child2, child3], 1):
            win_rate = child.value / child.visits if child.visits > 0 else 0
            ucb = child.ucb_score()
            
            print(f"\n子节点{i}:")
            print(f"  访问次数: {child.visits}")
            print(f"  累计价值: {child.value}")
            print(f"  胜率: {win_rate:.2%}")
            print(f"  UCB分数: {ucb:.4f}")
        
        print("\nUCB公式平衡了：")
        print("  1. 利用（Exploitation）：选择胜率高的节点")
        print("  2. 探索（Exploration）：选择访问次数少的节点")
        print("\n结论：虽然child2胜率最高(80%)，但child3因访问次数少")
        print("      而获得更高的UCB分数，会被优先探索")


# ==================== 可视化函数 ====================

def visualize_vector_retrieval_concepts():
    """可视化向量检索与AI系统的数学概念"""
    print("\n" + "="*60)
    print("生成向量检索与AI系统数学概念可视化...")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 向量相似度度量对比
    ax1 = fig.add_subplot(3, 3, 1)
    np.random.seed(42)
    # 生成两组向量
    v1 = np.array([1, 2, 3, 4, 5])
    v2_similar = v1 + np.random.randn(5) * 0.5  # 相似向量
    v3_different = np.random.randn(5) * 3  # 不相似向量
    
    x_pos = np.arange(len(v1))
    width = 0.25
    ax1.bar(x_pos - width, v1, width, label='向量v1', alpha=0.8)
    ax1.bar(x_pos, v2_similar, width, label='相似向量v2', alpha=0.8)
    ax1.bar(x_pos + width, v3_different, width, label='不相似向量v3', alpha=0.8)
    ax1.set_xlabel('维度')
    ax1.set_ylabel('值')
    ax1.set_title('向量相似度对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 余弦相似度可视化
    ax2 = fig.add_subplot(3, 3, 2)
    angles = np.linspace(0, np.pi, 100)
    cosine_sim = np.cos(angles)
    ax2.plot(np.degrees(angles), cosine_sim, 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(np.degrees(angles), cosine_sim, alpha=0.3)
    ax2.set_xlabel('向量夹角 (度)')
    ax2.set_ylabel('余弦相似度')
    ax2.set_title('余弦相似度 vs 向量夹角')
    ax2.grid(True, alpha=0.3)
    
    # 3. L1 vs L2 归一化对比
    ax3 = fig.add_subplot(3, 3, 3)
    v = np.array([1, 2, 3, 4, 5])
    v_l1 = VectorNormalization.l1_normalize(v)
    v_l2 = VectorNormalization.l2_normalize(v)
    
    x_pos = np.arange(len(v))
    width = 0.35
    ax3.bar(x_pos - width/2, v_l1, width, label='L1归一化', alpha=0.8)
    ax3.bar(x_pos + width/2, v_l2, width, label='L2归一化', alpha=0.8)
    ax3.set_xlabel('维度')
    ax3.set_ylabel('归一化后的值')
    ax3.set_title('L1 vs L2 归一化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LSH哈希碰撞概率
    ax4 = fig.add_subplot(3, 3, 4)
    similarities = np.linspace(-1, 1, 100)
    # LSH碰撞概率 ≈ 1 - θ/π，其中θ = arccos(similarity)
    angles = np.arccos(np.clip(similarities, -1, 1))
    collision_prob = 1 - angles / np.pi
    ax4.plot(similarities, collision_prob, 'g-', linewidth=2)
    ax4.fill_between(similarities, collision_prob, alpha=0.3)
    ax4.set_xlabel('余弦相似度')
    ax4.set_ylabel('哈希碰撞概率')
    ax4.set_title('LSH：相似度 vs 碰撞概率')
    ax4.grid(True, alpha=0.3)
    
    # 5. 量化误差分析
    ax5 = fig.add_subplot(3, 3, 5)
    np.random.seed(42)
    original_data = np.random.randn(1000) * 10 + 50
    
    bits = [4, 8, 16]
    colors = ['r', 'g', 'b']
    for bit, color in zip(bits, colors):
        _, dequantized, error = QuantizationMath.uniform_quantization(original_data, bit)
        ax5.hist(error, bins=50, alpha=0.5, label=f'{bit}-bit', color=color, density=True)
    
    ax5.set_xlabel('量化误差')
    ax5.set_ylabel('频率密度')
    ax5.set_title('不同位数的量化误差分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 量化信噪比对比
    ax6 = fig.add_subplot(3, 3, 6)
    bit_range = range(2, 17)
    snr_values = []
    
    for num_bits in bit_range:
        _, dequantized, error = QuantizationMath.uniform_quantization(original_data, num_bits)
        mse = np.mean(error ** 2)
        if np.var(original_data) > 0 and mse > 0:
            snr = 10 * np.log10(np.var(original_data) / mse)
            snr_values.append(snr)
        else:
            snr_values.append(0)
    
    ax6.plot(bit_range, snr_values, 'bo-', linewidth=2, markersize=6)
    ax6.set_xlabel('量化位数')
    ax6.set_ylabel('信噪比 (dB)')
    ax6.set_title('量化位数 vs 信噪比')
    ax6.grid(True, alpha=0.3)
    
    # 7. MCTS的UCB分数演示
    ax7 = fig.add_subplot(3, 3, 7)
    visits = np.arange(1, 101)
    parent_visits = 100
    
    # 不同胜率的UCB曲线
    win_rates = [0.3, 0.5, 0.7]
    colors = ['r', 'g', 'b']
    
    for win_rate, color in zip(win_rates, colors):
        values = visits * win_rate
        exploitation = values / visits
        exploration = 1.414 * np.sqrt(np.log(parent_visits) / visits)
        ucb = exploitation + exploration
        ax7.plot(visits, ucb, color=color, linewidth=2, label=f'胜率={win_rate}')
    
    ax7.set_xlabel('访问次数')
    ax7.set_ylabel('UCB分数')
    ax7.set_title('MCTS: UCB分数随访问次数变化')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 利用 vs 探索权衡
    ax8 = fig.add_subplot(3, 3, 8)
    visits = np.arange(1, 101)
    parent_visits = 100
    win_rate = 0.6
    
    exploitation = (visits * win_rate) / visits
    exploration = 1.414 * np.sqrt(np.log(parent_visits) / visits)
    
    ax8.plot(visits, exploitation, 'b-', linewidth=2, label='利用项')
    ax8.plot(visits, exploration, 'r-', linewidth=2, label='探索项')
    ax8.plot(visits, exploitation + exploration, 'g--', linewidth=2, label='UCB总分')
    ax8.set_xlabel('访问次数')
    ax8.set_ylabel('分数')
    ax8.set_title('MCTS: 利用 vs 探索权衡')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 向量空间中的相似度
    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    np.random.seed(42)
    
    # 生成聚类数据
    cluster1 = np.random.randn(30, 3) * 0.5 + np.array([2, 2, 2])
    cluster2 = np.random.randn(30, 3) * 0.5 + np.array([-2, -2, -2])
    cluster3 = np.random.randn(30, 3) * 0.5 + np.array([2, -2, 0])
    
    ax9.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2],
               c='r', marker='o', s=50, alpha=0.6, label='簇1')
    ax9.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2],
               c='g', marker='^', s=50, alpha=0.6, label='簇2')
    ax9.scatter(cluster3[:, 0], cluster3[:, 1], cluster3[:, 2],
               c='b', marker='s', s=50, alpha=0.6, label='簇3')
    
    ax9.set_xlabel('维度1')
    ax9.set_ylabel('维度2')
    ax9.set_zlabel('维度3')
    ax9.set_title('3D向量空间中的聚类')
    ax9.legend()
    
    plt.tight_layout()
    print("正在显示图形...")
    plt.show()
    print("图形已显示")


# ==================== 主函数 ====================

def main():
    """主函数：运行所有演示"""
    print("\n" + "="*70)
    print(" "*10 + "向量检索与AI系统数学基础")
    print(" "*5 + "Mathematical Foundations for Vector Retrieval and AI Systems")
    print("="*70)
    
    # 1. 向量相似度度量
    VectorSimilarity.compare_all_metrics()
    
    # 2. 向量归一化
    VectorNormalization.compare_normalizations()
    
    # 3. 局部敏感哈希
    LocalitySensitiveHashing.demonstrate_lsh()
    
    # 4. 量化误差分析
    np.random.seed(42)
    test_data = np.random.randn(1000) * 10 + 50
    QuantizationMath.quantization_error_analysis(test_data)
    
    # 5. 蒙特卡洛树搜索
    MonteCarloTreeSearch.demonstrate_ucb()
    
    # 6. 可视化
    visualize_vector_retrieval_concepts()
    
    print("\n" + "="*70)
    print("所有演示完成！")
    print("="*70)


if __name__ == "__main__":
    main()