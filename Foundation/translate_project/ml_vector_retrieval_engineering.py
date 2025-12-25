"""
向量检索系统工程实现
Vector Retrieval System Engineering Implementation

包含：
1. HNSW（Hierarchical Navigable Small World）索引
2. KV Cache（键值缓存）
3. 语义缓存（Semantic Cache）
4. 倒排索引（Inverted Index）
5. Top-K检索
6. 混合检索

Java对应：需要使用Lucene、Faiss等库
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq
import hashlib
import time
from typing import List, Tuple, Dict, Any, Optional
from ml_font_config import setup_chinese_font

# 配置中文字体
setup_chinese_font()


# ==================== 第一部分：HNSW索引 ====================

class HNSWIndex:
    """
    HNSW（Hierarchical Navigable Small World）索引
    
    核心思想：
    1. 多层图结构（类似跳表）
    2. 每层是一个NSW图（Navigable Small World）
    3. 上层稀疏，下层密集
    4. 搜索从上层开始，逐层下降
    
    时间复杂度：
    - 插入：O(log N)
    - 搜索：O(log N)
    
    空间复杂度：O(N * M)，M是每个节点的最大连接数
    """
    
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, max_level: int = 5):
        """
        初始化HNSW索引
        
        Args:
            dim: 向量维度
            M: 每层每个节点的最大连接数
            ef_construction: 构建时的搜索宽度
            max_level: 最大层数
        """
        self.dim = dim
        self.M = M  # 最大连接数
        self.ef_construction = ef_construction
        self.max_level = max_level
        
        # 存储结构
        self.vectors = []  # 存储所有向量
        self.graph = defaultdict(lambda: defaultdict(set))  # graph[level][node_id] = {neighbor_ids}
        self.entry_point = None  # 入口点
        self.node_levels = {}  # 每个节点的层数
        
    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算欧氏距离"""
        return np.linalg.norm(v1 - v2)
    
    def _select_level(self) -> int:
        """随机选择节点的层数（指数衰减）"""
        level = 0
        while np.random.random() < 0.5 and level < self.max_level:
            level += 1
        return level
    
    def insert(self, vector: np.ndarray) -> int:
        """
        插入向量到HNSW索引
        
        Args:
            vector: 要插入的向量
        Returns:
            node_id: 节点ID
        """
        vector = np.asarray(vector, dtype=float)
        node_id = len(self.vectors)
        self.vectors.append(vector)
        
        # 选择层数
        level = self._select_level()
        self.node_levels[node_id] = level
        
        # 如果是第一个节点
        if self.entry_point is None:
            self.entry_point = node_id
            for lv in range(level + 1):
                self.graph[lv][node_id] = set()
            return node_id
        
        # 从顶层开始搜索
        current = self.entry_point
        current_level = self.node_levels[self.entry_point]
        
        # 逐层下降，找到最近邻
        for lv in range(current_level, level, -1):
            current = self._search_layer(vector, current, 1, lv)[0][1]
        
        # 在每一层插入节点并建立连接
        for lv in range(level, -1, -1):
            candidates = self._search_layer(vector, current, self.ef_construction, lv)
            
            # 选择M个最近邻
            M = self.M if lv > 0 else self.M * 2
            neighbors = self._select_neighbors(candidates, M)
            
            # 建立双向连接
            self.graph[lv][node_id] = set()
            for _, neighbor_id in neighbors:
                self.graph[lv][node_id].add(neighbor_id)
                self.graph[lv][neighbor_id].add(node_id)
                
                # 如果邻居的连接数超过M，需要修剪
                if len(self.graph[lv][neighbor_id]) > M:
                    self._prune_connections(neighbor_id, M, lv)
            
            if lv > 0:
                current = neighbors[0][1]
        
        # 更新入口点（如果新节点层数更高）
        if level > self.node_levels[self.entry_point]:
            self.entry_point = node_id
        
        return node_id
    
    def _search_layer(self, query: np.ndarray, entry: int, ef: int, level: int) -> List[Tuple[float, int]]:
        """
        在指定层搜索最近邻
        
        Args:
            query: 查询向量
            entry: 入口节点
            ef: 搜索宽度
            level: 层数
        Returns:
            candidates: [(distance, node_id), ...]
        """
        visited = set([entry])
        candidates = [(self._distance(query, self.vectors[entry]), entry)]
        heapq.heapify(candidates)
        
        best = [(self._distance(query, self.vectors[entry]), entry)]
        
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            
            if current_dist > best[0][0]:
                break
            
            # 遍历邻居
            for neighbor in self.graph[level][current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self.vectors[neighbor])
                    
                    if dist < best[0][0] or len(best) < ef:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(best, (-dist, neighbor))
                        
                        if len(best) > ef:
                            heapq.heappop(best)
        
        # 返回最近的ef个节点
        result = [(-dist, node_id) for dist, node_id in best]
        result.sort()
        return result
    
    def _select_neighbors(self, candidates: List[Tuple[float, int]], M: int) -> List[Tuple[float, int]]:
        """选择M个最佳邻居"""
        return sorted(candidates)[:M]
    
    def _prune_connections(self, node_id: int, M: int, level: int):
        """修剪节点的连接数"""
        neighbors = list(self.graph[level][node_id])
        if len(neighbors) <= M:
            return
        
        # 计算距离并排序
        distances = [(self._distance(self.vectors[node_id], self.vectors[n]), n) 
                    for n in neighbors]
        distances.sort()
        
        # 保留最近的M个
        new_neighbors = set([n for _, n in distances[:M]])
        
        # 移除多余的连接
        for neighbor in neighbors:
            if neighbor not in new_neighbors:
                self.graph[level][node_id].discard(neighbor)
                self.graph[level][neighbor].discard(node_id)
    
    def search(self, query: np.ndarray, k: int = 10, ef: int = None) -> List[Tuple[float, int]]:
        """
        搜索Top-K最近邻
        
        Args:
            query: 查询向量
            k: 返回的结果数量
            ef: 搜索宽度（默认为k）
        Returns:
            results: [(distance, node_id), ...]
        """
        if ef is None:
            ef = max(k, 50)
        
        if self.entry_point is None:
            return []
        
        query = np.asarray(query, dtype=float)
        
        # 从顶层开始搜索
        current = self.entry_point
        current_level = self.node_levels[self.entry_point]
        
        # 逐层下降
        for lv in range(current_level, 0, -1):
            current = self._search_layer(query, current, 1, lv)[0][1]
        
        # 在底层搜索
        candidates = self._search_layer(query, current, ef, 0)
        
        return candidates[:k]


# ==================== 第二部分：KV Cache ====================

class KVCache:
    """
    KV Cache（键值缓存）
    
    用于LLM推理优化：
    1. 缓存attention的key和value
    2. 相同prefix可以复用
    3. 减少重复计算
    
    应用场景：
    - 对话系统（复用历史上下文）
    - 批量推理（共享prompt前缀）
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化KV Cache
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache = {}  # {prefix_hash: (key_cache, value_cache, timestamp)}
        self.access_count = defaultdict(int)
        self.timestamps = {}
    
    def _hash_prefix(self, prefix: str) -> str:
        """计算prefix的哈希值"""
        return hashlib.md5(prefix.encode()).hexdigest()
    
    def get(self, prefix: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取缓存的KV
        
        Args:
            prefix: 前缀文本
        Returns:
            (key_cache, value_cache) 或 None
        """
        prefix_hash = self._hash_prefix(prefix)
        
        if prefix_hash in self.cache:
            key_cache, value_cache, _ = self.cache[prefix_hash]
            self.access_count[prefix_hash] += 1
            self.timestamps[prefix_hash] = time.time()
            return key_cache, value_cache
        
        return None
    
    def put(self, prefix: str, key_cache: np.ndarray, value_cache: np.ndarray):
        """
        存储KV到缓存
        
        Args:
            prefix: 前缀文本
            key_cache: key缓存
            value_cache: value缓存
        """
        prefix_hash = self._hash_prefix(prefix)
        
        # 如果缓存已满，使用LRU策略淘汰
        if len(self.cache) >= self.max_size and prefix_hash not in self.cache:
            self._evict_lru()
        
        self.cache[prefix_hash] = (key_cache, value_cache, time.time())
        self.access_count[prefix_hash] = 1
        self.timestamps[prefix_hash] = time.time()
    
    def _evict_lru(self):
        """淘汰最久未使用的缓存"""
        if not self.timestamps:
            return
        
        # 找到最久未访问的
        lru_hash = min(self.timestamps.items(), key=lambda x: x[1])[0]
        
        del self.cache[lru_hash]
        del self.access_count[lru_hash]
        del self.timestamps[lru_hash]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_accesses': sum(self.access_count.values()),
            'avg_accesses': np.mean(list(self.access_count.values())) if self.access_count else 0
        }


# ==================== 第三部分：语义缓存 ====================

class SemanticCache:
    """
    语义缓存（Semantic Cache）
    
    核心思想：
    1. 使用向量相似度而非精确匹配
    2. 相似的query返回缓存的answer
    3. 支持置信度阈值
    
    应用场景：
    - 问答系统（相似问题复用答案）
    - 搜索引擎（相似查询复用结果）
    """
    
    def __init__(self, similarity_threshold: float = 0.85, max_size: int = 1000):
        """
        初始化语义缓存
        
        Args:
            similarity_threshold: 相似度阈值（余弦相似度）
            max_size: 最大缓存条目数
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        
        self.queries = []  # 存储query向量
        self.answers = []  # 存储对应的answer
        self.metadata = []  # 存储元数据（时间戳、访问次数等）
        self.access_count = []
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def get(self, query_embedding: np.ndarray) -> Optional[Tuple[Any, float]]:
        """
        获取缓存的答案
        
        Args:
            query_embedding: 查询的向量表示
        Returns:
            (answer, similarity) 或 None
        """
        if not self.queries:
            return None
        
        query_embedding = np.asarray(query_embedding, dtype=float)
        
        # 计算与所有缓存query的相似度
        similarities = [self._cosine_similarity(query_embedding, q) 
                       for q in self.queries]
        
        # 找到最相似的
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        # 如果相似度超过阈值，返回缓存的答案
        if max_sim >= self.similarity_threshold:
            self.access_count[max_idx] += 1
            self.metadata[max_idx]['last_access'] = time.time()
            return self.answers[max_idx], max_sim
        
        return None
    
    def put(self, query_embedding: np.ndarray, answer: Any, metadata: Dict = None):
        """
        存储query-answer对到缓存
        
        Args:
            query_embedding: 查询的向量表示
            answer: 答案
            metadata: 元数据
        """
        query_embedding = np.asarray(query_embedding, dtype=float)
        
        # 如果缓存已满，淘汰访问次数最少的
        if len(self.queries) >= self.max_size:
            self._evict_lfu()
        
        self.queries.append(query_embedding)
        self.answers.append(answer)
        self.access_count.append(1)
        
        if metadata is None:
            metadata = {}
        metadata['created_at'] = time.time()
        metadata['last_access'] = time.time()
        self.metadata.append(metadata)
    
    def _evict_lfu(self):
        """淘汰访问次数最少的缓存（LFU策略）"""
        if not self.access_count:
            return
        
        # 找到访问次数最少的
        lfu_idx = np.argmin(self.access_count)
        
        del self.queries[lfu_idx]
        del self.answers[lfu_idx]
        del self.access_count[lfu_idx]
        del self.metadata[lfu_idx]
    
    def clear(self):
        """清空缓存"""
        self.queries.clear()
        self.answers.clear()
        self.metadata.clear()
        self.access_count.clear()
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'size': len(self.queries),
            'max_size': self.max_size,
            'total_accesses': sum(self.access_count),
            'avg_accesses': np.mean(self.access_count) if self.access_count else 0,
            'threshold': self.similarity_threshold
        }


# ==================== 第四部分：倒排索引 ====================

class InvertedIndex:
    """
    倒排索引（Inverted Index）
    
    核心思想：
    1. 词 -> 文档列表的映射
    2. 支持快速关键词检索
    3. 可与向量检索混合使用
    
    应用场景：
    - 文本搜索
    - 混合检索（关键词 + 向量）
    """
    
    def __init__(self):
        """初始化倒排索引"""
        self.index = defaultdict(set)  # {term: {doc_ids}}
        self.documents = {}  # {doc_id: document}
        self.doc_id_counter = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词（按空格和标点分割）"""
        import re
        # 转小写并分词
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def add_document(self, document: str, doc_id: int = None) -> int:
        """
        添加文档到索引
        
        Args:
            document: 文档文本
            doc_id: 文档ID（可选）
        Returns:
            doc_id: 文档ID
        """
        if doc_id is None:
            doc_id = self.doc_id_counter
            self.doc_id_counter += 1
        
        self.documents[doc_id] = document
        
        # 分词并建立倒排索引
        tokens = self._tokenize(document)
        for token in tokens:
            self.index[token].add(doc_id)
        
        return doc_id
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
        Returns:
            results: [(doc_id, score), ...]
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # 找到包含查询词的文档
        doc_scores = defaultdict(float)
        
        for token in query_tokens:
            if token in self.index:
                for doc_id in self.index[token]:
                    # 简单的TF评分（词频）
                    doc_tokens = self._tokenize(self.documents[doc_id])
                    tf = doc_tokens.count(token) / len(doc_tokens)
                    
                    # IDF评分
                    idf = np.log(len(self.documents) / len(self.index[token]))
                    
                    doc_scores[doc_id] += tf * idf
        
        # 排序并返回Top-K
        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'num_documents': len(self.documents),
            'num_terms': len(self.index),
            'avg_doc_length': np.mean([len(self._tokenize(doc)) 
                                      for doc in self.documents.values()]) if self.documents else 0
        }


# ==================== 第五部分：混合检索 ====================

class HybridRetrieval:
    """
    混合检索（Hybrid Retrieval）
    
    结合：
    1. 向量检索（语义相似度）
    2. 关键词检索（精确匹配）
    
    融合策略：
    - 加权融合
    - RRF（Reciprocal Rank Fusion）
    """
    
    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        """
        初始化混合检索
        
        Args:
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        self.hnsw_index = None
        self.inverted_index = InvertedIndex()
        self.doc_embeddings = {}  # {doc_id: embedding}
    
    def add_document(self, doc_id: int, text: str, embedding: np.ndarray):
        """
        添加文档
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            embedding: 文档的向量表示
        """
        # 添加到倒排索引
        self.inverted_index.add_document(text, doc_id)
        
        # 添加到向量索引
        self.doc_embeddings[doc_id] = embedding
        
        # 初始化HNSW索引（如果还没有）
        if self.hnsw_index is None:
            self.hnsw_index = HNSWIndex(dim=len(embedding))
        
        self.hnsw_index.insert(embedding)
    
    def search(self, query_text: str, query_embedding: np.ndarray, 
              top_k: int = 10) -> List[Tuple[int, float]]:
        """
        混合检索
        
        Args:
            query_text: 查询文本
            query_embedding: 查询的向量表示
            top_k: 返回的文档数量
        Returns:
            results: [(doc_id, score), ...]
        """
        # 1. 向量检索
        vector_results = []
        if self.hnsw_index is not None:
            vector_results = self.hnsw_index.search(query_embedding, k=top_k * 2)
        
        # 2. 关键词检索
        keyword_results = self.inverted_index.search(query_text, top_k=top_k * 2)
        
        # 3. 融合结果（加权融合）
        doc_scores = defaultdict(float)
        
        # 归一化向量检索分数
        if vector_results:
            max_dist = max([dist for dist, _ in vector_results])
            for dist, node_id in vector_results:
                # 转换为相似度（距离越小，相似度越高）
                similarity = 1 - (dist / max_dist if max_dist > 0 else 0)
                doc_scores[node_id] += self.vector_weight * similarity
        
        # 归一化关键词检索分数
        if keyword_results:
            max_score = max([score for _, score in keyword_results])
            for doc_id, score in keyword_results:
                normalized_score = score / max_score if max_score > 0 else 0
                doc_scores[doc_id] += self.keyword_weight * normalized_score
        
        # 排序并返回Top-K
        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ==================== 可视化函数 ====================

def visualize_retrieval_systems():
    """可视化向量检索系统的各个组件"""
    print("\n" + "="*60)
    print("生成向量检索系统可视化...")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. HNSW层次结构示意图
    ax1 = fig.add_subplot(3, 3, 1)
    levels = [1, 3, 7, 15, 31]  # 每层的节点数
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    for i, (level, color) in enumerate(zip(levels, colors)):
        y = len(levels) - i
        x = np.linspace(0, 10, level)
        ax1.scatter(x, [y] * level, s=200, c=color, alpha=0.6, edgecolors='black')
        ax1.text(-0.5, y, f'层{i}', fontsize=10, va='center')
    
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(0, len(levels) + 1)
    ax1.set_title('HNSW层次结构')
    ax1.set_xlabel('节点分布')
    ax1.set_ylabel('层数')
    ax1.grid(True, alpha=0.3)
    
    # 2. KV Cache命中率模拟
    ax2 = fig.add_subplot(3, 3, 2)
    cache_sizes = [10, 50, 100, 200, 500, 1000]
    hit_rates = [0.3, 0.5, 0.65, 0.75, 0.82, 0.85]  # 模拟数据
    
    ax2.plot(cache_sizes, hit_rates, 'bo-', linewidth=2, markersize=8)
    ax2.fill_between(cache_sizes, hit_rates, alpha=0.3)
    ax2.set_xlabel('缓存大小')
    ax2.set_ylabel('命中率')
    ax2.set_title('KV Cache命中率 vs 缓存大小')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. 语义缓存相似度分布
    ax3 = fig.add_subplot(3, 3, 3)
    np.random.seed(42)
    # 模拟相似度分布
    hit_similarities = np.random.beta(8, 2, 500)  # 命中的相似度（偏高）
    miss_similarities = np.random.beta(2, 5, 500)  # 未命中的相似度（偏低）
    
    ax3.hist(hit_similarities, bins=30, alpha=0.6, label='缓存命中', color='green', density=True)
    ax3.hist(miss_similarities, bins=30, alpha=0.6, label='缓存未命中', color='red', density=True)
    ax3.axvline(x=0.85, color='black', linestyle='--', linewidth=2, label='阈值=0.85')
    ax3.set_xlabel('余弦相似度')
    ax3.set_ylabel('频率密度')
    ax3.set_title('语义缓存：相似度分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 倒排索引查询性能
    ax4 = fig.add_subplot(3, 3, 4)
    doc_counts = [100, 1000, 10000, 100000, 1000000]
    query_times = [0.001, 0.005, 0.02, 0.08, 0.3]  # 毫秒
    
    ax4.loglog(doc_counts, query_times, 'go-', linewidth=2, markersize=8)
    ax4.fill_between(doc_counts, query_times, alpha=0.3, color='green')
    ax4.set_xlabel('文档数量')
    ax4.set_ylabel('查询时间 (秒)')
    ax4.set_title('倒排索引查询性能')
    ax4.grid(True, alpha=0.3, which='both')
    
    # 5. 混合检索权重影响
    ax5 = fig.add_subplot(3, 3, 5)
    vector_weights = np.linspace(0, 1, 11)
    precision = 0.6 + 0.2 * vector_weights  # 模拟：向量检索更准确
    recall = 0.5 + 0.3 * (1 - vector_weights)  # 模拟：关键词检索召回率高
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    ax5.plot(vector_weights, precision, 'b-', linewidth=2, marker='o', label='精确率')
    ax5.plot(vector_weights, recall, 'r-', linewidth=2, marker='s', label='召回率')
    ax5.plot(vector_weights, f1_score, 'g-', linewidth=2, marker='^', label='F1分数')
    ax5.set_xlabel('向量检索权重')
    ax5.set_ylabel('性能指标')
    ax5.set_title