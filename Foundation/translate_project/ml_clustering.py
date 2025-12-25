"""
聚类算法实现
Clustering Algorithms Implementation

Java对应实现：可以使用Weka的聚类算法或自己实现
Java equivalent: Use Weka's clustering algorithms or implement from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()

# Java对应：import weka.clusterers.SimpleKMeans;
# Java对应：import weka.clusterers.DBSCAN;


class KMeansClustering:
    """
    K-Means聚类算法实现
    
    Java对应实现：
    public class KMeansClustering {
        private int k;                    // 聚类数量
        private int maxIterations;        // 最大迭代次数
        private double[][] centroids;     // 聚类中心
        private int[] labels;             // 样本标签
        
        public KMeansClustering(int k, int maxIterations) {
            this.k = k;
            this.maxIterations = maxIterations;
        }
        
        // 计算欧氏距离
        private double euclideanDistance(double[] point1, double[] point2) {
            double sum = 0.0;
            for (int i = 0; i < point1.length; i++) {
                sum += Math.pow(point1[i] - point2[i], 2);
            }
            return Math.sqrt(sum);
        }
    }
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=42):
        """
        初始化K-Means聚类
        
        Java对应：
        public KMeansClustering(int nClusters, int maxIterations, int randomState) {
            this.nClusters = nClusters;
            this.maxIterations = maxIterations;
            this.random = new Random(randomState);
            this.centroids = null;
            this.labels = null;
        }
        
        Args:
            n_clusters: 聚类数量
            max_iterations: 最大迭代次数
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def euclidean_distance(self, x1, x2):
        """
        计算欧氏距离
        
        Java对应：
        private double euclideanDistance(double[] x1, double[] x2) {
            double sum = 0.0;
            for (int i = 0; i < x1.length; i++) {
                double diff = x1[i] - x2[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
        
        Args:
            x1: 点1
            x2: 点2
        Returns:
            欧氏距离
        """
        # 欧氏距离: d = √(Σ(x1 - x2)²)
        # Java: distance = Math.sqrt(sum((x1[i] - x2[i])²))
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def initialize_centroids(self, X):
        """
        初始化聚类中心（随机选择k个样本）
        
        Java对应：
        private double[][] initializeCentroids(double[][] X) {
            double[][] centroids = new double[nClusters][X[0].length];
            
            // 随机选择k个样本作为初始中心
            Set<Integer> selectedIndices = new HashSet<>();
            while (selectedIndices.size() < nClusters) {
                int idx = random.nextInt(X.length);
                selectedIndices.add(idx);
            }
            
            int i = 0;
            for (int idx : selectedIndices) {
                centroids[i++] = X[idx].clone();
            }
            
            return centroids;
        }
        
        Args:
            X: 数据集
        Returns:
            初始聚类中心
        """
        np.random.seed(self.random_state)
        # 随机选择k个样本作为初始中心
        # Java: 使用Random.nextInt()随机选择索引
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def assign_clusters(self, X):
        """
        分配样本到最近的聚类中心
        
        Java对应：
        private int[] assignClusters(double[][] X) {
            int[] labels = new int[X.length];
            
            for (int i = 0; i < X.length; i++) {
                double minDistance = Double.MAX_VALUE;
                int closestCluster = 0;
                
                // 找到最近的聚类中心
                for (int j = 0; j < centroids.length; j++) {
                    double distance = euclideanDistance(X[i], centroids[j]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCluster = j;
                    }
                }
                
                labels[i] = closestCluster;
            }
            
            return labels;
        }
        
        Args:
            X: 数据集
        Returns:
            每个样本的聚类标签
        """
        labels = np.zeros(X.shape[0], dtype=int)
        
        # 对每个样本，找到最近的聚类中心
        # Java: 遍历所有样本和聚类中心，计算距离
        for i, sample in enumerate(X):
            distances = [self.euclidean_distance(sample, centroid) 
                        for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        
        return labels
    
    def update_centroids(self, X, labels):
        """
        更新聚类中心（计算每个簇的均值）
        
        Java对应：
        private double[][] updateCentroids(double[][] X, int[] labels) {
            double[][] newCentroids = new double[nClusters][X[0].length];
            int[] counts = new int[nClusters];
            
            // 累加每个簇的样本
            for (int i = 0; i < X.length; i++) {
                int cluster = labels[i];
                counts[cluster]++;
                for (int j = 0; j < X[i].length; j++) {
                    newCentroids[cluster][j] += X[i][j];
                }
            }
            
            // 计算均值
            for (int i = 0; i < nClusters; i++) {
                if (counts[i] > 0) {
                    for (int j = 0; j < newCentroids[i].length; j++) {
                        newCentroids[i][j] /= counts[i];
                    }
                }
            }
            
            return newCentroids;
        }
        
        Args:
            X: 数据集
            labels: 聚类标签
        Returns:
            新的聚类中心
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        # 计算每个簇的均值作为新的中心
        # Java: 累加每个簇的样本，然后除以样本数
        for k in range(self.n_clusters):
            cluster_samples = X[labels == k]
            if len(cluster_samples) > 0:
                centroids[k] = cluster_samples.mean(axis=0)
        
        return centroids
    
    def fit(self, X):
        """
        训练K-Means模型
        
        Java对应：
        public void fit(double[][] X) {
            // 初始化聚类中心
            centroids = initializeCentroids(X);
            
            for (int iter = 0; iter < maxIterations; iter++) {
                // 分配样本到聚类
                int[] newLabels = assignClusters(X);
                
                // 更新聚类中心
                double[][] newCentroids = updateCentroids(X, newLabels);
                
                // 检查是否收敛
                if (centroidsEqual(centroids, newCentroids)) {
                    System.out.println("Converged at iteration " + iter);
                    break;
                }
                
                centroids = newCentroids;
                labels = newLabels;
            }
        }
        
        Args:
            X: 训练数据
        """
        # 初始化聚类中心
        # Java: centroids = initializeCentroids(X)
        self.centroids = self.initialize_centroids(X)
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # 分配样本到最近的聚类中心
            # Java: labels = assignClusters(X)
            old_labels = self.labels
            self.labels = self.assign_clusters(X)
            
            # 更新聚类中心
            # Java: centroids = updateCentroids(X, labels)
            new_centroids = self.update_centroids(X, self.labels)
            
            # 检查是否收敛（聚类中心不再变化）
            # Java: if (centroidsEqual(centroids, newCentroids))
            if np.allclose(self.centroids, new_centroids):
                print(f"在第 {iteration} 次迭代时收敛")
                break
            
            self.centroids = new_centroids
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}")
    
    def predict(self, X):
        """
        预测新样本的聚类标签
        
        Java对应：
        public int[] predict(double[][] X) {
            return assignClusters(X);
        }
        
        Args:
            X: 新样本
        Returns:
            聚类标签
        """
        return self.assign_clusters(X)
    
    def plot_clusters(self, X, title="K-Means聚类结果"):
        """
        可视化聚类结果（仅适用于2D数据）
        
        Java对应：
        public void plotClusters(double[][] X, String title) {
            // 使用JFreeChart绘制散点图
            XYSeriesCollection dataset = new XYSeriesCollection();
            
            for (int k = 0; k < nClusters; k++) {
                XYSeries series = new XYSeries("Cluster " + k);
                for (int i = 0; i < X.length; i++) {
                    if (labels[i] == k) {
                        series.add(X[i][0], X[i][1]);
                    }
                }
                dataset.addSeries(series);
            }
            
            // 添加聚类中心
            XYSeries centroidSeries = new XYSeries("Centroids");
            for (double[] centroid : centroids) {
                centroidSeries.add(centroid[0], centroid[1]);
            }
            dataset.addSeries(centroidSeries);
        }
        """
        if X.shape[1] != 2:
            print("只能可视化2D数据")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 绘制每个簇的样本
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n_clusters))
        for k in range(self.n_clusters):
            cluster_samples = X[self.labels == k]
            plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1],
                       c=[colors[k]], label=f'簇 {k}', alpha=0.6)
        
        # 绘制聚类中心
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                   c='black', marker='X', s=200, label='聚类中心')
        
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) throws Exception {
        // 生成示例数据
        Random random = new Random(42);
        double[][] X = new double[300][2];
        
        // 生成3个簇的数据
        for (int i = 0; i < 100; i++) {
            X[i][0] = random.nextGaussian() + 0;
            X[i][1] = random.nextGaussian() + 0;
        }
        for (int i = 100; i < 200; i++) {
            X[i][0] = random.nextGaussian() + 5;
            X[i][1] = random.nextGaussian() + 5;
        }
        for (int i = 200; i < 300; i++) {
            X[i][0] = random.nextGaussian() + 10;
            X[i][1] = random.nextGaussian() + 0;
        }
        
        // K-Means聚类
        KMeansClustering kmeans = new KMeansClustering(3, 100, 42);
        kmeans.fit(X);
        
        // 可视化
        kmeans.plotClusters(X, "K-Means Clustering");
    }
    """
    print("=" * 50)
    print("聚类算法示例")
    print("=" * 50)
    
    # 生成示例数据（3个簇）
    # Java: 使用Random生成数据
    np.random.seed(42)
    
    # 簇1: 中心在(0, 0)
    cluster1 = np.random.randn(100, 2) + np.array([0, 0])
    # 簇2: 中心在(5, 5)
    cluster2 = np.random.randn(100, 2) + np.array([5, 5])
    # 簇3: 中心在(10, 0)
    cluster3 = np.random.randn(100, 2) + np.array([10, 0])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("\n1. 使用自定义K-Means实现")
    print("-" * 50)
    # Java: KMeansClustering kmeans = new KMeansClustering(3, 100, 42)
    custom_kmeans = KMeansClustering(n_clusters=3, max_iterations=100, random_state=42)
    custom_kmeans.fit(X)
    
    # 可视化
    print("\n绘制自定义K-Means聚类结果...")
    custom_kmeans.plot_clusters(X, "自定义K-Means聚类结果")
    print("聚类结果图形已显示")
    
    # 评估聚类质量
    # 轮廓系数（Silhouette Score）: 衡量样本与其簇的相似度
    # Java: 需要手动实现轮廓系数计算
    silhouette = silhouette_score(X, custom_kmeans.labels)
    print(f"轮廓系数: {silhouette:.4f}")
    
    print("\n2. 使用sklearn的K-Means")
    print("-" * 50)
    # Java: 使用Weka的SimpleKMeans
    sklearn_kmeans = KMeans(n_clusters=3, random_state=42)
    sklearn_labels = sklearn_kmeans.fit_predict(X)
    
    print("\n绘制Sklearn K-Means聚类结果...")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, 3))
    for k in range(3):
        cluster_samples = X[sklearn_labels == k]
        plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1],
                   c=[colors[k]], label=f'簇 {k}', alpha=0.6)
    
    plt.scatter(sklearn_kmeans.cluster_centers_[:, 0],
               sklearn_kmeans.cluster_centers_[:, 1],
               c='black', marker='X', s=200, label='聚类中心')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('Sklearn K-Means聚类结果')
    plt.legend()
    plt.grid(True)
    print("正在显示图形...")
    plt.show()
    print("图形已显示")
    
    silhouette_sklearn = silhouette_score(X, sklearn_labels)
    print(f"轮廓系数: {silhouette_sklearn:.4f}")
    
    print("\n3. DBSCAN聚类（基于密度）")
    print("-" * 50)
    # Java: 使用Weka的DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"估计的簇数量: {n_clusters_dbscan}")
    print(f"噪声点数量: {n_noise}")
    
    print("\n绘制DBSCAN聚类结果...")
    plt.figure(figsize=(10, 6))
    unique_labels = set(dbscan_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用黑色表示
            col = 'black'
        
        class_member_mask = (dbscan_labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col],
                   label=f'簇 {k}' if k != -1 else '噪声',
                   alpha=0.6)
    
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('DBSCAN聚类结果')
    plt.legend()
    plt.grid(True)
    print("正在显示图形...")
    plt.show()
    print("图形已显示")
    
    if n_clusters_dbscan > 1:
        # 排除噪声点计算轮廓系数
        mask = dbscan_labels != -1
        if np.sum(mask) > 0:
            silhouette_dbscan = silhouette_score(X[mask], dbscan_labels[mask])
            print(f"轮廓系数（不含噪声）: {silhouette_dbscan:.4f}")
    
    print("\n4. 层次聚类")
    print("-" * 50)
    # Java: 使用Weka的HierarchicalClusterer
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X)
    
    print("\n绘制层次聚类结果...")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, 3))
    for k in range(3):
        cluster_samples = X[hierarchical_labels == k]
        plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1],
                   c=[colors[k]], label=f'簇 {k}', alpha=0.6)
    
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('层次聚类结果')
    plt.legend()
    plt.grid(True)
    print("正在显示图形...")
    plt.show()
    print("图形已显示")
    
    silhouette_hierarchical = silhouette_score(X, hierarchical_labels)
    print(f"轮廓系数: {silhouette_hierarchical:.4f}")
    
    # 肘部法则：选择最佳K值
    print("\n5. 肘部法则选择最佳K值")
    print("-" * 50)
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    print("\n绘制肘部法则图...")
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('聚类数量 K')
    plt.ylabel('簇内平方和（Inertia）')
    plt.title('肘部法则')
    plt.grid(True)
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