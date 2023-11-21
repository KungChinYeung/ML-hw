import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    # 随机初始化聚类中心
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # 分配每个样本到最近的聚类中心
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # 更新聚类中心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 如果聚类中心没有变化，提前结束迭代
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# 生成一些示例数据
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=10, scale=2, size=(50, 2)),
                       np.random.normal(loc=20, scale=2, size=(50, 2)),
                       np.random.normal(loc=30, scale=2, size=(50, 2))])

# 执行 K-Means 聚类
k = 3
centroids, labels = kmeans(data, k)

# 绘制聚类结果
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
