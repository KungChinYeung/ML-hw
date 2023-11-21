import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
olddata = pd.read_csv("wine.data")

# 去除'class'列为3的数据
data = olddata[olddata['class'] != 3]

# 提取特征和标签
X = data.iloc[:, 1:]  # 特征从第二列开始
y = data['class']

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA进行降维
pca = PCA(n_components=2)  # 选择降维后的维度
X_pca = pca.fit_transform(X_scaled)

# 解释主成分
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# 将降维后的数据构建成DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['class'] = y

# 绘制降维后的数据
plt.figure(figsize=(10, 8))
colors = {1: 'red', 2: 'green'}  # 类别1用红色表示，类别2用绿色表示
for cls, color in colors.items():
    subset = pca_df[pca_df['class'] == cls]
    plt.scatter(subset['Principal Component 1'], subset['Principal Component 2'], color=color, label=f'Class {cls}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Data')
plt.legend()
plt.show()
