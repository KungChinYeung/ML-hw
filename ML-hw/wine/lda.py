import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# 确定允许的最大降维维度
max_components = min(X_scaled.shape[1], len(np.unique(y)) - 1)

# 使用LDA进行降维
lda = LinearDiscriminantAnalysis(n_components=max_components)  # 选择降维后的维度
X_lda = lda.fit_transform(X_scaled, y)

# 将降维后的数据构建成DataFrame
lda_df = pd.DataFrame(data=X_lda, columns=[f'LD {i+1}' for i in range(max_components)])
lda_df['class'] = y

# 绘制降维后的数据
plt.figure(figsize=(10, 8))
colors = {1: 'red', 2: 'green'}  # 类别1用红色表示，类别2用绿色表示
for cls, color in colors.items():
    subset = lda_df[lda_df['class'] == cls]
    plt.scatter(subset[f'LD 1'], subset[f'LD 2'], color=color, label=f'Class {cls}')

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.title('LDA of Wine Data')
plt.legend()
plt.show()
