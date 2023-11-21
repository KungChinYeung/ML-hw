import numpy as np

# 定义节点类
class Node:
    def __init__(self, feature=None, threshold=None, label=None, left=None, right=None):
        self.feature = feature  # 选择的特征索引
        self.threshold = threshold  # 特征的阈值
        self.label = label  # 叶子节点的类别
        self.left = left  # 左子树
        self.right = right  # 右子树

# 定义ID3决策树分类器类
class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1:  # 如果所有样本属于同一类别，则创建叶子节点
            return Node(label=y[0])
        if self.max_depth is not None and depth == self.max_depth:  # 达到最大深度，则创建叶子节点
            return Node(label=np.bincount(y).argmax())

        best_feature, threshold = self.find_best_split(X, y)  # 选择最佳的分裂特征和阈值

        if best_feature is None:  # 无法找到最佳分裂特征，则创建叶子节点
            return Node(label=np.bincount(y).argmax())

        left_indices = X[:, best_feature] <= threshold
        right_indices = ~left_indices

        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=threshold, left=left_subtree, right=right_subtree)

    def find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None  # 无法进行分裂

        num_classes = len(set(y))
        if num_classes == 1:
            return None, None  # 所有样本属于同一类别，无需分裂

        entropy_parent = self.calculate_entropy(y)  # 计算父节点的信息熵

        best_info_gain = 0
        best_feature = None
        best_threshold = None

        for feature in range(n):
            unique_values = np.unique(X[:, feature])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # 计算可能的阈值

            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue  # 忽略无法进行分裂的情况

                entropy_left = self.calculate_entropy(y[left_indices])
                entropy_right = self.calculate_entropy(y[right_indices])

                info_gain = entropy_parent - (np.sum(left_indices) / m * entropy_left +
                                              np.sum(right_indices) / m * entropy_right)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def calculate_entropy(self, y):
        m = len(y)
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def predict(self, node, x):
        if node.label is not None:
            return node.label

        if x[node.feature] <= node.threshold:
            return self.predict(node.left, x)
        else:
            return self.predict(node.right, x)

    def predict_all(self, root, X):
        return np.array([self.predict(root, x) for x in X])

# 计算准确率
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

# 加载Wine数据集
def load_wine_data():
    data = np.loadtxt("wine.data", delimiter=",", skiprows=1)
    X = data[:, 1:]  # Features start from the second column
    y = data[:, 0].astype(int)
    return X, y


from sklearn.model_selection import train_test_split
# 划分数据集
X, y = load_wine_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练ID3决策树
dt_id3 = DecisionTreeID3(max_depth=3)
root_node = dt_id3.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = dt_id3.predict_all(root_node, X_test)

# 计算准确率
accuracy = calculate_accuracy(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
