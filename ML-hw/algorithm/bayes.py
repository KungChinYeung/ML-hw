import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

# 加载Wine数据集
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义朴素贝叶斯分类器类
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_probs = counts / len(y)
        self.feature_probs = []

        for i in range(len(self.classes)):
            class_mask = (y == self.classes[i])
            class_data = X[class_mask]

            feature_probs_class = []
            for j in range(X.shape[1]):
                feature_values, feature_counts = np.unique(class_data[:, j], return_counts=True)
                feature_probs = (feature_counts + 1) / (len(class_data) + len(feature_values))
                feature_probs_class.append((feature_values, feature_probs))

            self.feature_probs.append(feature_probs_class)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            row_probs = []

            for j in range(len(self.classes)):
                class_prob = np.log(self.class_probs[j])

                for k in range(X.shape[1]):
                    feature_value = X[i, k]
                    feature_probs = self.feature_probs[j][k][1]

                    if feature_value in self.feature_probs[j][k][0]:
                        feature_prob = feature_probs[np.where(self.feature_probs[j][k][0] == feature_value)[0][0]]
                    else:
                        feature_prob = 1 / (len(self.feature_probs[j][k][0]) + len(self.classes))

                    class_prob += np.log(feature_prob)

                row_probs.append(class_prob)

            predictions.append(self.classes[np.argmax(row_probs)])

        return np.array(predictions)

# 创建并训练朴素贝叶斯分类器
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = nb_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

