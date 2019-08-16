from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 获取数据集
dataset = load_boston()
print(dataset.DESCR)
X = dataset.data
y = dataset.target
# 分割数据集 20percent 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# PCA降维 并归一化
pca = PCA(n_components='mle', whiten=True)
pca.fit(X_train)
X_r = pca.transform(X_train)

print('降维后的特征数\n', pca.n_components_)
print('降维后的各主成分的方差值\n', pca.explained_variance_)
print('降维后的各主成分占总方差值的比例\n', pca.explained_variance_ratio_)

LR = LinearRegression()
LR.fit(X_train, y_train)

print('Coefficients:\n', LR.coef_)

y_predict = LR.predict(X_test)

# 均方误差
print('MSE:', mean_squared_error(y_test,y_predict))

# R2 决定系数
# 越接近1 预测越好
print('r2_score:',r2_score(y_test,y_predict))
# 越接近中间的直线说明预测越准确
plt.scatter(y_test,y_test,marker='o')
plt.scatter(y_test,y_predict,marker='+' )
plt.show()