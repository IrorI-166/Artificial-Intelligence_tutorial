import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import csv

#データセットの読み込み、必要データの抽出
with open("Simple_Regression_Analysis/icecream_data.txt", "r") as data:
    reader = csv.reader(data, delimiter=" ")
    next(reader)
    training_data = []
    for row in reader:
        temperature = float(row[2])  # temperature
        icecream_sales = int(row[3])  # ice_cream_sales
        training_data.append([temperature, icecream_sales])

training_data = np.array(training_data)

print(training_data)

# データの分割
x = [d[0] for d in training_data]  # temperature
y = [d[1] for d in training_data]  # ice_cream_sales

# 散布図のプロット
plt.scatter(x, y)
plt.xlabel('Temperature') #X軸ラベル命名
plt.ylabel('Ice Cream Sales') #Y軸ラベル命名
plt.title('Relationship between Temperature and Ice Cream Sales')
plt.show()

#トレーニングデータをトレーニングセットとテストセットに分割
training_data_set, test_data_set = train_test_split(training_data, test_size=0.2, random_state=42)
print("Training Dataset:")
print(training_data_set)
print("\nTest Dataset:")
print(test_data_set)

# 説明変数と目的変数の分割
X_train = training_data_set[:, 0]  # 温度（説明変数）
y_train = training_data_set[:, 1]  # 売上金額（目的変数）
print(X_train)
print(type(X_train))

# 線形回帰モデルの作成と学習
model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)

# テストデータの予測
X_test = test_data_set[:, 0]  # 温度（説明変数）
y_test = test_data_set[:, 1]  # 売上金額（目的変数）
y_pred = model.predict(X_test.reshape(-1, 1))

# トレーニングデータの学習
slope, intercept = np.polyfit(X_train.flatten(), y_train, 1)

# 重みベクトルとバイアスの定義
w = np.array([slope])
b = intercept

# 予測結果の表示
print("予測結果:")
for i in range(len(X_test)):
    predicted = np.dot(w, X_test[i])[0] + b
    print(type(X_test[i]), type(predicted))
    print("温度: {:.2f}  ->  予測売上金額: {:.2f}".format(X_test[i], predicted))