import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import csv

#データセットの読み込み、必要データの抽出
def create_dataset():
    with open("Simple_Regression_Analysis/icecream_data.txt", "r") as data:
        reader = csv.reader(data, delimiter=" ")
        next(reader)
        training_data = []
        for row in reader:
            temperature = float(row[2])  # temperature
            icecream_sales = int(row[3])  # ice_cream_sales
            training_data.append([temperature, icecream_sales])
    training_data = np.array(training_data)
    return training_data

# データの分割
def split_data(training_data):
    x = [d[0] for d in training_data]  # temperature
    y = [d[1] for d in training_data]  # ice_cream_sales
    return x, y

#トレーニングデータをトレーニングセットとテストセットに分割
def split_training_and_test(training_data):
    training_data_set, test_data_set = train_test_split(training_data, test_size=0.2, random_state=42)
    return training_data_set, test_data_set

# 説明変数と目的変数の分割
def aplit_Exv_and_Tgv(training_data_set):
    X_train = training_data_set[:, 0]  # 温度（説明変数）
    Y_train = training_data_set[:, 1]  # 売上金額（目的変数）
    return X_train, Y_train

# 線形回帰モデルの作成と学習
def model_studying(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train.reshape(-1, 1), Y_train)
    return model

# テストデータの予測
def predict(test_data_set, model):
    X_test = test_data_set[:, 0]  # 温度（説明変数）
    Y_test = test_data_set[:, 1]  # 売上金額（目的変数）
    Y_pred = model.predict(X_test.reshape(-1, 1))
    return X_test, Y_test, Y_pred

# トレーニングデータの学習
def study(X_train, Y_train):
    slope, intercept = np.polyfit(X_train.flatten(), Y_train, 1)
    return slope, intercept

# 重みベクトルとバイアスの定義
def dec_slope_and_intercept(slope, intercept):
    w = np.array([slope])
    b = intercept
    return w, b

# 予測結果の表示
def get_result(X_test, w, b):
    print("予測結果:")
    predicted = np.dot(X_test.reshape(-1, 1), w) + b  # すべてのテストデータに対して予測を計算
    for i in range(len(X_test)):
        print("温度: {:.2f}  ->  予測売上金額: {:.2f}".format(X_test[i], predicted[i]))
    return predicted

def plot_graph(x, y, X_test, Y_test, predicted):
# 散布図のプロット
    plt.scatter(x, y, color='green')
    # 線形回帰直線のプロット
    plt.plot(X_test, predicted.flatten(), color='red', label='Linear Regression')
    plt.scatter(X_test, Y_test, color='yellow', label='Actual Test Data')
    plt.xlabel('Temperature')  # X軸ラベル命名
    plt.ylabel('Ice Cream Sales')  # Y軸ラベル命名
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    training_data = create_dataset()
    print(training_data)
    x, y = split_data(training_data)
    training_data_set, test_data_set = split_training_and_test(training_data)
    print("Training Dataset:")
    print(training_data_set)
    print("\nTest Dataset:")
    print(test_data_set)
    X_train, Y_train = aplit_Exv_and_Tgv(training_data_set)
    print(X_train)
    print(type(X_train))
    model = model_studying(X_train, Y_train)
    X_test, Y_test, Y_pred = predict(test_data_set, model)
    slope, intercept = study(X_train, Y_train)
    w, b = dec_slope_and_intercept(slope, intercept)
    predicted = get_result(X_test, w, b)
    plot_graph(x, y, X_test, Y_test, predicted)