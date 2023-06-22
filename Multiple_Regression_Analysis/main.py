import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 科学的表記を通常の浮動小数点表記に変更
np.set_printoptions(suppress=True)

#データセットの読み込み、整形
def create_dataset():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    print(diabetes)
    IndV = diabetes.data
    DepV = diabetes.target.reshape(-1, 1)
    training_data = np.hstack((IndV, DepV))
    return training_data

#トレーニングデータをトレーニングセットとテストセットに分割
def split_training_and_test(training_data):
    training_data_set, test_data_set = train_test_split(training_data, test_size=0.2, random_state=42)
    return training_data_set, test_data_set

#トレーニングデータセットの説明変数と目的変数の定義
def Set_IndV_and_DepV(training_data_set):
    X_train_age = training_data_set[:, 0].reshape(-1, 1)
    X_train_sex = training_data_set[:, 1].reshape(-1, 1)
    X_train_bmi = training_data_set[:, 2].reshape(-1, 1)
    X_train_bp = training_data_set[:, 3].reshape(-1, 1)
    X_train_tc = training_data_set[:, 4].reshape(-1, 1)
    X_train_ldl = training_data_set[:, 5].reshape(-1, 1)
    X_train_hdl = training_data_set[:, 6].reshape(-1, 1)
    X_train_tch = training_data_set[:, 7].reshape(-1, 1)
    X_train_ltg = training_data_set[:, 8].reshape(-1, 1)
    X_train_glu = training_data_set[:, 9].reshape(-1, 1)
    X_train = np.column_stack((X_train_age, X_train_sex, X_train_bmi, X_train_bp, X_train_tc, X_train_ldl, X_train_hdl, X_train_tch, X_train_ltg, X_train_glu))

    Y_train_disease_progression = training_data_set[:, 10].reshape(-1, 1)


    X_test_age = test_data_set[:, 0]
    X_test_sex = test_data_set[:, 1]
    X_test_bmi = test_data_set[:, 2]
    X_test_bp = test_data_set[:, 3]
    X_test_tc = test_data_set[:, 4]
    X_test_ldl = test_data_set[:, 5]
    X_test_hdl = test_data_set[:, 6]
    X_test_tch = test_data_set[:, 7]
    X_test_ltg = test_data_set[:, 8]
    X_test_glu = test_data_set[:, 9]
    X_test = np.column_stack((X_test_age, X_test_sex, X_test_bmi, X_test_bp, X_test_tc, X_test_ldl, X_test_hdl, X_test_tch, X_test_ltg, X_test_glu))

    Y_test_disease_progression = test_data_set[:, 10].reshape(-1, 1)

    return X_train, Y_train_disease_progression, X_test, Y_test_disease_progression

#モデルの学習(scikit-learn)
def model_fit(X_train, Y_train_disease_progression):
    model = LinearRegression()
    model.fit(X_train, Y_train_disease_progression)
    return model

#モデルの学習(Numpy)
#Numpyによる重回帰分析の実装方法が不明なため、コメントアウトとして残す
"""
def moddel_study(X_train, Y_train_disease_progression):
    slope, intercept = np.polyfit(X_train.flatten(), Y_train_disease_progression, 11)
    w = np.array([slope])
    b = intercept
    return w, b
"""

#モデルから予測値を出す(Numpy)
def get_result(model, X_test):
    predicted = model.predict(X_test) # すべてのテストデータに対して予測を計算
    return predicted

#グラフの描画
def plot_graph(X_test, Y_test_disease_progression, predicted):
    plt.plot(X_test, predicted.flatten(), color='red', label='Linear Regression')

if __name__ == "__main__":
    training_data = create_dataset()
    print("training_data :")
    print(training_data)

    training_data_set, test_data_set = split_training_and_test(training_data)
    print("training_data_set :")
    print(training_data_set)
    print("test_data_set :")
    print(test_data_set)

    X_train, Y_train_disease_progression, X_test, Y_test_disease_progression = Set_IndV_and_DepV(training_data_set)
    print(f"X_train\t:\n{X_train}\nY_train_disease_progression\t:\n{Y_train_disease_progression}")

    model = model_fit(X_train, Y_train_disease_progression)
    print(f"model\t:\t{model}")

    predicted = get_result(model, X_test)
    print(f"predicted\t:\n{predicted}")