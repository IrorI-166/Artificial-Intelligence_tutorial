import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import csv

#データセットの読み込み、必要データの抽出
def create_dataset():
    with open("Multiple_Regression_Analysis/airquarity_data.txt", "r") as data:
        reader = csv.reader(data, delimiter=" ")
        next(reader)
        training_data = []
        ozone_values = []
        solor_values = []
        wind_values = []
        temperature_values = []
        mounth_values = []
        day_values = []

        for row in reader:
            Ozone = int(row[1]) if row[1] != "NA" else np.nan
            Solor = int(row[2]) if row[2] != "NA" else np.nan
            Wind = float(row[3]) if row[3] != "NA" else np.nan
            Temperature = int(row[4]) if row[4] != "NA" else np.nan
            Mounth = int(row[5]) if row[5] != "NA" else np.nan
            Day = int(row[6]) if row[6] != "NA" else np.nan

            ozone_values.append(Ozone)
            solor_values.append(Solor)
            wind_values.append(Wind)
            temperature_values.append(Temperature)
            mounth_values.append(Mounth)
            day_values.append(Day)

            training_data.append([Ozone, Solor, Wind, Temperature, Mounth, Day])

    imputer = SimpleImputer(strategy='mean')
    training_data = np.array(training_data)
    training_data[:, 0] = imputer.fit_transform(np.array(ozone_values).reshape(-1, 1)).flatten()
    training_data[:, 1] = imputer.fit_transform(np.array(solor_values).reshape(-1, 1)).flatten()
    training_data[:, 2] = imputer.fit_transform(np.array(wind_values).reshape(-1, 1)).flatten()
    training_data[:, 3] = imputer.fit_transform(np.array(temperature_values).reshape(-1, 1)).flatten()
    training_data[:, 4] = imputer.fit_transform(np.array(mounth_values).reshape(-1, 1)).flatten()
    training_data[:, 5] = imputer.fit_transform(np.array(day_values).reshape(-1, 1)).flatten()

    return training_data

#トレーニングデータをトレーニングセットとテストセットに分割
def split_training_and_test(training_data):
    training_data_set, test_data_set = train_test_split(training_data, test_size=0.2, random_state=42)
    return training_data_set, test_data_set

#各データセットから説明変数と目的変数の抽出及び配列結合
def create_IndV_and_DepV_Array(training_data_set, test_data_set):
    ozone_train = training_data_set[:, 0]
    solor_train = training_data_set[:, 1]
    wind_train = training_data_set[:, 2]
    temperature_train = training_data_set[:, 3]
    mounth_train = training_data_set[:, 4]
    day_train = training_data_set[:, 5]

    ozone_test = test_data_set[:, 0]
    solor_test = test_data_set[:, 1]
    wind_test = test_data_set[:, 2]
    temperature_test = test_data_set[:, 3]
    mounth_test = test_data_set[:, 4]
    day_test = test_data_set[:, 5]

    #説明変数2個
    IndV2_Xtrain = np.column_stack((wind_train, temperature_train))
    IndV2_Ytrain = solor_train
    IndV2_Xtest = np.column_stack((wind_test, temperature_test))
    Indv2_Ytest = solor_test

    #説明変数3個
    IndV3_Xtrain = np.column_stack((wind_train, temperature_train, ozone_train))
    IndV3_Ytrain = solor_train
    IndV3_Xtest = np.column_stack((wind_test, temperature_test, ozone_test))
    IndV3_Ytest = solor_test

    #説明変数5個
    IndV5_Xtrain = np.column_stack((wind_train, temperature_train, ozone_train, mounth_train, day_train))
    IndV5_Ytrain =solor_train
    IndV5_Xtest = np.column_stack((wind_test, temperature_test, ozone_test, mounth_test, day_test))
    IndV5_Ytest = solor_test

    return IndV2_Xtrain, IndV2_Ytrain, IndV2_Xtest, Indv2_Ytest, IndV3_Xtrain, IndV3_Ytrain, IndV3_Xtest, IndV3_Ytest, IndV5_Xtrain, IndV5_Ytrain, IndV5_Xtest, IndV5_Ytest

#モデルの作成と学習
def IndV2model_studying(IndV2_Xtrain, IndV2_Ytrain):
    model_as_IndV2 = LinearRegression()
    model_as_IndV2.fit(IndV2_Xtrain, IndV2_Ytrain)
    return model_as_IndV2

def IndV3model_studying(IndV3_Xtrain, IndV3_Ytrain):
    model_as_IndV3 = LinearRegression()
    model_as_IndV3.fit(IndV3_Xtrain, IndV3_Ytrain)
    return model_as_IndV3

def IndV5model_studying(IndV5_Xtrain, IndV5_Ytrain):
    model_as_Indv5 = LinearRegression()
    model_as_Indv5.fit(IndV5_Xtrain, IndV5_Ytrain)
    return model_as_Indv5

#モデルの評価
def IndV2_predict(IndV2_Xtest, model_as_IndV2):
    IndV2_Ypred = model_as_IndV2.predict(IndV2_Xtest.reshape(-1, 1))
    return IndV2_Ypred

def IndV3_predict(IndV3_Xtest, model_as_IndV3):
    IndV3_Ypred = model_as_IndV3.predict(IndV3_Xtest.reshape(-1, 1))
    return IndV3_Ypred

def IndV5_predict(IndV5_Xtest, model_as_IndV5):
    IndV5_Ypred = model_as_IndV5.predict(IndV5_Xtest.reshape(-1, 1))
    return IndV5_Ypred

#Numpyで実装した場合の学習及びベクトル,バイアスの定義
def Study_as_IndV2(IndV2_Xtrain, IndV2_Ytrain):
    slope, intercept = np.polyfit(IndV2_Xtrain.flatten(), IndV2_Ytrain, 1)
    IndV2_w = np.array([slope])
    IndV2_b = intercept
    return IndV2_w, IndV2_b

def Study_as_IndV3(IndV3_Xtrain, IndV3_Ytrain):
    slope, intercept = np.polyfit(IndV3_Xtrain.flatten(), IndV3_Ytrain, 1)
    IndV3_w = np.array([slope])
    IndV3_b = intercept
    return IndV3_w, IndV3_b

def Study_as_IndV5(IndV5_Xtrain, IndV5_Ytrain):
    slope, intercept = np.polyfit(IndV5_Xtrain.flatten(), IndV5_Ytrain, 1)
    IndV5_w = np.array([slope])
    IndV5_b = intercept
    return IndV5_w, IndV5_b

#予測結果の表示
def get_IndV2model_result(IndV2_Xtest, IndV2_w, IndV2_b):
    print("予測結果:")
    IndV2_predicted = np.dot(IndV2_Xtest.reshape(-1, 1), IndV2_w) + IndV2_b
    for i in range(len(IndV2_Xtest)):
        print("風力: {:.2f}\n温度: {:.2f}->  予測日射量: {:.2f}".format(IndV2_Xtest[i], IndV2_predicted[i]))
    return IndV2_predicted

if __name__ == "__main__":
    training_data = create_dataset()
    print(f"training_data: {training_data}")

    training_data_set, test_data_set = split_training_and_test(training_data)
    print(f"training_data_set: {training_data_set}\ntest_data_set: {test_data_set}")

    (
        IndV2_Xtrain, IndV2_Ytrain, IndV2_Xtest, Indv2_Ytest,
        IndV3_Xtrain, IndV3_Ytrain, IndV3_Xtest, IndV3_Ytest,
        IndV5_Xtrain, IndV5_Ytrain, IndV5_Xtest, IndV5_Ytest
    ) = create_IndV_and_DepV_Array(training_data_set, test_data_set)

    model_as_IndV2 = IndV2model_studying(IndV2_Xtrain, IndV2_Ytrain)
    IndV2_Ypred = IndV2_predict(IndV2_Xtest, model_as_IndV2)
    IndV2_w, IndV2_b = Study_as_IndV2(IndV2_Xtrain, IndV2_Ytrain)
    IndV2_predicted = get_IndV2model_result(IndV2_Xtest, IndV2_w, IndV2_b)