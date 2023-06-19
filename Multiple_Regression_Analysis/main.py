import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import csv

# 科学的表記を通常の浮動小数点表記に変更
np.set_printoptions(suppress=True)

#データセットの読み込み、整形
def create_dataset():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    IndV = diabetes.data
    DepV = diabetes.target.reshape(-1, 1)
    training_data = np.hstack((IndV, DepV))
    return training_data

#トレーニングデータをトレーニングセットとテストセットに分割
def split_training_and_test(training_data):

if __name__ == "__main__":
    training_data = create_dataset()
    print(training_data)