import numpy as np
import sklearn
from sklearn import datasets as dt
import matplotlib.pyplot as plt

#データセットの読み込み、整形
def create_dataset():
    iris = dt.load_iris()
    print(iris)

if __name__ == "__main__":
    create_dataset()