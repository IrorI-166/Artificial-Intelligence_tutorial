import numpy as np
from sklearn.model_selection import train_test_split
import csv

#データセットの読み込み、必要データの抽出
def create_dataset():
    with open("Multiple_Regression_Analysis/airquarity_data.txt", "r") as data:
        reader = csv.reader(data, delimiter=" ")
        next(reader)
        training_data = []
        for row in reader:
            Ozone = int(row[1]) if row[1] != "NA" else None
            Solor = int(row[2]) if row[2] != "NA" else None
            Wind = float(row[3]) if row[3] != "NA" else None
            Temperature = int(row[4]) if row[4] != "NA" else None
            Mounth = int(row[5]) if row[5] != "NA" else None
            Day = int(row[6]) if row[6] != "NA" else None
            training_data.append([Ozone, Solor, Wind, Temperature, Mounth, Day])
    training_data = np.array(training_data)
    return training_data

#データセットの分割
def split_data(training_data):
    ozone = [d[0] for d in training_data]
    solor = [d[1] for d in training_data]
    wind = [d[2] for d in training_data]
    temprature = [d[3] for d in training_data]
    mounth = [d[4] for d in training_data]
    day = [d[5] for d in training_data]
    return ozone, solor, wind, temprature, mounth, day

#トレーニングデータをトレーニングセットとテストセットに分割
def split_training_and_test(training_data):
    training_data_set, test_data_set = train_test_split(training_data, test_size=0.2, random_state=42)
    return training_data_set, test_data_set

if __name__ == "__main__":
    training_data = create_dataset()
    print(f"training_data: {training_data}")
    training_data_set, test_data_set = split_training_and_test(training_data)
    print(f"training_data_set")