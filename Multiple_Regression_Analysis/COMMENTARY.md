# 重回帰分析とは
単回帰では1つであった説明変数が二個以上になった回帰分析のこと

# ソースコードの詳細解説
#### 25~55行目
```py
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
```