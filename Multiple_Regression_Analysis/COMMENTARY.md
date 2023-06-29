# 重回帰分析とは
単回帰では1つであった説明変数が二個以上になった回帰分析のこと

# ソースコードの詳細解説
概ね単回帰分析と変わらないのでほぼ省略。
変わった部分は説明変数が1個から10個になった点のみ。
### 58~61行目
```py
#モデルの学習(scikit-learn)
def model_fit(X_train, Y_train_disease_progression):
    model = LinearRegression()
    model.fit(X_train, Y_train_disease_progression)
    return model
```
scikit-learnを用いた重回帰分析モデルの学習では、X_trainにn行10列の行列として説明変数のトレーニングデータを格納し、それによって導出される目的変数1つをn行1列の行列としてY_trainに格納、`fit`メソッドに渡すことで、以下のプロセスを経て学習が行われる。
```
1. (m, n)行列のX_trainと長さmのベクトルY_trainを用意し、説明変数Xと目的変数Yのデータを代入する。
2. 行列X_trainとベクトルY_trainの内積を計算し、パラメータ（回帰係数）を仮決定する。
3. 最小二乗法を使用して、仮決定したパラメータを用いて損失関数を最小化する。
4. 損失関数が最小であればパラメーターを決定、そうでなければ2から再実行。
```