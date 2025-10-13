import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# 1. データ準備と前処理: Pandasの役割
# ----------------------------------------------------
# df = pd.read_csv('data.csv') 
# 実際にCSVファイルがある場合は使用

# (ここではCSV読み込みの代わりにDataFrameを作成し、データ整形をシミュレートする)
data = {
    '特徴量_X': [1, 2, 3, 4, 5, 6, 7, 8],
    '正解ラベル_y': [11, 22, 33, 44, 55, 66, 77, 88],
    '不要な列': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] # 汚いデータとして文字列を想定
}

df = pd.DataFrame(data)

print("PandasでDataFrameを読み込みました。")
print(df)
print("-" * 30) # 区切り線

# ----------------------------------------------------
# 2. データの可視化 (探索的データ分析: EDA)
# ----------------------------------------------------
# seabornのscatterplot（散布図）を使って、Xとyの関係をプロットする。
# x軸に'特徴量_X'、y軸に'正解ラベル_y'を指定。
sns.scatterplot(x='特徴量_X', y='正解ラベル_y', data=df)

plt.title('Relationship between X and y') # グラフのタイトル
plt.xlabel('Feature (X)') # x軸のラベル
plt.ylabel('Label (y)') # y軸のラベル
plt.grid(True) # グラフに見やすくグリッドを追加
plt.show() # グラフを表示

# 今回のデータは y = 11x という完璧な直線関係なので、散布図も綺麗な直線状に点が並ぶ。
# ----------------------------------------------------
# 3. NumPyへの変換と学習データの分割
# ----------------------------------------------------

# Pandas DataFrameから、NumPy配列 (nd.array) へと変換し、Xとyを確定する
# X (特徴量) はscikit-learnの仕様上、2次元配列 [サンプル数, 特徴量の数] である必要がある。
# df[['特徴量_X']] のように列名をリストで囲むと2次元のDataFrameとして抽出でき、.valuesで2次元のNumPy配列に変換できる。
X = df[['特徴量_X']].values 

# y (正解ラベル) は1次元配列 [サンプル数] で良いため、.valuesでそのまま変換する。
y = df['正解ラベル_y'].values

# データを学習用(75%)とテスト用(25%)に分割
# random_stateを使う理由は分割の仕方を固定し、同じ値なら何度実行しても同じように分割されるため再現しやすい。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ----------------------------------------------------
# 4. 機械学習モデルの学習・予測・評価: Scikit-learnの役割
# ----------------------------------------------------

# モデルのインスタンスを作成と学習
model = LinearRegression() 
model.fit(X_train, y_train) 

#モデルは「y = aX + b」の最適な a (係数) と b (切片) を見つけるものである。
print("モデルの学習が完了しました！")
print(f"学習後のモデルの傾き (係数 a): {model.coef_}")
print(f"学習後の切片 (b): {model.intercept_}")
print("-" * 30)

# 学習済みモデルで予測を実行
y_pred = model.predict(X_test)

print("モデルの予測が完了しました！")
print(f"テストデータ (X_test): \n{X_test}")
print(f"正解ラベル (y_test): {y_test}")
print(f"予測結果 (y_pred): {y_pred}")
print("-" * 30)

# ----------------------------------------------------
# 5. 予測結果の可視化
# ----------------------------------------------------
# モデルの予測がどれだけ正解に近いかを視覚的に確認。
# 元々の正解データを点でプロット（散布図）
plt.scatter(X_test, y_test, color='black', label='Actual Data (正解データ)')
# モデルが予測した結果を線でプロット（回帰直線）
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted Regression Line (予測直線)')

plt.title('Actual vs. Predicted')# タイトルを表示
plt.xlabel('Feature (X_test)')# X軸を表示
plt.ylabel('Label (y_test / y_pred)')# y軸を表示
plt.legend() # ラベルを表示
plt.grid(True)# グリッド線を表示
plt.show()

# 予測直線が正解データの点に重なるほど学習は成功している。

# ----------------------------------------------------
# 6. モデルの評価 (metrics)
# ----------------------------------------------------
# 平均二乗誤差 (Mean Squared Error, MSE) を計算。
# 予測値と正解値の差を二乗して平均したもので、値が0に近いほど誤差が小さいことを意味する。
MSE = mean_squared_error(y_test, y_pred) 
print(f"平均二乗誤差 (Mean Squared Error, MSE): {MSE}") 

# 決定係数 (R^2 Score) を計算。
# 0から1の値をとり、1に近いほどモデルの当てはまりが良いことを意味する。
R2 = r2_score(y_test, y_pred)
print(f"決定係数 (R^2 Score): {R2}")

# 今回のデータは完璧なため、誤差(MSE)はほぼ0になり、決定係数(R2)は1になる。
