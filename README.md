MLは、機械学習の基礎学習を証明するためのコードです。pandasのデータ処理、matplotlibでデータの視覚化、NumPyの学習データ分割、scikit-learnで機械学習の基礎である線形関数を実装した機械学習の基礎を詰め込んだ汎用型機械学習コードです。学習の理解を深めるためのコメントアウトやcsvファイル読み込みや複数のモデル評価など今後の汎用性を考慮したコード設計を目指しました。

## ✨ 主な機能
- **データハンドリング**: Pandas を使用してデータを機械学習に適した形式に整形します。
- **探索的データ分析 (EDA)**: Matplotlib と Seaborn を活用し、特徴量と目的変数の関係性を散布図で可視化します。
- **モデル学習**: Scikit-learn の LinearRegression モデルを使用し、学習データ（訓練データ）にフィットさせて回帰直線の係数と切片を算出します。
- **予測と評価**: 学習済みモデルを用いて、テストデータに対する予測値を算出します。その後、予測結果を平均二乗誤差 (MSE) と 決定係数 (R^2) という2つの評価指標で評価します。
- **結果の可視化**: 実際のデータ点とモデルが予測した回帰直線を同一グラフ上にプロットし、モデルの精度を直感的に確認できます。

## ⚙️ 必要な環境
- Python 3.8 以降

## 🚀 セットアップ方法
1. **リポジトリのクローン**
    ```bash
        git clone<https://github.com/kmtkzy1379/portfolio2-ML.git>
    cd portfolio2-ML
    ```

2. **仮想環境の作成とアクティベート**
    ```bash
    python -m venv evenv
    evenv\Scripts\activate
    ```

3. **依存関係のインストール**
    ```bash
    pip install -r requirements.txt
    ```

7. **アプリケーションの実行**  
      ```bash
      python app/model_training.py
      ```

## 🔧 技術スタック
- **データの読み込み、操作、前処理**: Pandas
DataFrameとして読み込みデータ整形をします。
- **高速な数値計算、特に多次元配列の操作**: NumPy
操作PandasのDataFrameからscikit-learnが要求するデータ形式への変換します。
- **モデルの構築**: Scikit-learn
（LinearRegression）、データ分割（train_test_split）、性能評価（mean_squared_error, r2_score）など、機械学習の主要な機能を担います。
- **データの可視化**: Matplotlib & Seaborn
学習データ、学習結果をグラフ上に表示します。

## 🤝 貢献
このプロジェクトは学習・ポートフォリオ目的で作成されています。改善提案やバグ報告は歓迎します。

## 📄 ライセンス
MIT License
