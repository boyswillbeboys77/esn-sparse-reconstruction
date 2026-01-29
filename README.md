# ESNによる疎観測状態の復元と信号再構成

　私の研究室では、光信号の測定を行っています。一般に、測定点数を増やすほど精度は向上しますが、その分測定時間が長くなるという課題があります。一方で、測定点数を減らせば短時間で測定できますが、精度が低下してしまいます。そこで私は、短時間で精度よく測定する方法はないかと考え、測定していない点を予測・復元する手法に着目しました。本研究では、Echo State Network（ESN）と呼ばれる機械学習手法を用いて、測定間の信号を推定する方法を検討しています。従来の測定では、各測定点はその時点の情報しか持ちませんが、ESNを用いることで、測定点に過去の情報を含めることが可能になります。そのため、限られた測定点からでも、信号全体を推定できると考えました。


---

## 手法概要

1. 真のESNに入力信号 `u(t)=y(t)` を与え、状態系列 `X(t)` を生成  
2. 疎に観測された状態から、線形状態遷移モデル  
   `x_{t+1} = A x_t + B u_t + b` を学習  
3. 観測時刻では状態をリセットし、観測間はロールアウトにより状態を復元  
4. 復元状態 `X̂(t)` から readout を学習し、出力信号 `y(t)` を再構成  
5. RMSE / nRMSE / R² により性能を評価

---

## 動作環境

- OS: **Linux / macOS**
- Python: **3.9 以上**
- bash / zsh などの Unix 系シェル

##以下のコマンドより実行できます

---

## リポジトリの取得

```bash
git clone https://github.com/boyswillbeboys77/esn-sparse-reconstruction.git
cd esn-sparse-reconstruction
```


セットアップ
（推奨）仮想環境の作成
```bash
python -m venv venv
source venv/bin/activate
```
依存ライブラリのインストール
```bash
pip install -r requirements.txt
```
実行方法

デフォルト設定で実験を実行します。
```bash
python3 -m scripts.run_experiment --plot 1 
```

### 結果が表示されない場合

WSLなどGUIが利用できない環境では、以下のコマンドで、Windows側のエクスプローラーから結果を確認できます。

```bash
explorer.exe esn_results
```
