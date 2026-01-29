# ESNによる疎観測状態の復元と信号再構成

本リポジトリは、Echo State Network（ESN）において  
**一部の時刻でのみ観測されたリザバー状態から、全時刻の状態を復元し、出力信号を再構成する**手法を実装したものです。

測定点数を減らすことで生じる情報欠落に対し、  
状態遷移モデルを学習し、観測点では状態をリセットしつつ、観測間はロールアウトによって補間することで、  
少ない観測点数でも信号を再構成できるかを検証します。

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
python3 -m scripts.run_experiment
```
実行例（オプション指定）

実験条件を明示して実行する場合の例です。
```bash
python3 -m scripts.run_experiment \
  --T 100 \
  --washout 20 \
  --step_obs 60 \
  --N_train 100 \
  --N_test 30 \
  --M 100 \
  --K_obs 100 \
  --plot 1
```
