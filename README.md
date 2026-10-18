# Graduation_Project

データを準備する方法
以下のコマンドによって選んだデータセットのクリーンと分割をおこなう。

BigVulデータセットをクリーンしてJsonlにする。
(venv) yudai@lyuyan-X570-AORUS-ELITE:~/Project/research/Graduation_Project$ python data/scripts/make_bigvul_json.py --input_csv data/raw/MSR_data_cleaned.csv --output_jsonl data/cleaned_data/bigvul_defect.jsonl

DiverseVulデータセットをクリーンしてJsonlにする。
(venv) yudai@lyuyan-X570-AORUS-ELITE:~/Project/research/Graduation_Project$ python data/scripts/make_diversevul_json.py --input_json data/raw/diversevul_20230702.json --output_jsonl data/cleaned_data/diversevul.jsonl

PrimeVulデータセットをクリーンする
(venv) yudai@lyuyan-X570-AORUS-ELITE:~/Project/research/Graduation_Project$ python data/scripts/clean_primevul.py --output_jsonl data/cleaned_data/primevul_defect.jsonl



クリーンして、分割後のBigVulとDiverseVulのデータセットはdata/cleaned_data/内にある。
DiverseVulでつかわれていた、過去のプロジェクトをたくさん合わせて、プロジェクトごとに訓練、検証、テスト内で重複しないようにしたデータセットはdata/raw/new_six_by_projectsフォルダ内にある。







Please change the amound of data to use for train by adjusting the number of "result = result.head(1000)"

Preparing CPG and embed
python run.py -cpg -embed -mode



training:
# 例1: データ10% を使って学習させたい(ハイパラチューニング時に便利)
python run.py -mode train -path ./data/model/ --subset_frac 0.1 

# 例2: フルデータ を 1 epoch で使って最終学習したい
python run.py -mode train -path ./data/model/ --no_subset



test:
python3 run.py -mode test -path ./data/model/bertggcn.pth



Trainingの後に以下のように一番良いパラメータが表示されますので、一番上のモデルを使ってください。
===== SWEEP SUMMARY (sorted by best_f1) =====
batch=16, lr=0.0001, best_f1=0.1848 (epoch 12), model=data/model/bert_rgcn_bs16_lr1e-04.pth
batch=32, lr=3e-05, best_f1=0.1817 (epoch 14), model=data/model/bert_rgcn_bs32_lr3e-05.pth
batch=8, lr=0.0001, best_f1=0.1686 (epoch 9), model=data/model/bert_rgcn_bs8_lr1e-04.pth
batch=32, lr=1e-05, best_f1=0.1436 (epoch 4), model=data/model/bert_rgcn_bs32_lr1e-05.pth
batch=16, lr=3e-05, best_f1=0.1369 (epoch 3), model=data/model/bert_rgcn_bs16_lr3e-05.pth
batch=8, lr=1e-05, best_f1=0.1237 (epoch 1), model=data/model/bert_rgcn_bs8_lr1e-05.pth
batch=8, lr=5e-05, best_f1=0.1230 (epoch 4), model=data/model/bert_rgcn_bs8_lr5e-05.pth
batch=16, lr=1e-05, best_f1=0.1171 (epoch 1), model=data/model/bert_rgcn_bs16_lr1e-05.pth
batch=16, lr=5e-05, best_f1=0.1038 (epoch 1), model=data/model/bert_rgcn_bs16_lr5e-05.pth
batch=32, lr=5e-05, best_f1=0.1010 (epoch 1), model=data/model/bert_rgcn_bs32_lr5e-05.pth
batch=32, lr=0.0001, best_f1=0.0334 (epoch 4), model=data/model/bert_rgcn_bs32_lr1e-04.pth
batch=8, lr=3e-05, best_f1=0.0063 (epoch 1), model=data/model/bert_rgcn_bs8_lr3e-05.pth

この例の場合は一番上のbert_rgcn_bs16_lr1e-04.pthが一番いいモデルなのでテストのときは以下のように実行します。
python3 run.py -mode test -path ./data/model/bert_rgcn_bs16_lr1e-04.pth




🚀 実行ステップ
1️⃣ CPG 生成
python main.py --cpg


Joern を使用して CPG（Code Property Graph）を構築

data/cpg/ 以下に保存

2️⃣ 埋め込み生成
python main.py --embed


CodeBERT により各ノード埋め込みを生成

data/embed/ に保存

3️⃣ モデル学習（Train）

モデル学習は以下の 2 モードで実行できます。

🧪 A. ハイパーパラメータ Sweep（デフォルト）

以下のみ指定した場合は、複数のハイパーパラメータを自動探索します：

python main.py --mode train


探索範囲：

パラメータ	候補値
batch_size	[8, 16, 32]
lr	[1e-4, 5e-5, 3e-5, 1e-5]
bert_lr_ratio	[0.1, 0.2, 0.3]

早期終了（EarlyStopping）は --patience により設定可能
デフォルトは 5 epoch（改良が無い場合に停止）

🎯 B. ハイパーパラメータ指定トレーニング

以下の引数を指定することで、単一設定に対してトレーニングが可能：

引数	説明
--batch_size	バッチサイズ指定
--lr	学習率指定
--bert_lr_ratio	CodeBERT 部分の学習率比率
--patience	EarlyStopping の patience
--subset_frac	1 epoch で使用するデータ割合（高速チューニング用）
--no_subset	常に全学習データ使用

例：

python main.py --mode train \
  --batch_size 16 \
  --lr 1e-4 \
  --bert_lr_ratio 0.2 \
  --patience 10 \
  --no_subset

🧾 ログ & 保存物

モデルパラメータ

models/bert_rgcn_bs{batch}_lr{lr}_br{ratio}.pth

学習曲線

plots/loss_curve_*.png

plots/training_metrics_*.png

全パラメータの結果要約は学習終了後に表示されます：

===== SWEEP SUMMARY (sorted by best_f1) =====
batch=16, lr=1e-4, bert_ratio=0.2 ...

🧩 推論・評価（Test）

Val でベスト閾値を探索 → Test 評価

python main.py --mode test --path ./models


出力例：

Test (thr=0.50): Loss=..., Acc=..%, P=..%, R=..%, F1=..%
Test (best thr=0.62): Loss=..., Acc=..%, P=..%, R=..%, F1=..%

🔍 便利オプション一覧
オプション	役割
--cpg	CPG 生成
--embed	埋め込み生成
--mode train	学習
--mode test	テスト評価
--path	モデル保存先 or 読み込み先指定
--subset_frac X	1 epoch 当たり X 割の学習データを使用
--no_subset	全データ使用（データ大時は時間要注意）
--patience K	EarlyStopping K Epoch
--batch_size	1 設定で学習
--lr	1 設定で学習
--bert_lr_ratio	1 設定で学習












---

## **① BigVul：CPG + 埋め込みを一括生成**

```bash
python main.py --dataset bigvul --cpg --embed
```

---

## **② BigVul：すでに CPG がある前提で埋め込みだけ実行**

```bash
python main.py --dataset bigvul --embed
```

---

## **③ BigVul：学習（デフォルト設定）**

```bash
python main.py --dataset bigvul --mode train
```

---

## **④ BigVul：学習（batch=16, lr=3e-5, patience=30）**

```bash
python main.py --dataset bigvul --mode train --batch_size 16 --lr 3e-5 --patience 30
```

---

## **⑤ BigVul：テスト**

```bash
python main.py --dataset bigvul --mode test
```

---

---

## **⑥ DiverseVul：CPG → Embed の一括実行**

```bash
python main.py --dataset diversevul --cpg --embed
```

---

## **⑦ DiverseVul：埋め込みを resume（途中から再開）**

```bash
python main.py --dataset diversevul --embed --resume
```

---

## **⑧ DiverseVul：学習（subset_frac=0.05 で 5% データのみ使用）**

```bash
python main.py --dataset diversevul --mode train --subset_frac 0.05
```

---

---

## **⑨ Six(new_six_by_projects)：CPG + 埋め込み同時実行**

```bash
python main.py --dataset six --cpg --embed
```

---

## **⑩ Six：学習（全データを 1 epoch で使いたいので no_subset 指定）**

```bash
python main.py --dataset six --mode train --no_subset
```

---

# ⭐ さらに便利な実務用セット（おまけ）

## **Six データセットを最初から最後まで一気に流す**

```bash
python main.py --dataset six --cpg --embed
python main.py --dataset six --mode train
python main.py --dataset six --mode test
```

## **DiverseVul を全部流す**


python main.py --dataset diversevul --cpg --embed
python main.py --dataset diversevul --mode train
python main.py --dataset diversevul --mode test



訓練のコマンドでデータセットを指定しても意味ない。ここでは、直近に作られたcpgとembedで使ったデータセットをつかう。そのため、diversevulを前回使って、今回はBigVulを使いたい場合はBigVulでCPGとEmbedを作る必要がある。



# 使い方例
Bigvul データセットを使用して、以下の条件でパラメータファインチューニングする時
patience=30, データセットは１エポックで全体を使う。

=> (venv) yudai@lyuyan-X570-AORUS-ELITE:~/Project/research/Graduation_Project$ python run.py -mode train -path ./data/trained_models/model_patience=30_subsetfrac_0.1_dataset_bigvul/ --patience 30 --no_subset --dataset bigvul

モデルを収納するファイルは-pathで指定する

