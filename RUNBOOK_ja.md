# GPU マシンでの実行手順（RUNBOOK）

卒論で使ったワークステーション（RTX 3090、Ubuntu）を想定。所要時間は目安で、必ず手順 6 の小規模実行で計測してから本番を回すこと。

## 0. 前提

- NVIDIA ドライバ + CUDA 12.x が動く GPU（`nvidia-smi` で確認）
- Python 3.10 または 3.11
- Java 17 以上（Joern 4 用。開発時に動作確認したのは OpenJDK 21）
- BigVul の `MSR_data_cleaned.csv`（旧リポジトリの `data/raw/` にあるもの）
- ディスク: CPG 生成の中間ファイルは既定で削除されるが、`data/input/*.pkl` は数十 GB になりうる

## 1. 取得と環境

```bash
git clone https://github.com/s1300105/Graduation_Project_fixed.git
cd Graduation_Project_fixed
python3.10 -m venv venv && . venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 3090 なら cu124 でよい
pip install -r requirements.txt                                        # torch_geometric は pure-Python 版で動く（torch_scatter 不要）
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"   # True を確認
pytest -q        # 67 件。初回は CodeBERT（約 500 MB）をダウンロードする。CPU で数分
```

## 2. Joern 4

```bash
mkdir -p joern && cd joern
curl -L -o joern-cli.zip https://github.com/joernio/joern/releases/download/v4.0.400/joern-cli.zip   # 約 1.9 GB
unzip -q joern-cli.zip && rm joern-cli.zip && cd ..
joern/joern-cli/joern --version
```

`run.py` は既定で `joern/joern-cli/joern` と `joern/joern-cli/joern-parse` を呼び、書き出しスクリプトは `joern/graph-for-funcs.sc` を使う。別の場所に入れた場合は `--joern_cli_dir /path/to/joern-cli/` を付ける。

## 3. BigVul → jsonl（約 10 分）

```bash
mkdir -p data/raw data/cleaned_data
ln -s /path/to/MSR_data_cleaned.csv data/raw/MSR_data_cleaned.csv     # コピーでもよい
python data/scripts/make_bigvul_json.py \
    --input_csv data/raw/MSR_data_cleaned.csv \
    --output_jsonl data/cleaned_data/bigvul.jsonl
ls data/cleaned_data        # bigvul_train.jsonl / bigvul_valid.jsonl / bigvul_test.jsonl
```

- 旧リポジトリの `bigvul_defect_*.jsonl` を流用することもできる（`--dataset bigvul_defect`）が、`lang` / `project` 列が無いので C++ 判定が構文ヒント頼みになる。作り直しを推奨。
- プロジェクト横断分割は `--split project --output_jsonl data/cleaned_data/bigvul_xproj.jsonl`（後で `--dataset bigvul_xproj`）。

## 4. CPG 生成と埋め込み（最も時間がかかる。tmux / nohup で）

```bash
mkdir -p logs
nohup python -u run.py --cpg --embed --dataset bigvul > logs/cpg_embed.log 2>&1 &   # -u: ログを即時に書く（git pull 後は不要）
tail -f logs/cpg_embed.log
```

- `--cpg`: 500 関数ずつ Joern を回す（JVM 起動 2 回/スライス）。15 万関数で約 300 スライス。1 スライス 1〜3 分なら合計 5〜15 時間。`--slice_size 1000` で JVM 回数を半減できる（メモリに余裕があれば）。
- `--embed`: 凍結 CodeBERT でノード特徴を作る。GPU なら fp16 で動く。中断したら同じフラグで `python run.py --embed --resume --dataset bigvul`（フラグが違うと拒否される）。
- 1200 字フィルタは既定で無効。論文と同じ条件に揃えたいときだけ `--max_chars 1200` を付ける。
- 終わったら必ず manifest を見る:

```bash
python -c "import json;print(json.dumps(json.load(open('data/cpg/cpg_manifest.json'))['splits'],indent=1))"
cat data/input/manifest.json
```

  - `rows_lost_in_joern`: Joern で解析できず除外された関数数（論文に書く）。`rescued_stage_A` / `rescued_stage_B` はシグネチャのマクロを除いて救済できた件数
  - 除外が多い（1 割超）ときは `python scripts/diagnose_joern_losses.py --split train --slice 0 --max 60` で内訳を見る（`method_dropped` が出たらパイプラインのバグなので報告）
  - `skipped_global`: `<global>` の除外数（0 でなければ修正が効いている）
  - `positives_after_filter`: 各分割の陽性数
  - `nodes_mean` / `edges_mean` / `num_relations`（3。`--reverse_edges` なら 6）

## 5. 任意: 変種の入力を作る

同じ `data/input` を全モデルが読むので、入力の変種ごとに `--data_root` を分ける。

```bash
# 逆辺あり
python run.py --embed --dataset bigvul --reverse_edges --data_root data_rev        # cpg は data/cpg をコピー or 再生成
# ノードの CODE を先頭行だけにする（M22 の緩和）
python run.py --embed --dataset bigvul --node_code_mode first_line --data_root data_fl
```

（`--embed` は `<data_root>/cpg/*_cpg.pkl` を読むので、`cp -r data/cpg data_rev/cpg` のように CPG を先に置く。）

## 6. 小規模で計測（30 分）

```bash
python run.py --mode train --model_type codebert --subset_frac 0.05 --epochs 2 --patience 2 --seeds 0 --runs_dir runs_smoke
python run.py --mode train --model_type hybrid   --subset_frac 0.05 --epochs 2 --patience 2 --seeds 0 --runs_dir runs_smoke
python -c "import json;[print(json.loads(l)['epoch'],round(json.loads(l)['elapsed_s'])) for l in open('runs_smoke/hybrid_bs16_lr1.0e-04_br0.30_seed0/history.jsonl')]"
```

`elapsed_s` から 1 epoch の時間を出す（5% で t 秒なら全データで約 20t 秒）。GPU メモリが足りなければ `--batch_sizes 8 --eval_batch_size 8`。

## 7. 本番（推奨順）

論文の設定は bs16 / lr 1e-4 / CodeBERT lr 比 0.3 / 200 epoch / patience 50。旧コードの曲線では検証損失の最小が epoch 30〜75、F1 は 100 epoch 以降ほぼ平坦だったので、まず `--epochs 40 --patience 10` で全腕を揃え、必要なら伸ばす。

```bash
COMMON="--epochs 40 --patience 10 --seeds 0 1 2"
# (a) グラフ枝に信号が届いているか
nohup python run.py --mode train --model_type graph_only $COMMON                    > logs/graph_only.log 2>&1
nohup python run.py --mode train --model_type hybrid --ablation shuffle_edges $COMMON > logs/hybrid_shuffle.log 2>&1
# (b) 本比較
nohup python run.py --mode train --model_type codebert $COMMON                      > logs/codebert.log 2>&1
nohup python run.py --mode train --model_type hybrid   $COMMON                      > logs/hybrid.log 2>&1
# (c) 融合の部品
nohup python run.py --mode train --model_type hybrid --ablation no_gate     $COMMON > logs/hybrid_nogate.log 2>&1
nohup python run.py --mode train --model_type hybrid --ablation no_cls_skip $COMMON > logs/hybrid_nocls.log 2>&1
```

- 1 コマンドが 3 seed を順に回す。GPU が 1 枚なら上から順に（同時に走らせない）。
- 各 run は `runs/<model>_bs16_lr1.0e-04_br0.30_seed<k>/` に `metrics.json`（閾値 0.5 と検証調整閾値の両方）、`history.jsonl`、`curves.png`、`best.pth`、`test_preds.npz` を書く。
- 同名 run があると止まる（上書きは `--overwrite`）。

## 8. 判定

```bash
# 構造は使われているか: hybrid とランダム辺対照に差があるか
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --b "runs/hybrid-shuffle_edges_bs16_lr1.0e-04_br0.30_seed*"
# 本比較
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --b "runs/codebert_bs16_lr1.0e-04_br0.30_seed*"
```

- 出力の `A - B` と 95% 区間を見る。区間が 0 をまたげば「差があるとは言えない」。
- `graph_only` が chance 付近（F1 が陽性率程度）なら、グラフ枝単体に信号が届いていない。
- `history.jsonl` の `frac_p1_mid`（確率が 0.05〜0.95 にある割合）が 0 に近づいたら出力が飽和している。`metrics.json` の `n_thresholds_tied` が大きいのも同じ兆候。

## 9. 追加実験

- 長さ別の層別: `test_preds.npz` の `idx` を jsonl の `func` 長と突き合わせて、512 トークン超とそれ以下で F1 を分ける。
- プロジェクト横断: 手順 3 の `--split project` で作った jsonl を `--dataset bigvul_xproj --data_root data_xproj` で 4〜8 をやり直す。
- ハイパーパラメータ探索は最終モデル・全データで: `--batch_sizes 8 16 32 --lrs 1e-4 5e-5 3e-5 1e-5 --bert_lr_ratios 0.1 0.2 0.3 --seeds 0 --epochs 20 --patience 5`。

## 10. よくある問題

- `CUDA out of memory`: `--batch_sizes 8 --eval_batch_size 8`。学習ループは OOM バッチを飛ばして数える（`history.jsonl` の `oom_skipped`）。0 でないなら設定を下げる。
- Joern が落ちる / 遅い: `logs/cpg_embed.log` の `[ERROR]` 行と manifest の `joern_failed_slices`。JVM のヒープは `JAVA_OPTS=-Xmx8g` などで調整。
- GPU の scatter 演算は非決定的なので同一 seed でも完全一致はしない。だから 3 seed 以上で区間を出す。
- 旧リポジトリのチェックポイントはモデル形状が違うので読めない（学習し直す）。

## 11. 課題スイート（POJ-104 / CodeNet C++1000 / Devign）

設計判断と根拠は `docs/TASK_SUITE_ja.md`。BigVul の `data/` と `runs/` は触らず、データセットごとに `--data_root` と `--runs_dir` と `--run_prefix` を分ける（`--cpg` は `<data_root>/cpg` を、`--embed` は `<data_root>/input` を**消して作り直す**）。

### 11.1 取得と前処理

```bash
mkdir -p data/raw
# POJ-104: HF の parquet 3 本（13 MB）。53,000 行 → 分割またぎの重複 1,000 行とラベル衝突 20 行を除いて 51,980 行、104 クラス、層化 80/10/10
python data/scripts/make_poj104_json.py --input data/raw/poj104 --hf --out_dir data_poj/cleaned_data
# Devign (CodeXGLUE defect detection): parquet 3 本（22 MB）。公式分割を保ち、ラベル衝突 162 行・分割またぎ 2 行・重複 1 行・空 2 行を除いて 21,710 / 2,723 / 2,718
python data/scripts/make_devign_json.py --input data/raw/codexglue_defect --hf --out_dir data_devign/cleaned_data
# CodeNet C++1000: tarball 136 MB（1000 問題 × 500 本）。既定は 1 問題 100 本 = 10 万本。メタデータ（294 MB）があれば Accepted のみに絞り、--split user で利用者横断分割
curl -L -o data/raw/Project_CodeNet_C++1000.tar.gz https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_C++1000.tar.gz
tar -xzf data/raw/Project_CodeNet_C++1000.tar.gz -C data/raw/
python data/scripts/make_codenet_json.py --input data/raw/Project_CodeNet_C++1000 --out_dir data_codenet/cleaned_data --per_problem 100
# 小さく試すなら --problems 50 --per_problem 100
```

各スクリプトは `<dataset>_classes.json`（target → 問題 ID）と `<dataset>_prep_stats.json`（各段階の件数、重複、拡張子の内訳、K&R の件数）を書く。**学習より先にこれらをコミットする**（分割の凍結）。

### 11.2 CPG と埋め込み

```bash
# プログラム全体: Joern のファイル単位 <global> を 1 グラフにする（main と補助関数、グローバル変数とその初期化子を含む。#define の関数風マクロと呼び先スタブは含まない）
nohup python -u run.py --cpg --embed --dataset poj104  --data_root data_poj     --graph_unit whole_file --max_nodes 1000 > logs/poj_cpg_embed.log 2>&1
nohup python -u run.py --cpg --embed --dataset codenet --data_root data_codenet --graph_unit whole_file --max_nodes 1000 > logs/codenet_cpg_embed.log 2>&1
# 単一関数: 既定（first_method）。救済段階 C（小文字マクロ）が自動で効く
nohup python -u run.py --cpg --embed --dataset devign  --data_root data_devign > logs/devign_cpg_embed.log 2>&1
```

- `--max_nodes` の切り捨ては**ソース位置順**なので、プログラム全体では後方の関数（しばしば main）から消える。`input/manifest.json` の `graphs_at_cap` を見て、無視できない割合なら上げる（CodeNet の 200 本試走では 1000 ノードで 4 本 = 2%）。
- `cpg_manifest.json` の `whole_files_without_real_method` は Joern が何も解析できなかったファイル（殻にはせず、救済パス → `rows_lost_in_joern`）。`rescued_stage_C` は小文字マクロで救えた件数。
- `--node_code_mode full`（既定）では `<global>` の BLOCK ノードがファイル全文の CLS を持つ。テキスト枝との冗長性を抑えたい場合は `first_line` を別の `--data_root` で作って比較する。
- 多クラスのクラス数は `--embed` が `target` から推定して `input/manifest.json` に書く（`--num_classes` で明示も可。食い違えば止まる）。

### 11.3 学習と判定

```bash
COMMON="--epochs 40 --patience 10 --seeds 0 1 2 3 4"
for ds in poj codenet; do
  for arm in "graph_only" "hybrid --ablation shuffle_edges" "codebert" "hybrid" "hybrid --ablation no_gate"; do
    name=$(echo $arm | tr ' ' '_' | tr -d '-')
    nohup python -u run.py --mode train --model_type $arm --data_root data_$ds --runs_dir runs_$ds --run_prefix ${ds}_ $COMMON > logs/${ds}_${name}.log 2>&1
  done
done
# 判定は macro-F1（均衡なので accuracy と同順）。対応つき bootstrap は同じ idx のテスト項目で揃える
python scripts/compare_runs.py --metric macro_f1 --a "runs_poj/poj_hybrid_bs16*" --b "runs_poj/poj_codebert_bs16*"
python scripts/compare_runs.py --metric macro_f1 --a "runs_poj/poj_hybrid_bs16*" --b "runs_poj/poj_hybrid-shuffle_edges_bs16*"
```

- 主比較は **5 seed 以上**（1〜2 点の差は 3 seed では見えない。`docs/TASK_SUITE_ja.md` §4）。
- 多クラスの `history.jsonl` には `val_accuracy` / `val_macro_f1` / `val_top5` が増え、`val_threshold` は null。`mean_p1` / `frac_p1_mid` は最大クラス確率に対する値。
- `--mode test` は run の `args.json` に記録した `data_root` と違うデータを拒否する（`--allow_other_data_root` で明示的に上書き）。
- 1000 クラスの CodeNet では `--loss ce` も試す（focal の α は均衡データでは全て ≈ 1 なので差は γ の項だけ）。

### 11.4 時間の目安（RTX 3090、BigVul 18 万関数の実績から外挿）

| 段階 | POJ-104（52k 本） | CodeNet（100k 本） | Devign（27k 関数） |
|---|---|---|---|
| Joern（`--cpg`） | 1 スライス 500 本あたり数十秒 → 1〜2 時間 | 2〜4 時間 | 1 時間弱 |
| ノード埋め込み（`--embed`） | ノード数は BigVul より多い（プログラム全体）ので BigVul と同程度〜倍 | BigVul の 1.5〜2 倍 | BigVul の 1/5 |
| 学習 1 epoch（hybrid） | BigVul の 1/3 程度 | BigVul の 2/3 程度 | BigVul の 1/6 程度 |

いずれも処理系の実測を取ってから本番の seed 数を決める（§6 の手順と同じ）。
