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
pytest -q        # 184 件。初回は CodeBERT（約 500 MB）をダウンロードする。CPU で数分
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
- `--embed`: 凍結 CodeBERT でノード特徴を作る。GPU なら fp16 で動く。中断したら同じフラグで `python run.py --embed --resume --dataset bigvul`（フラグが違うと拒否される）。各スライスの横に `<base>_input.meta.json`（元の cpg pickle のサイズと更新時刻、設定、件数）を書き、`--resume` はこれが一致するスライスだけを飛ばす。`--cpg` を作り直した後の `--resume` は、変わったスライスを自動で埋め込み直す（古い入力で学習しない）。記録の無い古いスライスは、入力が cpg pickle より新しいときだけ信用し、件数を数え直す。拒否される場合（フラグ違い、cpg が無い、`--graph_unit` 違い）は既存の `input/` を消さずに止まる。
- 1200 字フィルタは既定で無効。論文と同じ条件に揃えたいときだけ `--max_chars 1200` を付ける。
- 終わったら必ず manifest を見る:

```bash
python -c "import json;print(json.dumps(json.load(open('data/cpg/cpg_manifest.json'))['splits'],indent=1))"
cat data/input/manifest.json
```

  - `rows_lost_in_joern`: Joern で解析できず除外された関数数（論文に書く）。`rescued_stage_A` / `B` / `C` はシグネチャのマクロを除いて救済できた件数（C は小文字マクロと戻り値型マクロ。後から追加したので、古い CPG と比べるときは `--rescue_stages AB`）
  - 除外が多い（1 割超）ときは `python scripts/diagnose_joern_losses.py --split train --slice 0 --max 60` で内訳を見る（`method_dropped` が出たらパイプラインのバグなので報告）
  - `skipped_global`: `<global>` の除外数（0 でなければ修正が効いている）
  - `positives_after_filter`: 各分割の陽性数（ラベルが 0/1 のときだけ。多クラスでは null で、`label_counts` を見る）
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
# CodeNet C++1000: tarball 136 MB（1000 問題 × 500 本）。既定は 1 問題あたり最大 100 本（Accepted 絞り込み・dedup・空プログラム除去の後に抽出。配布版は既に Accepted・近似重複除去済みなので実測ではほぼ 100 本）。メタデータ（294 MB）があれば status で絞り、--split user で利用者横断分割（train の全クラス在籍だけを強制。valid / test に欠けるクラスは prep_stats の class_coverage を見る）
curl -L -o data/raw/Project_CodeNet_C++1000.tar.gz https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_C++1000.tar.gz
tar -xzf data/raw/Project_CodeNet_C++1000.tar.gz -C data/raw/
python data/scripts/make_codenet_json.py --input data/raw/Project_CodeNet_C++1000 --out_dir data_codenet/cleaned_data --per_problem 100
# 小さく試すなら --problems 50 --per_problem 100
```

各スクリプトは `<dataset>_classes.json`（target → 問題 ID）と `<dataset>_prep_stats.json`（各段階の件数、重複、拡張子の内訳、K&R の件数、問題ごとの本数）を書く。**学習より先にこれらをコミットする**（分割の凍結）。

注意: `input/*.pkl` と `cpg/*.pkl` は pandas のメジャー版をまたいで読めない（pandas 3 で書いた pickle は pandas 2.2 で `TypeError`）。`--cpg --embed` と `--mode train` は同じ環境で実行する。前処理の jsonl は移植できる。

### 11.2 CPG と埋め込み

```bash
# プログラム全体: Joern のファイル単位 <global> を 1 グラフにする（main と補助関数、グローバル変数とその初期化子を含む。#define の関数風マクロと呼び先スタブは含まない）
# --graph_unit は --cpg と --embed の両方に同じ値を渡す（別々に実行して食い違うと --embed が止まる）
nohup python -u run.py --cpg --embed --dataset poj104  --data_root data_poj     --graph_unit whole_file --max_nodes 1000 > logs/poj_cpg_embed.log 2>&1
nohup python -u run.py --cpg --embed --dataset codenet --data_root data_codenet --graph_unit whole_file --max_nodes 1000 > logs/codenet_cpg_embed.log 2>&1
# 単一関数: 既定（first_method）。救済段階 C（小文字マクロ）が自動で効く
nohup python -u run.py --cpg --embed --dataset devign  --data_root data_devign > logs/devign_cpg_embed.log 2>&1
```

- `--max_nodes` の切り捨ては**ソース位置順**なので、プログラム全体では後方の関数（しばしば main）から消える。`input/manifest.json` の `graphs_at_cap` を見て、無視できない割合なら上げる（CodeNet の 200 本試走では 1000 ノードで 4 本 = 2%）。
- `cpg_manifest.json` の `whole_files_without_real_method` は関数を 1 つも定義していないファイル（`#define` や構造体だけ、または K&R の `main()` を `.cpp` で渡したなどの解析失敗）。殻のグラフにはせず `rows_lost_in_joern` に数える。プログラム全体では救済パスを走らせない（`rescue_skipped_whole_file`）。C++ のラムダは Joern が別メソッドとして出すので、本体を `<global>` のグラフに合流させる（`lambda_methods_folded`）。
- `--node_code_mode full`（既定）では各関数の METHOD / BLOCK ノードがその関数全文の CLS を持つ（`<global>` 自身の BLOCK は Joern 4.0.400 では `<empty>`）。テキスト枝との冗長性を抑えたい場合は `first_line` を別の `--data_root` で作って比較する。
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
- `--mode test` は run の `args.json` に記録した `data_root` と違うデータを拒否する（`--allow_other_data_root` で明示的に許可）。許可した場合も run 自身の `metrics.json` / `*_preds.npz` は上書きせず、`<run>/reeval_<data_root の名前>/` に書く。クラス数・関係数・ノード特徴の次元がチェックポイントと違う入力は、フラグに関係なく拒否する。
- `metrics.json` の `best_val_selection_f1` は名前を保ったまま、`--select_on` の指標の値を持つ（多クラスでは macro-F1、`accuracy` なら正解率）。どの指標かは `best_val_selection_metric` に書く。
- `compare_runs.py` は、共有する idx でラベルが食い違う run 同士（別データセットの取り違え）、idx が重複・欠損（-1）の run、共通 idx が無い組を拒否する。共通部分が一部だけのときは件数を表示する。
- 1000 クラスの CodeNet では `--loss ce` も試す（focal の α は均衡データでは全て ≈ 1 なので差は γ の項だけ）。

### 11.4 時間の目安（RTX 3090、BigVul 18 万関数の実績から外挿）

| 段階 | POJ-104（52k 本） | CodeNet（100k 本） | Devign（27k 関数） |
|---|---|---|---|
| Joern（`--cpg`） | 1 スライス 500 本あたり数十秒 → 1〜2 時間 | 2〜4 時間 | 1 時間弱 |
| ノード埋め込み（`--embed`） | ノード数は BigVul より多い（プログラム全体）ので BigVul と同程度〜倍 | BigVul の 1.5〜2 倍 | BigVul の 1/5 |
| 学習 1 epoch（hybrid） | BigVul の 1/3 程度 | BigVul の 2/3 程度 | BigVul の 1/6 程度 |

いずれも処理系の実測を取ってから本番の seed 数を決める（§6 の手順と同じ）。

## 12. 化学（分子）: SMILES ＋ 分子グラフ

設計と事前登録は `docs/CHEM_PROTOCOL_ja.md`、結果は `docs/CHEM_RESULTS_ja.md`。BACE と BBBP は CPU でも回る（この repo の結果は 4 コア CPU で作った）。HIV と 10 seed・100 エポックの本番は GPU で回す。

**`docs/results/chem_cpu_v1/` は学習データの行順に欠陥があった登録 run**（追補 2）。この版より前の `run.py --molgraph` で作った入力（manifest に `row_order` が無いもの）は、作り直してから学習する。訂正後の結果は `docs/results/chem_cpu_v2/`。

```bash
pip install rdkit                      # 結果は RDKit 2026.09.1 で作った（版で解析できる SMILES と原子特徴が少し変わる）

# 一括（準備 → 全腕 → 集計）。中断しても同じコマンドで続きから再開する
python scripts/chem_grid.py --root experiments/chem --datasets ogb_molbace ogb_molbbbp --seeds 0 1 2 3 4 5 6 7 8 9 \
    --workers 1 --threads 8 --extra "--epochs 100 --patience 100"
python scripts/chem_grid.py --root experiments/chem --datasets ogb_molhiv --seeds 0 1 2 3 4 5 6 7 8 9 \
    --workers 1 --threads 8 --extra "--epochs 100 --patience 100"
cat experiments/chem/summary.md
```

個別に動かす場合:

```bash
python data/scripts/make_molnet_json.py --dataset ogb_molhiv --input data/raw/ogb --download --out_dir data_hiv/cleaned_data
python run.py --molgraph --dataset ogb_molhiv --data_root data_hiv --text_model DeepChem/ChemBERTa-77M-MTR
COMMON="--data_root data_hiv --runs_dir runs_hiv --run_prefix hiv_ --loss ce --select_on auc --epochs 100 --patience 100 --max_length 256 --eval_batch_size 64 --seeds 0 1 2 3 4 5 6 7 8 9"
python run.py --mode train --model_type codebert $COMMON                       # A1 テキストのみ
python run.py --mode train --model_type graph_only $COMMON                     # A2
python run.py --mode train --model_type hybrid $COMMON                         # A3
python run.py --mode train --model_type hybrid --ablation no_edges $COMMON     # A3e
python run.py --mode train --model_type hybrid --ablation shuffle_edges $COMMON  # A3s
python scripts/fingerprint_baseline.py --data_root data_hiv --dataset ogb_molhiv --runs_dir runs_hiv --run_prefix hiv_ --seeds 0 1 2 3 4
python scripts/compare_runs.py --metric auc --seed_bootstrap --a "runs_hiv/hiv_hybrid_bs16*" --b "runs_hiv/hiv_codebert_bs16*"
python scripts/compare_runs.py --metric auc --seed_bootstrap --a "runs_hiv/hiv_hybrid_bs16*" --b "runs_hiv/hiv_hybrid-no_edges_bs16*"
```

- **既定の focal loss と F1 閾値選択は使わない**（`--loss ce --select_on auc`）。scaffold 分割では valid と test の陽性率が大きく違う（BBBP: train 0.84 / valid 0.41 / test 0.53）ので、閾値つきの F1 は意味を失う。
- 指標は test ROC-AUC。HIV は陽性 3% なので PR-AUC も見る（`metrics.json` の `test_pr_auc`）。`final_epoch` は最後のエポックの値で、検証によるモデル選択に依存しない参照値。
- `--model_type codebert` は「テキストのみ」の腕の名前で、中身は manifest の `text_model`（ChemBERTa）になる。
- MoLFormer（`ibm-research/MoLFormer-XL-both-10pct`）で感度を見る場合は `--text_model ibm-research/MoLFormer-XL-both-10pct --trust_remote_code --max_length 202` を付ける。事前学習が立体情報なしの SMILES なので、`make_molnet_json.py` の出力の立体情報を落とすかどうかを先に決める（BACE では立体異性体でラベルが違うものがある）。
- HIV の時間の目安: CPU 4 コアでは 77M のテキスト枝つきの腕が 1 seed・30 エポックで約 4 時間。GPU なら数分〜十数分の見込み（未計測）。

## 13. 長さ別の確認: テキスト枝の窓の外にある欠陥を取りこぼしているか

新しい構造（LM が 512 トークンずつ読み、グラフが塊どうしをデータフロー・制御フローでつなぐ）に余地があるかを、手元の run の予測だけで先に確かめる。学習はしない（数分。大半は CSV の読み込み）。

```bash
python scripts/length_strata.py \
    --group codebert       "runs/codebert_bs16_lr1.0e-04_br0.30_seed*" \
    --group hybrid         "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" \
    --group hybrid-shuffle "runs/hybrid-shuffle_edges_bs16_lr1.0e-04_br0.30_seed*" \
    --group graph_only     "runs/graph_only_bs16_lr1.0e-04_br0.30_seed*" \
    --bigvul_csv data/raw/MSR_data_cleaned.csv --out runs/length_strata
cat runs/length_strata/summary.md
```

- 終わった run（`test_preds.npz` のある run）だけを使う。キューの途中でも、今ある run で先に見られる（codebert がまだなら hybrid 系と graph_only だけでもよい）。出力は `summary.md` / `summary.json` / `items.csv`（関数ごとのトークン数・分類・グループごとの平均確率）。
- 窓 W は run の `args.json` の text_model と max_length から決まる（CodeBERT・512 なら本文 510 トークン）。テキストは run が読んだテスト JSONL をそのまま、学習時と同じトークナイザで数える。
- `--bigvul_csv` は `make_bigvul_json.py` に渡したのと同じ CSV。脆弱な関数ごとに、修正で消えた・書き換えられた行（`func_before` と `func_after` の行の差分、空白の違いは無視）が窓の中か外かを判定する。行の位置は `make_bigvul_json.py` と同じ整形（空行・コメント・行頭空白の除去）を通して、モデルが読んだテキスト上のトークン位置で数える。追加だけの修正（チェックの追加）は、追加位置の直後の行で判定し、`insert:` を付けて別に数える。位置が決められない関数は `unknown:<理由>` として数え、黙って落とさない。
- **見るところ**は「Hidden vs visible flaws」の表。窓を超える関数のうち、欠陥が窓の外にある脆弱な関数（hidden）と窓の中にある関数（visible）について、窓を超える非脆弱な関数との AUC を比べる。hidden の AUC が明らかに低ければ、窓の外を読む構造に余地がある。同じくらいなら、モデルは欠陥そのものではなく長さや書き方で当てている（その場合、BigVul では新構造の差が出にくい）。窓を超える層の length-only AUC（長さだけで当てた AUC）も並べて見る。
- hybrid と graph_only はグラフで窓の外も見ている。hidden の AUC が codebert より hybrid で高ければ、今のグラフがすでに窓の外を補っている兆し。
- hidden の件数が少ないと区間が広い。件数（n）を必ず一緒に見る。区間はテスト項目の再標本化で、run（seed）は固定。
- 欠陥が窓の外にあるには関数が長い必要があるので、hidden の群は visible より長くなりやすい。モデルが長さで当てていると hidden の AUC が高めに出て、「差がない」側に寄る。各表の下の length-only AUC（visible / hidden）で、その寄りの大きさを見る。
- 古い run で `args.json` に `test_jsonl` が無いときは `--test_jsonl data/cleaned_data/bigvul_test.jsonl` を付ける（repo の根で実行する）。
