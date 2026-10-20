# Graduation_Project_fixed

卒論「Bidirectional Dual Cross-Attention for Graph and Language Model Fusion in
Vulnerability Detection」（R-GCN + CodeBERT, BigVul）の実装
[s1300105/Graduation_Project](https://github.com/s1300105/Graduation_Project) を、
監査（元リポジトリの `audit/REPORT_ja.md`）で見つかった欠陥を直して作り直した版です。
初回コミットは元リポジトリ `a7122a4` の**ソースファイルだけ**を無改変で取り込んだもの（生成物の `plots*/`、`lib/`、`data/data_for_checking/`、`results.txt`、pip ログ、`.deb` は入れていない）なので、ソースの修正は `git diff <初回コミット>..HEAD` で全て追えます。

## 何を直したか（監査 ID との対応）

| 監査 ID | 欠陥 | 修正 | 場所 |
|---|---|---|---|
| M19 | 辺の種別をリスト名から取り、辺自身の `type` を読んでいなかった（AST ≡ CFG） | 辺自身の `type` を読む。(src, dst, 関係) で重複除去。実サンプルで Joern の辺数（67/59/37）と一致することをテスト | `utils/functions/cpg.py` |
| M20 | Joern の `<global>` 疑似関数で全関数が二重化 | `graph_indexing` と `parse_to_functions` の両方で `<global>` を捨てる | `utils/process/cpg_generator.py`, `cpg.py` |
| M21 | 全関数を `.c` で書き出し、C++ 関数が 2 ノードの殻になる | `lang` 列または C++ 構文の検出で `.cpp` に書き出す | `utils/data/datamanager.py` |
| M13 | 未正規化の凍結 CLS をそのまま `input_proj` に入れていた | 入力に LayerNorm | `models/LMGNN.py` |
| M25 | ノードラベルを 1 ビットに潰し、辺は順方向のみ | Joern ラベルを埋め込み（`node_label`）、`--reverse_edges` で逆関係を追加可能 | `cpg.py`, `embeddings.py`, `LMGNN.py` |
| M22 | ノード特徴が入れ子の `CODE` 全文（BLOCK = 関数本体）の凍結 CLS で、テキスト枝と冗長 | **部分対応**: `--node_code_mode first_line` で各ノードの先頭行だけを埋め込む選択肢を追加。既定は論文どおり `full`。テキスト枝が持たない特徴（抽象データフロー等）への置き換えは未着手 | `embeddings.py` |
| M12 | ハイブリッドがベースラインの上位互換でない（CLS 読み出しを捨てていた） | `cls_skip`: CodeBERT の CLS を融合ベクトルに連結して分類器へ（既定 on） | `LMGNN.py` |
| M15 | ゲートに温度も正則化もなく崩壊する | `--gate_temperature`、`--ablation no_gate` | `layers.py` |
| M18 / M08 | 6 層 R-GCN に残差・dropout なし、勾配クリッピングなし | 層ごとに残差 + LayerNorm + dropout（`--ablation no_residual` で無効化）、`--clip_grad` | `LMGNN.py`, `run.py` |
| M01 | Focal loss がクラス重みを p_t の指数に入れていた | 重みなし CE から p_t を計算し α_t を外から掛ける（論文式 (12)） | `run.py: FocalLoss` |
| M02 | 毎 epoch 同一順序、`--subset_frac` 既定 0.10 が固定プレフィックス | (seed, epoch) で RNG を作り直し、ファイル順もシャッフル。サブセットは毎 epoch 再抽出。既定は全データ | `run.py: StreamGraphDataset` |
| M03 | seed なし単一 run | `--seeds 0 1 2 ...` で複数 run、`set_seed` + 決定論フラグ、seed ごとの平均 ± 標準偏差を出力 | `run.py` |
| M05 / M30 | 選択基準が版で違う、閾値探索が飽和後に無効 | 検証集合で閾値調整した F1 で選択（`--select_on tuned`、論文 §4.2）。閾値 0.5 と調整後の両方を常に記録 | `run.py` |
| M06 | HEAD で CodeBERT 単体が `NameError` | ゲート重みは `model.last_gate` から安全に読む。3 モデル種で学習ステップをテスト | `run.py`, `tests/test_training.py` |
| M28 | 未開示の 1200 字フィルタ | `--max_chars`（既定なし）。CPG 生成時に件数・陽性数を `cpg_manifest.json` に記録 | `run.py` |
| M32 / M34 | アブレーションなし | `--model_type graph_only`、`--ablation shuffle_edges / no_gate / no_cls_skip / no_input_norm / no_residual` | `LMGNN.py` |
| M36 / M37 | 数値・チェックポイントが残らない、test が別名のファイルを探す | run ごとに `runs/<name>/{args,metrics}.json, history.jsonl, best.pth, *_preds.npz, curves.png`。`--mode test --run_dir` | `run.py` |
| M17 / M40 | `os.chdir('/home/yudai/...')`、requirements の欠落 | 絶対パス・import 時の副作用を全廃。requirements を実測版に更新 | `models/layers.py`, `requirements.txt` |
| M39 | `graph-for-funcs.sc` が未コミット | 再構成版を同梱。Joern v4.0.400 で実行し、出力を回帰テストの固定データにした | `joern/graph-for-funcs.sc`, `tests/test_joern_export.py` |
| M23 | §3.2.2 の記号正規化が未適用 | `--normalize_graph_identifiers` でグラフ枝のみに適用可能（既定 off。正規表現ベースで壊れやすい） | `run.py` |
| M26 | Joern 失敗時に `zip` がずれて空結合 | 出力を stem で対応づけ、失敗は manifest に計数 | `run.py`, `cpg_generator.py` |
| M29 | 関数単位ランダム分割のみ | `make_bigvul_json.py --split project` でプロジェクト横断分割 | `data/scripts/make_bigvul_json.py` |
| — | 単一 run 同士の比較に検定がない | `scripts/compare_runs.py`: 同一テスト集合上の対応つき bootstrap で ΔF1 の 95% 区間 | `scripts/` |
| — | 二値・単一関数（BigVul）しか扱えない | **課題スイート**（`docs/TASK_SUITE_ja.md`）: 多クラス分類（`--num_classes`、argmax の accuracy / macro-F1 / top-5、`--select_on macro_f1\|accuracy`、`compare_runs.py --metric`）、プログラム全体のグラフ（`--graph_unit whole_file` = Joern のファイル単位 `<global>`）、POJ-104 / CodeNet C++1000 / Devign の前処理スクリプト。二値の経路は `tests/test_golden_binary.py` でビット同一を保証 | `run.py`, `utils/process/cpg_generator.py`, `utils/data/prep.py`, `data/scripts/make_{poj104,codenet,devign}_json.py` |
| — | 「最初の非 `<global>` メソッドを採る」規則が、複数関数のプログラムで main を捨てる（POJ-104 で 15%、CodeNet で 44%。実測） | `--graph_unit whole_file`。Joern が何も解析できなかったファイルは殻のグラフにせず損失として数える | `cpg_generator.py` |
| — | 小文字のシグネチャマクロ（`av_cold`, `coroutine_fn`, `APR_INLINE`, `APR_DECLARE(type)`）で Joern に落ちる（Devign 4.5%、D2A 5.5%。実測） | 救済段階 C（既知マクロの一覧と戻り値型マクロの展開。`manifest` の `rescued_stage_C`） | `utils/data/datamanager.py` |
| — | K&R 形式の `main()` を `.cpp` で渡すと main が丸ごと消える（Joern 4.0.400 で実測） | 前処理スクリプトが `ext` 列を書き、`to_files` がそれを優先する。規則: K&R main → `.c`、それ以外で C++ の手がかり → `.cpp` | `prep.py`, `datamanager.py` |

論文の記述から**意図的に変えた**点（全て引数で元に戻せます）: 入力 LayerNorm、ノードラベル埋め込み、R-GCN の残差、`cls_skip`、勾配クリッピング 1.0、重み減衰を LayerNorm / bias に掛けない、1200 字フィルタなし。
HEAD にだけあった pre-GCN attention と 768 次元化は論文にないので入れていません。

## 検証済み / 未検証

- `pytest`（CPU）: 実パイプライン形式のサンプル `data/check_data/cpg_head_check.json` に対する辺種別の一致（Joern が出した 67 AST / 59 CFG / 37 PDG と完全一致）、`<global>` の除外、3 モデル種 × アブレーションの forward / backward、損失の式、ストリームの順序、2 seed × 3 設定の端から端までの学習・評価・再評価、分割スクリプト、Joern v4.0.400 の実出力（`tests/fixtures/joern_v4_export.json`）の読み込み。。課題スイートで追加: 二値の学習出力のビット同一（`tests/test_golden_binary.py`）、3 クラスの端から端まで（manifest からのクラス数、`--num_classes`、`--select_on`）、Joern 4.0.400 のプログラム全体の実出力（`joern_v4_export_multi.json`、`joern_v4_export_crafted.json`: ラムダの合流、関数の無いファイルの除外、同じ番号の .c/.cpp）、救済段階 C、`ext` 列、前処理スクリプト 3 本の契約（合成入力）、`compare_runs` の多クラスと取り違え検出、Joern を模した `--cpg` の集計（`--rescue_stages`、whole_file の損失）、`--embed --resume` の来歴確認、`--mode test` の別データ評価
- Joern v4.0.400 を使って `tests/fixtures/joern_inputs/` の 12 関数で `run.py --cpg --embed --node_embedding fast` → `--mode train`（hybrid / graph_only、2 seed）→ `scripts/compare_runs.py` まで通した。12 ファイル中 7 件が `.cpp`（うち 101 は初期化子リストだけのコンストラクタ断片）、5 件が `.c` として書き出され、`CClient::ClearServerDependentCaps` が 80 ノード・AST 79 / CFG 64 / PDG 198 本の本物のグラフになる（元のコードでは 2 ノードの殻）。
- **Joern で解析できない関数の扱い。** シグネチャのマクロ（`__init`、`__user`、`UNUSED`、`__attribute__((...))` など）で Joern が関数を認識できない場合、`--cpg` はシグネチャからマクロを除いた版で自動的に再投入する（救済パス。段階 A は控えめ、段階 B は積極的。本文は変えず、テキスト枝は元のコードを読む。`--no_rescue_macros` で無効化）。それでも落ちる関数（例: クラス文脈のないコンストラクタ断片）は除外される。件数は `data/cpg/cpg_manifest.json` の `rescued_stage_A` / `rescued_stage_B` / `rescued_stage_C` / `rows_lost_in_joern` に出るので論文に書くこと（キーは 0 件でも常に書く）。両腕とも同じ入力ファイルを読むので比較は対称。**段階 C（小文字のシグネチャマクロ `av_cold` / `coroutine_fn` / `APR_INLINE`、戻り値型マクロ `APR_DECLARE(type)`）は後から追加した**ので、それ以前に作った BigVul の CPG を作り直すと行集合が増える（FFmpeg 由来の関数）。以前の挙動で作り直すなら `--rescue_stages AB`。プログラム全体（`--graph_unit whole_file`）では救済パスは走らない（先頭の `#define` / `#include` 行を関数シグネチャとして書き換えてしまうため。件数は `rescue_skipped_whole_file`）。
- **BigVul の `lang` 列は信頼しない。** 公開ミラーでは `CPP` と記録された行が 1.4% しかないのに、C++ 構文を含む関数は 10% 以上ある。`lang` は「C++ である」方向のヒントにだけ使い、`C` と書かれていても構文判定で C++ なら `.cpp` として Joern に渡す。
- 落ちた関数の内訳は `python scripts/diagnose_joern_losses.py --split train --slice 0` で調べられる（`--graph_unit` は `cpg_manifest.json` から読む）（落ちた関数を Joern に再投入し、「Joern がメソッドを出していない」か「出ているのに捨てている（バグ）」かを分類する）。
- **未検証**: BigVul 全体での CPG 生成・学習（データと GPU が要る）。CodeBERT ノード埋め込み（`--node_embedding codebert`）は 75 ノードのサンプルに対するテスト（`tests/test_node_embedding_codebert.py`）のみで、全データでは未実行。

## 使い方（GPU マシンでの詳細手順は RUNBOOK_ja.md）

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt     # CUDA 版 torch は環境に合わせて先に入れる
pytest -q

# 1. BigVul CSV -> jsonl（コメント除去、ラベル衝突の除去、完全一致 dedup、80/10/10）
python data/scripts/make_bigvul_json.py --input_csv data/raw/MSR_data_cleaned.csv \
    --output_jsonl data/cleaned_data/bigvul.jsonl            # --split project でプロジェクト横断

# 2. Joern 4.x を joern/joern-cli/ に置く（検証に使ったのは v4.0.400: github.com/joernio/joern/releases/download/v4.0.400/joern-cli.zip）

# 3. CPG と埋め込み
python run.py --cpg --embed --dataset bigvul                 # --reverse_edges で逆関係、--node_code_mode first_line で M22 の緩和
cat data/cpg/cpg_manifest.json data/input/manifest.json      # 件数・陽性数・<global> 除外数

# 4. 学習（論文の設定: bs16, lr1e-4, CodeBERT lr 比 0.3, 200 epoch, patience 50）を 3 seed で
python run.py --mode train --model_type codebert --seeds 0 1 2
python run.py --mode train --model_type hybrid   --seeds 0 1 2
python run.py --mode train --model_type graph_only --seeds 0 1 2
python run.py --mode train --model_type hybrid --ablation shuffle_edges --seeds 0 1 2
python run.py --mode train --model_type hybrid --ablation no_gate --seeds 0 1 2

# 5. 対応つき比較（同じテスト集合、各 run は自分の検証調整閾値）
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --b "runs/codebert_bs16_lr1.0e-04_br0.30_seed*"
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --b "runs/hybrid-shuffle_edges_bs16_lr1.0e-04_br0.30_seed*"

# 6. 既存 run の再評価
python run.py --mode test --run_dir runs/hybrid_bs16_lr1.0e-04_br0.30_seed0

# 7. 長さ別の確認: テキスト枝の窓（512 トークン）の外に欠陥がある関数を取りこぼしているか（RUNBOOK §13）
python scripts/length_strata.py --group codebert "runs/codebert_bs16_lr1.0e-04_br0.30_seed*" \
    --group hybrid "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --bigvul_csv data/raw/MSR_data_cleaned.csv --out runs/length_strata
```

### 化学（SMILES ＋ 分子グラフ。詳細は `docs/CHEM_PROTOCOL_ja.md` と RUNBOOK §12）

同じ分子を、化学言語モデル（ChemBERTa-2）が SMILES 文字列として、R-GCN が分子グラフ（原子＝ノード、結合の型＝関係）として読む。コードの CPG の代わりに `run.py --molgraph` で入力を作る以外は、学習・比較の仕組みはコードと同じ。

```bash
python scripts/chem_grid.py --root experiments/chem --datasets ogb_molbace ogb_molbbbp --seeds 0 1 2 3 4
```

### 課題スイート（非セキュリティの多クラス課題。詳細は `docs/TASK_SUITE_ja.md` と RUNBOOK §11）

```bash
# POJ-104（104 クラス、プログラム全体）: HF の parquet を取得 → jsonl（分割またぎの重複除去、dedup、層化 80/10/10）
python data/scripts/make_poj104_json.py --input data/raw/poj104 --hf --out_dir data_poj/cleaned_data
# CodeNet C++1000（既定は 1 問題あたり最大 100 本 ≈ 10 万本）。メタデータを渡すと Accepted のみ・利用者横断分割が可能
python data/scripts/make_codenet_json.py --input data/raw/Project_CodeNet_C++1000 --out_dir data_codenet/cleaned_data [--metadata_dir data/raw/Project_CodeNet/metadata --split user]
# Devign（CodeXGLUE defect detection、二値・均衡）: 公式分割を保ち、ラベル衝突・分割またぎの重複を除く
python data/scripts/make_devign_json.py --input data/raw/codexglue_defect --hf --out_dir data_devign/cleaned_data

# プログラム全体は <global> を 1 グラフにする。ノード上限は位置順の切り捨てなので上げる。--graph_unit は --cpg と --embed に同じ値を渡す（違えば --embed が止まる）
python run.py --cpg --embed --dataset poj104  --data_root data_poj     --graph_unit whole_file --max_nodes 1000
python run.py --cpg --embed --dataset codenet --data_root data_codenet --graph_unit whole_file --max_nodes 1000
python run.py --cpg --embed --dataset devign  --data_root data_devign                       # 単一関数は既定どおり

# 多クラスはクラス数を input/manifest.json から読む。run 名にデータセットの接頭辞を付けて BigVul の run と分ける
python run.py --mode train --model_type codebert --data_root data_poj --runs_dir runs_poj --run_prefix poj_ --seeds 0 1 2 3 4
python run.py --mode train --model_type hybrid   --data_root data_poj --runs_dir runs_poj --run_prefix poj_ --seeds 0 1 2 3 4
python run.py --mode train --model_type hybrid --ablation shuffle_edges --data_root data_poj --runs_dir runs_poj --run_prefix poj_ --seeds 0 1 2 3 4
python scripts/compare_runs.py --metric macro_f1 --a "runs_poj/poj_hybrid_bs16*" --b "runs_poj/poj_codebert_bs16*"
```

`--mode test --allow_other_data_root` の結果は `<run>/reeval_<名前>/` に書き、run 自身の出力は上書きしない。`compare_runs.py` は同じテスト集合でない組（idx は同じでもラベルが違う）を拒否する。

多クラスでは閾値調整は無く argmax。`metrics.json` の `test_tuned` / `test_05` は同じ argmax 指標（`threshold: null`、`accuracy`、`macro_f1`、`top5_accuracy`、`per_class_f1`）。選択基準は `--select_on tuned`（= macro-F1）か `accuracy`。

グリッドサーチは `--batch_sizes 8 16 32 --lrs 1e-4 5e-5 3e-5 1e-5 --bert_lr_ratios 0.1 0.2 0.3` のように
リストを渡します（最終モデル・最終損失・全データで行うこと。元のスイープは別モデル・10% データで行われていました）。

## 読み方

- `runs/<name>/metrics.json`: `test_05`（閾値 0.5）と `test_tuned`（検証で選んだ閾値）の P/R/F1/Acc と混同行列、`best_epoch`。
- `runs/<name>/history.jsonl`: epoch ごとの損失、検証 F1（0.5 / 調整後）、ゲート重みの平均、確率が中間帯 (0.05, 0.95) にある割合（飽和の監視用）。
- `runs/<name>/curves.png`: 損失・検証指標・ゲート重みの曲線。
- ランダム辺の対照（`--ablation shuffle_edges`）と同等なら、グラフ構造は使われていません。`graph_only` が chance 付近なら、グラフ枝単体に信号が届いていません。

## 推奨プロトコル

1. まず `graph_only` と `shuffle_edges` を回し、グラフ枝に信号が届いているかを確かめる。
2. 3 seed 以上で `codebert` と `hybrid` を回し、`compare_runs.py` の区間で判断する。
3. 1200 字フィルタは掛けない。長さ別（512 トークン超かどうか）に層別して報告する。
4. `--split project` または PrimeVul / DiverseVul でも評価する（構造が効くのは分布が変わる場面）。
