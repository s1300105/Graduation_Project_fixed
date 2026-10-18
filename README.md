# Graduation_Project_fixed

卒論「Bidirectional Dual Cross-Attention for Graph and Language Model Fusion in
Vulnerability Detection」（R-GCN + CodeBERT, BigVul）の実装
[s1300105/Graduation_Project](https://github.com/s1300105/Graduation_Project) を、
監査（元リポジトリの `audit/REPORT_ja.md`）で見つかった欠陥を直して作り直した版です。
初回コミットは元リポジトリ `a7122a4` の無改変コピーなので、`git diff <初回コミット>..HEAD` が修正の全体です。

## 何を直したか（監査 ID との対応）

| 監査 ID | 欠陥 | 修正 | 場所 |
|---|---|---|---|
| M19 | 辺の種別をリスト名から取り、辺自身の `type` を読んでいなかった（AST ≡ CFG） | 辺自身の `type` を読む。(src, dst, 関係) で重複除去。実サンプルで Joern の辺数（67/59/37）と一致することをテスト | `utils/functions/cpg.py` |
| M20 | Joern の `<global>` 疑似関数で全関数が二重化 | `graph_indexing` と `parse_to_functions` の両方で `<global>` を捨てる | `utils/process/cpg_generator.py`, `cpg.py` |
| M21 | 全関数を `.c` で書き出し、C++ 関数が 2 ノードの殻になる | `lang` 列または C++ 構文の検出で `.cpp` に書き出す | `utils/data/datamanager.py` |
| M13 | 未正規化の凍結 CLS をそのまま `input_proj` に入れていた | 入力に LayerNorm | `models/LMGNN.py` |
| M22 / M25 | ノードラベルを 1 ビットに潰し、辺は順方向のみ | Joern ラベルを埋め込み（`node_label`）、`--reverse_edges` で逆関係を追加可能 | `cpg.py`, `embeddings.py`, `LMGNN.py` |
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

論文の記述から**意図的に変えた**点（全て引数で元に戻せます）: 入力 LayerNorm、ノードラベル埋め込み、R-GCN の残差、`cls_skip`、勾配クリッピング 1.0、重み減衰を LayerNorm / bias に掛けない、1200 字フィルタなし。
HEAD にだけあった pre-GCN attention と 768 次元化は論文にないので入れていません。

## 検証済み / 未検証

- `pytest tests/`（36 件、CPU）: 実パイプライン形式のサンプル `data/check_data/cpg_head_check.json` に対する辺種別の一致（Joern が出した 67 AST / 59 CFG / 37 PDG と完全一致）、`<global>` の除外、3 モデル種 × アブレーションの forward / backward、損失の式、ストリームの順序、2 seed × 3 設定の端から端までの学習・評価・再評価、分割スクリプト、Joern v4.0.400 の実出力（`tests/fixtures/joern_v4_export.json`）の読み込み。
- Joern v4.0.400 を使って `tests/fixtures/joern_inputs/` の 12 関数で `run.py --cpg --embed --node_embedding fast` → `--mode train`（hybrid / graph_only、2 seed）→ `scripts/compare_runs.py` まで通した。C++ の 6 関数は `.cpp` として書き出され、`CClient::ClearServerDependentCaps` が 80 ノード・AST 79 / CFG 64 / PDG 198 本の本物のグラフになる（元のコードでは 2 ノードの殻）。
- **Joern で解析できない関数は除外される**（12 件中 2 件: クラス文脈のないコンストラクタ断片 `KillRing() : size(0), ...` と、シグネチャにマクロ `UNUSED` を含む関数）。元のコードはこれらを `<global>` の殻として学習に流していた。除外数は `data/cpg/cpg_manifest.json` の `rows_lost_in_joern` に出るので論文に書くこと。両腕とも同じ入力ファイルを読むので比較は対称。
- **未検証**: BigVul 全体での CPG 生成・学習（データと GPU が要る）。CodeBERT ノード埋め込み（`--node_embedding codebert`）はクラス単体でテスト済みだが全データでは未実行。

## 使い方

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt     # CUDA 版 torch は環境に合わせて先に入れる
pytest tests -q

# 1. BigVul CSV -> jsonl（コメント除去、ラベル衝突の除去、完全一致 dedup、80/10/10）
python data/scripts/make_bigvul_json.py --input_csv data/raw/MSR_data_cleaned.csv \
    --output_jsonl data/cleaned_data/bigvul.jsonl            # --split project でプロジェクト横断

# 2. Joern 4.x を joern/joern-cli/ に置く（検証に使ったのは v4.0.400: github.com/joernio/joern/releases/download/v4.0.400/joern-cli.zip）

# 3. CPG と埋め込み
python run.py --cpg --embed --dataset bigvul                 # --reverse_edges で逆関係を追加
cat data/cpg/cpg_manifest.json data/input/manifest.json      # 件数・陽性数・<global> 除外数

# 4. 学習（論文の設定: bs16, lr1e-4, CodeBERT lr 比 0.3, 200 epoch, patience 50）を 3 seed で
python run.py --mode train --model_type codebert --seeds 0 1 2
python run.py --mode train --model_type hybrid   --seeds 0 1 2
python run.py --mode train --model_type graph_only --seeds 0 1 2
python run.py --mode train --model_type hybrid --ablation shuffle_edges --seeds 0 1 2
python run.py --mode train --model_type hybrid --ablation no_gate --seeds 0 1 2

# 5. 対応つき比較（同じテスト集合、各 run は自分の検証調整閾値）
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1e-04_br0.30_seed*" --b "runs/codebert_bs16_lr1e-04_br0.30_seed*"
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1e-04_br0.30_seed*" --b "runs/hybrid-shuffle_edges_bs16_lr1e-04_br0.30_seed*"

# 6. 既存 run の再評価
python run.py --mode test --run_dir runs/hybrid_bs16_lr1e-04_br0.30_seed0
```

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
