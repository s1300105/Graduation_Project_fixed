# 化学（分子）での BDCA 検証：事前登録

作成: 2026-10-20。**この文書は、下記の比較で test の数値を見る前にコミットする。**
例外として、調査段階で graph_only の試走と ECFP+RF の値は既に見ている（下の「既に見た値」）。

## 問い

同じ分子を 2 通りに読む。テキスト枝は SMILES 文字列を化学言語モデルで読み、グラフ枝は分子グラフを R-GCN で読む。分子グラフは原子をノード、結合をエッジとし、結合の型を関係にする。このとき、融合（hybrid）はテキストだけ・グラフだけより良いか。良くなるなら、それは結合構造を使っているからか。

**前提の限界（先に書く）。** 2D の分子グラフは SMILES から一意に決まる。したがって融合の利得は「新しい情報」ではなく、置換不変性・局所性・環の明示という**帰納バイアス**の効果である。構造を使っているかどうかは、エッジを壊した対照・エッジ無しの対照との差で判断する。

## データと分割

| データ | 件数 | 陽性率（train / valid / test） | 実行場所 |
|---|---|---|---|
| ogbg-molbace | 1,210 / 151 / 152 | 0.397 / 0.861 / 0.533 | この CPU |
| ogbg-molbbbp | 1,631 / 204 / 204 | 0.839 / 0.407 / 0.529 | この CPU |
| ogbg-molhiv | 32,901 / 4,113 / 4,113 | 0.037 / 0.020 / 0.032 | GPU（RUNBOOK §12） |

- 分割は OGB 配布のファイルをそのまま使う。Hu et al. 2020 の決定論的 Bemis-Murcko scaffold 分割で、キラリティを含む。
- 行の重複除去はしない。OGB と揃えるためで、重複はすべて同じ分割の中にあり、分割をまたがない（実測）。
- RDKit 2026.09.1 で解析できない SMILES は全腕から除く。BACE と BBBP は 0 件。
- テキスト枝の入力は RDKit の canonical isomeric SMILES。グラフも同じ文字列から作る。

## 腕（同じ分割・同じ seed）

| 記号 | 腕 | 実装 |
|---|---|---|
| A0 | ECFP4（2048 bit）+ ランダムフォレスト 500 本 | `scripts/fingerprint_baseline.py` |
| A1 | テキストのみ（ChemBERTa-2 77M-MTR、最後の 2 層を微調整） | `--model_type codebert` |
| A2 | グラフのみ | `--model_type graph_only` |
| A2e | グラフのみ、エッジ無し | `--model_type graph_only --ablation no_edges` |
| A3 | 融合 | `--model_type hybrid` |
| A3s | 融合、エッジをシャッフル | `--model_type hybrid --ablation shuffle_edges` |
| A3e | 融合、エッジ無し | `--model_type hybrid --ablation no_edges` |

- テキストモデルは `DeepChem/ChemBERTa-77M-MTR` に固定する。理由は 3 つある。3 層・幅 384 で CPU で回せること、ChemBERTa-2 論文で MLM 版より強いこと、`trust_remote_code` が不要なこと。
- MoLFormer は GPU での感度確認に回す。

## 学習の設定（全腕共通）

```
--loss ce --select_on auc --epochs 30 --patience 30 --batch_sizes 16 --lrs 1e-4 --bert_lr_ratios 0.3
--eval_batch_size 64 --max_length 256 --seeds 0 1 2 3 4
```

- 既定の focal loss と F1 閾値選択は使わない。valid と test で陽性率が大きく違うので、閾値つきの F1 は意味を失う。
- エポック数は Hu et al. の 100 ではなく 30 とする（CPU の予算のため）。早期打ち切りはしない。
- 報告は 2 通り。主は valid ROC-AUC が最良のエポックの test 値、副は最終エポックの test 値。

## 指標と判定

- 主指標は test ROC-AUC。副指標は PR-AUC。
- 主比較は 2 つ。
  - **A3 − A1**：融合はテキストだけより良いか。
  - **A3 − A3e**：その差は結合構造によるか。
- 副比較は A3 − A3s、A2 − A2e、A3 − A0。
- 統計は 2 種類で、どちらも `scripts/compare_runs.py --metric auc` で計算する。
  - test 分子の対応つき bootstrap（2,000 回）で、seed ごとの AUC を平均したもの。
  - 二段 bootstrap（seed × 分子、`--seed_bootstrap`）。
- 「差がある」と言う条件は、二段 bootstrap の 95% 区間が 0 を含まないこと。
- 検出できる大きさの目安は 2〜3 ROC 点。単一モデルの test AUC の標準誤差が 0.03〜0.04 あるため。

## 既に見た値（調査段階、この設定とは別条件）

- graph_only の試走。CE、30 エポック、5 seed、別の特徴化スクリプトで行った。
  - BBBP: 0.661 ± 0.013
  - BACE: 0.773 ± 0.027
  - BBBP のシャッフル版: 0.617
  - BBBP のエッジ無し版: 0.640
- ECFP+RF: BBBP 0.677、BACE 0.853。
- 実験ドライバ `scripts/chem_grid.py` の動作確認として、BACE で 2 seed・2 エポック・訓練の 20% だけの試走をした。設定が登録とは別で、数値は結論に使わない。
- 文献値（同じ分割）。GIN を事前学習なしで使った場合、BBBP 65.8 ± 4.5、BACE 70.1 ± 5.4（Hu et al. 2020）。
- 文献での融合の利得（SMILES Transformer + GNN）は、最良の単一ビューに対して 0〜+1.5 点。BACE では、DVMP・UniMAP・MolPROP のいずれも利得がない。

## 既知の限界

- 化学言語モデルの事前学習コーパス（PubChem）には test の分子が含まれる可能性が高い。ラベルなしの露出だが、ゼロから学習する R-GCN との非対称がある。
- シャッフル対照は結合の双方向の対応を壊すので、構造の利用とノイズへの耐性が混ざる。エッジ無し対照を主に使うのはこのため。
- 原子特徴は次数・環・芳香族性・水素数を含むので、エッジ無しでも局所構造の一部は見える。
- BACE の valid は陽性 86%（陰性約 21 件）で、モデル選択がぶれやすい。最終エポックの値も必ず併記する。
