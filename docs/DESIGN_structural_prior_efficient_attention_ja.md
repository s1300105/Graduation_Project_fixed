# 研究設計: 効率的 attention における構造的事前分布の役割

作成: 2026-11-02。版 0.1（設計書。事前登録は予備実験の後に別文書として確定する）。
根拠とした文献調査: `reports/LLM GNN 融合 attention 先行研究.md`（調査ノートは `research_notes/LLM GNN 融合 attention 先行研究/`）。本文中の出典は全てその報告書で URL を確認したもの。

## 0. 一文で

> 構造情報は、線形時間の attention / 状態空間モデル（SSM）が失った鋭さを最も安く取り戻す手段か。それはグラフの大きさと、構造がテキストと相補的かどうかに、どう依存するか。

これは「GNN と LLM の融合」の研究ではない。**効率的な注意機構の性質**についての研究で、GNN 的な構造はその性質を調べる手段である。LLM は凍結したテキスト読み手として使い、効率の主張は融合経路のコストと入力トークン数に限定する。

## 1. 問いの背景（調査で確定した事実）

| 事実 | 出典 |
|---|---|
| 線形・カーネル化 attention は softmax に精度で劣る。原因は attention 分布の一様化と低ランク化と診断されている | [GraphGPS](https://arxiv.org/abs/2205.12454)、[GraphTARIF](https://arxiv.org/abs/2510.10631) |
| 構造マスクを厳密に入れた線形 attention は softmax と同精度に戻る。グラフランダム特徴（GRF）を φ の内部に入れると差の 75〜80% を回復する（ImageNet: 無マスク線形 0.693 → GRF 0.730 → 厳密マスク・softmax 0.741）。ただし視覚・点群のみ | [Topological Masking, ICLR 2025](https://arxiv.org/html/2410.03462v2) |
| 事前学習 LM のトークン列とグラフを構造つき線形 attention で結んだ研究は無い（13 クエリと被引用 94 件を走査） | 報告書 §3 |
| グラフ SSM は順序型（Graph-Mamba、GMN）と作用素型（GSSC、Chimera、MP-SSM）に分かれ、順序型は順序だけで精度が 12 点動く。LLM との融合は分子の HSA-Net 1 件で、そこでは Mamba 単独は cross-attention 単独に勝たない | [Chimera](https://arxiv.org/html/2510.12111v1)、[HSA-Net](https://arxiv.org/html/2508.08334v1) |
| テキストが豊富な同類性グラフでは構造チャネルの寄与はほぼゼロ。プールしたソフトトークンは学習済み定数で置換できる | [When Structure Doesn't Help](https://arxiv.org/html/2511.16767v2)、[Token Collapse](https://arxiv.org/html/2610.31572) |
| 構造エンコーダの優位は、テキストチャネルが弱い条件（低データ、小さい LM、疎なテキスト、直列化がコンテキストを飽和させる大きさ）に集中する | [Sanford et al.](https://arxiv.org/html/2405.18512)、[GraphLLM](https://arxiv.org/html/2310.05845) |
| 「構造が必要 / 相補的 / 導出可能」という層別を設計原理として使った研究は無い | 報告書 §7 |
| 注意機構を改変すると、パラメータが 0.015% でも壁時計が 2 倍になり得る。FlashAttention 非互換も起きる | [GTLM](https://arxiv.org/html/2605.10247)、[SLASH](https://arxiv.org/html/2605.10503v3) |

## 2. 仮説と反証条件

記号: F = 注意機構の形（softmax / 線形 / SSM）、S = 構造の入口（なし / φ の外 / φ の内 / 厳密マスク）、T = データの型（必要 / 相補 / 導出可能）、L = 入力の大きさ。

**H1（回復）** 線形 attention に構造を入れると softmax との精度差が縮む。回復量は 厳密マスク ≥ φ の内 > φ の外 > なし の順。
反証: 必要型のデータで、線形 ＋ φ の内 が差の 30% 未満しか回復しない。

**H2（事前分布か情報か）** 構造がテキストから導出可能な型では、softmax ＋ 構造 は利得を持たない（既知の否定結果と整合）が、線形 ＋ 構造 は利得を持つ。つまり構造は「情報」ではなく「鋭さの事前分布」として働く。
予測の表:

| | 導出可能型 | 必要型 |
|---|---|---|
| softmax ＋ 構造 | 利得なし | 利得あり |
| 線形 / SSM ＋ 構造 | **利得あり** | 利得あり、softmax との差が縮む |

反証: 導出可能型で 線形 ＋ 構造 − 線形 単体 の区間が分解能の中に収まる。この場合、構造は情報としてしか効かないという否定結果になる（それでも H1・H3・H4 は成立し得る）。

**H3（大きさ）** 線形 attention の一様化は入力が長いほど悪化するので、構造による回復量は L に単調に増える。これは精度側の「大きさ」である。コスト側の「大きさ」（線形が壁時計で softmax に勝つ交差点）は別の量で、別に測る。
反証: 回復量が L に対して平坦か減少する。

**H4（最も安いか）** 鋭さを取り戻す構造なしの手段（局所 softmax 窓 ＋ 線形大域、Gumbel による疎化、ランク回復ゲート、特徴写像の変更）と比べ、同じ壁時計・メモリ・追加パラメータの下で、構造の方が回復量が大きい。
反証: 構造なしの最良手段が同コストで同等以上。この場合「構造は鋭さの回復に必要ではない」が結論になり、それ自体が報告価値を持つ。

**H1〜H4 の全てで、精度だけでなく鋭さの指標（§3.1）が同じ向きに動くことを機構の証拠とする。** 精度だけなら相関の報告で終わる。

## 3. 操作化

### 3.1 鋭さ

融合層の attention 行列（線形 attention と SSM では、検証バッチの一部のクエリについて明示的に A_ij を復元する）に対して、層ごと・ヘッドごとに次を測る。

- 正規化エントロピー H(A_i) / log(支持の大きさ)。
- 実効ランク（特異値の参与比）。
- 構造質量: 1-hop、2-hop、整合辺の上に乗る attention 質量の割合。

これらは GraphTARIF の診断（一様化・低ランク化）をそのまま指標化したもの。学習の前後と、構造の有無で比較する。

### 3.2 大きさ

- 精度側: テキストトークン数 L_t とノード数 N。合成データでは生成時に制御し、実データでは長さ別の層に分けて報告する。
- コスト側: 融合層の順伝播＋逆伝播の壁時計とピークメモリを、L_t ∈ {1k, 2k, 4k, 8k, 16k, 32k, 64k} × N ∈ {100, 300, 1k, 3k} で計測する（§6.2）。線形が softmax に勝つ交差点を報告する。

### 3.3 相補性（データの型）

型は著者の印象ではなく、各データセットで**事前に決めた規則**で測って割り当てる。

| 測る量 | 意味 |
|---|---|
| a. 素の GNN（構造とノードラベルのみ、テキストなし）の精度 − 偶然 | 構造だけで解けるか |
| b. テキストのみ（凍結 LM ＋ 線形分類）の精度 − 偶然 | テキストだけで解けるか |
| c. softmax ＋ 構造 の精度が、辺をシャッフルすると落ちる量 | 構造が実際に使われたか |

規則: b が a より十分高く c が分解能の中 → **導出可能型**。a が b より高いか c が大きい → **必要型**。a と b が同程度で、両方を合わせた softmax 腕が各単独を上回る → **相補型**。TAG では同類性とノードテキスト長も併記する。この割り当ては予備実験で行い、事前登録に固定する。文献にこの操作化は無く、これ自体が貢献になる。

## 4. モデル空間

### 4.1 共通の骨格

```
テキスト: 凍結 LM（§4.4 の規模系列） → 最終層の隠れ状態をキャッシュ → 線形射影 d
構造:     ノード = 構造の単位（AST ノード、グラフのノード）
          ノード特徴 = ラベル埋め込み ＋ 整合するトークンの隠れ状態の平均（テキスト以上の情報を持たせない）
結合列:   [トークン列 ; ノード列]。関係種別 = 整合辺（トークン→所属ノード）、構造辺（AST 親子・兄弟・def-use、またはグラフの辺）、系列辺（隣接トークン）
融合:     結合列の上に F 層。各層 = 注意（§4.2 の形）＋ FFN ＋ 残差 ＋ LayerNorm
読み出し: マスク付き平均 → 分類器
```

今の `models/layers.py` の `FusionBlock`（code←graph と graph←code の 2 本の softmax cross-attention）は、この空間で「F = softmax、S = なし、support = 全対」の点であり、対照として残す。

### 4.2 因子 A: 注意機構の形 F

| 水準 | 実装 | 役割 |
|---|---|---|
| softmax | FlashAttention 2。加法バイアスとマスクは GDT の標準形 softmax(QKᵀ/√d ＋ B)V | 精度の上限、コストの基準 |
| 線形 | Performer 型の正のランダム特徴 φ（特徴数 r は予備実験で決め、全腕で固定） | 本命 |
| SSM（参照腕） | 作用素型。結合列に対するレゾルベント型の遷移（Chimera / GSSC に準ずる）。順序型は使わない | 「線形時間の別系譜でも同じか」の参照 |

### 4.3 因子 B: 構造の入口 S

| 水準 | softmax での実装 | 線形での実装 | SSM での実装 |
|---|---|---|---|
| なし | 全対 | 全対 | 構造なしの遷移（系列順） |
| φ の外 | 辺種別ごとの加法バイアス B（GDT 形） | 辺のある対にだけ疎な加法項（NodeFormer 形） | 遷移に辺の加法項 |
| φ の内 | （定義なし） | トークン・ノードの φ に GRF をテンソル積で掛ける（GKAT、ICLR 2025 形）。トークンは整合先ノードの GRF を継承する | 遷移作用素に隣接を入れる（作用素型の本来の形） |
| 厳密マスク | 支持外を −∞ | 支持外を厳密に 0 にした低ランク（木・格子では ICML 2022 の手法で線形時間、一般グラフでは上限参照として二次で計算） | （定義なし） |

主効果と、F × S の交互作用（特に 線形 × φ の内）が H1 の検定対象。

### 4.4 因子 C: 構造なしの鋭さ回復（H4 の対照）

線形 attention に対してのみ。各水準は追加パラメータと壁時計を 線形 ＋ φ の内 と ±10% に揃える。

- 局所 softmax 窓（幅 w）＋ 線形大域のハイブリッド。
- Gumbel ノイズによる疎化（NodeFormer 形）。
- ランク回復ゲート（GraphTARIF 形の値側ゲート）。
- 特徴写像の変更（多項式または学習型 φ）。

### 4.5 否定結果から要求される必須の対照

| 対照 | 守る主張 | 出典 |
|---|---|---|
| テキストのみ（凍結 LM ＋ 線形分類、同じキャッシュ） | 構造の利得がテンプレートや容量の差でない | [When Structure Doesn't Help](https://arxiv.org/html/2511.16767v2) |
| 学習済み定数トークン（構造を見ない追加列） | 追加列の存在そのものの効果を除く | [Token Collapse](https://arxiv.org/html/2610.31572) |
| 直列化した構造をテキストとして LM に LoRA で学習（トークン予算を揃える） | 「GNN 経路 対 直列化」の公平比較 | [G-Retriever](https://arxiv.org/html/2402.07630v3)、[Less is More](https://arxiv.org/html/2508.10115v1) |
| 素の GNN（GIN、テキストなし） | 構造だけで解ける課題の床 | [GraphInfer-Bench](https://arxiv.org/html/2606.11562) |
| MLP アダプタ（構造なしの射影） | GNN の寄与が射影の寄与でない | [When Structure Doesn't Help](https://arxiv.org/html/2511.16767v2) |
| 辺シャッフル、ノード順の置換 | 構造が実際に読まれている | [Huang et al.](https://arxiv.org/html/2309.16595v4)、[Mol-LLM](https://arxiv.org/html/2502.02810v2) |

### 4.6 揃える規則

- 学習可能パラメータ数は同じ F の中で ±5%。F をまたぐ比較はパラメータではなく壁時計とメモリで揃え、両方を報告する。
- 学習ステップ数、データ、seed、最適化設定は全腕で同一。
- 凍結 LM は全腕で同じキャッシュを読む。LM の中に LoRA を入れるのは「直列化 LoRA」対照だけ。

### 4.7 テキスト側 LM

- 主系列: Qwen2.5 の 0.5B / 1.5B / 3B（同一族で規模を振れる。コードと一般テキストの両方を読める）。
- 代替: Llama-3.2 1B / 3B。GTLM との対置に有利。
- 7B は 4bit の凍結推論でキャッシュのみ作り、1 点だけ置く（クラウド、§8）。
- Mamba 系 LLM（Falcon-Mamba 等）をテキスト側に置く拡張は、主実験が終わってから検討する。

## 5. データセット

構造は決定的に、失敗なしに、全サンプルで得られるものだけを使う（Joern は使わない）。各型に 1 つ以上、大きさを振れるものを最低 2 つ。

| 型（予想） | データ | 構造の取り方 | 大きさの振り方 | 備考 |
|---|---|---|---|---|
| 必要 | 合成グラフ推論（到達可能性、最短路長、サイクル、次数、コミュニティ）。ノードに短いテキスト属性を付け、問いは自然文 | networkx で生成。完全に正確 | N = 50〜5,000、密度 | 核。GraphQA 系は reason-then-code で飽和するので、[GraphArena](https://arxiv.org/html/2407.00379v2) 級の大きさと NP 的課題を含める |
| 必要 | 変数誤用検出（GREAT 形の合成バグ注入） | tree-sitter AST ＋ def-use | 関数長 | コードで「構造が必要」を作る |
| 相補 | 異類性 TAG（Roman-Empire、Amazon-Ratings）、題名のみの Cora / CiteSeer | 与えられた辺 | 近傍の hop 数 | [LLMNodeBed](https://arxiv.org/html/2502.00829v2) で LLM の利得が出る条件 |
| 相補 | コードの識別子マスク版（テキストから情報を削り、構造は保つ） | tree-sitter | 関数長 | 相補性を操作で作る。削る量が連続の軸になる |
| 導出可能 | POJ-104（関数・ファイル全体）、CodeNet C++1000 | tree-sitter AST（親子・兄弟・次トークン・def-use の 4 関係） | ファイル長（CodeNet で長いものを選ぶ） | 既存の前処理 `data/scripts/make_{poj104,codenet}_json.py` を流用し、Joern 部分だけ置換 |
| 導出可能 | PubMed、ogbn-arxiv（テキスト豊富） | 与えられた辺 | 近傍の大きさ | 否定結果の再現が出発点 |
| （任意） | QM9（3D 座標） | RDKit | 原子数 | 分子は時間が余れば。BACE / BBBP は [Mol-LLM](https://arxiv.org/html/2502.02810v2) の結果どおり不適 |

型は §3.3 の規則で測り直し、予想と違えば測った方を採る。
使わないもの: Devign / BigVul / D2A（ラベル精度 0.80 / 0.54 / 0.29、[Croft et al.](https://arxiv.org/abs/2301.05456v1)）、20 ノード以下の GraphQA、順序型 SSM の単純差し込み。

## 6. 測定

### 6.1 精度と効率の前線

- 精度: 課題ごとの標準指標（accuracy / macro-F1 / ROC-AUC / 回帰は MAE）。
- 効率: 学習 1 ステップと推論 1 サンプルの壁時計、ピークメモリ、入力トークン数（直列化腕）。FLOPs は副。
- 報告の形: 精度 × 壁時計の前線を、データ型ごと、大きさごとに描く。同一ハードウェア（RTX 3090、24 GB）で全て測り、その旨を明記する。
- 前処理コスト（最短路、GRF のランダムウォーク、固有分解）も計上する（[OpenGT](https://arxiv.org/html/2506.04765) の報告に倣う）。

### 6.2 マイクロベンチ

融合層単体で、L_t × N の格子上で順伝播＋逆伝播を 100 回計測。softmax は FlashAttention 2、マスクつき softmax は FlexAttention、線形は自前の φ 実装、SSM は選択的走査の実装。交差点の長さを報告する。GTLM の「2 倍のステップコスト」と同種の数字をこちらでも出す。

### 6.3 学習曲線（データ効率）

学習データの 5 / 10 / 25 / 50 / 100% で、主要な腕（softmax なし・softmax 外・線形なし・線形内・テキストのみ）を回す。指標は曲線下面積と、全データ精度の 90% に達するまでの割合。

### 6.4 規模系列

テキストのみを 0.5B / 1.5B / 3B（＋ 7B 1 点）で、0.5B ＋ 構造 と同じ図に載せる。「小さい LM ＋ 構造 が 大きい LM 単体 に届くか」は、同一族・同一データ・同一トークン予算でのみ主張する。

## 7. 統計と事前登録

- seed は主要セルで 5、その他で 3。
- 比較はテスト項目の対応つき bootstrap と、seed × 項目の二段 bootstrap（`scripts/compare_runs.py --seed_bootstrap`）。
- 主比較は仮説ごとに事前に固定し、仮説の族の中で Holm 補正する。
  - H1: 線形＋φ内 − 線形なし、および (線形＋φ内 − 線形なし) / (softmax外 − 線形なし) の回復率。
  - H2: 型 × 形 の交互作用。導出可能型での 線形＋構造 − 線形なし を主、softmax＋構造 − softmax なし を副。
  - H3: 回復量の L に対する傾き（合成データ）。
  - H4: 線形＋φ内 − 構造なし回復手段の最良。同コストの対。
- 分解能（95% 区間の半幅）を予備実験で見積もり、それより小さい差は「見分けられない」と書く。化学の事前登録（`docs/CHEM_PROTOCOL_ja.md`）と同じ運用。
- 本実験の前に事前登録文書をコミットし、予備実験は探索と明記する。

## 8. 予備実験（2〜3 週間）と中止基準

| 内容 | 規模 |
|---|---|
| 合成グラフ 1 課題（最短路長）× 2 サイズ、POJ-104 ファイル全体 | 2 データ |
| 腕: softmax なし / softmax 外 / 線形なし / 線形外 / 線形内 / 厳密マスク、テキストのみ | 7 腕 × 1〜2 seed |
| 測るもの: 精度、§3.1 の鋭さ、壁時計 | |

中止・変更の基準:

- 必要型で 線形＋φ内 が差の 30% 未満しか回復しない → GRF の分散（§10）を疑い、特徴数・重要度重み・GRFs++ 型の縫合を試してから規模を広げる。
- 導出可能型で 線形＋構造 の利得が分解能の中 → H2 は否定で確定とし、重心を H1・H3・H4 に移す。計画は続行。
- 壁時計の交差点が 64k トークンを超える → 実データの大きさでは効率の主張が成り立たないので、合成データのサイズ帯を上げるか、効率の主張をトークン数に限定する。

## 9. 計算資源と日程

- GPU: RTX 3090 1 枚で主実験は成立する。見積もり 250〜400 GPU 時間、デバッグ込みで 3〜5 週間の連続稼働。7B の 1 点と 3B の LoRA 対照はクラウド（A100 で 100〜200 ドル）。
- ストレージ: 凍結 LM のキャッシュは整合ノード単位にプールして保存する（トークン全部を保存すると 1 件 8 MB、全体で 500 GB になる）。200〜300 GB の SSD を確保する。
- 削る順番（資源不足のとき）: 7B → 3B の LoRA を凍結＋アダプタに格下げ → 学習曲線を 5 点から 3 点に → SSM 参照腕。融合の因子セル、§4.5 の対照、3 型、3 seed は削らない。
- 日程（相対）: 第 1〜3 週 予備実験と前処理、第 4 週 事前登録、第 5〜10 週 本実験、第 11〜12 週 解析と執筆。執筆は並行。

## 10. 技術的リスクと対策

| リスク | 根拠 | 対策 |
|---|---|---|
| GRF は疎で直径の大きいグラフで分散が大きく、歩みが交わらない対にゼロを返す。AST は木なのでまさにこの条件 | [Topological Masking](https://arxiv.org/html/2410.03462v2)、[GRFs++](https://arxiv.org/html/2510.07716v1) | 重要度重みつき GRF を必須にする（重みなしは無マスクより悪化した報告あり）。歩数と本数を予備実験で掃引。GRFs++ の縫合を実装候補に |
| テンソル積で特徴幅が d_s × d_g になる | 同上 | GRF を疎に保つ。アンカー点方式（KDLGT 形）を代替に |
| softmax 側は FlashAttention で実測が速く、線形が壁時計で勝つのは長い系列だけ | [GTLM](https://arxiv.org/html/2605.10247) | 交差点を測って報告する。FLOPs で勝ったと言わない |
| SSM 順序型は順序に敏感 | [Chimera](https://arxiv.org/html/2510.12111v1) | 作用素型のみ使う。順序型は入れない |
| ソフトトークンの崩壊（定数で置換可能） | [Token Collapse](https://arxiv.org/html/2610.31572) | 定数トークン対照を常に置く。構造は列として入れ、プールしない |
| 構造バイアスの種類で結果が逆転する（SPD が無効な場面） | [GTLM](https://arxiv.org/html/2605.10247)、[Attending to GTs](https://arxiv.org/abs/2302.04181) | φ の外の水準は辺種別バイアスと距離バイアスの 2 種を用意し、予備実験で選ぶ |
| 融合層だけの効率化では「LLM の効率化」と言いにくい | 報告書 §5 | 主張を「構造つき入力に対する融合経路のコストとトークン数」に限定して書く。Mamba 系 LLM は拡張 |

## 11. 位置づけ（必ず対置する先行）

| 軸 | 先行 |
|---|---|
| φ 内の構造 | GKAT / [ICML 2022](https://ar5iv.labs.arxiv.org/html/2107.07999)、[Topological Masking, ICLR 2025](https://arxiv.org/html/2410.03462v2)、[KDLGT](https://www.ijcai.org/proceedings/2023/0263.pdf)、[GRFs++](https://arxiv.org/html/2510.07716v1) |
| φ 外の線形グラフ Transformer | [NodeFormer](https://arxiv.org/html/2306.08385)、[SGFormer](https://arxiv.org/html/2306.10759v5)、[Polynormer](https://arxiv.org/html/2403.01232v3)、[GraphTARIF](https://arxiv.org/html/2510.10631v2) |
| softmax の標準形とバイアス | [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf)、[Graphormer](https://arxiv.org/abs/2106.05234)、[GRIT](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf) |
| LLM 内部への注入 | [GTLM](https://arxiv.org/html/2605.10247)、[GaLA](https://arxiv.org/html/2606.15633v2)、[Graph-KV](https://arxiv.org/html/2506.07334)、[NAG](https://arxiv.org/html/2601.22657)、[Plenz & Frank](https://arxiv.org/html/2401.07105v3) |
| 入力ソフトトークン | [GraphToken](https://arxiv.org/abs/2402.05862)、[GraphLLM](https://arxiv.org/html/2310.05845)、[LLaGA](https://arxiv.org/abs/2402.08170) |
| グラフ SSM | [Graph-Mamba](https://arxiv.org/html/2402.00789v1)、[GMN](https://arxiv.org/html/2402.08678v2)、[GSSC](https://arxiv.org/html/2406.05815v2)、[Chimera](https://arxiv.org/html/2510.12111v1)、[HSA-Net](https://arxiv.org/html/2508.08334v1) |
| 一致比較と否定結果 | [Tabula RASA](https://arxiv.org/html/2602.02834)、[When Structure Doesn't Help](https://arxiv.org/html/2511.16767v2)、[Token Collapse](https://arxiv.org/html/2610.31572)、[Less is More](https://arxiv.org/html/2508.10115v1)、[LLMNodeBed](https://arxiv.org/html/2502.00829v2) |
| コードの構造注入 | [GraphCodeBERT](https://ar5iv.labs.arxiv.org/html/2009.08366)、[Agarwal et al.](https://arxiv.org/html/2401.10716v1)、[UniXcoder](https://ar5iv.labs.arxiv.org/html/2203.03850) |

貢献の言い方: 「新しい融合機構の提案」ではなく、「効率的 attention に構造を入れるとき、どの入口が、どの大きさと相補性の条件で、何を節約するかを、統制された条件で測った」。

## 12. 実装計画（このリポジトリへの対応）

既存の学習・評価・比較の経路（`run.py` の run ディレクトリ、`metrics.json`、`scripts/compare_runs.py`）はそのまま使う。Joern の経路は触らず残す。

| 新規 / 変更 | 内容 |
|---|---|
| `utils/process/ast_tree_sitter.py`（新規） | tree-sitter で AST を取り、親子・兄弟・次トークン・def-use の 4 関係と、トークン→ノードの整合（文字オフセット）を書く。`--cpg` の代替として `--ast` を追加 |
| `utils/process/structure_features.py`（新規） | 打ち切り最短路、隣接の冪、重要度重みつき GRF、ランダムウォーク構造符号化。前処理時間を manifest に記録 |
| `utils/process/text_cache.py`（新規） | 凍結 LM の最終層隠れ状態を、整合ノード単位にプールして fp16 で保存。LM 名・層・プール法を manifest に記録 |
| `models/fusion/`（新規） | 結合列の融合層。引数 `--attn_form {softmax,linear,ssm}`、`--struct_entry {none,outside,inside,mask}`、`--sharpen {none,window,gumbel,rankgate,featmap}`。鋭さ指標を返すフック |
| `models/LMGNN.py`（変更） | 既存の `GraphCodeFusion` を `--attn_form softmax --struct_entry none` の 1 点として残し、新しい融合層を選べるようにする |
| `run.py`（変更） | 上の引数、壁時計・ピークメモリの記録、`--data_fraction`、定数トークン・MLP アダプタ・直列化 LoRA の対照腕 |
| `data/scripts/make_synth_graph_json.py`（新規） | 合成グラフ推論の生成（課題・N・密度・seed を引数に）。正解は networkx で計算 |
| `data/scripts/make_tag_json.py`（新規） | Cora / PubMed / ogbn-arxiv / 異類性グラフの読み込みと近傍部分グラフの切り出し |
| `scripts/microbench_fusion.py`（新規） | §6.2 の格子計測 |
| `scripts/learning_curve.py`（新規） | §6.3 の分割と集計 |
| `tests/`（追加） | 線形 φ が特徴数を増やすと softmax に収束すること、マスクの正しさ、作用素型 SSM の置換同変性、既存の二値経路のビット同一（`test_golden_binary.py` を維持） |

実装の順番は、予備実験に必要なものから: tree-sitter → キャッシュ → 融合層（softmax / 線形 / φ 内）→ 合成データ → マイクロベンチ。

## 13. 未決事項（着手前に決める）

1. テキスト側 LM の族（Qwen2.5 か Llama-3.2 か）。GTLM との対置を優先するなら Llama。
2. 修論の締切と、学習を始める期限。日程（§9）はそこから逆算して確定する。
3. QM9 を入れるか。入れるなら回帰指標の統計手順を追加する。
4. 7B の 1 点にクラウド予算を使えるか。
5. SSM 参照腕の実装方針（Chimera 形のレゾルベント近似の次数）。
