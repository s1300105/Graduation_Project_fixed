# LLM×GNN 構造融合: 部品は既出、因子分解×効率は未着手

**結論を先に述べる。** 「LLM と GNN 系の構造情報を attention の内部で融合する」研究は、構成要素ごとに見れば 2018–2026 年の間にほぼ出揃っている。スコアバイアス (Graphormer → GRIT → GDT)、support/マスク (Exphormer, GraphCodeBERT, Graph-KV)、正規化 (NodeFormer の Gumbel、SGFormer/Polynormer の線形カーネル)、値変換 (HGT, GRIT, Relational Transformer)、位置・構造符号化 (RWSE/RRWP/LapPE/Magnetic Laplacian)、線形 attention の φ 内部への構造挿入 (GKAT 2021 → ICLR 2025 topological masking)、グラフ向け SSM (Graph-Mamba/GMN → GSSC/Chimera/MP-SSM)、そして 2025–2026 年にはデコーダ LLM の attention に直接 SPD/RRWP バイアスや graph マスクを入れる研究 (GTLM, GaLA, Graph-KV, NAG, UniGTE) が登場した。しかし、計画されている修士研究の核である **「attention の構成要素を因子として分解し、どの要素が構造融合の利得を担うかを、構造–テキストの相補性で層別したデータセット横断で、精度ではなく効率 (データ効率・規模効率・トークン効率) を従属変数として測る」研究は、2026 年 11 月時点で存在しない**。最も近いのは、KGQA でパラメータを 67.1M に揃えてマスク vs バイアス vs 関係特異パラメータを比べた Tabula RASA (2026) と、Llama-3.2-1B の attention に 3 種のバイアスを入れて leave-one-out した GTLM (2026) だが、前者は LLM もテキスト豊富なノードも持たず明示的に「非因子計画」、後者はマスク・PE・正規化を固定しバイアス種のみを変えている ([Tabula RASA](https://arxiv.org/abs/2602.02834); [GTLM](https://arxiv.org/html/2605.10247))。正規化 (softmax vs カーネル/線形) と関係特異 W_V を LLM 融合の因子として扱った研究、構造を φ に入れた線形 attention で事前学習 LM とグラフを接続した研究、グラフ SSM を TAG/KG/コードで LLM と融合した研究はいずれも見つからなかった。一方で、2025–2026 年の否定結果 (When Structure Doesn't Help, Token Collapse, Less is More, LLMNodeBed) は、「テキストが豊富な同類性 TAG では構造は冗長」「プールされたソフトトークンは定数ベクトルで置換可能」という事実を数値で示しており、計画研究が置くべき必須ベースライン (テキストのみ CO、学習済み定数トークン、直列化テキスト LoRA、素の GNN、MLP アダプタ) を規定する。

---

## 1. 更新版三部表: 確立している部分 / 近い研究と残っている差分 / まだ無いもの

### 1-A. 確立している部分 (検証済み出典つき)

| 確立事項 | 根拠となる数値・記述 | 出典 |
|---|---|---|
| 純グラフ Transformer では、ペアごとのスコアバイアス (a) が最も一貫して効く | Graphormer: PCQM4M-LSC valid MAE 0.2276 (無) → 0.1304 (spatial+edge bias); GraphiT: ZINC 0.696 → 0.243 (隣接カーネル); GRIT attention → Graphormer attention に置換で ZINC 0.059 → 0.117 | [Graphormer](https://arxiv.org/abs/2106.05234); [GraphiT](https://ar5iv.labs.arxiv.org/html/2106.05667); [GRIT](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf) |
| バイアス付き softmax attention は一つの式 softmax(QKᵀ/√d + B)V に統一され、局所 GT・因果マスク・Shaw/ALiBi 型相対 PE を特殊例として含む。表現力は GD-WL で特徴づけられる | GDT (NeurIPS 2025) Theorem 1; PE なし GDT ≡ 1-WL; RRWP ≻ RWSE (同歩数), SPE ⪰ LPE | [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf) |
| バイアスの効果はデータセット依存で、SPD バイアスは異類性ノード分類で無効、ZINC では必須 | Attending to GTs: 異類性小規模データで Graphormer の attention bias は「no impact」、ZINC では無効化で 0.12 → 0.54 MAE | [Attending to Graph Transformers](https://arxiv.org/abs/2302.04181) |
| 線形/カーネル化 attention は softmax より精度で劣り、原因は分布の一様化と低ランク化 | GraphGPS: PCQM4Mv2 Transformer 0.1159 / Performer 0.1142 / BigBird 0.1237, CIFAR10 72.31 / 70.67 / 70.48; GraphTARIF が「overly uniform, low-rank」と診断 | [GraphGPS](https://arxiv.org/abs/2205.12454); [GraphTARIF](https://arxiv.org/abs/2510.10631) |
| 構造を φ の内部に入れる線形 attention は存在し理論的にも特徴づけ済み (テンソル積特徴、疎 GRF で O(N)、McDiarmid 集中) | ICLR 2025: ImageNet unmasked linear 0.693 → GRF-in-φ 0.730 → exact mask / softmax 0.741 | [Topological Masking with GRFs](https://arxiv.org/html/2410.03462v2) |
| デコーダ LLM の attention 内部への構造注入は極めて安価で、構造中心タスクでは大きく効く | GTLM: 構造パラメータ 173,056 (1B の 0.015%)、GraphQA 平均 33.2 → 85.7、needle-in-a-graph 98.0% vs 平文 8.5%; GaLA: 学習なしで Cora 半合成 69.3 → 87.9 | [GTLM](https://arxiv.org/html/2605.10247); [GaLA](https://arxiv.org/html/2606.15633) |
| テキスト豊富な同類性 TAG では構造チャネルの寄与はほぼゼロ | CO (テキストのみ) 平均 75.29% > ND テンプレート 69.48%; MLP アダプタ 76.26% > GCN 67.50; G-Retriever の graph token は学習済み定数で置換可能 (WebQSP 71.25 vs 70.33) | [When Structure Doesn't Help](https://arxiv.org/html/2511.16767v2); [Token Collapse](https://arxiv.org/html/2610.31572) |
| KGQA ではパラメータ一致下でマスクがバイアスより圧倒的に効く | Tabula RASA (67.1M 一致): Mask-only で MetaQA-3hop +72.5pp, WebQSP +45.5, CWQ +53.9; Bias-only は vanilla 同等以下 (CWQ 2.1% vs 2.7%) | [Tabula RASA](https://arxiv.org/html/2602.02834) |
| 分子では、グラフトークンは生成系タスク (キャプション・IUPAC・反応・QM9 回帰) で効き、MoleculeNet 二値分類では効かないか逆効果 | MolCA: 同一 Galactica-1.3B で BLEU-2 34.6 → 38.7; Mol-LLM: w/o Graph の方が BACE 80.8 vs 80.5, BBBP 84.3 vs 81.1 | [MolCA](https://arxiv.org/html/2310.12798); [Mol-LLM](https://arxiv.org/html/2502.02810v2) |
| コードでは、事前学習済みテキストモデルに対する明示構造の利得は全データ時に小さく (0.5–3 点)、低資源時と構造感応タスクで大きい | GraphCodeBERT MRR 0.713 vs 0.693; UniXcoder w/o AST は要約 BLEU に影響なし; Agarwal et al.: CoNaLa で構造 100 例 ≥ 平文 1.8k 例 | [GraphCodeBERT](https://ar5iv.labs.arxiv.org/html/2009.08366); [UniXcoder](https://ar5iv.labs.arxiv.org/html/2203.03850); [Agarwal et al.](https://arxiv.org/html/2401.10716v1) |
| グラフ SSM は順序型 (Graph-Mamba, GMN) と作用素型 (GSSC, Chimera, MP-SSM) に分裂し、順序型は LRGB で GraphGPS/Exphormer を上回るが置換同変性を失う | Peptides-func AP: GPS 0.6575 / GMN 0.7071 / Chimera 0.7021; GPS+Mamba 単純置換は Amazon-ratings で 45.13 (GPS 53.10 より低い) | [Chimera](https://arxiv.org/html/2510.12111v1); [MP-SSM](https://arxiv.org/html/2505.18728v1) |
| 構造エンコーダの優位は、テキストチャネルが弱い条件 (低データ・小/弱 LM・疎テキスト・直列化がコンテキストを飽和) に集中する | Sanford et al.: 1K サンプルで MPNN node degree 99.4 vs Transformer 31.6、100K で逆転; GraphLLM: 45 ノードで Graph2Text は 4,096 トークン超過 | [Sanford et al.](https://arxiv.org/html/2405.18512); [GraphLLM](https://arxiv.org/html/2310.05845) |

### 1-B. 近い研究と残っている差分 (1 論文 1 行)

#### B-1. attention 構成要素の設計空間 (純グラフ Transformer 側)

| 研究名・年・会場 | 何をしたか | 残っている差分 | URL |
|---|---|---|---|
| Self-Attention with Relative Position Representations — 2018, NAACL | 相対項を key 側 a^K と value 側 a^V に分け、WMT14 で a^K+a^V 25.8 / a^K のみ 25.8 / a^V のみ 25.3 / なし 12.5 BLEU: value 側は追加利得ゼロ | (a) vs (d) 分解の原点だが MT のみ。グラフ Transformer・LM 融合でこの分解は未再現 | https://arxiv.org/abs/1803.02155 |
| Heterogeneous Graph Transformer (HGT) — 2020, WWW | 型・関係特異の K/Q/V/ATT 行列の有無を ablation: OAG NDCG .615 (full) → .578 (共有重み); 論文特徴量は凍結 XLNet | 関係特異 W_V の唯一のクリーンな数値だが異種学術グラフのみ、LM は凍結特徴として使用、効率指標なし | https://arxiv.org/abs/2003.01332 |
| Do Transformers Really Perform Bad for Graph Representation? (Graphormer) — 2021, NeurIPS | spatial (SPD) bias・edge bias・centrality の積み上げ ablation (0.2276 → 0.1304) | 「バイアス」の中の積み上げのみ; マスク・正規化・W_V は固定; テキストなし | https://arxiv.org/abs/2106.05234 |
| Global Self-Attention as a Replacement for Graph Convolution (EGT) — 2022, KDD | エッジチャネルをスコアバイアス (a) と value の sigmoid ゲート (d) に同時使用; EGT-Constrained (1-hop のみ=マスク) 0.174 vs EGT 0.108 ZINC | ゲート (d) 単独の行別数値が抽出できず; テキストなし | https://arxiv.org/abs/2108.03348 |
| Structure-Aware Transformer (SAT) — 2022, ICML | 構造抽出器を Q/K の前段に入れ (構造条件付きスコア)、value は生特徴のまま; RWPE 0.310 → k-subgraph SAT 0.094 | 「スコア側のみ構造」の設計だが value 側との比較なし; OGBG-CODE2 は AST のみで LM なし | https://arxiv.org/abs/2202.03036 |
| Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS) — 2022, NeurIPS | 正規化 (Transformer/Performer/BigBird) と PE/SE (none/RWSE/LapPE/SignNet) の ablation; MPNN 除去は致命的、global attention 除去は軽微 | 正規化 × PE の交互作用はなし; 因子は一度に一つ; テキストなし | https://arxiv.org/abs/2205.12454 |
| NodeFormer — 2022, NeurIPS | カーネル化 Gumbel-softmax (c) + 観測エッジへの加法バイアス (a) + エッジ正則化; Gumbel vs 通常 softmax で OGB-Proteins 77.45 vs 75.50; 付録 Table 6: Cora w/o edge reg 81.98、w/o relational bias 88.06 (full 88.69) | 構造は φ の外 (加法項); 正規化の比較は Gumbel の有無のみ; 20News は TF-IDF で LM なし | https://arxiv.org/abs/2306.08385 |
| Relational Attention (Relational Transformer) — 2023, ICLR | エッジベクトルを Q, K, V すべてに注入 (a+d); CLRS で set-Transformer 42.34% → RT 81.30% | edge-in-Q/K と edge-in-V を分離 ablation していない; Graphormer 型スカラーバイアスとも未比較 | https://arxiv.org/abs/2210.05062 |
| Exphormer — 2023, ICML | support を局所エッジ ∪ expander ∪ 仮想ノードに限定 (b); 各パターン除去で CIFAR10 74.69 → 74.53–74.68 (ほぼ可換) | support のみの研究; バイアス・W_V と交差なし; テキストなし | https://arxiv.org/abs/2303.06147 |
| Graph Inductive Biases in Transformers without Message Passing (GRIT) — 2023, ICML | RRWP ペア表現を乗法 FiLM 条件付け + 加法バイアス + value 側 (W_Ev ê_ij) に同時注入; ablation は degree scaler, RRWP 更新, sparse 化, attention 形式, PE 種 | value 側 W_Ev 項の単独 ablation なし; 2024–2026 に GRIT 型 value 条件付け vs GDT 型加法バイアスの同一データ比較なし | https://proceedings.mlr.press/v202/ma23c/ma23c.pdf |
| SGFormer — 2023, NeurIPS | softmax なし単層線形 attention + GCN 分岐; softmax 版は 40K ノードで OOM、線形は 1.5 GB | softmax vs 線形の精度比較なし; PE ablation なし; テキストなし | https://arxiv.org/abs/2306.10759 |
| Attending to Graph Transformers — 2024, TMLR | PE/SE・補助 GNN・マスク・バイアスの分類と実験; SPD バイアスは異類性で無効、ZINC で必須; GPS の局所 GCN を無効化すると異類性で改善 | 設計空間の分類はあるが系統的な因子交差なし; LM なし | https://arxiv.org/abs/2302.04181 |
| Comparing Graph Transformers via Positional Encodings — 2024, ICML | APE (e) と RPE (a) の表現力等価性を証明 (無特徴グラフ); ノード特徴ありでは RPE ≥ APE | 理論は PE vs バイアスのみ; マスク・正規化・W_V を扱わず; 効率は非対象 | https://arxiv.org/abs/2402.14202 |
| OpenGT — 2025, arXiv (会場なし) | 13 種 GT + GNN を単一プロトコルで比較; PE の有用性は「データと構造に依存」; GPS+RWSE ZINC 0.0681 vs 0.1968; 前処理コスト (PubMed: RW 10 分, 全対 SPD 2 時間) | attention 形式はモデルごとに固定で因子化されていない; LM なし; 効率は前処理時間のみ | https://arxiv.org/abs/2506.04765 |
| Generalizable Insights for Graph Transformers (GDT) — 2025, NeurIPS | 統一式 softmax(QKᵀ/√d + B)V を提案し、PE 種 × 規模 (15M–160M, 8M グラフ) を比較; RRWP が 4/6 で最良だが高コスト、「PE efficiency can vary greatly while predictive performance differences are less pronounced」 | 式は MPNN・マスク・相対 PE を包含するのに、実験は PE 軸のみ; マスク/バイアス/W_V 軸を変えていない; LM なし (ogbg-code2 は AST トークン) | https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf |
| GraphTARIF — 2026, WWW | 線形 attention の一様化・低ランク化を診断し、value 側ゲート付き GAT 分岐 (d) と log-power 焦点化 (c) を追加; Minesweeper 91.62 → 99.03 | (c)(d) を同時に加えており単独効果は分離されない; LM なし | https://arxiv.org/abs/2510.10631 |
| X-LogSMask — 2026, arXiv (会場なし) | log(Ã^k+ε) を logits に加算 (ソフトなマスク/バイアス混成); Cora 82.70 → 58.80 (除去) | PE vs マスクの直接比較なし; LM なし | https://arxiv.org/html/2607.01553 |

#### B-2. 線形 attention のカーネル特徴写像 φ

| 研究名・年・会場 | 何をしたか | 残っている差分 | URL |
|---|---|---|---|
| Graph Kernel Attention Transformers (GKAT) — 2021, arXiv v1 (後に ICML 2022 論文へ改題) | softmax カーネル特徴 ⊗ RWGNK グラフカーネル特徴のテンソル積で φ 内部に構造を入れる; 誘導閉路検出 83.6% vs GAT 77.0%; GAT 比 Pubmed で 9.5× 高速 | テキスト・LM なし; TU/引用グラフでは SAT/LSAT に劣る (KDLGT Table 2: Cora GKAT 73.84 vs SAT 83.06) | https://arxiv.org/abs/2107.07999v1 |
| From block-Toeplitz matrices to differential equations on graphs — 2022, ICML | マスク付き低ランク attention A = M ⊙ 𝒦(Q,K) の一般理論; 因果・Toeplitz RPE・木・拡散カーネルマスクの効率的実装; ViT で +2.5–3.4 点 | 実験は ViT と GKAT タスクのみ; 言語モデルなし | https://proceedings.mlr.press/v162/choromanski22a.html |
| KDLGT — 2023, IJCAI | 低ランク RPE を [Q, B_q][K, B_k]ᵀ の形でカーネル引数に連結し Performer 化; LSAT vs SAT ZINC 0.130 vs 0.082、Cora 83.37 vs 83.06 | 「φ 内部」の弱い形; LM なし; 効率は速度 (9.63×) のみ | https://www.ijcai.org/proceedings/2023/0263.pdf |
| Linear Transformer Topological Masking with Graph Random Features — 2025, ICLR | 疎 GRF ⊗ q/k 特徴で O(N) マスク付き線形 attention; ImageNet 0.693 → 0.730 (softmax 0.741); 重要度重みなし GRF は 0.689 (無マスクより悪化) | 格子/kNN グラフのみ; 任意グラフ・テキスト・LM なし; softmax+exact mask との比較なし | https://arxiv.org/abs/2410.03462 |
| GRFs++ (Refined Graph Random Features) — 2026, ICML | 短い歩みの縫合と Poisson 停止で遠隔ノード対のゼロ推定問題を解消 (直径 159 まで) | Transformer への適用なし; 言語グラフ (AST, 依存構造) での分散測定なし | https://arxiv.org/html/2510.07716v1 |
| Rotary Position Encodings for Graphs (WIRE) — 2025, arXiv | Laplacian スペクトルによる q,k 回転; 「線形 attention と互換、バイアス型 RPE は非互換」 | PRE-φ (φ の前段); LM 適用なし | https://arxiv.org/abs/2509.22259 |
| Analytic-Walk Rotary Positional Encodings (AW-RoPE) — 2026, arXiv | エッジ上の回転を歩みに沿って輸送し Performer φ の前に適用; ogbg-code2 F1 0.173 → 0.185 | φ は回転と可換でないため厳密性は生 dot-product のみ; LM なし | https://arxiv.org/html/2609.32178v1 |
| GIMLET — 2023, arXiv | T5 の softmax attention に SPD + 経路エッジ埋め込みバイアスを加算 (graph→text は −∞ マスク); GKAT を引用するが機構は不使用 | softmax 加法バイアスであり φ 注入ではない; 線形 attention の議論なし | https://arxiv.org/html/2306.13089v3 |

#### B-3. グラフ状態空間モデル

| 研究名・年・会場 | 何をしたか | 残っている差分 | URL |
|---|---|---|---|
| Graph-Mamba — 2024, arXiv (査読会場なし) | GraphGPS の attention を Mamba ブロックに置換、次数順ソート + ランダム置換; GPS-Transformer 比 GPU メモリ −74%, FLOPs −66%; 順序 ablation で PascalVOC F1 0.3105 → 0.4314 | 構造は入力順序と GatedGCN のみ; 置換同変性なし; LM なし; データ効率なし | https://arxiv.org/abs/2402.00789 |
| Graph Mamba Networks (GMN) — 2024, KDD | ランダムウォーク部分グラフトークン化 + 双方向 Mamba; ogbn-arxiv 3.85 GB vs Exphormer 36.18 GB; 双方向除去が最大の低下 (−4.4〜−5.0 点) | 著者自身が表現力は「サンプリングと符号化であり Mamba 自体ではない」と述べる; LM なし | https://arxiv.org/abs/2402.08678 |
| What Can We Learn from State Space Models for ML on Graphs? (GSSC) — 2024, arXiv (NeurIPS 2024 帰属は未検証) | SSM カーネルを Laplacian 固有ベクトルで因子化した距離カーネルに置換 (順序不要、置換同変); ZINC 0.064 (GRIT 0.059), Peptides-func 0.7081 | 構造カーネル単独の因子化で特徴類似カーネルとの積ではない; LM なし | https://arxiv.org/abs/2406.05815 |
| Best of Both Worlds: Hybrid Graph Sequence Models (GSM++) — 2025, ICML | 9 系列モデル × 6 トークナイザ × 7 データで「no free lunch」; SSM は近傍トークンに敏感・深さで表現崩壊; Mamba → Transformer ハイブリッド | 効率理論はあるがデータ効率・LM 融合なし | https://arxiv.org/abs/2411.15671 |
| Chimera: State Space Models Beyond Sequences — 2025, TMLR | SSM マスクを (I−A)⁻¹ レゾルベントに一般化 (任意位相); LRGB で GMN 同等以上; GLUE で BERT +0.7 | グラフと言語は別実験で融合ではない; 一般グラフでは二次〜三次コスト | https://arxiv.org/abs/2510.12111 |
| Message-Passing State-Space Models (MP-SSM) — 2025, NeurIPS | メッセージパッシング深さ方向に線形再帰; 置換同変; Roman-empire 90.91 vs GMN 87.69 | LM なし; LRGB 数値は付録 (集約サイトの 0.2458 は未検証) | https://arxiv.org/abs/2505.18728 |
| HSA-Net — 2025, arXiv (会場なし) | 分子 GNN と LLaMA-2-7B-chat / Galactica-1.3B の間に「Graph-Mamba projector」(断片+次数順ソート) を置く; ablation で Mamba 単独 (37.5 BLEU) は cross-attention 単独 (37.9) を上回らない | 唯一の graph-SSM × LLM 融合だが分子 (≤数百原子) のみ、順序型、時間/メモリ/FLOPs 未報告、ソート規則や隣接入力の ablation なし | https://arxiv.org/html/2508.08334v1 |
| AdaptQA (Knowledge-Based Systems 2025) | KG から「Graph-based Mamba」で回答生成 → LLM が後段で書き換え (HotpotQA 94.47%) | LLM は後処理で共同学習ではない; **全文未取得・構成未検証** | https://ro.ecu.edu.au/ecuworks2022-2026/6144 |
| NeuroMambaLLM — 2026, arXiv | fMRI の ROI 時系列に Mamba を走らせ LLaMA-3-8B (LoRA) に投影 | SSM は時間方向でグラフノード方向ではない | https://arxiv.org/abs/2602.13770 |
| Towards Understanding What State Space Models Learn About Code — 2026, arXiv | SSM コード LM を AST 兄弟/距離/DFG エッジで probing; 事前学習では Transformer より構造を捉えるが fine-tuning で忘却 | SSM 系 LM を probing した研究で、グラフ SSM と LM の融合ではない | https://arxiv.org/abs/2602.06774 |

#### B-4. LLM 側注入法 2024–2026

| 研究名・年・会場 | 何をしたか | 残っている差分 | URL |
|---|---|---|---|
| Graph Language Models — 2024, ACL | T5 の相対位置パラメータをグラフ上ホップ距離で再利用 + ℓGLM/gGLM マスク; ConceptNet 線形プローブ 59.5% vs 直列化 55.3%; マスク時 41.8 vs 20.3; REBEL 全データでは差消失 (85.28 vs 85.36)、低データで差拡大 | エンコーダ LM (T5)・KG のみ; 相対 PE とマスクを分離 ablation していない; 効率は「同複雑度」の記述のみ | https://arxiv.org/abs/2401.07105 |
| GraphLLM — 2023, arXiv | グラフ Transformer 出力を 5 トークン prefix として全層 K/V に注入; 平均 98.19% vs Graph2Text 47.35%; コンテキスト −96.45%、推論 3.42× 高速 | 合成グラフ (15–45 ノード) のみ; 13B は 7B を上回らない; マスク/バイアスとの比較なし | https://arxiv.org/abs/2310.05845 |
| ENGINE — 2024, IJCAI | 凍結 LLaMA2-7B の各層に G-Ladder GNN 側枝; キャッシュで学習 10h18m (LoRA) → 21 分、早期終了で推論最大 5×; message passing 除去で Cora 91.48 → 76.94 | LLM-as-encoder 構成で attention 内部は非改変; データ効率曲線なし | https://arxiv.org/abs/2401.15569 |
| G-Retriever — 2024, arXiv (NeurIPS 2024 は arXiv 上で未確認) | graph token + 直列化テキスト + PCST 検索; WebQSP トークン 100,627 → 610; GraphToken 57.05 < LoRA-on-text 66.03 < G-Retriever 70.49 | 後に Token Collapse で graph token が定数で置換可能と判明; attention 内部は非改変 | https://arxiv.org/abs/2402.07630 |
| Attention Mechanisms Perspective — 2025, ICML | 直列化 2-hop 近傍での LLaMA2-7B の attention を解析; node→node attention はグラフ構造に従わない; 可視範囲 k の中間窓が最良 (WikiCS 双方向 67.81 → 78.43) | 因子は attention 窓 (support) のみ; バイアス・PE・W_V なし | https://arxiv.org/abs/2505.02130 |
| Graph-KV — 2025, NeurIPS | 各チャンクを独立 KV 化し、対象チャンクはグラフ上ソースの KV のみ参照; ソースは同一位置範囲を共有; Cora 71.03 vs sequential 67.09 | 1-hop のみ; バックボーン非 fine-tune; マスクと位置 ID 共有を分離せず | https://arxiv.org/html/2506.07334 |
| UniGTE — 2025, NeurIPS | Vicuna-7B + LoRA、graph token に共有位置 ID、SPD バイアス、経路エッジ記述バイアス、text→graph 遮断マスク; PubMed 0.870 vs TEA-GLM 0.781; Cora 0.6 s/sample vs GOFA 5.1 s | バイアス・マスク・位置共有を同時導入し個別 ablation なし; 効率は per-sample 時間のみ | https://arxiv.org/abs/2510.16885 |
| NAG — 2026, arXiv | Qwen3-0.6B、エンコーダなし、閉じタグをハブにしたマスク + 位置 ID 再調整; 位相タスク 8/9 で Qwen3-LoRA を上回る; 位置再調整 ablation は +2.95pp (connected nodes) / −0.5pp (ExplaGraphs) | パラメータ・遅延未報告; TAG データなし; マスク vs バイアス比較なし | https://arxiv.org/html/2601.22657 |
| Teaching LLMs to See Graphs (GTLM) — 2026, arXiv (会場なし) | Llama-3.2-1B の全層 attention に SPD/RRWP/Magnetic Laplacian バイアス (173k パラメータ) + ノードごと RoPE リセット; leave-one-out: Magnetic 除去で Edge Count 56.9 → 7.1、SPD 除去は誤差内 (Shortest Path 90.1 → 92.0); 1B→8B で分類は ≤0.3pp; FlexAttention で FlashAttention 比 ≈2× ステップコスト | マスク (PrefixLM) と PE リセットは固定で ablation なし; 正規化・W_V 因子なし; GNN ソフトトークン腕なし; 学習データ割合曲線なし; 「1B ≈ 7B InstructMol」はベースライン別論文で非一致比較 | https://arxiv.org/html/2605.10247 |
| Formalizing and Mitigating Structural Distortion in LLM Attention (GaLA) — 2026, KDD | 凍結 Qwen2.5-3B/7B, Ministral-3B に推論時のみ λ_h/d_G(i,j) バイアス (前半層); PubMed 77.5 → 82.5; 利得は小型で大きく 7B で縮小 | 単一因子 (バイアス) のみ; マスクと未比較; 学習・データ効率なし | https://arxiv.org/html/2606.15633 |
| Slash the Sink (SLASH) — 2026, arXiv | 学習なしで sink の attention 重みを γ 倍し再正規化 (c 類似の介入); GraphInstruct 平均 Llama-3.2-3B 0.085 → 0.214; 完全 attention 行列が必要で FlashAttention 非互換 (36.1 GB vs 13.3 GB) | 正規化介入だが構造因子との交差なし; fine-tuned GraphWiz では効果なし | https://arxiv.org/html/2605.10503v3 |
| GaRA — 2026, ICML | GCN からグラフ条件付き LoRA 因子を生成し layer 21/32 の V 射影に注入; 層スイープ {1,11,21,31,32} で中間層が最良 | 重み注入は入力ソフトトークンと併用必須; 予算一致の注入点比較なし | https://arxiv.org/html/2606.22429v1 |
| GaLoRA — 2026, arXiv (ワークショップ) | GPT-2 (124M) の層 5–7 / 9–11 に GraphSAGE 埋め込みをゲート注入、0.295M 学習パラメータ; Reddit で GraphAdapter-Llama-2-13B (0.6461) を上回る 0.6611 | 注入点 ablation なし (理由づけのみ); 規模効率の主張は非一致比較 | https://arxiv.org/html/2603.10298 |
| Scaling Graph Reasoning with Learnable Graph Positional Encodings — 2026, arXiv | GNN+GT が計算した置換同変 PE で LRM の attention logits を変調、隣接行列直列化を除去しコンテキスト O(N) | **LLM サイズ・データ・数値は abstract 上で未検証** | https://arxiv.org/abs/2610.27439 |
| One LLM Token for Explicit Graph Structural Understanding (SOG_k) — 2026, arXiv (ICLR 2026 は未検証) | VQ 離散構造トークン 1 個を語彙に追加; LLaMA3-3B で Cora 91.58 vs GPT-4o 68.62 | 「単一トークンで効率的」と主張するがトークン・遅延・メモリ数値なし; 3B 学習 vs 70B+ 非学習で非一致 | https://arxiv.org/html/2602.01771v1 |

#### B-5. 三応用領域

| 研究名・年・会場 | 何をしたか | 残っている差分 | URL |
|---|---|---|---|
| GraphCodeBERT — 2021, ICLR | 事前学習 LM にデータフロー誘導マスク attention (b); 検索 MRR full 0.713 / w/o data flow 0.693 | マスクのみ; バイアス・W_V と未比較; 125M エンコーダ | https://arxiv.org/abs/2009.08366 |
| Language-Agnostic Representation Learning of Source Code (Code Transformer) — 2021, ICLR | AST 最短路・祖先・兄弟・PPR の 4 距離に関係別 key 射影 (a); 4 種すべてで 48.75 F1 vs 単独 46.17–48.47 | バイアス種内の ablation のみ; 低資源言語での利得は記述レベル | https://arxiv.org/abs/2103.11318 |
| RAT-SQL — 2020, ACL | 関係埋め込みを K と V に加算 (a+d), BERT 使用; w/o schema-linking 55.13 → 40.37 | K 側と V 側の分離 ablation なし | https://aclanthology.org/2020.acl-main.677/ |
| StructCoder — 2022/2023, ACM TKDD | AST+DFG を構造項として attention に加算; 小型 T5 の ablation で DFG-enc は Java→C# をわずかに下げる; 入力長 +75–83%, 推論 +28–29% | 効率コストは副次報告; LoRA/凍結デコーダへの適用なし | https://arxiv.org/html/2206.05239v3 |
| Structured Code Representations Enable Data-Efficient Adaptation — 2024, arXiv (会場なし) | CST 直列化 + 継続事前学習で学習曲線: CoNaLa 100 例 ≥ 平文 1.8k 例; 60M→770M で利得は消えない; 「+trees without tokenizer」は大幅悪化 | 構造はトークン列で attention 内部ではない; 継続事前学習そのものの効果とテキストのみ継続事前学習で統制していない | https://arxiv.org/abs/2401.10716v1 |
| GALLa — 2025, ACL | GNN (AST+DFG) を cross-attention で 7 LLM (350M–14B) に整合、推論時は破棄; 平均相対利得 +2%〜+36% で規模で単調減少せず; DFG > AST | LoRA vs full FT の交絡; データ効率なし; 構造は学習信号で実行時入力ではない | https://arxiv.org/html/2409.04183v4 |
| CGBridge — 2025, KDD | 凍結 Qwen2.5-Coder 7B にグラフ Transformer + Q-Former 橋; 翻訳 98.26% vs LoRA 97.17%; GraphText プロンプトは zero-shot より悪化 | 180.8M 学習パラメータ; データ効率なし; attention 内部非改変 | https://arxiv.org/html/2512.07666v1 |
| Code Graph Model (CGM) — 2025, arXiv | Qwen2.5-72B にリポジトリグラフのマスク + アダプタ; SWE-bench Lite 43.00% | **グラフなし ablation・LoRA 設定・グラフ構築は未取得** | https://arxiv.org/abs/2505.16901 |
| MolCA — 2023, EMNLP | Q-Former で GINE → Galactica; 同一 LM で SMILES のみ 34.6 → +graph 38.7 BLEU-2; BBBP は 70.8 → 70.0 | 分類での利得は分散内; LM 規模 × グラフ有無のスイープなし | https://arxiv.org/html/2310.12798 |
| Mol-LLM — 2025, NeurIPS | 「graph bypass」を診断し MolPO (摂動グラフとの選好) を導入; w/o Graph 対照が MoleculeNet 4/5 で上回る | グラフの寄与が小さい理由 (短い分子) は仮説; 規模・データ効率なし | https://arxiv.org/html/2502.02810v2 |
| 3D-MoLM — 2024, ICLR | 2D-MoLM 対照腕あり: 2D は SMILES のみとほぼ同等 (27.15 vs 27.01 BLEU-2)、3D は +2 点・HOMO MAE 1.24 → 0.26 | 2D 冗長性を示す唯一の統制腕だが分子のみ | https://arxiv.org/html/2401.13923 |
| Sanford et al., Understanding Transformer Reasoning Capabilities via Graph Algorithms — 2024, arXiv | GNN vs ≤60M Transformer を 1K vs 100K サンプルで比較: 低データ・局所タスクで GNN、高データ・大域タスクで Transformer | 事前学習 LLM なし (T5-11B fine-tune は 1 点); 構造注入機構の因子化なし | https://arxiv.org/html/2405.18512 |
| Let Your Graph Do the Talking (GraphToken) — 2024, arXiv (会場なし) | 凍結 PaLM 2 S にソフトトークン; GraphQA 最大 +73pp; エンコーダ種 (GCN/GIN/MPNN/HGT/MHA) と特徴 (IDX vs LPE) を ablation | トークン・推論コスト未測定; パラメータ総数の記述が不整合; GraphArena では単独 GIN に劣る | https://arxiv.org/abs/2402.05862 |
| GraphArena — 2025, ICLR | 実グラフ 10 タスク、GNN 腕あり: 大規模 NP タスクで GNN > Claude-3.5; GraphToken < 単独 GIN | 構造注入機構の比較ではない; 大規模 NP の詳細数値は未読部分 | https://arxiv.org/abs/2407.00379 |
| GraphInfer-Bench — 2026, arXiv | 実 TAG の open-ended 推論 5 タスク; GraphSAGE NMI ≈0.46 vs 最良 LLM ≈0.31 | 効率数値なし; 機構因子なし | https://arxiv.org/html/2606.11562 |
| Can LLMs Effectively Leverage Graph Structural Information through Prompts (Huang et al.) — TMLR | 直列化変更で精度 ≤0.2 点、再配線で LLM −0.25% vs MPNN −5.55%: LLM はプロンプトを「段落として」読む; 利得は同類性と相関 | プロンプトレベルのみ; 学習済み融合モデル内で情報 vs 帰納バイアスを分離していない | https://arxiv.org/abs/2309.16595 |

#### B-6. 効率を従属変数にした統制研究・ベンチマーク・否定結果

| 研究名・年・会場 | 何をしたか | 残っている差分 | URL |
|---|---|---|---|
| What Structural Inductive Bias Helps Transformers Reason Over KGs? (Tabula RASA) — 2026, arXiv (ICML 2026 掲載は未相互確認) | 67.1M に一致させ Full / Mask-only / Bias-only / vanilla を比較; マスク単独で大半の利得、関係パラメータは +3.3–8.3pp; 遅延 6× | 明示的に非因子計画 (成分 2–4 を個別 ablation せず「今後の課題」); LLM なし、テキスト豊富ノードなし; 1k/5k スイープは付録のみ | https://arxiv.org/abs/2602.02834 |
| GLBench — 2024, NeurIPS D&B | 12 GraphLLM 法を単一プロトコルで; 「no clear scaling law」; A800 上の時間・メモリ | 手法は公開時のまま (パラメータ非一致); トークン数なし; 注入機構の因子なし | https://arxiv.org/abs/2407.07457 |
| When Do LLMs Help With Node Classification? (LLMNodeBed) — 2025, ICML | 14 データ・>2,700 モデル; LLM 法の優位は半教師あり・異類性に集中; Qwen-7B/14B 以降は限界利得、32B は 5× 遅い | ラベル割合は 2–3 レジームで学習曲線なし; 小 LLM+構造 vs 大 LLM 単体の計算一致比較なし | https://arxiv.org/abs/2502.00829 |
| Exploring the Potential of LLMs in Learning on Graphs — 2023/2024, SIGKDD Explorations | エンコーダ種・ラベル率・2-hop 近傍要約を比較; Deberta→GCN 59% (低ラベル) vs 86%; 近傍要約は Citeseer/Pubmed で悪化; GLEM 68,071 s vs Sentence-BERT 239 s | パラメータ非一致; API コストで 200 ノード評価 | https://arxiv.org/abs/2307.03393 |
| When Structure Doesn't Help — 2025, arXiv (会場なし) | テンプレート・アダプタ・規模 (7B→13B)・テキスト量を変化; 構造テンプレートは 13B でも不利; 分子では GIN が BBBP のみ優位 | attention 構成要素の ablation なし; パラメータ数未報告; 査読済み再現なし | https://arxiv.org/abs/2511.16767 |
| On the Token Collapse Phenomenon in Textual Graph Soft Prompting — 2026, arXiv | G-Retriever 系のトークンが入力間で cos 0.985–0.998; 学習済み定数で同等; 予算一致で改良エンコーダを提案 | 入力レベルのみ; attention 注入腕なし | https://arxiv.org/html/2610.31572 |
| A Graph Talks, But Who's Listening? (CLEGR) — 2025, arXiv | ソフトプロンプト LLM が full-GNN GLM と同等; 「構造十分」な Cora のみ GraphToken 86.72 vs 28.69 | 数値は図のみ; 機構因子なし | https://arxiv.org/abs/2508.20583 |
| Less is More: Learning Graph Tasks with Just LLMs — 2025, arXiv | Graph Tokens (400 固定) vs +text vs P-Tuning vs LoRA-on-text; OOD 140–160 ノードで LoRA .992 vs Graph Tokens .254; 30 例ではエンコーダ学習不可 | 合成タスク; Pareto はトークン数の「優位」を認めるが精度固定の予算曲線なし | https://arxiv.org/html/2508.10115v1 |
| Agentic Graph Token Reasoning — 2026, arXiv | 精度 vs トークン消費の Pareto を 3B/7B で描く (OGBN-Products); zero-shot 転移 57.4% vs 直列化 28.1% | 単一エージェント法; 注入機構横断ではない; Pareto 詳細は **snippet のみ** | https://arxiv.org/abs/2608.00542 |
| GL-Fusion — 2024, arXiv | cross-attention/ゲート/集約器/予測器/LoRA rank の leave-one-out; 複雑度 O((nL_n+L_t)²) → O(nL_n·L_t) | 実測時間・トークンなし; マスク・PE は未分離 | https://arxiv.org/abs/2412.06849 |
| Graph Serializations for Reasoning LLMs — 2026, arXiv | 15 形式 × 9 構成; GraphQA-Hardened で形式差 4.2–28.4 点; 選択器 +9.52 点 at 3.71× コスト | 学習済み graph token 腕は「未評価」と明記 | https://arxiv.org/html/2610.22734 |

### 1-C. まだ無いもの (検索・引用グラフ走査で見つからなかった項目)

| 未着手項目 | 最も近い既存研究と、なぜ足りないか | 根拠 |
|---|---|---|
| {スコアバイアス} × {support/マスク} × {正規化} × {値変換} × {PE/SE} の因子計画 (≥4 因子) | RASA は 3 セル (Full/Mask/Bias) で非因子; GDT は式を持つが PE 軸のみ; GTLM はバイアス種のみ | [RASA](https://arxiv.org/html/2602.02834); [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf) |
| LLM–グラフ融合内での softmax vs カーネル/線形/Gumbel 正規化の比較 | 純 GT (GraphGPS, NodeFormer, SGFormer, GraphTARIF) のみ | [GraphGPS](https://arxiv.org/abs/2205.12454); [GraphTARIF](https://arxiv.org/abs/2510.10631) |
| 関係特異 W_V を単独因子として孤立 (Shaw 型 K-only vs V-only のグラフ版) | HGT は異種グラフ・凍結 XLNet; GRIT/RT は V 項未 ablation; RASA の value gating はバンドル | [HGT](https://ar5iv.labs.arxiv.org/html/2003.01332); [GRIT](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf) |
| 事前学習 LM のトークン列とグラフを、構造を φ に入れた線形/カーネル化 attention で接続 | 構造-in-φ は視覚・点群・TU のみ; LM+graph 系は softmax バイアス/マスクのみ (13 クエリ・4 論文の被引用走査で正例ゼロ) | [Topological Masking](https://arxiv.org/html/2410.03462v2); [GIMLET](https://arxiv.org/html/2306.13089v3) |
| グラフ SSM と LLM の融合 (TAG/KG 補完/コード AST) および Mamba 系 LLM による graph 読解 | HSA-Net (分子のみ)、AdaptQA (後処理・未検証)、NeuroMambaLLM (時間方向) | [HSA-Net](https://arxiv.org/html/2508.08334v1) |
| 「構造が必要 / 相補的 / テキストから導出可能」で意図的に層別したデータセット設計 | 同類性 vs 異類性、テキスト豊富 vs 題名のみの分割として暗黙にのみ存在 | [LLMNodeBed](https://arxiv.org/html/2502.00829v2); [Structure Doesn't Help](https://arxiv.org/html/2511.16767v2) |
| 同一バックボーンで融合モデル vs テキストのみの学習曲線 (1%–100%) | 2–3 レジーム (半教師/教師あり、低/高ラベル、30-shot) のみ; コードの Agarwal et al. は継続事前学習を未統制 | [Agarwal et al.](https://arxiv.org/html/2401.10716v1); [Less is More](https://arxiv.org/html/2508.10115v1) |
| 小 LLM+構造 vs 大 LLM 単体の計算/パラメータ一致比較 | GTLM 1B vs InstructMol 7B、SOG_k 3B vs GPT-4o、GaLoRA 124M vs GraphAdapter 13B はすべて別論文・別学習設定 | [GTLM](https://arxiv.org/html/2605.10247); [GaLoRA](https://arxiv.org/html/2603.10298) |
| 注入機構横断の「精度固定でのトークン予算」曲線 | 計数したのは Less is More (固定 400)、GTLM (needle 65k)、AGT (単一法) のみ | [GTLM](https://arxiv.org/html/2605.10247); [AGT](https://arxiv.org/abs/2608.00542) |
| 入力ソフトトークン vs 中間層側枝 vs 層ごとの attention バイアスの予算一致スイープ | GaRA 層スイープ、GreaseLM M スイープ、GaLA 前半層など単軸のみ | [GaRA](https://arxiv.org/html/2606.22429v1) |
| 効率そのものを attention 因子の従属変数にした研究 | 皆無 | — |
| 凍結/LoRA デコーダ code LLM の自己 attention に GraphCodeBERT 型 DFG マスクやスコアバイアスを直接入れる研究 | CGM (リポジトリグラフ・72B・ablation 未取得)、PRISM (cross-attention・HLS) | [CGM](https://arxiv.org/abs/2505.16901); [PRISM](https://arxiv.org/html/2609.38601) |
| 分子で LM 規模 × グラフ有無、学習量 × グラフ有無のスイープ | MolCA/LLaMo/Mol-LLM は単一データレジーム; InstructMol 7B→13B はグラフのみ | [Mol-LLM](https://arxiv.org/html/2502.02810v2) |
| sparsemax/entmax のグラフ Transformer 適用 | NLP のみ (Adaptively Sparse Transformers) | [arXiv 1909.00015](https://arxiv.org/abs/1909.00015v2) |
| R-GCN / GAT / Transformer を単一パラメータ化設計空間に置いた統制因子比較 | GDT が式レベルで包含を主張するのみ; RGAT→R-GCN の −1〜2% は二次情報で未検証 | [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf) |

---

## 2. attention 構成要素の設計空間: スコアバイアスが最も効き、W_V と正規化は孤立検証がない

純グラフ Transformer 系の統制 ablation は、**(a) ペアごとのスコアバイアスが最も一貫して利得を担う**ことを示している。Graphormer の積み上げ (0.2276 → 0.1304)、GraphiT の隣接カーネル (0.696 → 0.243)、GRIT attention と Graphormer attention の差 (0.059 vs 0.117)、Relational Transformer の set-Transformer 対照 (81.3% vs 42.3%) がその証拠である ([Graphormer](https://arxiv.org/abs/2106.05234); [GraphiT](https://ar5iv.labs.arxiv.org/html/2106.05667); [GRIT](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf); [RT](https://arxiv.org/html/2210.05062v3))。**(b) support/マスク単独が決定的だったのは Dwivedi–Bresson の 2020 年設定 (sparse ≫ full、ZINC 0.226 vs 0.598) のみ**で、Exphormer の ablation では局所・expander・仮想ノードの各パターンが互いにほぼ可換になる ([Dwivedi & Bresson](https://arxiv.org/abs/2012.09699); [Exphormer](https://arxiv.org/abs/2303.06147))。GDT は softmax(d^{-1/2}QKᵀ + B)V という一つの式が局所 GT (エッジ上で B=0、それ以外 −∞)・因果マスク・Shaw/ALiBi 型相対 PE を包含すると明記しており、これは計画研究が因子計画を載せる自然な足場になる。ただし GDT の実験自体は PE 種 × 規模しか変えていない ([GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf))。

**(d) 値変換の孤立証拠は極端に少ない。** Shaw et al. (NLP) は a^V が a^K の上に何も足さないことを示し、HGT は型/関係特異パラメータの共有化で NDCG .615 → .578 を報告するが、GRIT と RT は value 側にペア特徴を入れながら単独 ablation をしておらず、EGT の value ゲートは「significant improvement」と記述されるのみで行別数値が抽出できなかった ([Shaw](https://ar5iv.labs.arxiv.org/html/1803.02155); [HGT](https://ar5iv.labs.arxiv.org/html/2003.01332); [EGT](https://arxiv.org/html/2108.03348v3))。これらを合わせると、関係特異パラメータは関係型が異なる意味を持つ (異種グラフ) ときに効き、関係が幾何しか符号化しないときはスコア側だけで足りる、という仮説が立つが、グラフ Transformer でも LM 融合でも検証されていない。

**(c) 正規化**については、softmax が精度の基準であり、カーネル化/線形化はメモリのために採用される。GraphGPS の Performer/BigBird は softmax に劣り、GraphTARIF は線形 attention が「overly uniform attention distributions and low-rank projection structures」で表現力を失うと診断し、value 側 GAT 分岐と log-power 焦点化で補う ([GraphGPS](https://arxiv.org/abs/2205.12454); [GraphTARIF](https://arxiv.org/abs/2510.10631))。非 softmax 正規化が精度を上げた唯一の例は NodeFormer の Gumbel ノイズ (OGB-Proteins 77.45 vs 75.50) で、これは効率ではなく構造学習として位置づけられる ([NodeFormer](https://arxiv.org/html/2306.08385))。DIFFormer の cosine-linear vs sigmoid-quadratic は混在 (Cora 85.9 vs 84.1、Citeseer 73.5 vs 75.7) ([DIFFormer](https://arxiv.org/html/2301.09474v4))。sparsemax/entmax をグラフ Transformer に適用した論文は見つからなかった。

**(e) PE/SE**では、Black et al. が無特徴グラフで APE ≡ RPE を証明しつつノード特徴ありでは RPE ≥ APE と注意し、GDT は RRWP ≻ RWSE、SPE ⪰ LPE を示しながら「PE efficiency can vary greatly while predictive performance differences are less pronounced」と効率側の差を強調する ([Black et al.](https://arxiv.org/abs/2402.14202); [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf))。OpenGT は PubMed で RW 統計 10 分、スペクトル 1 時間、全対 SPD 2 時間という前処理コストを報告しており、PE 選択は計画研究の「効率」軸に直接つながる ([OpenGT](https://arxiv.org/html/2506.04765))。

| 成分 | 統制 ablation の有無 (純 GT) | LM 融合内での ablation | 主要な未解決点 |
|---|---|---|---|
| (a) スコアバイアス | あり (Graphormer, GraphiT, GRIT, GDT 理論) | バイアス種のみ (GTLM, Code Transformer); 単一 (GaLA) | バイアス vs マスク vs W_V の同一データ比較 |
| (b) support/マスク | あり (Dwivedi–Bresson, Exphormer, Spexphormer) | 窓幅 (ICML 2025); マスク vs バイアス 3 セル (RASA, LLM なし) | バイアスとの交互作用 (RASA: bias-only は vanilla 以下) |
| (c) 正規化 | 部分 (GraphGPS, NodeFormer, DIFFormer; SGFormer は精度比較なし) | なし | LM 内で softmax vs カーネル/線形; sparsemax/entmax は皆無 |
| (d) 値変換 | HGT のみクリーン; GRIT/RT/EGT は未分離 | RAT-SQL は K+V 同時; RASA value gating はバンドル | Shaw 型 K-only vs V-only のグラフ/LM 版 |
| (e) PE/SE | あり (GraphGPS, GDT, OpenGT, MoSE, SPSE) | GraphToken IDX vs LPE (合成); Structure Doesn't Help は交絡 | PE 計算コストを効率従属変数に含めた比較 |

重要な注意点として、同じ「構造バイアス」でも何を入れるかで結果が逆転する。GTLM では SPD バイアス除去が Shortest Path をむしろ改善 (90.1 → 92.0) し Magnetic Laplacian が支配的で、Attending to GTs では SPD は異類性ノード分類で無効だった ([GTLM](https://arxiv.org/html/2605.10247); [Attending to GTs](https://arxiv.org/abs/2302.04181))。因子水準の選び方を誤ると主効果が洗い流される。

---

## 3. 線形 attention のカーネル特徴写像 φ: 構造を φ に入れる系譜は一本で、LM との接続はゼロ

φ(Q)φ(K)ᵀ の内部にグラフ構造を入れた論文は、Choromanski/Reid グループの一本の系譜しかない。GKAT (2021) は softmax カーネル特徴 φ_K と RWGNK グラフ特徴 φ_T のテンソル積 (φ_K(q_k) ⊗ φ_T(k)) で A_i(k,l) = K(q_k,k_l)·T(k,l) を線形化し、時間 O(N·m·r·d)、空間 O(Nmr) を主張した ([GKAT v1](https://arxiv.org/pdf/2107.07999v1))。同じ arXiv 番号は改題を経て ICML 2022 の「マスク付き低ランク attention」の一般理論になり (Lemma 3.1: マスク M の行列ベクトル積が T_M(L) なら O((T_M(L)+L)md))、ImageNet ViT で +2.5–3.4 点を示した ([ICML 2022](https://ar5iv.labs.arxiv.org/html/2107.07999))。2023–2024 年の GRF 理論 (Taming graph kernels, q-GRFs, General GRFs, Repelling Random Walks) を経て、ICLR 2025 の topological masking が φ(q) ⊗ GRF(node) の疎クロネッカー特徴で O(N) を達成し、ImageNet で unmasked linear 0.693 → GRF-in-φ 0.730 (softmax・exact mask 0.741) を報告した。重要度重みなしの ad-hoc GRF は 0.689 で無マスクより悪く、O(N²) の密な学習可能グラフ特徴でも 0.696 にとどまる ([Topological Masking](https://arxiv.org/html/2410.03462v2))。KDLGT (IJCAI 2023) は低ランク RPE を [Q, B_q][K, B_k]ᵀ の形でカーネル引数に連結する弱い形の φ 内注入で、LSAT vs SAT が ZINC 0.130 vs 0.082、Cora 83.37 vs 83.06 と、ノードタスクでは線形化の損失が小さい ([KDLGT](https://www.ijcai.org/proceedings/2023/0263.pdf))。

これに対し、「線形グラフ Transformer」として知られる NodeFormer / DIFFormer / SGFormer / Polynormer / GraphTARIF はすべて φ を構造フリーに保ち、構造は加法エッジ項・正則化・GCN 分岐・局所 GAT モジュールとして φ の外に入る ([NodeFormer](https://arxiv.org/html/2306.08385v1); [SGFormer](https://arxiv.org/html/2306.10759v5); [Polynormer](https://arxiv.org/html/2403.01232v3); [GraphTARIF](https://arxiv.org/html/2510.10631v2))。2025–2026 年の新しい経路は RoPE-for-graphs (WIRE, AW-RoPE) で、構造を q,k の回転として φ の前段に入れ線形 attention と互換にするが、AW-RoPE 自身が「φ は回転と可換でないため厳密な相対位相恒等式は生 dot-product のみ」と認めている ([WIRE](https://arxiv.org/abs/2509.22259); [AW-RoPE](https://arxiv.org/html/2609.32178v1))。

**事前学習 LM のトークン列とグラフを構造認識線形/カーネル化 attention で接続した論文は、13 本の検索クエリと GKAT・Taming・General GRFs・topological masking の被引用リスト (Semantic Scholar、計 ~94 件) を題名単位で走査しても見つからなかった。** LM+graph 系で見つかったのは GIMLET (T5 に SPD+経路エッジバイアス、softmax)、UTGDiff、DuConTE、Graph-KV、UniGTE、GL-Fusion で、いずれも softmax バイアス/マスクである ([GIMLET](https://arxiv.org/html/2306.13089v3); [UTGDiff](https://arxiv.org/abs/2408.09896); [DuConTE](https://arxiv.org/abs/2604.17411))。MoLFormer は線形 attention + RoPE だがグラフ構造を使わない ([MoLFormer](https://www.nature.com/articles/s42256-022-00580-7.pdf))。このギャップは検索の人工物ではなく実在すると判断できる: 技術を持つグループは視覚・点群・ロボティクス・カーネル機械のみに発表し、LM+graph コミュニティは softmax 設計に標準化している。

一致比較の観点では、{構造-in-φ, 加法バイアス softmax, ハードマスク} をパラメータ一致で三者比較した論文はない。利用できる部分証拠は、ICLR 2025 の ViT 表 (格子グラフで構造-in-φ が線形–softmax 間ギャップの約 75–80% を回復)、GraphGPS/Exphormer の同一レシピ表 (CIFAR10: softmax 72.305 / Performer 70.670 / BigBird 70.480 / Exphormer 疎マスク 74.69)、TokenGT の同一 PE での softmax vs Performer (0.0910 vs 0.0935) である ([Topological Masking](https://arxiv.org/html/2410.03462v2); [Exphormer](https://arxiv.org/html/2303.06147v2); [TokenGT](https://arxiv.org/html/2207.02505v2))。これらは、小中規模グラフでは疎マスク経路の方が強いベースラインであり、φ 注入が報われるのは N が大きく (>30k トークン) 二次マスクが使えないときに限られることを示唆する。コスト面では、テンソル積は片方の因子が疎 (GRF 支持 ≈ n/p_halt) か小 (アンカー点 s) であるときにのみ現実的で、密な d_s × d_g 特徴はトークンあたり d_s·N を要する。疎なテキストトークングラフ (AST や依存構造、次数 ≤3) では GRF の分散はエッジあたり小さいが遠隔対の被覆が崩れ、GRFs++ 型の縫合が必要になる。言語グラフでこれを測った論文はない ([GRFs++](https://arxiv.org/html/2510.07716v1))。

---

## 4. グラフ状態空間モデル: 順序型と作用素型に分裂し、LLM 融合は分子 1 件のみ

2024–2026 年に検証可能なグラフ SSM は約 20 本あり、二つの族に分かれる。**(A) 順序型**: Graph-Mamba (次数順ソート + ランダム置換、GraphGPS の attention を置換) と GMN (ランダムウォーク部分グラフトークン化 + 双方向 Mamba) に代表され、構造は入力順序と局所 GNN/PE 経由でのみ入り、選択ゲート B, C, Δ は素の Mamba のままである。**(B) 作用素型**: GSSC (Laplacian 固有ベクトルで因子化した距離カーネル)、GraphSSM (Laplacian 正則化付き再帰)、Chimera (レゾルベント (I−A)⁻¹)、MP-SSM (メッセージパッシング深さ方向の再帰)、GRAMA (graph-ARMA) で、位相が遷移作用素そのものに入り置換同変が構成的に保証される ([Graph-Mamba](https://arxiv.org/html/2402.00789v1); [GMN](https://arxiv.org/html/2402.08678v2); [GSSC](https://arxiv.org/html/2406.05815v2); [Chimera](https://arxiv.org/html/2510.12111v1); [MP-SSM](https://arxiv.org/html/2505.18728v1))。

精度では、LRGB で順序型が GraphGPS/Exphormer を 2–7 点上回り Performer/BigBird 版 GraphGPS には大差をつける (Peptides-func: GPS 0.6575 / Exphormer 0.6527 / Graph-Mamba 0.6739 / GMN 0.7071 / Chimera 0.7021; GPS 内で Graph-Mamba vs Performer vs BigBird は PascalVOC 0.4191 vs 0.3724 vs 0.2762) が、GRIT には CLUSTER (76.8 vs 80.0) と ZINC (GSSC 0.064 vs GRIT 0.059) で負け、GPS に Mamba を単純に差し込んだ「GPS+Mamba」は GMN に全データで負け Amazon-ratings では素の GPS より低い (45.13 vs 53.10) ([Chimera](https://arxiv.org/html/2510.12111v1); [GSSC](https://arxiv.org/html/2406.05815v2); [MP-SSM](https://arxiv.org/html/2505.18728v1))。順序の寄与は大きく、Graph-Mamba では置換/優先順位の有無で PascalVOC F1 が 0.3105 → 0.4314 と 12 点動き、GMN では後方スキャン除去が 4–5 点を失わせ、Chimera の ImageNet ablation は 2-D DAG 77.8% vs 双方向 1-D 76.5% vs 前方 1-D 73.8% と平坦化のコストを定量化する。GMN の著者は表現力が「サンプリングと符号化であり Mamba 自体ではない」と述べており、順序型における SSM は本質的に効率装置である ([GMN](https://arxiv.org/html/2402.08678v2))。効率は Graph-Mamba が GPS-Transformer 比で GPU メモリ −74%・FLOPs −66%、GMN が ogbn-arxiv で 3.85 GB vs Exphormer 36.18 GB を報告するが、同変型は固有分解 (GSSC) や逆行列 (Chimera: 厳密は三次、近似は最悪二次) を払う。**どのグラフ SSM 論文もデータ効率や規模スケーリングを報告せず、NodeFormer/SGFormer/Polynormer をベースラインに含めていない** ([Graph-Mamba](https://arxiv.org/html/2402.00789v1); [GMN](https://arxiv.org/html/2402.08678v2))。

LLM との融合は、検証できた範囲で HSA-Net ただ一件が「分子 GNN と LLaMA-2-7B-chat / Galactica-1.3B の間に Graph-Mamba projector を置く」構成であり、その ablation では Mamba 単独 projector (BLEU 37.5) が cross-attention 単独 (37.9) を上回らず、価値は深層 GNN 特徴を補完する点にある。時間・メモリ・FLOPs は未報告で、ソート規則や隣接/距離行列入力の ablation もない ([HSA-Net](https://arxiv.org/html/2508.08334v1))。AdaptQA は LLM を後処理に使う KGQA で全文未取得、NeuroMambaLLM は fMRI 時系列方向の Mamba である ([AdaptQA](https://ro.ecu.edu.au/ecuworks2022-2026/6144); [NeuroMambaLLM](https://arxiv.org/abs/2602.13770))。TAG・KG 補完・コード/AST でグラフ SSM と LM を融合した研究、Mamba 系 LLM (Mamba, Jamba, Falcon-Mamba) で graph を読ませた研究は見つからなかった。計画研究が「隣接する線」としてグラフ SSM を因子に含めるなら、順序型を使う限り置換同変性の批判 (GSSC, MP-SSM, GSM++) を継承するため、作用素型 (GSSC/Chimera/MP-SSM) を選ぶ必要がある。二つの専門サーベイ (Atitallah et al. 2024; Xu, CMES 2026) はいずれも順序/置換同変性問題と LLM 融合に触れていない ([Atitallah et al.](https://arxiv.org/html/2412.18322v1); [Xu 2026](https://www.techscience.com/CMES/v148n2/68573/html))。

---

## 5. LLM 側注入法 2024–2026: 入力ソフトトークンから attention 内部へ移行

2023–2024 年の「graph-to-LLM」法 (GraphToken, LLaGA, GraphGPT, G-Retriever, TEA-GLM, GraphPrompter, GNP) はほぼすべて凍結 ~7B デコーダの入力埋め込み層にソフトトークンを前置する INPUT-SOFT 型で、学習するのは GNN と射影のみである。例外として GraphLLM (全層 K/V に 5 トークン prefix)、ENGINE (各層 G-Ladder 側枝)、GOFA (最終数層に GNN 挿入)、GraphAdapter (最終隠れ状態で融合)、Plenz & Frank の GLM (T5 内で相対 PE + マスク) がある。この世代の「効率」はほぼ常に学習パラメータ数 (0.1M–130M vs 7B) を意味し、計算やトークンを定量化したのは ENGINE (キャッシュで学習 12×、早期終了で推論 5×)、GraphLLM (コンテキスト −96%、推論 3.42×)、G-Retriever (検索でトークン −99%) だけである ([ENGINE](https://arxiv.org/html/2401.15569v2); [GraphLLM](https://arxiv.org/html/2310.05845v1); [G-Retriever](https://arxiv.org/html/2402.07630v3))。G-Retriever の表は 2024 年で最もクリーンな三者比較で、WebQSP では直列化テキスト LoRA (66.03) がソフトトークン単独 GraphToken (57.05) に勝ち、ソフトトークンは直列化テキストと併用して初めて 70.49 に達する。これは 2026 年の Token Collapse 批判を予告している ([Token Collapse](https://arxiv.org/html/2610.31572))。

2025–2026 年には注入点が **attention 内部**へ明確に移った。設計は次の 6 類型に整理できる。

| 類型 | 代表 (年) | 機構 | バックボーン | 効率主張 |
|---|---|---|---|---|
| 学習済み加法バイアス + 位置リセット | GTLM (2026), Learnable Graph PE (2026, 未検証) | 層×ヘッド×SPD のルックアップ、RRWP MLP、Magnetic Laplacian; ノードごと RoPE リセットで prefix ノード置換同変 | Llama-3.2-1B/8B + LoRA | 構造パラメータ 0.015%; FlexAttention で ≈2× ステップコスト; コンテキスト O(N) |
| 学習なし推論時バイアス | GaLA (KDD 2026) | 前半層のみ λ_h/d_G(i,j); エントロピー較正 | 凍結 Qwen2.5-3B/7B, Ministral-3B | 学習ゼロ; Cora で BFS 225 s vs GaLA 265 s vs CoT >5,000 s |
| マスク + 位置 ID 共有/再調整 | Graph-KV (NeurIPS 2025), NAG (2026), UniGTE (NeurIPS 2025) | 対象チャンクはグラフ上ソースの KV のみ参照; 閉じタグハブ; graph token 間で RoPE 同一 | Llama-3.1-8B, Qwen3-0.6B, Vicuna-7B | 位置スパン T·L; UniGTE Cora 0.6 s/sample vs GOFA 5.1 s |
| 学習なし attention 再配分 (正規化介入) | SLASH (2026), LoReC (2026) | sink 重み縮小・再正規化; graph token の logits 増幅 + FFN 再注入 | Llama-3.2-3B〜Qwen3-14B; GraphGPT/GraphPrompter 上 | SLASH は FlashAttention 非互換 (36.1 GB vs 13.3 GB); LoReC 82.69 ms/token vs 45.23 |
| グラフ条件付き重み (生成 LoRA) | GaRA (ICML 2026) | GCN → 低ランク行列を layer 21 の V 射影に加算 | Vicuna-7B | 1.46 s/sample 学習; 入力ソフトトークン併用必須 |
| 中間層ゲート側枝 | GaLoRA (2026), ENGINE (2024) | 1-hop → 層 5–7、2-hop → 層 9–11 | GPT-2 124M, RoBERTa | 0.238% パラメータ |

出典: [GTLM](https://arxiv.org/html/2605.10247); [GaLA](https://arxiv.org/html/2606.15633v2); [Graph-KV](https://arxiv.org/html/2506.07334); [NAG](https://arxiv.org/html/2601.22657); [UniGTE](https://arxiv.org/html/2510.16885v1); [SLASH](https://arxiv.org/html/2605.10503v3); [LoReC](https://arxiv.org/html/2604.17897); [GaRA](https://arxiv.org/html/2606.22429v1); [GaLoRA](https://arxiv.org/html/2603.10298)。

これらの 2026 年「ネイティブ」設計は、Plenz & Frank が 2024 年に T5 で使った二つの原始操作 (グラフ距離で索引した相対 attention 項と、ブロック/グラフマスク) を、RoPE デコーダで (a) ノードごとの位置 ID リセット/共有と (b) ノード対バイアスのトークン対へのブロードキャストによって実現し直したものである。事前学習済みのノード内計算を保ったまま厳密還元 (GTLM: 単一ノード入力で logit 差 ≈2e-5) と同変性の証明が可能になる ([GTLM](https://arxiv.org/html/2605.10247); [Plenz & Frank](https://arxiv.org/html/2401.07105v3))。**しかし各論文は他の成分を固定している**: GTLM はマスク (PrefixLM) と PE リセットを固定しバイアス種のみ、NAG はマスクと位置再調整、UniGTE はバイアス・マスク・位置共有を同時導入して個別 ablation なし、GaLA は単一バイアス。注入点の予算一致スイープも存在せず、部分証拠 (GaRA layer 21/32 最良、GreaseLM M=5、SLASH の中間層ヘッド集中、ENGINE 層 5 で >90% 早期終了) は「中間層が最良の単一注入点」で一致するが、入力ソフトトークン vs 中間層側枝 vs 層ごと attention バイアスを同一バックボーン・同一データで比べた研究はない ([GaRA](https://arxiv.org/html/2606.22429v1); [GreaseLM](https://arxiv.org/pdf/2201.08860))。

「小 LLM + 構造 ≈ 大 LLM 単体」の証拠はすべて間接的である。GTLM は分類が 1B で飽和し 8B は生成のみ改善、GaLA の利得は 3B で大きく 7B で縮小、GaLoRA (124M) は GraphAdapter-Llama-2-13B に 3 データ中 1 つで勝つ、NAG は 0.6B を 8B の zero-shot 参照に対置する。いずれも同一ベース LLM を構造なしで拡大した対照ではない ([GaLA](https://arxiv.org/html/2606.15633v2); [GaLoRA](https://arxiv.org/html/2603.10298))。また「パラメータ効率」は「カーネル効率」と交換されている: GTLM の score_mod は FlashAttention の ≈2×、SLASH は完全 attention 行列を要する。計画研究で効率を従属変数にするなら、この壁時計・メモリコストを必ず計上しなければならない。

サーベイの分類軸は依然として「LLM がパイプラインのどこに座るか」(Enhancer/Predictor/Aligner; GNN-as-prefix/LLM-as-prefix; Graph2text/Graph2token; Graph-as-Text/Embedding/Merging) であり、Su et al. (2025)、Yu et al. (TOIS 2025)、PKU の 2026 年 position paper を含め、**デコーダ内 attention バイアス/位置符号化注入のカテゴリを持つサーベイはない** ([Su et al.](https://arxiv.org/html/2510.21131); [Yu et al.](https://arxiv.org/html/2501.01124); [Position paper](https://arxiv.org/html/2605.02452v1))。計画研究の機構レベル分解は既存の全サーベイ分類と直交する。

---

## 6. 三応用領域: コード・分子・グラフ推論/TAG で「構造が効く条件」は一致する

### 6-1. コード

正典系列 (GREAT → GraphCodeBERT → Code Transformer / TPTrans / AST-Trans → TreeBERT / SynCoBERT / UniXcoder / StructCoder → AST-T5) は、構造を attention (マスクまたはスコアバイアス)、入力 (追加位置・直列化木)、事前学習目的のいずれかに入れる ≤250M のエンコーダ/エンコーダ–デコーダで、CodeXGLUE 系での利得は典型的に 0.5–3 点である。GraphCodeBERT は検索 MRR 0.713 vs 0.693、BigCloneBench F1 0.950 vs 0.941; UniXcoder の w/o AST は POJ-104 で −1.8 だが要約 BLEU (19.33 vs 19.30) と生成 EM (22.60 vs 22.60) には影響なし; StructCoder は DFG-enc が Java→C# をわずかに下げ、AST+DFG 入力で入力長 +75–83%、推論 +28–29% ([GraphCodeBERT](https://ar5iv.labs.arxiv.org/html/2009.08366); [UniXcoder](https://ar5iv.labs.arxiv.org/html/2203.03850); [StructCoder](https://arxiv.org/html/2206.05239v3))。最も一貫した正の信号は構造整合の事前学習目的 (AST-T5, AST-FIM, GrammarCoder, UniXcoder) からくる。

LLM 規模では、≥30B でのクリーンな構造有無比較は存在しない。GALLa は 350M–14B の 7 LLM で +2%〜+36% の相対利得を報告し規模で単調減少しないが、LoRA vs full FT が交絡し、DFG > AST であった。CGBridge は凍結 Qwen2.5-Coder-7B で翻訳 98.26% vs LoRA 97.17% だが 180.8M パラメータを学習し、直列化 GraphText プロンプトは zero-shot より悪い。一方 Su & McMillan は静的解析事前学習が CodeLlama-13B の下流を信頼できる形では改善しないとし、probing 研究 (Anand et al.; Ma et al. TOSEM) は大型/fine-tuned モデルが構文・データフローを小型エンコーダより悪く符号化すると報告する ([GALLa](https://arxiv.org/html/2409.04183v4); [CGBridge](https://arxiv.org/html/2512.07666v1); [Su & McMillan](https://arxiv.org/html/2505.12118v1); [Anand et al.](https://arxiv.org/html/2406.11930v1))。**データ効率曲線を持つのは Agarwal et al. (2024) と SAT (2024) だけ**で、前者は CoNaLa で構造 100 例 ≥ 平文 1.8k 例、Concode で 500 例 ≈ 10k 例、60M→770M で利得が消えないと報告するが、ablation で MSP 継続事前学習目的が大半を担い、テキストのみの継続事前学習対照は走らせていない。SAT の 20–80% データでの利得は 0–1.4 点で非単調である ([Agarwal et al.](https://arxiv.org/html/2401.10716v1); [SAT](https://arxiv.org/html/2404.07471))。**凍結/LoRA デコーダ code LLM の自己 attention に GraphCodeBERT 型 DFG マスクやスコアバイアスを直接入れた論文はなく**、最も近いのは CGM (リポジトリグラフマスク、72B、ablation 未取得) と PRISM (layer 0 のゲート付き cross-attention、HLS、3× 低速) である ([CGM](https://arxiv.org/abs/2505.16901); [PRISM](https://arxiv.org/html/2609.38601))。

データセット選択に関する重要な注意: Devign/BigVul/D2A のラベル精度は 0.80/0.543/0.286 (Croft et al. ICSE 2023)、PrimeVul では 7B モデルが BigVul 68.26 F1 に対し 3.09 F1 であり、脆弱性データは構造効果の分離に使えない。ほぼ厳密なラベルを持つ POJ-104 と CodeNet が適切で、そこでは事前学習テキストモデルが GNN に追いつくか逆転 (C-BERT Java250 97.40 vs GIN-V 92.77) しつつ、長い C++ (C++1000/1400) では GIN-V が勝つ (96.64 vs 93.79) ([Croft et al.](https://arxiv.org/abs/2301.05456v1); [PrimeVul](https://arxiv.org/html/2403.18624v2); [CodeNet](https://ar5iv.labs.arxiv.org/html/2105.12655))。

### 6-2. 分子

2023–2025 年の支配的設計は凍結 5 層 GIN (MoleculeSTM/GraphCL、hidden 300) → Q-Former / 線形 / 多層 cross-attention 射影 → Galactica/Llama-2/Mistral/Llama-3.1 で、ほぼ全てが SMILES/SELFIES も LM に与えるため、グラフは文字列の「追加」である。同一エンコーダの再利用により、報告利得の差は射影器と学習レシピからくる。証拠はタスク依存に分裂する: 生成系 (キャプション、IUPAC、反応、QM9 回帰) では同一 LM でのグラフ追加が一貫して効き (MolCA 34.6 → 38.7; LLaMo 26.1 → 37.8 BLEU; EDT-Former MoleculeQA 58.77 → 65.11)、MoleculeNet 二値分類では小さい・不均一・負である (MolCA BBBP 70.8 → 70.0; Mol-LLM の w/o Graph が BACE/BBBP/ClinTox/HIV で上回る) ([MolCA](https://arxiv.org/html/2310.12798); [LLaMo](https://arxiv.org/html/2411.00871v1); [Mol-LLM](https://arxiv.org/html/2502.02810v2); [EDT-Former](https://arxiv.org/html/2602.02742))。Mol-LLM は SFT モデルが真のグラフと MACCS 摂動グラフを「hardly distinguishes」する「graph bypass」を文書化し、Ariguib–Niepert–Manolache は「SMILES and 2D molecular graphs encode largely redundant information」と明言する ([Ariguib et al.](https://arxiv.org/html/2509.22468))。3D-MoLM は 2D-MoLM 対照腕を持つ唯一の論文で、2D は SMILES のみとほぼ同等 (27.15 vs 27.01)、3D が +2 点と HOMO MAE 1.24 → 0.26 をもたらす ([3D-MoLM](https://arxiv.org/html/2401.13923))。LM 規模 × グラフ有無、学習量 × グラフ有無をスイープした分子論文はなく、複数シード・有意性検定つきで BACE/BBBP/HIV/Tox21 を統制した SMILES のみ vs SMILES+graph も存在しない。

### 6-3. グラフ推論と TAG

NLGraph → GraphQA → GraphInstruct → GraphArena → GraphOmni を通じて失敗様式は安定している: 計数/次数タスクはテキスト直列化で失敗し、精度はノード数とともに単調に落ち (GraphArena で GPT-4o の幻覚は 5 ノード 16% → 30 ノード >80%)、形式だけで GPT-4o の diameter 精度が最大 40 点動く ([GraphArena](https://arxiv.org/html/2407.00379v2); [GraphOmni](https://arxiv.org/html/2504.12764))。テキストを学習済み構造 prefix で置換した GraphLLM と GraphToken だけが計数/次数で天井近くに達するが、≤20–45 ノードの合成グラフに限られる。GNN 腕を含むベンチマーク (GraphArena, Sanford et al., GraphInfer-Bench) は素の GNN が大規模/NP/多ノードタスクで LLM 系に匹敵または上回ることを一貫して示し (GraphInfer-Bench: GraphSAGE NMI ≈0.46 vs 最良 LLM ≈0.31)、Hu et al. は GPT-4o-mini の reason-then-code が既存 GraphQA 系ベンチをほぼ飽和させると報告する ([GraphInfer-Bench](https://arxiv.org/html/2606.11562); [Hu et al.](https://arxiv.org/abs/2509.24260))。トークンコストはエッジ数と形式冗長性で増え、GraphWiz は 4,096 トークンに収めるため 100 ノードで打ち切り、GraphLLM は 45 ノードで Graph2Text が上限を超える一方 5 トークンで ≈96% を維持する ([GraphWiz](https://arxiv.org/html/2402.16029v2); [GraphLLM](https://arxiv.org/html/2310.05845))。

三領域を横断すると、**構造エンコーダの優位はテキストチャネルが弱いところに集中する**: 低データ (Sanford et al. の 1K)、弱い/小さい LM (Huang et al.: LLaMA-2-7B +21.7% vs ChatGPT +2.78%)、疎なノードテキスト (Cora: GraphToken 86.72 vs ソフトプロンプト 28.69)、直列化がコンテキストを飽和する大きさ、OOD サイズ (ただし構造が attention/位置項として入る場合のみで、プールされたソフトトークンは OOD で崩壊: Less is More DFS .254 vs LoRA .992) ([Huang et al.](https://arxiv.org/html/2309.16595); [CLEGR](https://arxiv.org/html/2508.20583); [Less is More](https://arxiv.org/html/2508.10115v1))。これは計画研究の「必要/相補/導出可能」という三型と整合し、文献はこの軸を暗黙にしか持っていない。

---

## 7. 効率を従属変数にした統制研究・ベンチマーク・否定結果

GLBench (NeurIPS 2024) と LLMNodeBed (ICML 2025) が TAG ノード分類の二大単一プロトコルベンチマークである。両者とも壁時計時間とメモリを報告し、LLM 規模でのクリーンなスケーリングを否定する (GLBench「no clear scaling law」; LLMNodeBed は Qwen-7B/14B 以降の利得が限界的で 32B は 5× 遅い)。LLMNodeBed は LLM 法の優位が半教師あり (平均 ENGINE 78.40 vs テキストのみ LLMIT 65.76) と異類性 (Texas で ≈+12%) に集中し、教師ありでは限界的と示す ([GLBench](https://arxiv.org/html/2407.07457v4); [LLMNodeBed](https://arxiv.org/html/2502.00829v2))。いずれもパラメータを条件間で一致させず、トークン数を報告せず、attention 内部の注入機構を変えていない。「データ割合」は 2–3 の離散レジームとしてのみ運用され、学習曲線はない。

パラメータ一致を実現した研究は Tabula RASA (67.1M) と Less is More (P-Tuning に 400 仮想トークン) だけである。RASA の結論は計画研究にとって決定的に重要で、**マスク単独が利得の大半 (MetaQA-3hop +72.5pp) を与え、バイアス単独はマスクなしでは vanilla 同等以下 (CWQ 2.1% vs 2.7%)、関係特異パラメータは +3.3–8.3pp を追加し、遅延は 6×** である。著者自身が成分 2–4 の個別 ablation を今後の課題としている ([RASA](https://arxiv.org/html/2602.02834))。トークン効率は主張されることが測定されることよりはるかに多い: GraphToken, SOG_k, GL-Fusion, HS2C はトークン数を報告せず、計数したのは Less is More (固定 400 トークン vs エッジリスト)、GTLM (needle-in-a-graph ≈65k トークンで 98.0% vs 8.5%)、Agentic Graph Token Reasoning (単一法の Pareto) のみである。精度を固定して各注入機構に必要なトークン予算を測った研究はない。

否定結果の系列は 2023 年から一貫している。Chen et al. (2023) は近傍要約が Citeseer/Pubmed で悪化し「Simply enlarging the model size of LLMs may not help」; Huang et al. は直列化変更で ≤0.2 点、再配線で LLM −0.25% vs MPNN −5.55%; When Structure Doesn't Help は CO 75.29% > ND 69.48%、MLP アダプタ 76.26% > GCN 67.50、School (異類性) で ND 66.43 vs CO 91.13、13B でも差が残る; Token Collapse は学習済み定数が G-Retriever と同等 (ExplaGraphs 85.56 vs 85.56) で直列化グラフ除去は 19–51 点の損失; Less is More は graph token が OOD で脆く 30 例では学習不能; ICML 2025 の attention 解析は LLM の node→node attention が構造に従わない ([Chen et al.](https://arxiv.org/html/2307.03393v4); [Huang et al.](https://arxiv.org/html/2309.16595v4); [Structure Doesn't Help](https://arxiv.org/html/2511.16767v2); [Token Collapse](https://arxiv.org/html/2610.31572); [Attention Mechanisms Perspective](https://arxiv.org/abs/2505.02130))。反証となるのは構造中心タスクである: GraphInfer-Bench で素の GNN が全タスクで最良 LLM 行に匹敵/上回る、RASA で一致パラメータ下のマスクが +45–73pp、GTLM の needle-in-a-graph 98.0% vs 8.5%、GTLM の Table 19 で Cora/PubMed のテキストのみ版は <2.2pp の損失だが WebQSP F1 48.5 → 73.5、GraphQA 平均 33.2 → 85.7 ([GTLM](https://arxiv.org/html/2605.10247))。

要約すると、文献における「構造が必要 / 相補的 / 導出可能」の分割は暗黙である: 導出可能 (同類性引用 TAG: Cora/PubMed/arXiv、豊富なテキスト) → 構造は冗長; 相補的 (異類性/疎テキスト/低ラベル) → 新しいテキストをもたらすとき効く; 必要 (コミュニティ検出、多ホップ KGQA、needle-in-graph、BBBP) → 構造認識 attention か GNN が勝つ。この軸で意図的にデータセットを構築した論文は一つもない。

---

## 8. 判定: 既出 / 部分的 / 未着手、および修士研究設計への含意

### 8-1. 被覆マトリクス

| 計画研究の構成要素 | 判定 | 根拠 |
|---|---|---|
| スコアバイアスを因子に | **部分的** — 単一因子かバイアス種内のみ | GTLM (3 種 leave-one-out), GaLA (単一), RASA Bias-only (KGQA, LLM なし) |
| support/マスクを因子に | **部分的** — RASA Mask-only (67M, 一致); ICML 2025 の窓幅 | [RASA](https://arxiv.org/html/2602.02834); [2505.02130](https://arxiv.org/abs/2505.02130) |
| マスク × バイアスの交互作用 | **部分的** — RASA の 3 セルのみ; LLM/TAG では皆無 | [RASA](https://arxiv.org/html/2602.02834) |
| 正規化 (softmax vs カーネル/線形/Gumbel) を LM 融合の因子に | **未着手** — 純 GT のみ | [GraphGPS](https://arxiv.org/abs/2205.12454); [GraphTARIF](https://arxiv.org/abs/2510.10631) |
| 関係特異 W_V を因子に | **未着手** (孤立) — HGT は異種グラフ; RASA はバンドル | [HGT](https://ar5iv.labs.arxiv.org/html/2003.01332) |
| PE/SE を因子に | **部分的** — GraphToken IDX vs LPE (合成); GDT/OpenGT (LM なし); Structure Doesn't Help は交絡 | [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf) |
| ≥4 成分の因子計画 | **未着手** — RASA が明示的に非因子 | [RASA](https://arxiv.org/html/2602.02834) |
| 構造-in-φ 線形 attention で LM と接続 | **未着手** — 系譜は視覚/点群のみ | [Topological Masking](https://arxiv.org/html/2410.03462v2) |
| グラフ SSM を融合因子に (TAG/KG/コード) | **未着手** — HSA-Net は分子・順序型のみ | [HSA-Net](https://arxiv.org/html/2508.08334v1) |
| 相補性で層別したデータセット設計 | **未着手** (設計原理として) — 同類/異類、豊富/題名のみの分割に暗黙 | [LLMNodeBed](https://arxiv.org/html/2502.00829v2) |
| データ効率 (学習曲線) | **部分的** — 2–3 レジーム; コードの Agarwal et al. は継続事前学習が未統制 | [Agarwal et al.](https://arxiv.org/html/2401.10716v1) |
| 規模効率 (小 LLM+構造 vs 大 LLM 単体) | **部分的** — 別論文間の非一致比較; 同一族内の限界利得 | [GTLM](https://arxiv.org/html/2605.10247); [LLMNodeBed](https://arxiv.org/html/2502.00829v2) |
| コンテキスト/トークン効率 | **部分的** — 固定 400 トークン (Less is More)、needle 65k (GTLM)、単一法 Pareto (AGT); 大半は主張のみ | [Less is More](https://arxiv.org/html/2508.10115v1) |
| パラメータ一致条件 | **部分的** — RASA と Less is More のみ | [RASA](https://arxiv.org/html/2602.02834) |
| 効率を attention 因子の従属変数に | **未着手** | — |

**総合判定**: 計画研究の各軸には部分的先例があるが、どの論文も、どの論文の組も、それらを組み合わせていない。attention 構成要素の因子計画を、LLM を含むテキスト豊富グラフ上で、相補性層別データセットと効率アウトカムに交差させた研究は、2026 年 11 月時点の文献に存在しない。

### 8-2. 必須ベースラインと対照 (否定結果から導かれる)

| ベースライン/対照 | 理由 | 出典 |
|---|---|---|
| テキストのみ (中心ノードのみ CO; 分子は SMILES/SELFIES のみ; コードはトークンのみ) を同一バックボーンで | CO が構造テンプレートに平均 +5.8pp で勝つ; Mol-LLM の w/o Graph が分類で上回る | [Structure Doesn't Help](https://arxiv.org/html/2511.16767v2); [Mol-LLM](https://arxiv.org/html/2502.02810v2) |
| 学習済み定数トークン (グラフを見ない) と直列化グラフのみ | 定数が G-Retriever と同等; 直列化除去が真の損失 | [Token Collapse](https://arxiv.org/html/2610.31572) |
| 直列化テキスト上の LoRA (同一トークン予算) | GraphToken 57.05 < LoRA-on-text 66.03; OOD で LoRA .992 vs Graph Tokens .254 | [G-Retriever](https://arxiv.org/html/2402.07630v3); [Less is More](https://arxiv.org/html/2508.10115v1) |
| 素の GNN (GCN/GraphSAGE/GIN) | GraphInfer-Bench で全タスク最良 LLM 行に匹敵/上回る; GraphArena で GraphToken < 単独 GIN | [GraphInfer-Bench](https://arxiv.org/html/2606.11562); [GraphArena](https://arxiv.org/html/2407.00379v2) |
| MLP アダプタ (構造なし射影) | MLP 76.26% > GCN/GAT/GIN 67–68% | [Structure Doesn't Help](https://arxiv.org/html/2511.16767v2) |
| GDT 型 softmax 加法バイアス (「標準形」) と Mask-only 腕 | 分野の収束形; RASA でマスクが大半を担いバイアス単独は vanilla 以下 | [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf); [RASA](https://arxiv.org/html/2602.02834) |
| グラフ摂動対照 (ランダム再配線、MACCS 型摂動、ノード順序置換) | LLM はプロンプトを段落として読む (再配線 −0.25%); graph bypass | [Huang et al.](https://arxiv.org/html/2309.16595v4); [Mol-LLM](https://arxiv.org/html/2502.02810v2) |
| 同一ベース LLM の規模系列 (例: Llama-3.2-1B/3B、Qwen 0.6B–7B) を構造なしで | 既存の「1B ≈ 7B」は別論文比較; 分類は 1B で飽和し生成のみ 8B で改善 | [GTLM](https://arxiv.org/html/2605.10247) |
| 壁時計・ピークメモリ・FlashAttention 互換性の計上 | カスタム attention は ≈2× ステップコスト (GTLM) / 完全行列必須 (SLASH 36.1 vs 13.3 GB); RASA 遅延 6× | [GTLM](https://arxiv.org/html/2605.10247); [SLASH](https://arxiv.org/html/2605.10503v3) |
| PE 前処理コストの計上 | PubMed で RW 10 分 / スペクトル 1 時間 / 全対 SPD 2 時間; RRWP は最良だが最高コスト | [OpenGT](https://arxiv.org/html/2506.04765); [GDT](https://proceedings.neurips.cc/paper_files/paper/2025/file/f79df6cbc6e5f708440004fad7ef64cc-Paper-Conference.pdf) |

### 8-3. 位置づけるべき先行手法

| 計画研究の因子/軸 | 必ず対置すべき手法 |
|---|---|
| スコアバイアス | GDT (統一式)、GTLM (LLM 内 SPD/RRWP/Magnetic)、GaLA (学習なし 1/d)、Graphormer/GRIT、Code Transformer、GIMLET |
| support/マスク | Tabula RASA (Mask-only)、GraphCodeBERT (DFG マスク)、Graph-KV・NAG・UniGTE (マスク+位置共有)、Exphormer、ICML 2025 attention 窓 |
| 正規化 | NodeFormer (Gumbel)、SGFormer/Polynormer/GraphTARIF (線形)、SLASH (sink 再正規化)、GraphGPS Performer/BigBird |
| 値変換 | HGT、GRIT (W_Ev)、Relational Transformer、RAT-SQL (r^V)、RASA value gating、GaRA (V 射影 LoRA)、EGT ゲート |
| PE/SE | GDT/Black et al. (理論)、OpenGT、GTLM の RoPE リセット、NAG の位置再調整、WIRE/AW-RoPE |
| φ 内構造 (線形 attention) | GKAT/ICML 2022、ICLR 2025 topological masking、KDLGT、GRFs++ |
| グラフ SSM | Graph-Mamba、GMN、GSSC/Chimera/MP-SSM (同変型)、HSA-Net (唯一の LLM 融合) |
| 注入点 | Plenz & Frank (T5)、GraphLLM (prefix-KV)、ENGINE (側枝)、GaRA 層スイープ、GOFA/GreaseLM |
| 効率ベンチ | GLBench、LLMNodeBed、Less is More、Agentic Graph Token Reasoning、Graph Serializations (Oracle) |

### 8-4. データセット選択の具体案 (三型 × 領域)

| 型 | TAG/グラフ推論 | 分子 | コード | 根拠 |
|---|---|---|---|---|
| 構造が必要 | GraphQA-Hardened (35–75 ノード)、GraphArena large-NP、コミュニティ検出 (GraphInfer-Bench)、WebQSP/CWQ 多ホップ、needle-in-a-graph | BBBP (GIN が優位)、QM9/HOMO-LUMO 回帰、反応予測 | 変数誤用 (GREAT 型合成)、長い C++ (CodeNet C++1000/1400)、zero-shot コード間検索 | [GraphInfer-Bench](https://arxiv.org/html/2606.11562); [Structure Doesn't Help](https://arxiv.org/html/2511.16767v2); [CodeNet](https://ar5iv.labs.arxiv.org/html/2105.12655) |
| 構造がテキストと相補的 | 異類性 (Roman-Empire, Amazon-Ratings, School, Texas)、題名のみ Cora/CiteSeer、半教師あり低ラベル | 大きな分子 (60+ 原子; HSA-Net・MolBasic で誤差増大)、OOD AqSol/ORDerly | 低資源言語・低資源 fine-tune (Agarwal 100–500 例)、Java→C# 翻訳 | [LLMNodeBed](https://arxiv.org/html/2502.00829v2); [Mol-LLM](https://arxiv.org/html/2502.02810v2); [Agarwal et al.](https://arxiv.org/html/2401.10716v1) |
| 構造がテキストから導出可能 | PubMed、ogbn-arxiv、ogbn-products (豊富なテキスト、ND≈HN p=0.20/0.49)、ExplaGraphs/SceneGraphs (定数トークンで同等) | 短い MoleculeNet 分子 (BACE/HIV/ClinTox; SMILES で読める) | 全データ CodeXGLUE 要約/生成 (UniXcoder w/o AST で差なし)、POJ-104 (事前学習 LM が GNN に追いつく) | [Structure Doesn't Help](https://arxiv.org/html/2511.16767v2); [Token Collapse](https://arxiv.org/html/2610.31572); [UniXcoder](https://ar5iv.labs.arxiv.org/html/2203.03850) |

避けるべきもの: Devign/BigVul/D2A (ラベル精度 0.80/0.543/0.286)、reason-then-code で飽和する素の GraphQA (n ≤ 20)、GPS+Mamba 型の単純置換 (GMN より下、GPS より下の例あり)。

### 8-5. 設計上のリスク

- 因子水準の選択: SPD バイアスは GTLM で寄与ゼロ、Attending to GTs で異類性では無効。バイアス水準は Magnetic Laplacian / RRWP / 1/d のように複数用意し、マスクとの交互作用セルを必ず含める (RASA: Bias-only は vanilla 以下)。
- 線形 attention を因子水準に入れるなら、表現力損失 (一様化・低ランク) を GraphTARIF 型の補正ありで測るか、φ 内構造 (GRF) を水準に加えないと「正規化」因子が「構造の有無」因子と交絡する。
- 効率指標はトークンだけでなくカーネル効率を含める。attention を改変するとパラメータは 0.015% でも壁時計は 2× になりうる。
- 「小 LLM + 構造 vs 大 LLM 単体」は同一モデル族・同一学習データ・同一トークン予算で行い、別論文の数値 (InstructMol 7B 等) との比較を主張の根拠にしない。
- 相補性の型は事前に操作化する (同類性、テキスト長、ラベル率、構造のみで解けるかの GNN 単独精度) — 文献にはこの操作化が存在しないため、それ自体が貢献になる。

---

## 9. 未検証フラグ一覧 (ノートから引き継ぎ)

以下の項目は一次資料で確認できなかったか、二次情報・検索スニペット・abstract のみに依拠している。本報告はこれらを主張の根拠にしていない。

| 項目 | 状態 |
|---|---|
| GREAT (ICLR 2020) の attention 機構・ablation | OpenReview が検証ウォールで未確認; Google Research ページの記述のみ |
| GSSC の NeurIPS 2024 帰属 | 未検証 (索引は CoRR 2024; TPAMI 2026 版は S2 経由) |
| Graph-Mamba (Wang et al.) の査読会場 | arXiv のみ |
| GMN の KDD 2024 (ACM DL 403) | DL エントリ URL は検索結果で確認、ページ取得は失敗 |
| GraMO、Mamba-GTC、FGMMa | スニペットのみ / URL なし |
| MP-SSM の LRGB 数値 (Peptides-struct 0.2458) | 集約サイト、未検証 |
| AdaptQA (KBS 2025) の GBM 構成・LLM | 全文未取得 |
| CrossLLM-Mamba、Manifold Random Features、Random Wavelet Features、Variance-Reducing Couplings の詳細 | スニペット / 題名のみ |
| SCGT (TPAMI 2026)、MetaGT、Restricted Global-Aware Graph Filters、DUALFormer、G2LFormer、GT-SNT が φ に構造を入れるか | 未開封 (サーベイ/スニペットは否定的) |
| Green/Martin kernels (ICML 2025) の注入点 | 未検証 |
| GCKN (arXiv 2003.05189) の機構 | GraphiT の記述経由のみ |
| Taming graph kernels (ICML 2023) の著者リスト | arXiv ページは Choromanski のみ表示 |
| Polynormer に関する cosFormer 失敗の記述 | 二次 (Lacuna) |
| RGAT→R-GCN の −1〜2% | 二次 (EmergentMind)、一次 (arXiv 2109.05922) は未取得 |
| EMNLP 2023 KG-to-text 相対距離バイアス、MGSA (2409.10294) | 未取得 |
| LLaGA ICML 2024、G-Retriever NeurIPS 2024、GraphTranslator WWW 2024 | arXiv ページで会場未確認 (広く引用はされている) |
| GraphToken の学習パラメータ総数 (17k–199k vs ≈11M head) と会場 | HTML 抽出で不整合; arXiv のみ |
| HiGPT の LLM サイズ・数値、GraphPrompter の Llama-2 サイズ | abstract のみ |
| Learnable Graph PE (2610.27439) の LLM サイズ・データ・数値 | abstract のみ |
| GTLM のベースライン「RGLM」 | 同定不能 |
| SOG_k の ICLR 2026 掲載 | スニペットのみ、arXiv ページに記載なし |
| MDPI Applied Sciences 15(5):2418 (SLM+グラフエンコーダ ≈ LLM) | HTTP 403、スニペットのみ |
| TU Delft 2026 修士論文 (GNN-LLM ハイブリッド) | 503、スニペットのみ |
| AskGNN の 1/5/10% 設定、LLM-GNN の 2,124× コスト削減 | 二次 (alphaxiv) |
| 2026 LLM-GNN co-teaching (2606.11583) の +7.86pp | スニペットのみ |
| "Language Models are Graph Learners"、"Can GNNs still help in the LLM era" | 該当論文見つからず (sotaverified 掲載のみ) |
| Tabula RASA の ICML 2026 掲載 | 掲載リストあり、相互確認なし; When Structure Doesn't Help は会場なし |
| InstructGLM の表の列 (Llama-7b 75.70 等) | 列対応未検証 |
| MoleculeSTM の Nature Machine Intelligence 掲載 | 出版社ページがログイン壁 |
| Llamole (ICLR 2025) のグラフ ablation | 未読 |
| MolMix の「1D 寄与最小」 | arXiv 2509.22468 経由の二次 |
| Zhong et al. 2024 の GNNTrio 数値 | 二次要約のみ |
| KG-LLM-Bench のトークン数 (2,645 vs 4,505) | 二次 |
| Talk like a Graph の ICLR 2024 (OpenReview 未発見) | Google blog / DBLP 経由 |
| LLM4DyG の LLM リスト・DST2 数値、Graph Linearization の手法別精度 | 未取得 |
| GraphArena の GraphToken vs GIN、GNN vs Claude 大規模 NP の詳細数値 | HTML 未読部分 |
| CLEGR の正確な数値 | 図のみ |
| FusionVul、CodingTeachLLM の詳細 | スニペット / 未検証 |
| CGFuse の融合層・LLM サイズ・凍結/LoRA | abstract に記載なし |
| CGM のグラフなし ablation・LoRA 設定 | 未取得 |
| SynCoBERT の会場; Agarwal et al.、SAT、Vul-LMGNNs、CGM の査読会場 | 未確認 |
| Pyda / Jahanshahi–Mockus (LLM4Code 2025) の実験 | abstract のみ |
| Troshin & Chirkova のモデルサイズ別所見 | 未抽出 |
| "The Gate Always Closes"、"From Noise to Diversity" | スニペットのみ (非グラフ) |
| Agentic Graph Token Reasoning の Pareto 数値、HS2C の圧縮率 | スニペット / abstract に数値なし |
| Graph2text or Graph2token の「parameter adaptations may be more effective」帰属 | 取得本文で未確認 |
| Semantic Scholar 被引用走査 | GraphToken 160 件は取得; LLaGA / Plenz & Frank / ENGINE / GraphAdapter / GraphCodeBERT / GALLa は 429 で未走査; Google Scholar 不可; 中国語会場未検索 |

---

## 結論

文献の全体像を一言で言えば、「attention に構造を入れる」技術は 2025 年までに GDT の一式に収束し、2026 年にはその式がそのままデコーダ LLM の内部に移植された (GTLM, GaLA, Graph-KV, NAG, UniGTE)。しかし移植の各論文は、他の成分を固定して一つの成分だけを動かすという、純グラフ Transformer 時代と同じ単因子スタイルを踏襲している。その結果、「事前学習 LM が存在するとき、どの attention 成分が構造融合の利得を担うか」という問いは、成分ごとの断片的な答え (マスクが大半を担う — RASA/KGQA; Magnetic Laplacian バイアスが支配的で SPD は無効 — GTLM/GraphQA; value 側は key 側に対して冗長 — Shaw/MT) が互いに異なる条件で得られているだけで、同一条件下で比較されたことがない。計画研究の因子計画は、この断片を一枚の設計空間に載せ直す試みとして位置づけられる。

より本質的な含意は、2025–2026 年の否定結果が「構造が効くかどうか」の問いを「どの型のデータで、どの効率指標で効くか」の問いへ置き換えたことである。テキストが豊富な同類性 TAG で構造が冗長なのは、もはや反論ではなく前提であり、定数トークン・テキストのみ・直列化 LoRA・素の GNN の四つの対照を置かない融合研究は信頼されない。従って計画研究の新規性は「因子分解」そのものよりも、相補性の型を事前に操作化したデータセット設計と、パラメータ・計算・トークン予算を一致させた上で効率を y 軸に置く測定枠組みにある。これは既存のどのサーベイ分類とも直交し、Tabula RASA の著者が自ら「今後の課題」とした成分別 ablation を、LLM・テキスト豊富ノード・効率アウトカムに拡張する形で、空白を正確に埋める。
