# 読むべき論文 20 本

作成: 2026-11-03。選定元は第 1 回・第 2 回の文献調査報告書（`reports/`）で、全て URL と内容を確認済みのもの。設計書 `docs/DESIGN_structural_prior_efficient_attention_ja.md` の問い「構造は線形 attention / SSM が失った鋭さを、窓・ゲート・周期 softmax と同コストで比べてどれだけ戻すか」に直接効く順に並べた。

選び方: (1) 仮説 H1〜H4 のどれかの根拠か反証候補になる、(2) 本実験の水準や指標をそのまま借りる、(3) 査読で「これと比べたか」と必ず聞かれる、のいずれかを満たすもの。サーベイと教科書的な基礎（Linear Transformer、Performer、Mamba 第 1 版）は外した。

## 読む順番

上から順に読む。A で問題を、B で既存の回復手段を、C〜E で構造の入れ方を、F で最も近い先行と否定結果を押さえる。

### A. 鋭さが失われる理由（問題の定義）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 1 | [Softmax is not Enough (for Sharp Size Generalisation)](https://arxiv.org/html/2410.01104v3) | Veličković, Perivolaropoulos, Barbero, Pascanu。ICML 2025 | 「鋭さ」の形式的定義と、有界ロジットの softmax は長さ n で全ての重みが Θ(1/n) に落ちるという定理。線形 attention には同じ定理が無いことを自分の言葉で説明できるようにする | Lemma 2.1、Theorem 2.2、適応温度 | H3、§3.1 |
| 2 | [The Devil in Linear Transformer](https://ar5iv.labs.arxiv.org/html/2210.10340) | Qin ら。EMNLP 2022 | 線形 attention が劣る二つの原因（勾配の非有界性と attention dilution）を初めて切り分けた論文。dilution の測り方の原型 | Proposition 3.1、局所累積 attention の指標、TransNormer の設計 | §1、§3.1 |
| 3 | [The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry](https://arxiv.org/html/2402.04347) | Zhang, Bhatia, Kumbong, Ré。ICLR 2024 | 線形 attention に欠けるのは「低エントロピー（spiky）」と「単調性」だという診断と、特徴写像を学習して softmax を真似る修正。H4 の対照 (d) の元 | spikiness と単調性の定義、連想想起の実験、長さ別の KL | H4 (d)、§3.1 |
| 4 | [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/html/2402.01032v2) | Jelassi, Brandfonbrener, Kakade, Malach。ICML 2024 | 固定サイズ状態のモデルは長さ L の複写に失敗するという容量の下界。H3 を「エントロピー法則」ではなく「検索容量」として書く根拠 | Corollary 2.8、電話帳課題 | H3 |
| 5 | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/html/2402.18668v2)（Based） | Arora, Eyuboglu, Zhang ら。ICML 2024（会場はノードで未確認） | 検索に Ω(N) ビットの状態が要るという定理と、局所 softmax 窓 64〜128 が検索精度の 90.8% を取り戻すという結果。H4 の対照 (a) と前線図（精度 対 状態バイト）の手本 | Theorem 3.1、Fig. 2（状態サイズ 対 MQAR）、窓と特徴写像の ablation | H3、H4 (a)、§6.1 |

### B. 構造を使わない回復手段（H4 の帰無仮説）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 6 | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v3) | Yang, Kautz, Hatamizadeh。ICLR 2025 | ゲートと delta 則で状態側の鋭さを戻し、1.3B で Transformer++ を上回る。S-NIAH の長さ別崩落（4K → 8K で 92.2 → 29.6）は H3 の長さ依存の具体例 | 式 (10) の再帰、S-NIAH の表、SWA ハイブリッド | H4 (b)、H3 |
| 7 | [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/html/2410.10254v3) | Zhang, Arora ら、Ré。ICLR 2025 | Llama 3 8B を線形化したとき、窓 64 の有無だけで MMLU が 23.8 → 52.8 に動く。凍結した LLM の上で attention を差し替える本研究の状況に最も近い | attention transfer、窓の ablation | H4 (a) |

### C. 線形 attention の内部に構造を入れる（H1 の直接の先行）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 8 | [Linear Transformer Topological Masking with Graph Random Features](https://arxiv.org/html/2410.03462v2) | Reid, Dubey ら、Weller, Choromanski。ICLR 2025 | **最重要。** φ(q) とグラフランダム特徴のテンソル積で位相マスクを線形時間にし、線形と softmax の差の 75〜80% を回復した。本研究はこれを LM とトークン↔ノードの結合列に持ち込む。重要度重みなしでは無マスクと同等（0.689 対 0.693）という失敗例も必読 | マスク M_α = Σ α_k W^k、濃度不等式、重要度重みの ablation、設定（n = 100、p_halt = 0.1） | H1、§4.3 |
| 9 | [From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers](https://ar5iv.labs.arxiv.org/html/2107.07999)（旧題 GKAT） | Choromanski ら。ICML 2022 | マスクの行列ベクトル積が速ければマスク付き線形 attention も速い、という一般理論。木と格子で厳密マスクを線形時間にする方法の出典 | Lemma 3.1、木と格子の構成 | H1 の厳密マスク水準 |
| 10 | [GraphTARIF: Linear Graph Transformer with Augmented Rank and Improved Focus](https://arxiv.org/html/2510.10631v2) | Hu, Li, Fan, Yang。WWW 2026 | グラフ上の線形 attention が「分布の一様化と低ランク化」で失敗するという診断と、その指標（PSE エントロピー、ランク）。第 1 期で再現する 3 本の一つ | Theorem 1（ランクとクラス分離）、log-power 鋭化、GAT 分岐 | §3.1、P4 |

### D. グラフ Transformer の attention に構造を入れる標準形

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 11 | [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454)（GraphGPS） | Rampášek, Galkin, Dwivedi, Luu, Wolf, Beaini。NeurIPS 2022 | 局所 MPNN ＋ 大域 attention の設計空間と、Performer / BigBird が softmax に負ける同一レシピの比較。実験の組み方（パラメータ予算、seed 数）の手本 | attention モジュールの ablation 表、PE / SE の分類 | §4.2、§7 |
| 12 | [Graph Inductive Biases in Transformers without Message Passing](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf)（GRIT） | Ma ら。ICML 2023 | メッセージ伝播なしで attention の中だけに構造を入れる、という本研究の発想の直接の先行。既定の位置符号化 RRWP の出典 | RRWP の定義と Prop. 3.2、K の感度 | §4.1 の PE 既定 |
| 13 | [Comparing Graph Transformers via Positional Encodings](https://arxiv.org/html/2402.14202v4) | Black, Wan, Mishne, Nayyeri, Wang。ICML 2024 | 絶対 PE と相対 PE の等価性、冪の積み重ねがスペクトル核と同等という定理。PE を 1 つに固定する理由の理論的な裏付け | Thm 3.8 / 3.10、Example 1、Thm 4.6〜4.9 | §4.1 |
| 14 | [Generalizable Insights for Graph Transformers in Theory and Practice](https://arxiv.org/html/2511.08028v1)（GDT） | Stoll, Müller, Morris。NeurIPS 2025 | softmax(QKᵀ/√d ＋ B)V という一つの式が局所 GT・マスク・相対 PE を含むことを示した。本研究の softmax 側の「標準形」。AST 課題で PE の差は F1 0.2 だが RRWP はコスト 2 倍、という効率の数字 | Theorem 1、OGB-Code2 の PE 比較表 | §4.2 の softmax 水準、§6 |

### E. 状態空間モデルの作用素に構造を入れる（第 2 の系譜）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 15 | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/html/2405.21060v1)（Mamba-2） | Dao, Gu。ICML 2024 | 選択的 SSM が「構造化マスク付きの線形 attention」Y = (L ⊙ CBᵀ)V と等価であることを示した。線形 attention と SSM を一つの枠で比べる本研究の理論的な土台 | SSD の双対性、半分離マスク、カーネルの速度 | §4.2 の SSM 水準 |
| 16 | [Chimera: State Space Models Beyond Sequences](https://arxiv.org/html/2510.12111) | Lahoti, Marwah, Puduppully, Gu。TMLR 2025 | SSD のマスクを任意グラフのレゾルベント (I − A)⁻¹ に置き換え、DAG なら線形時間の 1 パスになる。本研究の SSM 融合層はこの形で確定済み。LM とは結ばれておらず、そこが本研究の空白 | DAG の再帰式、巡回グラフの近似、平坦化の ablation | §4.2、§4.3 の SSM 列 |

### F. LM と構造を結ぶ最も近い先行と、否定結果

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 17 | [Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning](https://arxiv.org/html/2605.10247)（GTLM） | Vajda。arXiv 2026（会場なし） | 凍結した Llama-3.2-1B の attention に SPD / RRWP / 磁気ラプラシアンのバイアスを入れた、LLM 側で最も近い先行。構造パラメータは 0.015% だが壁時計は 2 倍。softmax の範囲に留まっている点が本研究との差 | バイアスの leave-one-out、テキストのみ版との比較表、ステップコスト | §4.2、§11 |
| 18 | [What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA](https://arxiv.org/html/2602.02834) | Petersen ら、Maggioni。arXiv 2026（ICML 2026 掲載は相互確認なし） | パラメータを揃えてマスク・バイアス・関係パラメータを比べた唯一の研究。マスクだけで +45〜73 点、バイアスだけでは vanilla 以下。本研究の因子計画で何を揃えるべきかの手本であり、比較を因子計画にしなかった点が本研究の差 | Full / Mask-only / Bias-only / vanilla の表、パラメータ一致の方法 | §4.6、§11 |
| 19 | [When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected](https://arxiv.org/html/2511.16767v2) | Xu, You, Ma。arXiv 2025 | テキストのみが構造テンプレートに勝ち、MLP アダプタが GNN に勝つという否定結果。必須対照（テキストのみ、MLP アダプタ）を置く理由で、H2 の softmax 行「導出可能型で利得なし」の根拠。第 1 期で再現する 3 本の一つ | テンプレート比較の表、アダプタ比較、規模を上げても残る差 | §4.5、H2 |
| 20 | [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://ar5iv.labs.arxiv.org/html/2009.08366) | Guo ら。ICLR 2021 | コードのトークンと構造ノードを一つの列に並べ、整合マスクで結ぶ方法の原型。本研究の整合規則（葉 ↔ サブワード区間）と tree-sitter の使い方をここから借りる | グラフ誘導マスクの式、データフローの抽出、`run.py` の整合処理 | §4.1 の整合規則、P1 |

## 読み方の目安

- **1 本目は 8（位相マスク）を通読し、残りは「見る箇所」を中心に読む。** 8 が本研究の H1 そのものの先行で、他の論文はその前提（A、B）か比較対象（D〜F）になる。
- **A の 5 本を読み終えた時点で、設計書 §3.1 の鋭さ指標と H3 を自分で説明できるか確かめる。** できなければ 1 と 4 を読み直す。
- **各論文で「本研究との差」を一行で書く。** 査読で「これと何が違うか」と聞かれる論文は 8、16、17、18 の 4 本。この 4 本の差分の一行は、そのまま論文の関連研究節になる。
- 会場が未確認のもの（5、17、18、19）は、引用前に会場を確かめる。報告書の未検証一覧（第 1 回 §9、第 2 回 §8）に入っている。

## 第 3 回文献調査の後に足す枠（設計書 v0.5）

設計書 v0.5 で、疎な attention を三つ目の系譜にし、実在モデルの attention 分析（H5）を柱に加えた。この二つは第 1 回・第 2 回の調査の対象外だったので、第 3 回の報告書が出たら、それぞれから 2〜3 本をこの一覧に足す。それまでの暫定の候補は、第 2 回で確認済みの次の 3 本（attention sink の分析）。

- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)（Xiao ら。ICLR 2024）
- [When Attention Sink Emerges in Language Models](https://arxiv.org/html/2410.10781v2)（Gu ら。ICLR 2025）
- [Why do LLMs attend to the first token?](https://arxiv.org/html/2504.02732v4)（Barbero ら。会場はノードで未確認）

## 次に読む候補（20 本の後）

20 本で足りない部分が出たときの補欠。Zoology（MQAR の定義）、Kimi Linear（ハイブリッド比率の ablation）、GRFs++（AST のような直径の大きいグラフでの GRF）、Attention is not all you need（ランク崩壊）、Accounting for Variance in Machine Learning Benchmarks（統計の手順）。URL は第 2 回報告書にある。
