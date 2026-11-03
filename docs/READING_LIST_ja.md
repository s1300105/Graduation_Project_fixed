# 読むべき論文 30 本

作成: 2026-11-03（版 2。版 1 の 20 本を設計書 v0.5 に合わせて作り直した）。
選定元: 第 1 回〜第 3 回の文献調査（`reports/` と `research_notes/`）。全て URL と内容を調査ノートで確認したもの。会場が確認できていないものは表に明記した。

設計書 `docs/DESIGN_structural_prior_efficient_attention_ja.md`（v0.5）の題「効率的 attention は何を失い、何が安く取り戻すか」に合わせて並べた。研究の柱は二本で、凍結 LM の上の融合層で因子を振る統制実験（H1〜H4）と、公開モデルの中の attention を測る分析（H5）である。

選び方は次のいずれかを満たすこと。

- 仮説 H1〜H5 のどれかの根拠か、反証の候補になる。
- 本実験の水準・指標・課題・チェックポイントをそのまま借りる。
- 査読で「これと比べたか」と必ず聞かれる。

## 版 1 からの変更

- **足した系譜**: 疎な attention（5 本）と、実在モデルの attention 分析（6 本）。設計書 v0.5 で加えた二本柱に対応する。
- **足した論文**: 検索容量の課題を定義した Zoology と、同じデータで softmax と比べられる NVIDIA の 8B ハイブリッド研究。
- **外した論文**: グラフ寄りの 3 本（When Structure Doesn't Help、GraphCodeBERT、Black らの位置符号化の比較）と、GraphTARIF 以外の線形グラフ Transformer。いずれも末尾の補欠に回した。研究の主語を attention に寄せたためで、論文の価値が下がったわけではない。

## 読む順番

上から順に読む。A で問題を定義し、B〜D で三系統の回復手段を押さえ、E〜G で構造の入れ方を見て、H で実在モデルの測り方を学び、I で最も近い先行を確かめる。

### A. 鋭さが失われる理由（6 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 1 | [Softmax is not Enough (for Sharp Size Generalisation)](https://arxiv.org/html/2410.01104v3) | Veličković, Perivolaropoulos, Barbero, Pascanu。ICML 2025 | 「鋭さ」の形式的な定義と、有界ロジットの softmax は長さ n で全ての重みが Θ(1/n) に落ちるという定理。線形 attention に同じ定理が無いことを説明できるようにする。任意の理論目標の出発点 | Lemma 2.1、Theorem 2.2、適応温度 | H3、§2 の理論目標 |
| 2 | [The Devil in Linear Transformer](https://ar5iv.labs.arxiv.org/html/2210.10340) | Qin ら。EMNLP 2022 | 線形 attention が劣る二つの原因（勾配の非有界性、attention dilution）を初めて切り分けた。dilution の測り方の原型 | Proposition 3.1、局所累積 attention の指標 | §1、§3.1 |
| 3 | [The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry](https://arxiv.org/html/2402.04347) | Zhang, Bhatia, Kumbong, Ré。ICLR 2024 | 線形 attention に欠けるのは「低エントロピー（spiky）」と「単調性」という診断と、特徴写像を学習して softmax を真似る修正。H4 の対照 (d) | spikiness と単調性の定義、長さ別の KL | H4 (d)、§3.1 |
| 4 | [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927) | Arora, Eyuboglu, Timalsina ら、Ré。ICLR 2024 | 効率的なモデルと softmax の差の大半が「連想想起」で説明できると示し、合成課題 MQAR を定義した。本研究の合成検索課題の設計の手本 | MQAR の生成手順、想起の差の分解 | H3、§3.1 の行動指標、§6.5 |
| 5 | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/html/2402.18668v2)（Based） | Arora, Eyuboglu, Zhang ら、Ré。ICML 2024（会場はノードで未確認） | 検索に Ω(N) ビットの状態が要るという定理と、局所 softmax 窓 64〜128 が検索精度の 90.8% を戻すという結果。前線の図（精度 対 状態バイト）の手本 | Theorem 3.1、Fig. 2、窓と特徴写像の ablation | H3、H4 (a)、§6.1 |
| 6 | [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/html/2402.01032v2) | Jelassi, Brandfonbrener, Kakade, Malach。ICML 2024 | 固定サイズの状態は長さ L の複写に失敗するという下界。H3 を「エントロピーの法則」ではなく「検索容量」として書く根拠。電話帳課題は H5 にも使う | Corollary 2.8、電話帳課題 | H3、H5 |

### B. 線形と SSM の系譜の回復手段（3 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 7 | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v3) | Yang, Kautz, Hatamizadeh。ICLR 2025 | ゲートと delta 則で状態側の鋭さを戻す。S-NIAH が 4K から 8K で 92.2 から 29.6 に崩れる表は H3 の長さ依存の具体例 | 式 (10)、S-NIAH の表、SWA ハイブリッド | H4 (b)、H3 |
| 8 | [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/html/2410.10254v3) | Zhang, Arora ら、Ré。ICLR 2025 | Llama 3 8B を線形化したとき、窓 64 の有無だけで MMLU が 23.8 から 52.8 に動く。凍結 LLM の attention を差し替える本研究の状況に最も近い | attention transfer、窓の ablation | H4 (a) |
| 9 | [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887) | Waleffe ら、Dao, Gu ら（NVIDIA）。arXiv 2024 | 同じ 3.5T トークンで学習した 8B の Transformer・Mamba-2・ハイブリッド（56 層中 attention 4 層）の比較。純 SSM は電話帳で約 1K から失敗し、ハイブリッドは約 5.5K まで 100%。H5 で同じデータの softmax 対照として使えるチェックポイント | ハイブリッドの比率、電話帳と RULER の長さ別の表 | H5、§6.5 |

### C. 疎な attention の系譜（5 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 10 | [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) | Zaheer ら。NeurIPS 2020 | 窓＋大域＋ランダムの固定パターンで線形の計算量にしつつ、普遍近似性を保つという理論。疎な attention の表現力を語るときの基準。GraphGPS の疎な選択肢の元 | 普遍近似の定理、必要な層数の下界、ブロック実装 | 因子 A の疎の水準 |
| 11 | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/html/2502.11089v2)（NSA） | Yuan ら（DeepSeek）。ACL 2025 | 圧縮・ブロック選択（64 トークンのブロックを上位 16 個）・窓の 3 分岐を学習で持つ疎な attention。64K で FlashAttention-2 比 順伝播 9.0 倍、逆伝播 6.0 倍。選択分岐だけを取り出すと H4 (e) の水準になる | 選択分岐の設計、GQA 群でのブロック共有、Triton カーネル、LongBench の表 | H4 (e)、§4.2 |
| 12 | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | Lu ら（Moonshot）。arXiv 2025 | ブロックの平均キーとの内積で上位 k 個のブロックを選ぶ、最も単純な学習による選択。H4 (e) の最有力の実装候補。教師つき微調整で劣化し、最後の数層を全対に戻すと直るという失敗例も重要 | ブロックの大きさと k の設定、RULER の比較、失敗と対処 | H4 (e)、§4.2 |
| 13 | [Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction](https://arxiv.org/html/2510.20787) | He, Garner。arXiv 2025 | 疎（NSA）と線形（Gated DeltaNet）と softmax を、同じパラメータ数と同じトークン数で比べた唯一の表。壁時計とメモリは揃えていないので、そこが本研究の差になる | 0.4B と 1.4B の NIAH と想起課題の表 | H4、§4.6 |
| 14 | [Understanding Long Programming Languages with Structure-Aware Sparse Attention](https://arxiv.org/abs/2205.13730)（SASA） | Liu, Wang, Chen, Gao, Zhou。SIGIR 2022（短報） | **疎な支持を構造（AST）で選んだ最も近い先行。** 学習統計で選ぶブロックと AST の隣接で選ぶブロックの両方を持ち、ablation では学習統計の方が効いた（F1 0.911 から 0.828、AST 側は 0.874）。両者を同じ k の別の腕として比べてはいない点が本研究の差 | ブロックの構成、ablation の表 | §4.3 の疎の列、H4 (e) |

### D. 線形 attention の内部に構造を入れる（3 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 15 | [Linear Transformer Topological Masking with Graph Random Features](https://arxiv.org/html/2410.03462v2) | Reid, Dubey ら、Weller, Choromanski。ICLR 2025 | **最重要。** φ(q) とグラフランダム特徴のテンソル積で位相マスクを線形時間にし、線形と softmax の差の 75〜80% を回復した。重要度重みなしでは無マスクと同等（0.689 対 0.693）という失敗例も必読 | マスク M_α = Σ α_k W^k、濃度不等式、重要度重みの ablation、設定（n = 100、p_halt = 0.1） | H1、§4.3 |
| 16 | [From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers](https://ar5iv.labs.arxiv.org/html/2107.07999)（旧題 GKAT） | Choromanski ら。ICML 2022 | マスクの行列ベクトル積が速ければマスク付き線形 attention も速い、という一般理論。木と格子で厳密マスクを線形時間にする方法の出典 | Lemma 3.1、木と格子の構成 | H1 の厳密マスク水準 |
| 17 | [GraphTARIF: Linear Graph Transformer with Augmented Rank and Improved Focus](https://arxiv.org/html/2510.10631v2) | Hu, Li, Fan, Yang。WWW 2026 | グラフ上の線形 attention が分布の一様化と低ランク化で失敗するという診断と、その指標（PSE エントロピー、ランク）。第 1 期で再現する 3 本の一つ | Theorem 1、log-power 鋭化 | §3.1、P4 |

### E. softmax 側の標準形と設計空間（3 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 18 | [Generalizable Insights for Graph Transformers in Theory and Practice](https://arxiv.org/html/2511.08028v1)（GDT） | Stoll, Müller, Morris。NeurIPS 2025 | softmax(QKᵀ/√d ＋ B)V という一つの式が局所 GT・マスク・相対位置符号化を含むと示した。本研究の softmax 側の標準形。AST 課題で位置符号化の差は F1 0.2 だが、RRWP はコストが 2 倍という効率の数字 | Theorem 1、OGB-Code2 の位置符号化の比較表 | §4.2、§6 |
| 19 | [Graph Inductive Biases in Transformers without Message Passing](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf)（GRIT） | Ma ら。ICML 2023 | メッセージ伝播なしで attention の中だけに構造を入れる、という発想の直接の先行。既定の位置符号化 RRWP の出典 | RRWP の定義と Prop. 3.2、K の感度 | §4.1 |
| 20 | [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454)（GraphGPS） | Rampášek, Galkin, Dwivedi, Luu, Wolf, Beaini。NeurIPS 2022 | 同じレシピで softmax・Performer（線形）・BigBird（疎）を差し替えて比べた表がある。三系統を一つの実験で並べる組み方の手本 | attention モジュールの ablation 表、パラメータ予算と seed 数 | §4.2、§7 |

### F. SSM の作用素に構造を入れる（2 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 21 | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/html/2405.21060v1)（Mamba-2） | Dao, Gu。ICML 2024 | 選択的 SSM が構造化マスク付きの線形 attention Y = (L ⊙ CBᵀ)V と等価であることを示した。三系統を一つの式で比べる土台で、H5 で Mamba-2 の暗黙の attention を明示する方法でもある | SSD の双対性、半分離マスク | §4.2、§6.5 |
| 22 | [Chimera: State Space Models Beyond Sequences](https://arxiv.org/html/2510.12111) | Lahoti, Marwah, Puduppully, Gu。TMLR 2025 | SSD のマスクをグラフのレゾルベント (I − A)⁻¹ に置き換え、DAG なら線形時間の 1 パスになる。本研究の SSM 融合層はこの形で確定済み。LM と結ばれていないことが本研究の空白 | DAG の再帰式、巡回グラフの近似、平坦化の ablation | §4.2、§4.3 |

### G. 実在モデルの attention の測り方（6 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 23 | [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574) | Wu, Wang, Xiao, Peng, Fu。ICLR 2025 | 検索ヘッドを「生成したトークンが針の中にあり、そのヘッドの argmax も同じトークンにある」割合で同定する方法。全ヘッドの 3〜6% が検索ヘッドで、上位 50 個を消すと針の課題が 50 を割る。H5 の「検索に使われるヘッド」の定義をここから借りる | 検索スコアの定義（閾値 0.1）、検出の手順、マスクによる因果の検証 | H5、§6.5 |
| 24 | [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/abs/2410.10781) | Gu, Pang, Du ら、Lin。ICLR 2025（Spotlight） | sink の定量指標 Sink_k^ε（閾値 0.3）を定義し、sink は鍵のバイアスとして働き、sigmoid attention ではほぼ消えると示した。本研究で sink 質量を別に報告し、除外して鋭さを測り直す手順の根拠 | 指標の定義、softmax 対 sigmoid の比較 | §3.1 |
| 25 | [Why do LLMs attend to the first token?](https://arxiv.org/html/2504.02732v4) | Barbero, Arroyo, Gu ら、Veličković, Pascanu。arXiv 2025（会場は未確認） | sink は過剰な混合を防ぐ仕組みだという説明と、モデル規模や学習時の文脈長とともに sink が増えるという測定。Gemma 7B で先頭トークンを除くと RULER-4K が 82.57 から 0 になる | 過剰混合の定理、規模別の sink の表 | §3.1、H5 |
| 26 | [The Hidden Attention of Mamba Models](https://arxiv.org/abs/2403.01590) | Ali, Zimerman, Wolf。arXiv 2024（会場は未確認） | Mamba の選択的 SSM から暗黙の attention 行列を取り出す式。softmax が無いので正規化を自分で決める必要があり、その判断の出発点。後続の統一形（arXiv 2405.16504）も併せて読む | 暗黙の attention の式、可視化の正規化 | §6.5、P10 |
| 27 | [Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism](https://arxiv.org/abs/2504.18574) | Bick, Xing, Gu。arXiv 2025（ICML 2025 掲載は未確認） | 検索を担う「集めて集約する」ヘッドが SSM とハイブリッドにもあり、Zamba2-7B では attention ヘッド 9 個を消すと MMLU が 64.3 から 34.9 に落ちる。H5 (ii)「ハイブリッドでは鋭い読み出しが softmax 層に集中する」の直接の先行 | ヘッドの同定法、ablation の表 | H5 (ii) |
| 28 | [Stuffed Mamba: Oversized States Lead to the Inability to Forget](https://arxiv.org/abs/2410.07145) | Chen, Zhang, Hu, Han, Liu, Sun。COLM 2025 | Mamba-2 の passkey 検索は 8K 以内でほぼ完全、16K を超えるとほぼ 0 になり、状態の大きさと検索できる長さの関係を法則として当てはめた。H5 の長さ依存の検証で手元の測定と照合する数値 | 状態の大きさと passkey の長さの法則、層別の保持率 | H3、H5 |

### H. 最も近い先行（2 本）

| # | 論文 | 著者・会場 | 読む理由 | 見る箇所 | 効く節 |
|---|---|---|---|---|---|
| 29 | [Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning](https://arxiv.org/html/2605.10247)（GTLM） | Vajda。arXiv 2026（会場なし） | 凍結した Llama-3.2-1B の attention に SPD・RRWP・磁気ラプラシアンのバイアスを入れた、LLM 側で最も近い先行。構造パラメータは 0.015% だが壁時計は 2 倍。softmax の範囲に留まっている点が本研究との差 | バイアスの leave-one-out、テキストのみ版との比較、ステップコスト | §4.2、§11 |
| 30 | [What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA](https://arxiv.org/html/2602.02834) | Petersen ら、Maggioni。arXiv 2026（ICML 2026 掲載は相互確認なし） | パラメータを揃えてマスク・バイアス・関係パラメータを比べた唯一の研究。マスクだけで 45〜73 点上がり、バイアスだけでは何もしない版以下。因子計画で何を揃えるかの手本 | Full / Mask-only / Bias-only / vanilla の表 | §4.6、§11 |

## 読み方の目安

- **最初に 15（位相マスク）を通読する。** H1 そのものの先行で、他の論文はその前提か比較対象になる。残りは「見る箇所」を中心に読めば足りる。
- **A の 6 本を読んだら、設計書 §3.1 の鋭さ指標と H3 を自分の言葉で説明できるか確かめる。** できなければ 1 と 6 を読み直す。
- **C と G は第 1 期の後半（第 4〜5 か月）に読む。** P7（構造なしの回復手段の比較）と P10（実在モデルの分析の試行）の直前に読むと、そのまま実装に移れる。
- **各論文で「本研究との差」を一行で書く。** 査読で差分を聞かれる論文は 14（SASA）、15（位相マスク）、22（Chimera）、29（GTLM）、30（Tabula RASA）の 5 本。この 5 行はそのまま関連研究の節になる。
- **会場が未確認の論文は引用前に確かめる。** 5、9、12、13、25、26、27、29、30 の 9 本。報告書の未検証一覧（第 1 回 §9、第 2 回 §8、第 3 回の報告書）に入っている。

## 補欠（30 本の後）

足りない部分が出たときに読む。URL は各報告書にある。

- **グラフ寄り（版 1 から外したもの）**: When Structure Doesn't Help（テキストのみ対照の強さ）、GraphCodeBERT（トークンとノードの整合の原型）、Comparing Graph Transformers via Positional Encodings（位置符号化の理論）。
- **線形と SSM**: Kimi Linear（ハイブリッドの比率の ablation）、GRFs++（AST のような直径の大きいグラフでの GRF）、A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models（暗黙の attention の統一形）。
- **疎**: DeepSeek-V3.2 の DSA（トークン単位の上位 k 選択）、SeerAttention（学習によるブロック選択のゲート）。
- **分析と評価**: RULER（長さを制御した検索課題）、DuoAttention（検索ヘッドとストリーミングヘッドの分離）、Accounting for Variance in Machine Learning Benchmarks（統計の手順）、Attention is not all you need（ランク崩壊）。
