# 疎な attention は選択で失敗し、実在モデルの対照は 4K で尽きる

作成: 2026-11-03。第 3 回文献調査の報告書（第 1 期計画 §2.3 の系譜 F「疎な attention」と系譜 G「実在 LLM の attention の分析」）。調査ノートは `research_notes/疎な attention と実在モデルの分析 文献調査/` の 3 本（`sparse_attention_lineage.md`、`attention_head_analysis_methods.md`、`linear_ssm_model_analysis_and_checkpoints.md`）。対象は設計書 `docs/DESIGN_structural_prior_efficient_attention_ja.md`（v0.5）と第 1 期計画 `docs/PHASE1_PLAN_survey_and_pilot_ja.md`（P7、P10）。第 1 回報告書 `reports/LLM GNN 融合 attention 先行研究.md` と第 2 回報告書 `reports/効率的 attention の鋭さと構造 文献調査.md` は再掲せず、新しいノートが結論を変える・鋭くする箇所だけ参照する。

印の意味: **〔K〕** はノートの調査者が原論文の記憶から書き、今回の調査で数値・文言を再取得していない項目（URL は一次資料）。**〔U〕** は二次情報のみ・未検証。**〔S〕** は集約サイト・検索要旨のみ。無印は一次資料（arXiv の要旨・HTML、Anthology、公式ブログ、HF API）で確認済み。印のついた項目と、ノートが「unverified」とした項目は全て §10 の未検証一覧に引き継いだ。HF の ID はノートの記載どおり（2026-11-03 に HF API で確認されたもの）。

**要旨。** 疎な attention の 2019〜2026 年の系譜は、内容によらない固定パターン（窓・ストライド・大域トークン・ランダム辺）から、**キーのブロック（またはトークン）を安い代理得点で採点し、上位 k 個を選び、選んだ支持の上だけで厳密な softmax を掛ける**という一つの型に収束した（MoBA、NSA、DeepSeek の DSA）。この型は支持上の softmax をそのまま残すので、正しいキーを選べた限り argmax 的な鋭さを失わない。失うのは**選択の誤り**であり、ブロック平均に薄まった単一の鍵を取り逃がせば、その鍵の情報は全て失われる。パラメータとトークンを揃えた唯一の比較（0.4B / 1.4B）では、純 NSA は Transformer にやや劣り、Gated DeltaNet との上下は規模で入れ替わり、ハイブリッドが最良だった。**壁時計とメモリを揃えた 疎 対 線形 対 SSM の比較は見つからなかった。** 実カーネルでの速度向上は 32K 以上・疎性 75〜90% 以上でしか大きくならず、学習時の疎 attention は KV を減らさない。構造で疎な支持を選んだ研究はコードの SASA（AST 隣接ブロック）・LongCoder とグラフの Exphormer にあるが、**「学習による選択」と「構造による選択」を同じ k で別々に学習して比べた研究は無い**。実在モデルの分析では、検索ヘッド（全ヘッドの約 3〜6%）、sink 率、帰納・echo スコアの定義が揃っている。いずれも argmax か 1 列の質量だけで計算できるので、16K でも 24 GB に収まる。SSM・線形モデルの暗黙の attention は符号つきで和が 1 にならず、正規化の先例は DeciMamba の「行ごとの |α| の L1 正規化」しかない。同じデータ・同じ規模の softmax 対照を持つ公開チェックポイントは五組あるが、**全て 2K〜4K で学習されている**。したがって H5 の 4K〜16K は両族にとって外挿になる。ヘッドごとの指標が正答を予測する AUROC を長さ別に測った研究も無く、ここが H5 の空白になる。

---

## 1. 疎な attention の系譜は「代理得点 → 上位 k → 支持上の厳密 softmax」に収束した

### 1.1 三つの段階

**固定パターン（2019〜2020）。** Sparse Transformer は局所＋ストライドの因子分解パターンで O(n√n) を実現し、専用のブロック疎カーネルを書いた〔K〕（[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)）。Longformer は窓＋拡張窓＋課題ごとの大域トークンを使う〔K〕（[Longformer](https://arxiv.org/abs/2004.05150)）。BigBird はブロック 64 の窓＋大域＋ランダムブロックを使う〔K〕（[Big Bird](https://arxiv.org/abs/2007.14062)）。ETC は追加の大域トークンを文や段落に結びつけた〔K〕（[ETC](https://arxiv.org/abs/2004.08483)）。このうち製品の LLM に生き残ったのは二つの原始要素だけである。一つは**全対層と交互に置く窓**で、Mistral 7B の W = 4096〔K〕（[Mistral 7B](https://arxiv.org/abs/2310.06825)）、Gemma 3 の局所 5 : 大域 1（窓 1024）〔K〕（[Gemma 3](https://arxiv.org/abs/2503.19786)）、gpt-oss の帯幅 128〔K〕（[gpt-oss](https://arxiv.org/abs/2508.10925)）がこれにあたる。もう一つは **sink ＋ 窓**で、先頭 4 トークンと直近窓を残す（[StreamingLLM](https://arxiv.org/abs/2309.17453)）〔K〕。BigBird のランダム辺は製品から消え、大域的な到達は深さ方向の周期的な全対層で賄われている。

**内容に基づく選択（2020〜2023）。** 初期の試みはトークン単位だった。Reformer は LSH で、Routing Transformer はオンライン k-means で、Sinkhorn attention は学習したブロック並べ替えで支持を作る〔K〕（[Reformer](https://arxiv.org/abs/2001.04451)、[Routing Transformer](https://arxiv.org/abs/2003.05997)、[Sparse Sinkhorn Attention](https://arxiv.org/abs/2002.11296)）。現在の型の直接の祖先は Landmark Attention で、ブロックごとに学習した landmark トークンへの attention でブロック全体の取り出しを決める〔K〕（[Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300)）。

**ネイティブに学習するブロック上位 k（2025）。** NSA は三つの枝をゲートで合成する。一つ目は圧縮枝（ブロック長 l = 32、ストライド 16）、二つ目は選択枝（ブロック 64、上位 n = 16 個で、うち先頭 1 個と局所 2 個は常に入る）、三つ目は窓 512 の枝である。選択の得点は圧縮枝の attention から作り、GQA グループ内の全ヘッドが同じ KV ブロックを選ぶ（[Native Sparse Attention](https://arxiv.org/html/2502.11089v2)）。MoBA の選択はもっと単純で、ゲート得点を ⟨q, ブロック内キーの平均⟩ とし、上位 k ブロックに現ブロックを必ず加える。FlashAttention varlen の上で動き、Kimi の長文脈要求に使われている（[MoBA](https://arxiv.org/html/2502.13189v1)）。DeepSeek-V3.2 の DSA はトークン単位である。lightning indexer I_{t,s} = Σ_j w^I_{t,j}·ReLU(q^I_{t,j}·k^I_s) で採点して上位 k = 2048 トークンを選ぶ。KV を全ヘッドで共有する MLA の MQA 形の上でだけ成り立つ設計である（[DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)）。

### 1.2 選択の基準は四つに分かれる

| 基準 | 何で採点するか | 代表 | 性格 |
|---|---|---|---|
| 内容によらない | 位置だけ（窓・ストライド・先頭・ランダム） | Sparse Transformer、Longformer、BigBird、Mistral SWA、StreamingLLM | 支持が入力に依らず、カーネルが自明に速い。窓外の任意トークンを 1 層で取り出せない |
| 圧縮・プールしたキー | ブロックの代表（平均キー、圧縮トークン、要素ごとの min / max 上界、landmark） | MoBA、NSA、Quest、Landmark、InfLLM | 主モデルの q で直接採点する。追加の学習器が要らない |
| 別に学習した採点器 | 小さな indexer / gate を密 attention への KL や自己蒸留で学習 | DSA（lightning indexer）、SeerAttention（AttnGate） | 採点器と主モデルの学習が切り離される（DSA は indexer の入力を detach） |
| 蓄積した attention の統計 | 累積 attention、観測窓での投票、ヘッド単位の役割 | H2O、SnapKV、DuoAttention | 推論時のみ。除去は不可逆 |

粒度にも規則がある。トークン単位の選択（DSA）は KV をヘッド間で共有する MLA を前提にし、1 回の gather で全ヘッドに供給する。GQA / MHA のモデルではブロック単位（NSA、MoBA）が選ばれる（[DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)、[NSA](https://arxiv.org/html/2502.11089v2)）。

### 1.3 目録

| 方式 | 原題 | 年・会場 | 選択 | 粒度・予算 | 学習 | 公表コスト |
|---|---|---|---|---|---|---|
| Sparse Transformer〔K〕 | [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) | 2019 arXiv | 固定ストライド | トークンのパターンをブロック疎で実装、O(n√n) | ネイティブ | — |
| Longformer〔K〕 | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) | 2020 arXiv | 窓＋拡張＋大域 | トークン、w ≈ 512 | ネイティブ / 継続 | 線形 |
| BigBird〔K〕 | [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) | NeurIPS 2020 | 窓＋大域＋ランダム | ブロック 64 | ネイティブ | 線形 |
| ETC〔K〕 | [ETC: Encoding Long and Structured Inputs in Transformers](https://arxiv.org/abs/2004.08483) | EMNLP 2020 | 大域・局所（追加大域トークン） | トークン | ネイティブ | 線形 |
| Reformer〔K〕 | [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) | ICLR 2020 | LSH バケット | トークンのチャンク | ネイティブ | O(n log n) |
| Routing〔K〕 | [Efficient Content-Based Sparse Attention with Routing Transformers](https://arxiv.org/abs/2003.05997) | TACL 2021 | オンライン k-means | トークンのクラスタ | ネイティブ | O(n^1.5) |
| Sinkhorn〔K〕 | [Sparse Sinkhorn Attention](https://arxiv.org/abs/2002.11296) | ICML 2020 | 学習したブロック並べ替え | ブロック | ネイティブ | — |
| Mistral SWA〔K〕 | [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | 窓 W = 4096 | トークンの帯 | ネイティブ | 16K で 2 倍 |
| StreamingLLM〔K〕 | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | ICLR 2024 | sink 4 ＋ 窓 | トークン | 推論時（sink 事前学習も） | 再計算窓比 22.2 倍 |
| Landmark〔K〕 | [Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300) | NeurIPS 2023 | landmark で上位ブロック | ブロック | 微調整 | — |
| H2O〔K〕 | [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) | NeurIPS 2023 | 累積 attention ＋ 直近 | トークン、除去 | 推論時 | スループット最大 29 倍 |
| Quest〔K〕 | [Quest: Query-Aware Sparsity](https://arxiv.org/abs/2406.10774) | ICML 2024 | ページの min / max 上界で上位 k | KV ページ | 推論時 | self-attention 7.03 倍 |
| MInference〔K〕 | [MInference 1.0](https://arxiv.org/abs/2407.02490) | NeurIPS 2024 | ヘッドごとの A 形 / 縦斜線 / ブロック | 混合 | 推論時 | A100 で 1M 前処理 10 倍 |
| DuoAttention | [DuoAttention](https://arxiv.org/abs/2410.10819) | ICLR 2025 | 検索ヘッド（全 KV）対 ストリーミングヘッド（sink ＋ 窓） | ヘッド | 推論時（軽い最適化） | メモリ 2.55 / 1.67 倍減 |
| RetrievalAttention〔K〕 | [RetrievalAttention](https://arxiv.org/abs/2409.10516) | 2024 arXiv | KV 上の ANN | トークン（キーの 1〜3%） | 推論時 | RTX 4090 で 8B・128K を 0.188 s/トークン |
| InfLLM〔K〕 | [InfLLM](https://arxiv.org/abs/2402.04617) | NeurIPS 2024 | 代表トークンで上位ブロック ＋ sink ＋ 窓 | ブロック | 推論時 | — |
| SeerAttention | [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](https://arxiv.org/abs/2410.13276) | 2024/25 arXiv | 学習した AttnGate（プールした Q・K） | ブロック | gate のみ学習、主モデルは凍結 | 32K・疎性 90% で FA2 比 5.67 倍 |
| NSA | [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | ACL 2025（Anthology で Best Paper） | 圧縮（32 / 16）＋ 上位 16 ブロック（64）＋ 窓 512 | ブロック 64、GQA 共有 | ネイティブ | 64K・8×A100 で順 9.0 倍・逆 6.0 倍 |
| MoBA | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | 2025 arXiv | ⟨q, 平均キー⟩ で上位 k、現ブロック強制 | ブロック 512（上位 3）/ 4096（上位 12） | ネイティブか継続 | 1M で 6.5 倍、10M で 16 倍 |
| FlashMoBA | [Optimizing Mixture of Block Attention](https://arxiv.org/abs/2511.11571) | 2025 arXiv | MoBA ＋ キーの短い畳み込み、小ブロック | 小さい B | ネイティブ | FA2 比 最大 14.7 倍 |
| DSA | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | 2025 技術報告 | lightning indexer で上位 k | トークン、k = 2048 | 密モデルの継続学習 | 中核 O(L²) → O(Lk) |
| BSFA | [Block Sparse Flash Attention](https://arxiv.org/abs/2512.07011) | 2025 arXiv | 較正した閾値でブロック最大値を判定 | ブロック | 推論時 | 最大 1.24 倍 |
| BLASST | [BLASST](https://arxiv.org/abs/2512.12087) | 2025 arXiv | online softmax 内のブロック刈り込み λ = a/L | ブロック | 推論時（近似） | H200 / B200 で前処理 1.62 倍 |

表の全方式が、選んだ支持の上では厳密な softmax を取る。例外は BLASST で、小さいブロックを飛ばすので近似になる。NSA の会場は Anthology の記載で確認したが、ACL の賞のページが当初これを載せていなかったという掲示板の指摘が未解決で残っている〔S〕（[ACL Anthology 2025.acl-long.1126](https://aclanthology.org/2025.acl-long.1126)、[HN スレッド](https://hnews-reader.vercel.app/jobs/44761548)）。NSA の学習トークン数は本文で 270B、序論で 260B と食い違う（[NSA](https://arxiv.org/html/2502.11089v2)）。

設定値の要点は次のとおり。MoBA は 568M〜2.1B を系列長 8K・ブロック 512・上位 3 で学習した。Llama-3.1-8B の 128K → 1M の継続学習（100B トークン）ではブロック 4096・上位 12 とし、最後の 3 層を全対のまま残した（[MoBA](https://arxiv.org/html/2502.13189v1)）。DSA は二段で学習する。密な warm-up では indexer 以外を凍結し、KL(p_main ‖ softmax(I)) を 1,000 ステップ・2.1B トークン最小化する。疎の段階では全パラメータを 15,000 ステップ・943.7B トークン学習し、KL は選んだ集合の上に限る（[DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)）。

### 設計への含意（§4.2 疎の水準、§2 H4 (e)、§13 項目 7）

1. **学習で選ぶ水準は MoBA 形を主にする。** ゲートは ⟨q, ブロックの平均キー⟩、上位 k、現ブロックを強制し、選んだ支持の上で厳密な softmax を取る。単一の機構でパラメータが増えず、公開カーネルがある。NSA の**選択枝だけ**（ブロック 64、先頭 1・局所 2 を含む上位 n）は副の候補として P7 で比べる。NSA の全体は圧縮枝と窓枝が選択の効果と交絡するので使わない。
2. **先頭ブロックと現ブロックは常に支持に含め、その分も k に数える**（NSA の数え方）。sink の受け皿を確保すると同時に、§2.3 の到達性の条件を満たすためである。
3. **トークン単位の DSA は対象外にする。** 前提が MLA の KV 共有だからである。ただし DSA の学習法（密 warm-up と KL 蒸留）は、ルータが冷えたまま始まる問題の対策として §10 のリスク表に入れる。

---

## 2. 疎な attention は支持上の softmax を厳密に保ち、失うのは選択の誤りである

### 2.1 何を保ち、何を失うか

NSA・MoBA・DSA・Quest は、選んだ支持の上で密な softmax をそのまま正規化し直す。したがって**正しいキーが支持に入っている限り**、argmax 的な読み出しは保たれる。第 2 回報告書は線形 attention の損失を三つの機構にまとめた。一様化、ランク ≤ d、固定状態の検索容量である。疎な attention は支持の選択が当たればこの三つを全て避ける。支持の大きさ k·B は d を超えてよく、KV は O(N) のまま残るからである。その代わり、**採点器が鍵を取り逃がすという四つ目の失敗**を新たに持つ。

この失敗の機構を最も精密に述べたのは FlashMoBA の信号対雑音モデルである。平均プールしたキーでは、1 個の関連キーが B − 1 個の雑音キーに薄まる。このためルータの検索精度は、ブロックの大きさ B が頭の次元 d に比べて大きいほど落ちる。設計原理は d/B 比（小さいブロックほど良い）であり、キーに短い畳み込みを掛けると関連信号がまとまる（[Optimizing Mixture of Block Attention](https://arxiv.org/abs/2511.11571)）。SNR ∝ √(d/2B) という式そのものは二次要約からの引用である〔S〕。MoBA の粒度 ablation もこれと整合する。疎性 75% に固定したとき、最も粗い「8 個中 2 個」の分割は細かい分割より LM 損失が約 1e-2 悪い。さらに、系列末尾のトークン（長距離側）の損失差は平均の損失差より大きい（[MoBA](https://arxiv.org/html/2502.13189v1)）。

固定パターンの失敗はもっと単純で、窓の外は取り戻せない。StreamingLLM は除去した中間トークンを回復できない〔K〕（[StreamingLLM](https://arxiv.org/abs/2309.17453)）。He & Garner の 1.4B では、GDN ＋ SWA の S-NIAH が 62.4 と、素の GDN の 88.9 を下回った（[He & Garner](https://arxiv.org/html/2510.20787)）。

学習の仕方による差は大きい。同じモデルの LongBench では、ネイティブに学習した NSA が 0.469、全対が 0.437、事後の厳密 Top-k が 0.423、Quest 0.392、InfLLM 0.383、H2O 0.303 だった（[NSA](https://arxiv.org/html/2502.11089v2)）。累積 attention で除去する方式が最も悪く、事後の厳密上位 k でも密より約 1.4 点落ちる。学習なしの疎化を最も大規模に調べた The Sparse Frontier（6 方式、最長 128K、疎性 0.95 まで、9 課題）の結論は三つある。第一に、長い系列では大きな高疎性モデルが同じ計算量の小さな密モデルに勝つ。第二に、デコードは前処理より、長い系列は短い系列より高い疎性に耐える。第三に、**中程度の疎性でも少なくとも 1 課題は大きく劣化する**（[The Sparse Frontier](https://arxiv.org/abs/2504.17768)）。学習なしの数字から学習ありの疎モデルを予測してはいけない。

もう一つの弱点は微調整にある。MoBA は SFT で劣った。プロンプト部の損失をマスクすると勾配が疎になるためで、最後の数層を全対に戻すと直った（[MoBA](https://arxiv.org/html/2502.13189v1)）。

### 2.2 線形・SSM との比較: パラメータとトークンを揃えた比較は 1 件だけ

He & Garner は FineWeb-Edu で、0.4B（10B トークン）と 1.4B（30B トークン）の各腕をパラメータとトークンを揃えて学習した。純 NSA、GDN、二種類のハイブリッド（GDN ＋ NSA、GDN ＋ 全対）を Transformer++ と比べている（[Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction](https://arxiv.org/html/2510.20787)、2025 arXiv、会場の記載なし）。

| モデル | RULER S-NIAH 平均 0.4B / 1.4B | EVAPORATE 平均（FDA・SWDE・NQ・SQuAD・TriviaQA・DROP）0.4B / 1.4B | 常識推論平均 0.4B / 1.4B |
|---|---|---|---|
| GDN（線形） | 51.7 / 88.9 | 19.47 / 29.50 | 48.7 / 56.2 |
| GDN ＋ SWA | 54.7 / 62.4 | 22.81 / 32.42 | — |
| laNSA（GDN ＋ NSA） | 74.8 / 77.2 | 25.58 / 33.80 | — |
| GDN ＋ 全対 attention | 80.4 / 86.8 | 27.19 / **37.85** | — |
| **NSA（疎）** | **72.8 / 74.8** | **24.18 / 32.38** | 48.8 / 54.7 |
| Transformer++（softmax） | 75.1 / **91.4** | 27.01 / 34.19 | 49.3 / 54.0 |

読み方は三つある。第一に、純 NSA は両規模で Transformer++ にやや劣り、**規模を上げても S-NIAH がほとんど伸びない**（72.8 → 74.8。Transformer++ は 75.1 → 91.4、GDN は 51.7 → 88.9）。第二に、NSA は 0.4B では GDN を大きく上回るが、1.4B では S-NIAH で逆転される。ただし検索集約課題（EVAPORATE）では両規模とも NSA が上である。第三に、ハイブリッドが最良で、GDN ＋ 全対は 1.4B の EVAPORATE で Transformer++ を超える。著者は線形 attention の限界を「可変長の履歴は固定サイズの状態に可逆に圧縮できない」と述べ、NSA のような問い依存の疎法はこれを避ける代わりに O(N) の KV を要すると整理している。

2026 年の NAtS-L は、NSA・MoBA・GDN ＋ SWA(1024) の疎な変種が全対と同程度の検索性能を持つと報告しているが、数値は抽出していない〔S〕（[Neural Attention Search Linear](https://arxiv.org/pdf/2602.03681)）。

**この比較はパラメータとトークンで揃えたものであり、壁時計やメモリでは揃えていない。壁時計・メモリを揃え、同じカーネル基盤で 疎 対 線形 対 SSM を比べた一次資料は見つからなかった。** NSA・MoBA の論文はどちらも線形・SSM の基準線を持たない（[NSA](https://arxiv.org/html/2502.11089v2)、[MoBA](https://arxiv.org/html/2502.13189v1)）。NSA・MoBA の選んだ支持の上で attention のエントロピーや鋭さを、密な attention と比べて測った研究も無い。

### 2.3 理論: 保証は固定パターンの到達性についてだけある

Yun らは、各トークンが自分を見ること、ハミルトン路のような連鎖を含むこと、p 層で全トークンに到達できることを条件に、層あたり O(n) 本の接続を持つ疎 Transformer が連続な系列対系列関数の普遍近似子になることを示した〔K〕（[O(n) Connections are Expressive Enough](https://arxiv.org/abs/2006.04862)）。BigBird は O(1) 個の大域トークンで普遍近似性とチューリング完全性を示した。その裏面として、全対なら O(1) 層で解ける課題に、Õ(n) 辺の疎 attention では Ω̃(n) 層を要するものがあることも示している〔K〕（[Big Bird](https://arxiv.org/abs/2007.14062)）。Exphormer はエキスパンダと仮想大域ノードでグラフ版の結果を与えた〔K〕（[Exphormer](https://arxiv.org/abs/2303.06147)）。**内容依存の上位 k（NSA / MoBA / DSA）の表現力の結果は見つからなかった。** 証明はいずれも入力に依らない固定パターンを仮定しており、学習で選ぶ支持の保証は経験的なものしかない。

### 設計への含意（§3.1 鋭さ、§2 H4 (e)、§1 事実表）

1. **§3.1 に「支持再現率（support recall）」を足す。** 正解キーの集合 G（needle の区間、構造上の目標ノード、整合先トークン）に対して、|G ∩ 選択支持| / |G| をクエリごとに測り、下流の精度と並べて報告する。これで疎の腕の失敗を「選択の誤り」（再現率が低い）と「読み出しの誤り」（再現率は高いのに外れる）に分解できる。文献はこの分離をほとんど報告していない。疎の水準では、Ĥ を log(|選択支持|) で正規化した版も併記する。
2. **H4 (e) の予測を書き足す。** 「(e) の検索の損失は主に支持再現率の不足から来る。ブロックが粗いほど（d/B が小さいほど）、また鍵が単一トークンであるほど大きい」。これは FlashMoBA と MoBA の ablation から直接出る予測で、事前登録で検定できる。
3. **§1 の事実表に 2 行足す。** 「疎な attention は支持上の softmax を厳密に保ち、失敗は選択の誤りとして現れる」と、「パラメータ・トークンを揃えた唯一の比較（0.4B / 1.4B）では純 NSA は Transformer++ にやや劣り、GDN との上下は規模で入れ替わる。壁時計・メモリを揃えた比較は存在しない」（出典は He & Garner と NSA）。
4. **疎の列に「固定」の水準を床として置く。** 窓 ＋ 先頭ブロックで、内容にも構造にも依らない支持である。学習選択と構造選択の利得をこの床から測ると、窓の効果を選択の効果と取り違えずに済む。

---

## 3. 実カーネルでは 32K 超・疎性 75% 超でしか速くならず、学習時の KV は減らない

### 3.1 公表された速度

| 方式 | 条件 | 速度 | GPU・実装 | 出典 |
|---|---|---|---|---|
| NSA | 64K、ブロック 64 | 順伝播 9.0 倍・逆伝播 6.0 倍（Triton FA2 比）。短い系列では小さい（図のみ） | 8×A100、Triton | [NSA](https://arxiv.org/html/2502.11089v2) |
| NSA デコード | 8K / 16K / 32K / 64K | メモリアクセス量からの**予測**で 4 / 6.4 / 9.1 / 11.6 倍。実測ではない | — | [NSA](https://arxiv.org/html/2502.11089v2) |
| MoBA | 1M 前処理 / 10M（疎性 95.31%） | attention 部分が 6.5 倍 / 16 倍 | GPU の種類は記載なし。FA varlen | [MoBA](https://arxiv.org/html/2502.13189v1) |
| FlashMoBA | 小ブロック | FA2 比 最大 14.7 倍 | CUDA | [FlashMoBA](https://arxiv.org/abs/2511.11571) |
| SeerAttention | 32K、ブロック疎性 90% | FA2 比 5.67 倍（128K で 7.3 倍という報告は〔S〕） | ブロック疎 FA | [SeerAttention](https://arxiv.org/abs/2410.13276)、[review](https://www.themoonlight.io/en/review/seerattention-learning-intrinsic-sparse-attention-in-your-llms) |
| BLASST | 疎性 74.7% / 73.2% | 前処理 1.62 倍 / デコード 1.48 倍 | H200 / B200 | [BLASST](https://arxiv.org/abs/2512.12087) |
| BSFA | 計算の約 50% を省略 | 端から端まで 1.10〜1.24 倍（Llama-3.1-8B、精度 99% 超） | 記載なし | [BSFA](https://arxiv.org/abs/2512.07011) |
| FlexAttention（密マスク） | 因果 | FA2 の順伝播 90%・逆伝播 85% | A100 | [FlexAttention blog](https://pytorch.org/blog/flexattention/) |
| Mistral SWA〔K〕 | 16K、W = 4096 | 約 2 倍 | FA ＋ xFormers | [Mistral 7B](https://arxiv.org/abs/2310.06825) |
| MInference〔K〕 | 1M 前処理 | 最大 10 倍 | A100 1 枚 | [MInference](https://arxiv.org/abs/2407.02490) |
| RetrievalAttention〔K〕 | 8B、128K | 0.188 s/トークン | RTX 4090（24 GB）、CPU の ANN | [RetrievalAttention](https://arxiv.org/abs/2409.10516) |
| DSA | 128K | 中核 O(Lk) だが indexer は O(L²)（少数ヘッド・ReLU・FP8 で安い）。図 3 の数値は抽出できず | H800 | [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1) |

**ブロックの大きさは 64〜128 に集まる**（NSA 64、BigBird 64、FlexAttention の既定 128）。テンソルコアと SRAM のタイルに合うからである。MoBA の 512〜4096 はカーネルには安いがルータの精度を落とす（§2.1）。小さいブロックは融合カーネルを書いて初めて速くなる（[FlashMoBA](https://arxiv.org/abs/2511.11571)）。NSA は GQA グループの全クエリをまとめて読み、共有する疎な KV ブロックを順に SRAM に流す。FlashAttention 流にクエリをブロックで分割すると、疎な選択の下では読み出しが散らばるからである（[NSA](https://arxiv.org/html/2502.11089v2)）。

**理想の速度 1/(1 − 疎性) は上限にすぎない。** BLASST の 1.62 倍は、疎性 74.7% の理想 3.95 倍の約 41% にとどまる。BSFA は計算を半分省いても端から端まで 1.10〜1.24 倍である（後者は attention 以外の時間も含む）。採点と不規則なメモリアクセスの費用が差を食う。NSA 自身も短い系列では効率を引き出せないと認めている（[BLASST](https://arxiv.org/abs/2512.12087)、[BSFA](https://arxiv.org/abs/2512.07011)、[NSA](https://arxiv.org/html/2502.11089v2)）。疎性と GPU を与えたとき FlexAttention のブロック疎が FA2 の密な因果計算に勝つ閾値を示した一次資料は見つからなかった。**RTX 3090 / 4090 でブロック疎と FA2 を比べた測定も見つからなかった。** FA3 は Hopper 専用なので、3090 での基準は FA2 になる。

### 3.2 FlexAttention の BlockMask の費用構造

第 2 回報告書 §5.1 は、FlexAttention がブロック単位の疎性が無いと速くならないこと、要素ごとのマスクに 15〜20% の費用がかかることをすでに示した。今回のノートは、**データ依存のマスク**に固有の費用を足す。BlockMask は `mask_mod` から `create_block_mask` で作る。既定の BLOCK_SIZE は 128 で、空のブロックは飛ばし、満ちたブロックはマスクの適用を省き、部分ブロックだけマスクを掛ける。Mistral 形の窓（1024）を BlockMask で書くと、疎性が高いので FA2 の因果計算より「かなり速い」。メタデータは [B, H, Q_LEN/128, KV_LEN/128] の大きさで、1M トークンで約 60 MB、BLOCK_SIZE = 1024 なら 1 MB 未満になる。BlockMask の再計算は「数百マイクロ秒」かかり（再コンパイルなら秒単位）、`_compile=True` にすると実行時間とメモリが「しばしば一桁」減る。そして**層ごと・入力ごとに変わるデータ依存のマスクは、構築費用を償却できないので「最も難しい設定」**とブログ自身が明記している。執筆時点の制約は三つあった。H100 で FA3 の水準に届かないこと、系列長が 128 の倍数であること、GQA 対応が未完であることである（[FlexAttention blog](https://pytorch.org/blog/flexattention/)）。FlexAttention 論文の詳細は今回再取得していない〔K〕（[FlexAttention: A Programming Model for Generating Optimized Attention Kernels](https://arxiv.org/abs/2412.05496)）。

### 3.3 メモリ: 疎は KV を減らさない

学習時のブロック疎 attention は、全 KV を保持したまま計算と活性の転送量だけを減らす（He & Garner の言う O(N) の KV）。線形 attention はヘッドあたり O(d²) の状態で済む（[He & Garner](https://arxiv.org/html/2510.20787)）。元の FlashAttention のブロック疎版は、非零ブロックの割合に比例する IO 量を持つ〔K〕（[FlashAttention](https://arxiv.org/abs/2205.14135)）。したがって「同じメモリ」とは、KV / 状態のバイト数を揃えるのか、ピークの活性メモリを揃えるのかを明示しない限り定義できない。

### 設計への含意（§4.2、§4.4、§4.6、§6.2、P8）

1. **実装は FlexAttention の BlockMask にし、学習選択のマスクは上位 k の添字から直接組む。** `create_block_mask` を経由せず、kv_num_blocks / kv_indices のテンソルを作る。構築の時間は壁時計に含め、別の列でも記録する。BLOCK_SIZE は 128（ネイティブ）を主にし、64 は自前の構築で副とする。
2. **§4.4 / §4.6 の「揃える規則」を疎については変える。** (e) と疎の列は、壁時計と**ピークの活性メモリ**で揃える。KV / 状態のバイト数は報告するが揃えない（疎は O(N·d)、線形は O(d²) で、原理的に揃えられない）。このことを事前登録に明記する。
3. **P8 に疎の腕を加える。** 3090 の格子で疎性 75% / 87.5% / 93.75% を測り、ブロック疎が FA2 に勝つ系列長と疎性を自分で求める。文献に無い数字であり、それ自体が結果になる。
4. **予想を事前に書く。** 融合層の長さ（16K 以下）でブロック 64〜128 なら、疎の腕は FA2 に壁時計で勝てない可能性が高い（NSA の短系列の注記、BLASST・BSFA の効率）。その場合、壁時計を揃えるための k は大きくなる（疎性が低くなる）。これを (e) にとって不利な条件として報告する。

---

## 4. 構造で疎な支持を選んだ研究は SASA が最も近く、同じ k での比較はまだ無い

### 4.1 既存の研究

**SASA**（[Understanding Long Programming Languages with Structure-Aware Sparse Attention](https://arxiv.org/abs/2205.13730)、SIGIR 2022 short、[会場記録](https://pure.ecnu.edu.cn/en/publications/understanding-long-programming-languages-with-structure-aware-spa/)）は、BigBird 形のブロック単位（b = 32）で四つの成分を合わせる。窓 3 ブロック、大域キー 2 ブロック、上位 k = 3 ブロック、AST ブロック attention である。上位 k の 3 ブロックは、CodeSearchNet で事前に集計した attention 頻度表から選ぶ。入力ごとに学習した選択ではなく統計である。AST ブロック attention は、tree-sitter の AST を隣接行列にしてブロック化し、AST 得点の和で上位 k ブロックを選ぶ。どの AST 関係を数えるかと、融合したブロック疎カーネルを使うかは論文に書かれていない。骨格は長さ 1024 の CodeBERT で、メモリを 23〜33% 節約し、32 GB の V100 でバッチ 8 のとき 7 GB 超を節約する。遅延の数値は無い（[ar5iv](https://ar5iv.labs.arxiv.org/html/2205.13730)）。

| モデル | BCB F1 | Defect Acc | Search MRR | Summ BLEU-4 |
|---|---|---|---|---|
| **SASA** | **0.911** | **57.44** | **46.36** | 12.88 |
| BigBird | 0.895 | 56.79 | 41.89 | 12.42 |
| Longformer | 0.820 | 56.90 | 44.09 | 12.20 |
| GraphCodeBERT | 0.871 | 55.39 | 43.13 | **12.93** |
| CodeBERT | 0.789 | 55.71 | 38.37 | 12.26 |
| SASA − 上位 k | 0.828（−0.083） | −0.6 | — | — |
| SASA − AST | 0.874（−0.037） | −0.22 | — | — |

同じモデルの中で同じ k = 3 の二成分を外した ablation では、**データから選んだ上位 k ブロックの方が AST ブロックより効いている**（[ar5iv](https://ar5iv.labs.arxiv.org/html/2205.13730)）。

ほかの例は次のとおりである。LongCoder は窓に加え、一定間隔の「橋」トークンと、import・クラス定義・関数定義のような長距離に重要な文に置く「記憶」トークンを持つ。構文で大域支持を選ぶ例である〔K〕（[LongCoder](https://arxiv.org/abs/2306.14893)）。Exphormer は局所のグラフ辺、エキスパンダ（ランダム d 正則グラフ）、仮想大域ノードを支持とする〔K〕（[Exphormer](https://arxiv.org/abs/2303.06147)）。GraphGPS は大域 attention の選択肢に BigBird（疎）と Performer（線形）を持つ〔K〕（[GraphGPS](https://arxiv.org/abs/2205.12454)）。ETC は大域トークンを文書構造に結びつけられる〔K〕（[ETC](https://arxiv.org/abs/2004.08483)）。

NLP の構文誘導 attention はいずれも**密なマスク**であり、ブロック疎カーネルを使わない。SG-Net（依存木の祖先に限る）、Syntax-Guided Transformers、Dependency Transformer Grammars、Tree Transformer がこれにあたり、Tree Transformer は後に構成素の模型として無効だと示された。いずれも要旨の水準で確認した（[SG-Net](https://arxiv.org/abs/2012.13915)、[Syntax-Guided Transformers](https://arxiv.org/abs/2311.04364)、[Dependency Transformer Grammars](https://arxiv.org/abs/2407.17406)、[Tree Transformer の再検討](https://arxiv.org/html/2411.16993v1)）。GraphCodeBERT のデータフロー誘導マスクも、長さ 512 の密なマスクである〔K〕（[GraphCodeBERT](https://arxiv.org/abs/2009.08366)）。

### 4.2 判定: 「同じ k での学習選択 対 構造選択」は比べられていない

**別々に学習した二つの腕として、学習による選択と構造による選択を同じ k で比べた研究は見つからなかった。** 近いものはそれぞれ次の点で足りない。SASA は両者を 1 つのモデルに混ぜ、k = 3 ブロックずつに固定して ablation しただけで、しかも学習側は入力ごとの採点ではなくコーパスの頻度表である。LongCoder は同数の大域トークンを持つ学習上位 k ではなく、他の疎な基準線と比べている。Exphormer はエキスパンダを他のグラフ Transformer と比べており、学習によるルーティングとは比べていない。この判定には但し書きがつく。Semantic Scholar の被引用の走査は行っていない。2024〜2026 年の、規模のあるコード LLM での構造誘導の疎 attention も見つからなかった。AST / 依存構造を使う融合ブロック疎カーネルをデコーダ LLM で使った例も無い。

第 2 回報告書は「構文誘導 attention は全て二次の softmax」と書いたが、これは一点だけ精密にできる。**エンコーダには構造で選ぶブロック疎の支持がある（SASA）。線形 attention・SSM・デコーダ LLM には無い。**

### 設計への含意（§4.3 疎の列、§2 H4 (e)、§11）

1. **「構造で選ぶ」を SASA 形に定義する。** 学習選択と同じ B と k を使い、各ブロックの得点を、クエリのブロックからそのブロックへの構造得点の和とする。構造得点には、設計書がすでに計算する打ち切り冪級数 Σ α_k W^k（または RRWP）の質量を使う。選んだ上位 k の上で厳密な softmax を取る。
2. **「併用」は和集合ではなく予算の分割にする**（学習 k/2 ＋ 構造 k/2）。和集合にすると支持が最大 2k になり、同じコストの比較が崩れる。
3. **学習側は入力ごとに採点する（MoBA のゲート）。** SASA の頻度表は使わない。使うと SASA の弱点をそのまま引き継ぐ。
4. **窓を越えて増えた支持を報告する。** AST や依存の隣接はトークン順で局所的になりやすい（親子の範囲が重なる）ので、構造で選んだブロックは窓のブロックとかなり重なる。「構造で選んだブロックのうち窓の外にある割合」を記録し、窓の効果を構造の効果と取り違えないようにする。
5. **SASA から出る予想を事前登録に書く。** 導出可能型（POJ-104 など）では学習選択が構造選択と同等以上になる見込みである。構造選択が勝つと予想できるのは必要型（合成グラフ、変数誤用）であり、これは H2 の予測表を疎の列に写したものになる。
6. **§11 の「疎な attention」の行を埋める**（§9 の表）。

---

## 5. 事前学習 Transformer のヘッド分析: 検索ヘッドは 3〜6%、指標は argmax と 1 列で足りる

### 5.1 同定の方法と式

| 指標 | 定義 | 設定・閾値 | 因果の検証 | 出典 |
|---|---|---|---|---|
| **検索スコア**（Wu ら） | \|g_h ∩ k\| / \|k\|。k は needle のトークン集合、g_h はヘッド h が copy-paste した needle トークンの集合。1 回の copy-paste は、生成トークン w が needle 内にあり、かつ h の **argmax attention** の入力位置が needle 区間内で同じトークンを持つこと | 0.1 以上なら検索ヘッド。needle 組 3 × 長さ 20 点（1K〜50K の等間隔）× 深さ 10 ≈ 600 例。0.1 を超えるのは全ヘッドの約 3〜6%。族内の相関 r > 0.8、族間 r < 0.1 | 上位 K = 50（全ヘッドの約 5%）をマスクすると全モデルで NIAH が 50 未満。抽出型 QA の F1 は −9.2% / −23.1%。ランダムなヘッドのマスクは小さい（図 7 のみ）。CoT は大きく損なうが直接回答はほぼ影響なし | [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/html/2404.15574v1)（ICLR 2025、[proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Conference.pdf)） |
| **最適化ゲート**（DuoAttention） | KV ヘッドごとに α ∈ [0, 1]、出力 = α·全 attention ＋ (1 − α)·ストリーミング（sink ＋ 直近）。損失 = passkey の最後の l トークンでの最終隠れ状態の L2 距離 ＋ λ·L1(α)、λ = 0.05 | 32 トークンの passkey 10 個を BookSum に挿入、長さ 1K〜最大を 50 区間、識別時の窓は sink 128 ＋ 直近 256、AdamW 2,000 ステップ、学習率 0.02 → 0.002。分位閾値 τ で二値化。検索ヘッドの比率は MHA で 25%、GQA で 50% | その比率で 1,048K まで NIAH が全対と一致。attention プロファイルで選ぶより良い。Wu の検索スコアとは直接比べていない | [DuoAttention](https://arxiv.org/html/2410.10819) |
| **帰納スコア** | `[BOS] + r + r`（r は長さ n のランダムトークン列）で、2 回目の位置 i から i − n + 1 への attention の平均〔K〕。定義は「前置一致」と「複写」の二条件で、数値の閾値は無い | 長さ 512 の文脈。ICL スコア = トークン 500 での損失 − トークン 50 での平均損失 | 34 モデル、5 万回超のヘッド ablation。相転移は 2 層以上でのみ現れ、帰納ヘッドの形成と一致 | [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)、[TransformerLens](https://transformerlensorg.github.io/TransformerLens/) |
| **echo スコア** | 現トークンの前回の出現位置への attention（帰納は前回の出現の次のトークン） | 4 回繰り返したランダムトークン。echo 上位 1%・帰納上位 14% に全キャッシュを残す〔U〕 | KV を最大約 70% 削減〔U〕 | [RazorAttention](https://arxiv.org/abs/2407.15891)、[HF papers](https://huggingface.co/papers/2407.15891) |
| **sink 率**（Gu ら） | α_k^{l,h} = (1/(T − k + 1)) Σ_{i=k}^{T} A_{i,k}^{l,h}。Sink_k^ε = (1/L) Σ_l (1/H) Σ_h 𝟙(α_k^{l,h} > ε) | ε = 0.3（厳しく T に鈍い）。Barbero らは LLaMA 3.1 で ε = 0.8 | 学習可能な KV バイアスが sink を吸収する。正規化しない sigmoid attention は 1B まで sink を作らない | [When Attention Sink Emerges in Language Models](https://arxiv.org/html/2410.10781v2)、[Why do LLMs attend to the first token?](https://arxiv.org/html/2504.02732v4) |
| **問い焦点の質量**（QRHead） | 少数の実課題例で、問いトークンから正解文脈への attention 質量を集計して選んだヘッドの質量 | — | 検索器・再順位づけとして使える（下記） | [Query-Focused Retrieval Heads](https://arxiv.org/abs/2506.09944) |
| 信頼度（Voita ら）〔K〕 | トークン平均の最大 attention 重み（top-1 質量の祖先） | Hard-Concrete ゲート ＋ L0 | WMT En–Ru で 48 個中 38 個のエンコーダヘッドを除いて BLEU −0.15 | [Analyzing Multi-Head Self-Attention](https://aclanthology.org/P19-1580/) |
| 重要度（Michel ら）〔K〕 | I_h = E_x \|∂L(x)/∂ξ_h\| | 層ごとに正規化 | 約 20%（WMT）〜40%（BERT）のヘッドを除いても大きくは落ちない | [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) |
| ヘッドのエントロピー（Clark ら）〔K〕 | トークン平均の attention エントロピー、特殊トークンへの質量 | BERT-base | [SEP] への attention は「no-op」 | [What Does BERT Look At?](https://arxiv.org/abs/1906.04341) |
| 位置バイアスの較正 | 観測した attention = 関連度 ＋ 位置バイアス。各位置にダミー文書を置いてバイアスを推定し、差し引く | — | RAG で最大 +15 pt | [Found in the Middle](https://arxiv.org/html/2406.16008v2) |

Gu らの Sink₁（%）を入力の種類ごとに見ると、自然文 / ランダムトークン / 繰り返しトークンの順で、GPT2-XL が 77.00 / 70.29 / 62.28、Mistral-7B が 97.49 / 75.21 / **0.00**、LLaMA2-7B 基盤が 92.47 / 90.13 / **0.00**、LLaMA3-8B 基盤が 99.02 / 91.23 / **0.00** である。sink は Pythia-14M にも現れる。最初のトークンは隠れ状態のノルムが巨大だが、キーと値のノルムは小さい（[Gu et al.](https://arxiv.org/html/2410.10781v2)）。つまり、帰納・echo スコアに使う繰り返しランダムトークンの探針では、Llama / Mistral の sink は消える。**sink 質量は探針の種類で変わる**ので、帰納の探針と NIAH の探針で別々に記録する必要がある。巨大活性（中央値の約 10⁴ 倍）が sink に結びつくという指摘もある〔K〕（[Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)）。

### 5.2 検索の失敗と内部指標の関係

成否で条件づけた証拠は四つあり、いずれも部分的である。

**検索ヘッドの不活性。** NIAH で幻覚を起こすとき、検索ヘッドは部分的にしか働いていないか、働いていない。弱い検索ヘッドも needle の一部を保持する。この点は ICLR 版の要約経由で確認した（[Wu et al., ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Conference.pdf)）。

**位置バイアス。** 文書の先頭と末尾は関連度に関係なく多くの attention を受ける（U 字）。位置バイアスを差し引いた attention は正解文書をより正しく順位づける（[Found in the Middle](https://arxiv.org/html/2406.16008v2)）。生の attention は、偏ってはいるが情報を持つ信号である。

**質量は検索器になる。** 選んだヘッドの、問いから候補文書への質量の累積は、零ショットの検索器・再順位づけとして働く。BEIR で RankGPT を上回り、LongMemEval・CLIPPER で全文脈より 10% 超良い。copy-paste の得点は意味的・言い換えの検索ヘッドを取りこぼす、と著者は指摘する（[QRHead](https://arxiv.org/abs/2506.09944)）。

**sink は害ではなく必要である。** Gemma 7B から BOS を除くと、ARC-e が 80.77 → 28.49、**RULER-4K が 82.57 → 0.00** に落ちる。固定 BOS で事前学習したモデルでは、BOS を除くと sink 指標が 90.84% → 0.05%、検証損失が 2.69 → 7.56 になる。sink は規模とともに強まり（8B 45.97、70B 73.49、405B 78.29）、**事前学習の文脈長とともに強まる**。文脈長 128 ではほぼ現れず、5B トークンの固定予算では文脈長に応じて増えて飽和する（[Barbero et al.](https://arxiv.org/html/2504.02732v4)。arXiv に会場の記載なし、COLM 2025 との情報は〔U〕）。

分散の理論は、長さとともに最大重みが落ちることを予言する。有界ロジットの softmax は項目数が増えると分散し、適応温度で緩和できる（[Softmax is not Enough](https://arxiv.org/html/2410.01104v3)）。SSMax の s·log n 倍のロジットは鍵情報の検索を改善する（要旨と二次要約で確認。[Scalable-Softmax](https://arxiv.org/html/2501.19399v1)）。ASEntmax は学習長の 8 倍で softmax・SSMax・固定 α-entmax より良く検索する（[Long-Context Generalization with Sparse Attention](https://arxiv.org/abs/2506.16640)）。成否の条件づけに近い未検証の仕事として、隠れ状態は答えを誤っても目標位置を符号化しているという探針の研究〔U〕（[Insights into LLM Long-Context Failures](https://arxiv.org/abs/2406.14673)）と、受けた attention で文書を並べ替える Attention Sorting〔U〕（[Attention Sorting](https://arxiv.org/abs/2310.01427)）がある。エントロピーそのものを扱う未検証の仕事も三つある。並列文脈符号化での問いトークンのエントロピー上昇（[Attention Entropy is a Key Factor](https://arxiv.org/html/2412.16545)）〔U〕、エントロピーの振る舞いが「硬い / 動く」に分かれるヘッド（[From Rigid to Dynamic](https://arxiv.org/pdf/2606.09508)）〔U〕、QK-norm の有無でエントロピーと sink がまとまるという報告（[Cracks in the Foundation](https://arxiv.org/pdf/2608.10296)）〔U〕である。

**ヘッドごとのエントロピーや needle 質量が NIAH の正答を予測する力（AUROC など）を長さ別に報告した研究は見つからなかった。** 「集中が正答を予測し、その予測力が長さとともに弱まる」という主張はもっともらしいが、確立していない。正答・誤答で needle への質量を比べる 2025 年の専用論文を探した検索も空振りだった。ここが H5 の空白である。第 2 回報告書 §3.2 は sink を「中身のない鋭さ」として除外する対象と書いた。新しいノートはこれを一歩進める。sink は**検索の成功に因果的に必要**であり、学習文脈長とともに強まる。したがって、除外するだけでなく**共変量として**モデルに入れなければならない。sink に質量が行ったせいでエントロピーが低いことと、haystack に分散したせいでエントロピーが高いことは、別の現象だからである。

### 5.3 16K を 24 GB で測る手順

長さ 16,384 の attention 行列 1 枚は 2.68×10⁸ 要素あり、**1 ヘッドで fp32 なら約 1.07 GB、bf16 なら約 0.54 GB** になる。32 ヘッドの層 1 枚で 34 GB / 17 GB なので、7B の bf16 重み（約 14〜15 GB）の横で `output_attentions=True` を 1 層分でも使うことはできない。4K ならこの 16 分の 1（bf16 で層あたり約 1 GB）で、層ごとに回せる（ノートの算術）。FlashAttention のインタフェースは出力と `softmax_lse`（クエリごとの log-sum-exp）を返す。全 softmax 行列を返す `return_attn_probs` はテスト用である（[flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)）。LSE だけではエントロピーは出ない。16K 以上のヘッドごとのエントロピーを LSE で計算する、保守されたツールも見つからなかった。

ノートが組み立てた手順は次のとおりである（単一の出典に基づく確立手順ではなく、推論による提案）。

**ロジットの再計算。** 各 attention モジュールにフックを掛け、RoPE 適用後の Q と K / V を取る（GQA なら展開する）。クエリを b = 256〜1,024 のブロックに分け、因果マスクつきで S = QKᵀ/√d を fp32 で計算し、p = softmax(S) を作る。

**蓄積する量。** 正規化エントロピー H̃_i = H(p_i)/log(i)（因果の行長 i で割る。N では割らない）、top-k 質量、sink 質量 p_{i,1}（または先頭 s トークンへの質量）、needle 質量 Σ_{j∈needle} p_{i,j} の四つを蓄積する。必要なメモリは b × N × H × 4 バイトで、512 × 16K × 32 × 4 ≈ 1.07 GB になる。

**ストリーミングのエントロピー。** L_i = logsumexp_j s_ij（= `softmax_lse`）を使うと、H_i = L_i − Σ_j p_ij s_ij と書ける。2 回目のキーブロックの走査では Σ exp(s − L)·s だけを蓄積すればよく、メモリは O(N·d) で済む。top-k 質量は、ブロックごとの上位 k を逐次併合して求める。

**クエリの標本。** 決定的なクエリは、問いの最後のトークンと回答生成の最初の数ステップである。これに背景用のランダムな haystack クエリ約 256 個を加える。Wu の得点自体もデコード時のクエリしか使わない（[Wu et al.](https://arxiv.org/html/2404.15574v1)）。

**実効ランクと GQA。** 実効ランクは、b × N のブロックか、標本クエリでのヘッド出力に、ランダム化 SVD を掛けて求める。DuoAttention は KV ヘッド単位、Wu は クエリヘッド単位なので、両方の粒度で報告する（[DuoAttention](https://arxiv.org/html/2410.10819)）。

道具については、TransformerLens の `hook_pattern` / `hook_attn_scores` が B×H×N×N を露出し、約 7B までが想定である〔K〕（[TransformerLens](https://transformerlensorg.github.io/TransformerLens/)）。nnsight は任意のモジュール出力（`q_proj`、`k_proj`）を保存できる〔K〕（[nnsight](https://nnsight.net/)）。検出の参照コードとして [Retrieval_Head](https://github.com/nightdessert/Retrieval_Head) と [duo-attention](https://github.com/mit-han-lab/duo-attention) がある。

**課題の生成と落とし穴。** RULER の S-NIAH / MK-NIAH は「One of the special magic {type_needle_k} for {key} is: {value}.」の雛形を使い、値は 7 桁の数・単語・UUID、haystack は繰り返しの雑音文か Paul Graham の随筆である。実効長は Llama-2-7B の 4K の得点 85.6% を超える最大長と定義される〔K〕（[RULER](https://arxiv.org/abs/2404.06654)、[コード](https://github.com/NVIDIA/RULER)）。passkey は 5 桁の鍵を繰り返しの充填文に埋める〔K〕（[Landmark](https://arxiv.org/abs/2305.16300)）。MQAR〔K〕（[Zoology](https://arxiv.org/abs/2312.04927)）と電話帳〔K〕（[Repeat After Me](https://arxiv.org/abs/2402.01032)）は小型モデル向けである。落とし穴は五つある。U 字の位置バイアス（[Found in the Middle](https://arxiv.org/html/2406.16008v2)）、copy-paste ヘッドの意味的検索への移りにくさ（[QRHead](https://arxiv.org/abs/2506.09944)）、BOS の欠落による崩壊（[Barbero et al.](https://arxiv.org/html/2504.02732v4)）、argmax 型の得点の局在化の人工物〔U〕（[RoPE retrieval heads](https://arxiv.org/pdf/2606.21249)）、そしてトークナイザの差である。最後の点については、7 桁の数のトークン化が Llama-2・Llama-3・Qwen で大きく異なり、Wu の |k| が族ごとに変わるとノートは推論しているが、出典での確認はない。

### 設計への含意（§6.5、§3.1、§2 H5）

1. **§6.5 のヘッドのラベルを三つに固定する。** 長さに依らない帰納 ＋ echo スコア（繰り返しランダムトークン）、分析の長さ格子上での Wu の検索スコア（閾値 0.1）、Gu の sink α₁^{l,h}（ε = 0.3、T と系列数を明記、Llama-3.1 系なら ε = 0.8 も）である。加えて、同じ K で Wu・QRHead・DuoAttention のヘッド集合の重なり（Jaccard）を報告する。安く得られる貢献になる。
2. **長さ格子は Wu の 20 長さ × 10 深さ × 3 needle ではなく、5 長さ × 10 深さ × 20 needle 以上にする。** AUROC を出すには、セルごとに十分な数の失敗が要る。
3. **主分析は「ヘッドごとの指標が正答を予測する AUROC の、log L に対する傾き」とする。** 指標は、回答の最初のステップでの needle 質量、問いの最後のトークンの Ĥ、top-k 質量、sink 質量の四つである。深さで層別し、sink 質量を共変量にし、妨害 needle への質量を別に記録する（「分散による失敗」と「別の目標を選んだ失敗」を分けるため）。これは文献に無い分析である（§8）。
4. **因果の確認。** 各長さで、上位 K 個の検索ヘッドのマスクとランダムな K 個のマスクを比べる（Wu は K = 50、規模に応じて全ヘッドの約 5%）。
5. **`scripts/analyze_pretrained_attention.py` の実装方針。** 順伝播は SDPA で行い、フックで取った Q / K からクエリブロック単位で再計算する。ストリーミングのエントロピーの恒等式を使う。BOS の有無は manifest に記録し、全条件で揃える。課題は**トークン長**で生成し、needle のトークン幅をトークナイザごとに記録し、族をまたぐ比較では単語か UUID の値を使う。

---

## 6. SSM・線形モデルの暗黙の attention は符号つきで和が 1 にならないので、|α| の L1 正規化を主にする

### 6.1 取り出し方

選択的 SSM の暗黙の attention は Ali・Zimerman・Wolf が定義した。α̃_{i,j} = C_i (∏_{k=j+1}^{i} Ā_k) B̄_j で、離散化を展開すると S_C(x_i)·exp(Σ_{k=j+1}^{i} softplus(S_Δ(x_k)))·A·softplus(S_Δ(x_j))·S_B(x_j) になる。各チャネル d は状態座標ごとに N 個の行列を持ち、それを合計してチャネルあたり L×L の行列 1 枚にする。層あたり D 枚になる。著者はこれを Transformer の層の約 D·N/H ≈ 100N 倍の枚数と見積もるが、序論では「三桁」と書いており、両者は整合していない。可視化では Transformer の行に softmax を、Mamba には**絶対値の min-max 正規化**を掛け、和が 1 になることも非負であることも主張しない（[The Hidden Attention of Mamba Models](https://arxiv.org/html/2403.01590v2)、会場の記載なし）。

Zimerman らはこれを、conv1d・ゲート・正規化を含む層全体の作用素 Y = H·X に広げた。Mamba は H = W′·α̂·Z′·M、Mamba-2 は H = ℕ·W′·α̂·Z′·M（ℕ は GroupRMSNorm を対角化したもの）で、Griffin・RWKV・RetNet・HGRN にも同じ形を与えた。忠実度の ablation では、セグメンテーション mIoU が全体で 62.51、Conv1D なしで 50.64、ゲートなしで 55.78、**S6 のみで 53.19** だった。S6 以外の部品が忠実度に効いている（[Explaining Modern Gated-Linear RNNs via a Unified Implicit Attention Formulation](https://arxiv.org/html/2405.16504v2)。ICLR 2025 との情報は未確認）。Mamba-2 では SSD の二次形 L ⊙ (CBᵀ) がヘッドごとに同じ対象を与える（[Transformers are SSMs](https://arxiv.org/abs/2405.21060)）。HF `transformers` の純 PyTorch 経路は、チャンク内の減衰マスク `L = torch.exp(segment_sum(A))` と、チャンク間の減衰 `decay_chunk` を明示的に作る（[modeling_mamba2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)）。

**長さ別に測った唯一の先例は DeciMamba である。** 各行の絶対値を L1 正規化して分布にし（Eq. 7: N(x)_j = |x_j| / Σ_k |x_k|）、最後のトークンからの期待距離「Mamba Mean Distance」Σ_j N(α)_{L,j}·(L − j) を WikiText-103 の 100 例で測った。隠れた attention は 2K では密だが、16K（学習長の 8 倍）では疎になり、下の行に向かって成分が消える。生の平均距離は長さとともに伸びるが、**長さで正規化した文脈の利用率は「劇的に」落ちる**。数表は無い（[DeciMamba](https://arxiv.org/html/2406.14528v3)。ICLR 2025 との情報は未確認）。2025〜2026 年に、暗黙の写像の正規化エントロピー・top-k 質量・sink 質量・実効ランクを 1K〜16K の長さの関数として報告した論文は見つからなかった。関連する仕事は三つある。SSM の強い新近性バイアスと深さによる過平滑化を証明した [Understanding and Mitigating Bottlenecks of State Space Models](https://arxiv.org/abs/2501.00658)（ICLR 2025）、LRP による代替の帰属法である [MambaLRP](https://arxiv.org/abs/2406.07592)、そして大域チャネルの受容野が学習長を超えると縮むという [LongMamba](https://arxiv.org/pdf/2504.16053)〔U〕である。

### 6.2 機構の知見: 検索は少数の Gather-and-Aggregate ヘッドに集まる

Bick・Xing・Gu は、Transformer でも SSM でも、検索が少数の Gather-and-Aggregate（G&A）ヘッドを通ることを示した。Llama-3.1-8B を刈り込んだ最小モデルでは、L16H22（Gather）と L17H24（Aggregate）を残すと MMLU が 66.32% になる。どちらか一方を除くと 24.36% / 25.59%（偶然の水準）に落ち、KV 検索（20 対）は 85.6% から 76% / 62% に落ちる。Zamba2-7B（Mamba-2 と attention が 6:1）では G&A が 9 個の attention ヘッド（L47H{2,4,6,8,18,21,25}、L59H{17,21}）にあり、これを ablate すると MMLU が 64.3% → 34.9% に落ちる（表 2 では基準が 65.1 で、論文内で食い違う）。一方、蒸留した Mamba-2 である Llamba-8B では、30 ヘッドを除いても −4.3% しか落ちない。**SSM の G&A の写像は「より滑らか」で区間の境界を越えて漏れるが、これは図 5 の目視だけで、エントロピーや鋭さの統計は報告されていない**（[Understanding the Skill Gap in Recurrent Language Models](https://arxiv.org/html/2504.18574v2)。ICML 2025 との情報は未確認）。

ハイブリッドについては二つの知見がある。Jet-Nemotron は Qwen2.5-1.5B で、MMLU に効く attention 層は約 2 層、検索に効く層は 2〜3 層で、両者が異なることを見つけた。Jet-Nemotron-2B は層 15・20 だけを全対 attention にし、層 21・22 を SWA にしている（[Jet-Nemotron](https://arxiv.org/html/2508.15884v3)）。Waleffe らの 8B Mamba-2-Hybrid は、56 層のうち 4 層の attention で電話帳と RULER に十分だった。どの層が検索するかの分析は無い（[An Empirical Study of Mamba-based Language Models](https://arxiv.org/html/2406.07887v1)）。

SSM 固有の知見もある。Stuffed Mamba は、層 38 の 3 ヘッドで t = 8K の先頭トークンの累積保持率が 0.997 を超えることを示した。ほとんど忘れないということである。passkey の想起長は状態サイズの指数関数で伸びる（T_recall = 4.756·(1.365^{N_S} − 1) − 0.742、R² > 0.999。単位は本文で明示されていない）（[Stuffed Mamba](https://arxiv.org/html/2410.07145v4)）。**softmax が無くても、1 に近い減衰から sink に似た効果が生じうる**ことを示す数字である。Hymba は「attend を強いられる」負担を学習可能なメタトークンで逃がす設計で、sink 質量の分析に直接関わる（[Hymba](https://arxiv.org/abs/2411.13676)）。Mamba の事実想起の因果追跡は、Transformer と同じく中間層の最後の主語トークンに局在する（[Locating and Editing Factual Associations in Mamba](https://arxiv.org/abs/2404.03646)）。

### 6.3 推奨する正規化と単位

ノートの推論によれば、Mamba-1 の C_i と B̄_j は入力依存で符号を問わず、Ā_k ∈ (0, 1) である。第 2 回報告書の線形 attention の暗黙の重み φ(q)ᵀφ(k_j) / Σ φ(q)ᵀφ(k_i) は、φ が正の写像（融合層の Performer 型）のときにだけ分布になる。ノートが導いた GLA の q_iᵀ diag(∏ ゲート) k_j と、DeltaNet / Gated DeltaNet の WY / UT 逆行列 (I + tril(diag(β)KKᵀ))⁻¹ を含む形も、同じく符号つきである（どちらもノートの調査者の導出で、fla の参照実装との照合が要る。[DeltaNet](https://arxiv.org/abs/2406.06484)）。正規化は次の三つから一つを主として事前登録し、残りを頑健性の確認にする。

| 正規化 | 式 | 先例 | 位置づけ |
|---|---|---|---|
| (a) 絶対値の L1 | \|α\| / Σ\|α\| | DeciMamba Eq. 7 | **主** |
| (b) 正の部分 | ReLU(α) / Σ ReLU(α) | Ali らの帰属で使う (·)⁺ 作用素 | 頑健性 |
| (c) エネルギー | α² / Σα² | なし（ノートの提案） | 頑健性 |

暗黙の写像には、softmax には無い統計が二つある。一つは**行の生の L1 質量** Σ_j |α_ij| で、softmax の行は常に 1 だが、これは長さとともに増えも減りもする。もう一つは**負の質量の割合** Σ|α⁻| / Σ|α| である。正規化エントロピーは H(行 i) / log(i) とし、1K〜16K で比べられるようにする。単位の選び方も結果を左右する。Mamba-1 はチャネルごとに 1 枚（D = 2·d_model、130M で 1536、2.8B で 5120）、Mamba-2 は SSD のヘッドごとに 1 枚（ヘッド次元 P で共有、M = L ⊙ (CBᵀ) を Δ_j でスケール）である。Transformer のヘッドに最も近いのは Mamba-2 のヘッドだが、Zimerman の ablation（S6 のみ 53.19 対 層全体 62.51）が示すとおり、**検索と結びつけるなら層全体の作用素 H の方が忠実**である。sink 質量は正規化の後でしか定義できず、トークン 0（または先頭 k トークン）が正規化 |α| に占める割合とするのが自然である。

計算面では、HF の Mamba / Mamba-2 はどちらも `output_attentions` を持たない。x・Δ・A・B・C・ゲート・conv の活性をフックで取って再構成する必要がある。Mamba-1 では、HiddenMambaAttn・UnifiedImplicitAttnRepr・DeciMamba の参照コードがすでにこれを実装している（[HiddenMambaAttn](https://github.com/AmeenAli/HiddenMambaAttn)、[UnifiedImplicitAttnRepr](https://github.com/Itamarzimm/UnifiedImplicitAttnRepr)、[DeciMamba](https://github.com/assafbk/DeciMamba)）。元の `state-spaces/*`（`-hf` でないもの）は `mamba_ssm` 形式で、`mamba_ssm.models.mixer_seq_simple` から読み込む。HF の Auto クラスでは読めず、`-hf` 版は Mamba-1 にしかない（[HF state-spaces](https://huggingface.co/state-spaces)）。`mamba-ssm` は `MAMBA_KEEP_CUDA_BUILD=TRUE` を付けずに入れると CUDA カーネルなしで入る（[mamba README](https://github.com/state-spaces/mamba)）。fla のモデルの既定は `attn_mode: "chunk"`（Triton）で、トークンごとの状態を返さない（[gla-1.3B-100B config](https://huggingface.co/fla-hub/gla-1.3B-100B/blob/main/config.json)）。

メモリはノートの見積もりで次のとおりである。Mamba-2 の 1 ヘッド・16K の写像は fp32 で約 1.07 GB になる。`mamba2-2.7b` を既定の headdim 64 / expand 2 と仮定すると層あたり 80 ヘッド（未確認）で、層全体では約 86 GB になるから、ヘッドごとに回して統計だけを残す。Mamba-1 は全行列の実体化が不可能なので、必要な行（最後のトークン、回答トークン、needle の位置）だけを計算する。1 行あたりのコストはチャネルごとに O(L·N) である。累積の対数減衰は fp32 で計算し、exp(segment_sum) のアンダーフローは真のゼロとして 0·log 0 := 0 とする。

### 設計への含意（§3.1、§6.5、§2 H5、P10）

1. **§3.1 の暗黙の重みの定義を二つに分ける。** 融合層の線形 attention（正の φ）では φ(q)ᵀφ(k_j) / Σ をそのまま使う。事前学習の Mamba / Mamba-2 / GLA / GDN では、(a) |α| / Σ|α| を主にし、(b)(c) を頑健性の確認にする。行の生の L1 質量と負の質量の割合を必ず併記する。
2. **単位を事前登録する。** Mamba-2 ではヘッドごと、Mamba-1 ではチャネル平均（Ali の rollout と同じ）を主にする。検索と結びつける分析では層全体の H を主にし、S6 のみの写像は副にする。ハイブリッドでは層の種類（softmax / 窓 / SSM・線形）ごとに集計する。
3. **§3.1 に「正規化平均距離」を足す。** DeciMamba Eq. 8 を長さ L で割ったもので、新しいトークンへの集中（新近性の崩壊）を測る。
4. **H5 の向きを族ごとに書き分ける。** softmax の理論は長さとともに Ĥ が上がる（分散）と予言する。一方 DeciMamba の観察は、SSM が学習長を超えると写像が新しいトークンに集中して疎になる、つまり **Ĥ が下がりながら検索が失敗する**可能性を示す。H5 (i) の「Ĥ が長さとともに上がる」は、SSM については逆向きかもしれない。第 2 回報告書は「線形 attention にはエントロピーの長さ法則が無い」と書いたが、SSM については新近性の崩壊という別の向きの定性的な証拠が出てきたことになる。主の指標は向きに依存しない needle 質量と AUROC にする（§9 の H5 改訂案）。
5. **H5 (ii) で Bick らの「より滑らか」を数値にする。** 同じデータの三つ組（§7）で、softmax 層と SSM 層の G&A 相当ヘッドの Ĥ・top-k 質量を並べる。ヘッドの ablation による検索の低下との相関も取る。

---

## 7. 同じデータ・同じ規模の softmax 対照を持つ公開チェックポイントは五組あり、全て 2K〜4K で学習されている

### 7.1 一致した組

| 組 | HF ID（ノートの記載どおり） | データ | トークン | ライセンス | 形式 | 24 GB で 16K 推論 | 学習文脈長 |
|---|---|---|---|---|---|---|---|
| **A. state-spaces 2.7B 三つ組** | `state-spaces/mamba2-2.7b`、`state-spaces/transformerpp-2.7b`、`state-spaces/mamba2attn-2.7b` | Pile | 300B | Apache-2.0 | `mamba_ssm`（`mixer_seq_simple`、HF Auto 不可） | 可。fp16 で約 5.4 GB ずつ。Transformer++ はヘッドごとに再計算（32 ヘッド × 80 次元） | 2K（ノートの推論、config で未確認） |
| A′. state-spaces 小型 ＋ Pythia（近似一致） | `state-spaces/mamba2-{130m,370m,780m,1.3b}`、`state-spaces/mamba-{130m,370m,790m,1.4b,2.8b}`（Mamba-1 は `-hf` 版あり）、対照 `EleutherAI/pythia-160m`、`EleutherAI/pythia-2.8b` | Pile | Mamba 300B。Pythia 約 300B〔K〕 | 小型 Mamba のライセンスは未確認（Apache-2.0 の tag を確認したのは 2.7B の三つのみ）。Pythia はノートに記載なし | `mamba_ssm` / HF | 余裕あり | Mamba-130M は 2K（DeciMamba）。Pythia は 2048〔K〕 |
| **B. HazyResearch 1B-50B 三つ組** | `hazyresearch/based-1b-50b`、`hazyresearch/attn-1b-50bn`（Llama 型）、`hazyresearch/mamba-1b-50b`。360M 版は `based-360M-30B`、`attn-360M-30B`、`mamba-360M-30B` | Pile、**「完全に同じデータ順」**（1B 版のカード） | 50B / 30B | 1B の三つは表示なし。`based-360M-30B` と `transformer-pp-1.3b` は Apache-2.0 | ノートに記載なし | 余裕あり | 未確認 |
| **C. fla-hub 1.3B / 2.7B** | `fla-hub/{transformer,gla,delta_net,retnet,hgrn,hgrn2,gsa,mamba}-1.3B-100B`、同じ並びの `-2.7B-100B`、`fla-hub/delta_net-1.3B-8K-100B` | SlimPajama（transformer・gla のカードの tag） | 100B | transformer・gla は MIT。delta_net などのカードは空 | fla（`trust_remote_code` / `import fla`、chunk Triton） | 可。1.3B は約 2.6 GB | config で 2048（GLA / DeltaNet の論文は未確認） |
| **D. m-a-p ハイブリッド線形群** | `m-a-p/1.3B-100B-{GatedDeltaNet,GLA,DeltaNet,HGRN,HGRN2,RetNet}-pure`、`-hybrid-{3,6,12,24}-1`、対照 `m-a-p/transformer_1.3B_baseline`。340M 版は `m-a-p/340M-20B-*-pure(-baseline)`、`-hybrid-*-1`、対照 `m-a-p/transformer_340M_baseline` | FineWeb-Edu | 100B / 20B | 論文に記載なし | fla | 余裕あり | 論文は 4,096、config は 2048（矛盾） |
| **E. NVIDIA 8B** | `nvidia/mamba2-8b-3t-4k`、`nvidia/mamba2-hybrid-8b-3t-4k`、`-32k`、`-128k`、対照 `nvidia/gpt3-8b-multi-3.5t-base` | 英語 70%・非英語 15%・コード 15%、SentencePiece 256K | 3.5T | Apache-2.0 | Megatron-LM（`model_optim_rng.pt`）。HF 変換は未確認 | 窮屈。fp16 で約 16 GB。16K は分割処理か 4bit（変換の道具は未確認） | 4K（ハイブリッドは 32K / 128K も） |

出典: [HF state-spaces](https://huggingface.co/state-spaces)、[mamba README](https://github.com/state-spaces/mamba)、[transformerpp-2.7b config](https://huggingface.co/state-spaces/transformerpp-2.7b/blob/main/config.json)、[mamba2-2.7b config](https://huggingface.co/state-spaces/mamba2-2.7b/blob/main/config.json)、[Pythia](https://arxiv.org/abs/2304.01373)〔K〕、[HF hazyresearch](https://huggingface.co/hazyresearch)、[based-1b-50b](https://huggingface.co/hazyresearch/based-1b-50b)、[attn-1b-50bn](https://huggingface.co/hazyresearch/attn-1b-50bn)、[mamba-1b-50b](https://huggingface.co/hazyresearch/mamba-1b-50b)、[HF fla-hub](https://huggingface.co/fla-hub)、[transformer-1.3B-100B](https://huggingface.co/fla-hub/transformer-1.3B-100B)、[gla-1.3B-100B](https://huggingface.co/fla-hub/gla-1.3B-100B)、[delta_net-1.3B-100B](https://huggingface.co/fla-hub/delta_net-1.3B-100B)、[A Systematic Analysis of Hybrid Linear Attention](https://arxiv.org/html/2507.06457v2)、[m-a-p コレクション](https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e)、[GatedDeltaNet-pure config](https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure/blob/main/config.json)、[NVIDIA ssms コレクション](https://huggingface.co/collections/nvidia/ssms-666a362c5c3bb7e4a6bcfb9c)、[gpt3-8b-multi-3.5t-base](https://huggingface.co/nvidia/gpt3-8b-multi-3.5t-base)、[mamba2-hybrid-8b-3t-4k](https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k)、[Waleffe et al.](https://arxiv.org/html/2406.07887v1)。

組ごとの注意点は四つある。

**A′ は厳密な一致ではない。** state-spaces の Mamba と Pythia はどちらも Pile で学習したが、Pythia は独自の固定データ順を使い（標準版と重複除去版がある）、Mamba のデータ順が同じだという記載は無い。最適化と学習率の計画も違う。Mamba の語彙 50277 は GPT-NeoX のトークナイザの大きさと同じだが、これは config からの推論である。

**A の三つ組は幅と語彙を共有する。** `transformerpp-2.7b` は d_model 2560、32 層、32 ヘッド × 80 次元、rotary 40、語彙 50277 である。`mamba2-2.7b` は d_model 2560、64 層、語彙 50277 である。`mamba2attn-2.7b` の attention 層の数と位置は未確認である。

**D は Gated DeltaNet とハイブリッド比率の軸を持つ唯一の組である。** 340M（20B トークン）と 1.3B（100B トークン）で各 36 モデル、6 族 × 純粋と 4 比率（24:1、12:1、6:1、3:1）を揃えている。fla-hub には Gated DeltaNet のチェックポイントが無い。Pile 50B で学習した `simarora/gated-deltanet-{d256,k4}-p1b-t50b-pile-re0` が B の組と一致する可能性はあるが、カードを読んでおらず未確認である（[HF 検索](https://huggingface.co/models?search=gated-deltanet)）。

**一致しない組。** Falcon-H1、Hymba、RecurrentGemma、Zamba2、Jet-Nemotron には、同じデータの softmax の双子が無い。

| モデル | HF ID | 構成 | ライセンス | 一致の欠如 |
|---|---|---|---|---|
| Falcon-H1 | `tiiuae/Falcon-H1-0.5B-Base`、`-1.5B-Base`、`-1.5B-Deep-Base`、`-3B-Base`、`-7B-Base`、`-34B-Base` | attention と Mamba-2 の並列ハイブリッド | Falcon-LLM License | 双子なし。文脈長は未確認（[HF](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base)、[技術報告](https://arxiv.org/abs/2507.22448)） |
| Hymba | `nvidia/Hymba-1.5B-Base`、`-Instruct` | Mamba と attention のヘッドが並列、メタトークン、attention 層の 90% が SWA | NVIDIA Open Model License | 双子なし。8K は未確認（[HF](https://huggingface.co/nvidia/Hymba-1.5B-Base)） |
| RecurrentGemma | `google/recurrentgemma-2b`、`-9b`（`-it` も） | RG-LRU ＋ 局所 attention | Gemma（承認制） | Gemma より少ないトークンで学習しており、データが一致しない（[HF](https://huggingface.co/google/recurrentgemma-2b)、[論文](https://arxiv.org/abs/2404.07839)） |
| Zamba2 | `Zyphra/Zamba2-1.2B`、`-2.7B`、`-7B` | Mamba-2 ＋ 共有 attention、G&A が解析済み | ノートに記載なし | 双子なし（[HF](https://huggingface.co/Zyphra/Zamba2-2.7B)） |
| Jet-Nemotron | `jet-ai/Jet-Nemotron-2B`、`-4B` | JetBlock（GDN 系）＋ 全対 2 層 ＋ SWA 2 層 | `ncslv1`（非商用）。`transformers<=4.53.0` に固定 | 最も近い参照は親の Qwen2.5-1.5B。学習文脈長の記載なし（[HF](https://huggingface.co/jet-ai/Jet-Nemotron-2B)） |

Nemotron-H の基盤チェックポイントは HF 検索で見つからなかった（未確認）。

### 7.2 再現の目標になる公表値

| 設定 | 公表値 | 出典 |
|---|---|---|
| Mamba-130M の passkey（2K で学習） | 1K〜8K は成功、16K で 0.8、32K 以上で失敗。Pythia-160M は 1K〜2K でのみ成功 | [DeciMamba](https://arxiv.org/html/2406.14528v3) |
| Mamba の複数文書検索（約 2K で学習、10 → 160 文書 ≈ 2K → 32K） | 68.3、74.0、64.7、45.3、24.7、5.7、1.0、0.3、0.3 | [DeciMamba](https://arxiv.org/html/2406.14528v3) |
| PG-19 の perplexity | Mamba-2.8B は 2K で 9.17、4K で 11.6、8K 以降は発散。Mamba-1.4B は 2K で 10.31、8K で 14.43、10K で 18.4、20K 以降は発散 | [DeciMamba](https://arxiv.org/html/2406.14528v3) |
| `state-spaces/mamba2-*` の passkey | 8K 以内はほぼ完全、16K を超えると 130M 以外は「悪いかゼロ」。継続学習で Mamba-2-370M は 256K でほぼ完全 | [Stuffed Mamba](https://arxiv.org/html/2410.07145v4) |
| 電話帳（Pile） | 70 項目以上で Pythia-410M が Mamba-2.8B を上回る。最大の Mamba は 300 トークンの並べ替え文字列の複写で 0% | [Repeat After Me](https://arxiv.org/html/2402.01032v2) |
| NVIDIA 8B の RULER（13 課題平均） | 4K: Transformer 77.62、Hybrid 81.99、Mamba-2 52.14。16K: 62.33 / 74.19。32K: 56.49 / 69.93 | [Waleffe et al.](https://arxiv.org/html/2406.07887v1) |
| NVIDIA 8B の電話帳 | Transformer は 4K まで約 100%。Mamba-2 は約 1K を超えると失敗。Hybrid-4K は約 5.5K まで 100% | [Waleffe et al.](https://arxiv.org/html/2406.07887v1) |
| m-a-p の RULER 平均 | 純粋 0.256、24:1 0.338、12:1 0.340、6:1 0.390、3:1 0.397、Transformer 0.423。最良は GatedDeltaNet 3:1 の 0.440。純粋は GDN 0.356、DeltaNet 0.320、HGRN2 0.203、GLA 0.144 | [Hybrid Linear Attention](https://arxiv.org/html/2507.06457v2) |
| Jet-Nemotron の検索（FDA / SWDE / SQuAD） | Jet-Nemotron-2B 74.2 対 Qwen3-1.7B-Base 76.1 | [Jet-Nemotron](https://arxiv.org/html/2508.15884v3) |

m-a-p の論文は自分の表と合わない「nearly doubles」という言い方をしている（24:1 → 3:1 は +17%）。fla-hub 1.3B / m-a-p のモデルを 1K〜16K で S-NIAH / passkey 掃引した公表値は見つからなかった。公開された Hazy のチェックポイントでの MQAR の結果も見つからなかった。

### 7.3 学習長の壁

**一致した組は全て 2K（または 4K）で学習されている。** 16K の評価は両族にとって外挿であり、しかも失敗の機構が族ごとに違う。Transformer 側は RoPE の外挿で崩れる（Pythia の passkey は 1K〜2K でしか成功しない）。SSM 側は受容野の縮小と忘却の失敗で崩れる（§6）。DeciMamba の表では、2K で学習した Mamba-130M が 8K まで passkey に成功するのに対し、Pythia-160M は 2K を超えると失敗する。**外挿領域での softmax 対 SSM の差は、主に位置符号化の失敗を測っている**ことになる。分布内の長文脈を取りたいなら、`nvidia/mamba2-hybrid-8b-3t-32k`（ただし 32K の Transformer の双子は無い）、`fla-hub/delta_net-1.3B-8K-100B`、Falcon-H1 / Hymba を使い、一致しないことを受け入れるしかない。

### 7.4 推奨する選択

**§6.5 の本分析** では、主に二組を使う。

**主 1: state-spaces 2.7B 三つ組**（`mamba2-2.7b` / `transformerpp-2.7b` / `mamba2attn-2.7b`）。SSM 対 softmax 対 ハイブリッドで最も厳密な対であり、データ・トークン・語彙・幅が揃い、Apache-2.0 である。bf16 で 16K の推論が 24 GB に収まる。

**主 2: m-a-p 1.3B-100B の部分集合。** GatedDeltaNet と GLA について、pure、hybrid-24-1、hybrid-3-1 の三つ（比率軸の両端と純粋）と `m-a-p/transformer_1.3B_baseline` を使う。線形の族とハイブリッド比率の軸を与え、公表済みの RULER 値（§7.2）と突き合わせられる。

副として三つを使う。**HazyResearch 1B-50B 三つ組**はデータ順まで一致し、Based（線形 ＋ 窓）が H4 (a) の実在版になる。**fla-hub 1.3B-100B** は SlimPajama での再現に使う。**NVIDIA hybrid 32K** は、一致しないことを明記した上で分布内の長文脈の参照にする。設計書が候補に挙げた Falcon-H1 1.5B と Hymba-1.5B は、同じデータの softmax の双子が無いので主の一覧から外し、補足の分析に格下げする。2.7B 以下では量子化をしない。設計書 §6.5 の「4bit で推論」は attention の統計そのものを乱しうる（推論）うえ、bf16 で収まるからである。

**P10 の試行** では二系統を使う。一つは近似一致の系で、`state-spaces/mamba2-130m` / `-370m` と `EleutherAI/pythia-160m`（と同系列の 410M）である。既知の曲線（DeciMamba・Stuffed Mamba）がある。もう一つは厳密一致の系で、m-a-p の 340M-20B の GatedDeltaNet pure / hybrid-3-1 と `m-a-p/transformer_340M_baseline` である。fla の経路を試し、暗黙の重みの復元が GDN でも働くかを確かめる（正確な ID の末尾 `-pure` か `-pure-baseline` かはコレクションで確かめる）。長さは 1K / 2K / 4K / 8K とし、学習長の境界である 2K を加える。SSM だけ 16K も測る。最初の健全性の確認は、DeciMamba の Mamba-130M の passkey 曲線（16K で 0.8、32K で 0）と、2K で密・16K で疎という写像の変化を再現することである。

### 設計への含意（§6.5、§2 H5、§13 項目 8、P10）

1. **§6.5 の「対象」を §7.4 の一覧に置き換える。** 主は A と D の部分集合、副は B・C・E、補足は Falcon-H1 / Hymba / Zamba2 / Jet-Nemotron とする。§13 項目 8 は「方針は決定、ID の最終確認と長さの上限を P10 で決める」に改める。
2. **H5 を二つの領域に分ける。** 学習長以内（1K〜2K、m-a-p と NVIDIA は 4K まで）を分布内、それを超える範囲を外挿とする。外挿では「softmax 側の失敗は位置符号化による」ことを明記し、族の差を鋭さの機構として解釈しない。
3. **2.7B 以下は bf16 で推論し、量子化しない。** 8B の Megatron チェックポイントは、HF 変換が確認できるまで副の扱いにとどめる。
4. **ライセンスを manifest に記録する。** Jet-Nemotron は非商用である。m-a-p と Hazy 1B のライセンスは未表示なので、公開の前に確かめる。

---

## 8. 新規性の判定: 二本の新しい柱はどちらも空白を突いている

**柱 1（統制実験の疎の水準）。** 空白は三つある。(i) 学習による選択と構造による選択を、同じ k で別々に学習した腕として比べた研究は無い（§4.2）。(ii) 疎・線形・SSM を壁時計とメモリを揃えて比べた研究も無い。パラメータとトークンを揃えた比較が 1 件あるだけである（§2.2）。(iii) 支持再現率を下流の精度から切り離して報告する慣行が無い（§2）。三つとも、設計書の因子 A × 疎の列 × §6.2 の測定でそのまま埋まる。

ただし但し書きが四つある。第一に、Semantic Scholar の被引用の走査（NSA・MoBA・SASA）をしていない。第二に、2025 年の後続（InfLLM-v2、SeerAttention-R、Flash Sparse Attention）を読んでいない。第三に、NAtS-L の数値を抽出していない。第四に、SASA の ablation は学習選択が勝つ方向を示しているので、新規性は「構造が勝つ」という予言ではなく、**同じコストで測ったという事実**に置くべきである。これは第 2 回報告書の判定（新規性は「構造を入れること」ではなく、同コストで構造と系列の局所性を比べることにある）を、疎の系譜に延ばしたものである。

**柱 2（実在モデルの分析）。** 空白は四つある。(i) ヘッドごとの指標が検索の正答を予測する AUROC を長さ別に測った研究は無い（§5.2）。(ii) SSM・線形モデルの暗黙の写像の正規化エントロピー・top-k 質量・sink 質量・実効ランクを、長さの関数として報告した研究も無い。DeciMamba の定性的な観察と平均距離があるだけである（§6.1）。(iii) Bick らの「SSM の G&A はより滑らか」は目視にとどまる（§6.2）。(iv) SSM の暗黙の写像で sink 質量を測った研究も無い。一致したチェックポイントは公開資源なので、新規性は資源ではなく分析の方にある。m-a-p の論文自体も、比率ごとの RULER を持つだけで内部の指標を持たない。

**新規性への脅威** は五件あり、事前登録の前に全文を読む。ACL 2026 の [Retrieval Heads are Dynamic](https://arxiv.org/abs/2602.11162)（方法が未取得）、[Attention Entropy is a Key Factor](https://arxiv.org/html/2412.16545)〔U〕、[From Rigid to Dynamic](https://arxiv.org/pdf/2606.09508)〔U〕、[Cracks in the Foundation](https://arxiv.org/pdf/2608.10296)〔U〕、[LongMamba](https://arxiv.org/pdf/2504.16053)〔U〕である。いずれかが長さ別の成否条件づけを含んでいれば、§5 と §6 の主張を「線形・SSM・ハイブリッドの一致した組での比較」に絞り直す。

---

## 9. 設計書への変更提案

| 節 | 現行（v0.5） | 変更案 | 根拠 |
|---|---|---|---|
| §1 事実表 | 疎の行なし | 2 行を追加。「疎な attention は支持上の softmax を厳密に保ち、失敗は選択の誤りとして現れる（ブロック平均で単一の鍵が薄まる）」「パラメータ・トークンを揃えた唯一の比較（0.4B / 1.4B）で、純 NSA は Transformer++ にやや劣り、GDN との上下は規模で入れ替わる。壁時計・メモリを揃えた 疎 対 線形 対 SSM の比較は存在しない」 | §2.1、§2.2 |
| §2 H4 (e) | 「学習で選ぶ疎な softmax（上位 k 個のブロック選択）。水準は第 3 回で確定」 | 「MoBA 形のブロック上位 k: ゲート ⟨q, ブロックの平均キー⟩、先頭ブロックと現ブロックを強制（k に数える）、選んだ支持の上で厳密な softmax。ブロック B ∈ {64, 128}（主は 128 = FlexAttention ネイティブ）、キー予算 k·B ∈ {256, 512, 1024}（B = 64 なら k ∈ {4, 8, 16}、B = 128 なら k ∈ {2, 4, 8}）。NSA の選択枝だけ（B = 64、先頭 1・局所 2 を含む上位 n）を P7 で副として比べる。最終の k は 線形＋φ内 と壁時計が揃う値を P7 で選ぶ」。予測を追加: 「(e) の検索の損失は主に支持再現率の不足から来る」 | §1.3、§2.1、§3.1 |
| §4.2 疎の行 | 「具体的な方式は第 3 回文献調査で確定する」 | 上の (e) と同じ方式を書く。実装は FlexAttention の BlockMask を上位 k の添字（kv_num_blocks / kv_indices）から直接組み、構築時間を壁時計に含めて別にも記録する。DSA（トークン単位）は MLA 前提なので対象外 | §1.2、§3.2 |
| §4.3 疎の列 | なし / 構造で選ぶ / 併用 | 4 水準にする。**固定**（窓 ＋ 先頭ブロック。内容にも構造にも依らない床）、**学習**（MoBA 形）、**構造**（SASA 形: 同じ B と k で、ブロックごとの Σ α_k W^k（または RRWP）の質量の和で上位 k）、**併用**（学習 k/2 ＋ 構造 k/2。和集合にしない）。構造で選んだブロックのうち窓の外にある割合を記録する | §4.2 含意 |
| §3.1 鋭さ | 支持再現率なし。暗黙の重みは φ(q)ᵀφ(k_j)/Σ のみ | 行を追加: **支持再現率** \|G ∩ 選択支持\| / \|G\|（疎の水準。G は needle・構造目標・整合先）、**支持内 Ĥ**（log \|選択支持\| で正規化）、**正規化平均距離**（DeciMamba Eq. 8 / L）。暗黙の重みの定義を分ける: 正の φ（融合層）はそのまま、事前学習の Mamba / GLA / GDN は \|α\| / Σ\|α\| を主、ReLU 版と α² 版を頑健性に、行の生の L1 質量と負の質量の割合を併記 | §2 含意 1、§6.3 |
| §3.1 sink 質量 | Gu の Sink_k^ε（ε = 0.3、T 明記）、除外して Ĥ を再計算 | 加えて: sink を除外するだけでなく共変量として扱う（因果的に必要、学習文脈長とともに強まる）。探針（繰り返しトークン / NIAH）ごとに別々に記録する（Llama / Mistral では繰り返しトークンで Sink₁ = 0）。Llama-3.1 系は ε = 0.8 も | §5.1、§5.2 |
| §4.4 / §4.6 揃える規則 | 因子 C は壁時計・ピークメモリ・KV / 状態バイトを ±10% で揃える | (e) と疎の列は、**壁時計とピークの活性メモリ**で揃える。KV / 状態のバイト数は報告するが揃えない（疎 O(N·d) 対 線形 O(d²) は原理的に揃わない）。事前登録に明記する | §3.3 |
| §2 H5 | 「(i) 検索ヘッドの正規化エントロピーが長さとともに上がり、同じ長さで正答率が落ちる。(ii) ハイブリッドでは鋭い読み出しが softmax 層と窓の層に集中する」 | 「同じデータ・同じ規模で学習した softmax・線形・SSM・ハイブリッドの公開モデルで、(i) 層・ヘッドごとの指標（主: 回答の最初のステップでの needle 質量、副: Ĥ、top-k 質量、sink 質量、正規化平均距離）が検索の成否を予測する AUROC は、検索ヘッド・層で最も高い。その AUROC と正答率の log L に対する傾きを、**学習長以内（分布内: 1K〜2K、m-a-p と NVIDIA は 4K まで）と学習長の外（外挿: 4K〜16K）で別々に**推定する。指標の向きは族で異なりうる（softmax は分散で Ĥ が上がり、SSM は新近性の崩壊で正規化平均距離が下がる）ので、Ĥ の向きは族ごとに予測する。(ii) ハイブリッドでは、needle 質量と低い Ĥ が softmax 層と窓の層に集中する」。反証: (i) 分布内で、全ての検索ヘッドの AUROC の 95% 区間が 0.5 を含む。(ii) needle 質量が softmax 層に集中しない。外挿での softmax 側の失敗は位置符号化によるものとして別に扱う | §5.2、§6 含意 4、§7.3 |
| §6.5 対象 | Mamba-2 130M〜2.8B、fla の小型 GDN / GLA、Falcon-H1 1.5B、Hymba-1.5B、Pythia | 主: `state-spaces/mamba2-2.7b` / `transformerpp-2.7b` / `mamba2attn-2.7b`、`m-a-p/1.3B-100B-{GatedDeltaNet,GLA}-{pure,hybrid-24-1,hybrid-3-1}` ＋ `m-a-p/transformer_1.3B_baseline`。副: `hazyresearch/{based-1b-50b,attn-1b-50bn,mamba-1b-50b}`、fla-hub 1.3B-100B、`nvidia/mamba2-hybrid-8b-3t-32k`（非一致を明記）。補足: Falcon-H1、Hymba、Zamba2、Jet-Nemotron | §7.4 |
| §6.5 課題と分析 | 長さ 1K / 2K / 4K / 8K / 16K、S-NIAH・MQAR・電話帳、成否で条件づけた指標 | 格子を 5 長さ × 10 深さ × 20 needle 以上に。妨害 needle の三種（なし / 同形式の別キー / 近似重複キー）。深さで層別。**主分析は「AUROC 対 log L」**（§8 の空白）。ヘッドのラベルは帰納・echo・Wu の検索スコア（0.1）・sink。Wu / QRHead / DuoAttention のヘッド集合の Jaccard。因果の確認は上位 K 個の検索ヘッドのマスク対ランダム K 個。課題はトークン長で生成し、needle のトークン幅を記録し、族をまたぐときは単語か UUID の値を使う。BOS を全条件で揃えて記録する | §5.3 含意 |
| §6.5 計算量 | 「3B 以下を 4bit で推論」「16K では暗黙の重みの復元をクエリの標本に限る」 | 2.7B 以下は bf16（量子化は attention の統計を乱しうる）。Transformer は SDPA で順伝播し、フックした RoPE 後の Q / K からクエリブロック（b = 512 で約 1.07 GB）ごとに再計算し、エントロピーはストリーミングの恒等式 H = L − Σ p s で求める。Mamba-2 はヘッドごとに L ⊙ CBᵀ を組んで統計だけを残す。Mamba-1 は必要な行だけを O(L·N) で計算する。累積の対数減衰は fp32、0·log 0 := 0 | §5.3、§6.3 |
| §10 リスク | — | 追加 5 件。(1) 学習選択のマスクは層ごと・入力ごとに変わり、BlockMask の構築費を償却できない → 添字から直接組み、構築時間を計上する。(2) 16K 以下・ブロック 64〜128 では、疎が 3090 の FA2 に勝てない可能性 → P8 で測り、壁時計を揃える k が大きくなることを (e) に不利な条件として報告する。(3) ルータの冷えた始動と、疎な勾配（MoBA の SFT の劣化）→ 密 warm-up（DSA 形の KL 蒸留）を任意で入れる。(4) BOS / sink の欠落で検索が崩れる（RULER-4K 82.57 → 0）→ BOS を固定・記録する。(5) 一致した組は全て 2K〜4K 学習で、外挿の比較が位置符号化の失敗と交絡する → H5 を二領域に分ける | §3.2、§3.1、§2.1、§5.2、§7.3 |
| §11 位置づけ | 「疎な attention: 第 3 回で埋める」「実在 LLM の attention の分析: StreamingLLM、Gu、Barbero、Repeat After Me」 | 疎: [NSA](https://arxiv.org/abs/2502.11089)、[MoBA](https://arxiv.org/abs/2502.13189)、[FlashMoBA](https://arxiv.org/abs/2511.11571)、[DSA](https://arxiv.org/abs/2512.02556)、[SeerAttention](https://arxiv.org/abs/2410.13276)、[The Sparse Frontier](https://arxiv.org/abs/2504.17768)、[He & Garner](https://arxiv.org/html/2510.20787)、[SASA](https://arxiv.org/abs/2205.13730)、[Exphormer](https://arxiv.org/abs/2303.06147)、[BigBird](https://arxiv.org/abs/2007.14062)。実在モデル: 既存の 4 件に加え [Retrieval Head](https://arxiv.org/abs/2404.15574)、[DuoAttention](https://arxiv.org/abs/2410.10819)、[QRHead](https://arxiv.org/abs/2506.09944)、[Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)、[Found in the Middle](https://arxiv.org/abs/2406.16008)、[Hidden Attention of Mamba](https://arxiv.org/abs/2403.01590)、[Unified Implicit Attention](https://arxiv.org/abs/2405.16504)、[DeciMamba](https://arxiv.org/abs/2406.14528)、[Gather-and-Aggregate](https://arxiv.org/abs/2504.18574)、[Stuffed Mamba](https://arxiv.org/abs/2410.07145)、[Hybrid Linear Attention](https://arxiv.org/abs/2507.06457) | §1、§5、§6、§7 |
| §12 実装 | `--attn_form sparse`、`analyze_pretrained_attention.py` | `--sparse_select {fixed,learned,struct,mixed}`、`--sparse_block {64,128}`、`--sparse_budget`（キー数）を追加。融合層は支持再現率を返すフックを持つ。分析スクリプトは `mamba_ssm` 形式と fla 形式の読み込み、暗黙の写像の 3 種の正規化、AUROC 対 log L の集計を持つ | §3、§5.3、§6.3 |
| §13 項目 7 | 未決 | 方針は決定（MoBA 形、B ∈ {64, 128}、k·B ∈ {256, 512, 1024}、NSA 選択枝を副）。残るのは P7 での B・k の最終値と、壁時計を揃える規則 | §1、§3 |
| §13 項目 8 | 未決 | 方針は決定（§7.4）。残るのは P10 での ID の最終確認と長さの上限 | §7.4 |
| 第 1 期 P7 | (a)〜(d) の 4 水準 | (e) を加える。B ∈ {64, 128} × 予算 3 点、MoBA 形 対 NSA 選択枝、支持再現率と BlockMask の構築費を記録する。決めること: (e) の B・k と、壁時計とピーク活性メモリで揃える規則 | §2、§3 |
| 第 1 期 P8 | 線形・SSM 対 FA2 / FlexAttention | ブロック疎（疎性 75 / 87.5 / 93.75%、ブロック 64 / 128）を加え、3090 で FA2 に勝つ系列長と疎性を求める | §3.1 |
| 第 1 期 P10 | Mamba-2 小型 ＋ 同データの softmax 1 つ、長さ 1K / 4K / 8K | `state-spaces/mamba2-130m` / `-370m` ＋ `EleutherAI/pythia-160m`（と 410M）、加えて m-a-p 340M-20B の GDN pure / hybrid-3-1 ＋ `m-a-p/transformer_340M_baseline`。長さ 1K / 2K / 4K / 8K（SSM は 16K も）。最初に DeciMamba の Mamba-130M の passkey と、Stuffed Mamba の mamba2 の passkey を再現する。決めること: §6.5 の ID、長さの上限、正規化の主、AUROC が分布内で 0.5 から離れるか | §7.4 |

---

## 10. 未検証一覧（ノートから引き継ぎ。事前登録前に解消する）

印: 〔K〕= 記憶由来で再取得していない、〔U〕= 二次情報のみ、〔S〕= 集約サイト・検索要旨のみ、「未確認」= ノートが unverified / gap とした項目。

| # | 項目 | 状態 | 解消方法 |
|---|---|---|---|
| **S. 疎な attention の系譜** | | | |
| S1 | Sparse Transformer の strided / fixed パターン、O(n√n)、blocksparse カーネル（[arXiv](https://arxiv.org/abs/1904.10509)） | 〔K〕 | 原論文 |
| S2 | Longformer の窓・拡張窓・大域トークン、TVM カーネル、層ごとの窓の計画（[arXiv](https://arxiv.org/abs/2004.05150)） | 〔K〕、層ごとの値は未取得 | 原論文の付録 |
| S3 | BigBird のブロック 64・窓 3 / 大域 2 / ランダム 3、課題ごとの値、普遍近似・チューリング完全・Ω̃(n) 層の下界（[arXiv](https://arxiv.org/abs/2007.14062)） | 〔K〕、課題ごとの値は未取得 | 付録と定理 |
| S4 | ETC の大域・局所の構成と構造への結びつけ（[arXiv](https://arxiv.org/abs/2004.08483)） | 〔K〕 | 原論文 |
| S5 | Mistral の W = 4096、受容野約 131K、キャッシュ 8 倍減、16K で 2 倍（[arXiv](https://arxiv.org/abs/2310.06825)） | 〔K〕 | 原論文 |
| S6 | Mistral SWA と全対の、同規模での検索ベンチの比較 | 一次資料なし | 追加調査 |
| S7 | StreamingLLM の sink 4、4M トークン、22.2 倍、sink 事前学習（[arXiv](https://arxiv.org/abs/2309.17453)） | 〔K〕 | 原論文 |
| S8 | Gemma 3 の 5:1・窓 1024（[arXiv](https://arxiv.org/abs/2503.19786)）、gpt-oss の帯幅 128（[arXiv](https://arxiv.org/abs/2508.10925)） | 〔K〕 | 技術報告 |
| S9 | Reformer、Routing（WT-103 15.8 対 18.3、PG-19 33.2）、Sinkhorn、Combiner（[arXiv](https://arxiv.org/abs/2107.05768)）、H-Transformer-1D（[arXiv](https://arxiv.org/abs/2107.11906)）、Memorizing Transformers（[arXiv](https://arxiv.org/abs/2203.08913)） | 〔K〕 | 各原論文 |
| S10 | Landmark の LLaMA-7B 32K 延長（[arXiv](https://arxiv.org/abs/2305.16300)）、Unlimiformer（[arXiv](https://arxiv.org/abs/2305.01625)） | 〔K〕 | 原論文 |
| S11 | NSA の ACL 2025 Best Paper（賞のページが当初載せていなかったという指摘、[HN](https://hnews-reader.vercel.app/jobs/44761548)） | 〔S〕、未解決 | ACL 2025 の賞の公式ページ |
| S12 | NSA の学習トークン 270B 対 260B の食い違い | 論文内の不整合 | 著者の最新版 |
| S13 | MoBA の GPU の種類とカーネル単位のメモリ数値 | 抽出テキストに記載なし | 全文・コード |
| S14 | FlashMoBA の SNR ∝ √(d/2B) の式 | 〔S〕 | 原論文の該当節 |
| S15 | DSA の indexer のヘッド数、図 3 のコスト数値 | 抽出不能 | PDF |
| S16 | SeerAttention の 128K・疎性 90% で 7.3 倍（[review](https://www.themoonlight.io/en/review/seerattention-learning-intrinsic-sparse-attention-in-your-llms)） | 〔S〕 | 原論文 |
| S17 | SeerAttention の 5.67 倍・90%・32K | 要旨の断片で確認 | 全文 |
| S18 | H2O の 29 倍・3 倍、SnapKV の 3.6 倍・8.2 倍・380K（[arXiv](https://arxiv.org/abs/2404.14469)）、Quest の 7.03 倍・2.23 倍、MInference の 10 倍、InfLLM の構成 | 〔K〕 | 各原論文 |
| S19 | RetrievalAttention のキーの 1〜3%、RTX 4090 で 0.188 s/トークン | 〔K〕 | 原論文 |
| S20 | BSFA の GPU と数値、BLASST の数値 | 検索要旨で確認、GPU 不明（BSFA） | 全文 |
| S21 | The Sparse Frontier の結論（Findings of ACL 2026、[Anthology preview](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1926/)） | 検索要旨で確認 | 全文 |
| S22 | NAtS-L の「NSA・MoBA・GDN ＋ SWA は全対と同程度の検索」と数値（[PDF](https://arxiv.org/pdf/2602.03681)） | 〔S〕、数値未抽出 | 全文 |
| S23 | Gated DeltaNet の「ハイブリッドが純 attention を検索で上回る」（[arXiv](https://arxiv.org/abs/2412.06464)） | 〔S〕 | 本文 |
| S24 | Based / Zoology の SWA 基準線（[Based](https://arxiv.org/abs/2402.18668)、[Zoology](https://arxiv.org/abs/2312.04927)）、Based の ICML 2024 | 〔K〕、会場未確認 | 原論文・プロシーディング |
| S25 | InfLLM-v2（[arXiv](https://arxiv.org/abs/2506.07900)）、SeerAttention-R（[arXiv](https://arxiv.org/abs/2506.08889)）、Flash Sparse Attention（[arXiv](https://arxiv.org/abs/2508.18224)） | 未取得 | 全文 |
| S26 | NSA / MoBA / SASA の Semantic Scholar 被引用走査、2026 年の後続 | 未実施 | API キー取得後 |
| S27 | FlexAttention 論文の詳細（[arXiv](https://arxiv.org/abs/2412.05496)）、[attention-gym](https://github.com/pytorch-labs/attention-gym) | 〔K〕 | 論文 |
| S28 | 元の FlashAttention のブロック疎版の IO 量（[arXiv](https://arxiv.org/abs/2205.14135)） | 〔K〕 | 原論文 |
| S29 | RTX 3090 / 4090 での FlexAttention ブロック疎 対 FA2 の測定、FA2 に勝つ疎性の閾値 | 一次資料なし | P8 で自分で測る |
| S30 | SASA が数える AST 関係の定義、融合ブロック疎カーネルの有無 | 論文に記載なし | 著者・コード |
| S31 | LongCoder の橋・記憶トークン（[arXiv](https://arxiv.org/abs/2306.14893)）、Exphormer、GraphGPS、GraphCodeBERT のマスク | 〔K〕 | 原論文 |
| S32 | SG-Net、Syntax-Guided Transformers、Dependency Transformer Grammars、Tree Transformer の再検討 | 要旨の断片で確認 | 全文 |
| S33 | Yun ら 2020 の十分条件（[arXiv](https://arxiv.org/abs/2006.04862)） | 〔K〕 | 原論文 |
| S34 | 内容依存の上位 k の表現力の結果、NSA / MoBA の支持上のエントロピー測定、デコーダ LLM での AST 融合ブロック疎カーネル、2024〜2026 年の規模のある構造誘導疎 attention | 見つからず（否定的な検索結果） | 被引用走査で再確認 |
| **H. 事前学習 Transformer のヘッド分析** | | | |
| H1 | Clark らのエントロピー・特殊トークンの質量・no-op の議論、コード（[GitHub](https://github.com/clarkkev/attention-analysis)） | 〔K〕 | 原論文 |
| H2 | Voita らの信頼度の定義、48 中 38 ヘッドで BLEU −0.15、コード（[GitHub](https://github.com/lena-voita/the-story-of-heads)） | 〔K〕 | 原論文 |
| H3 | Michel らの I_h の式、20% / 40%、コード（[GitHub](https://github.com/pmichel31415/are-16-heads-really-better-than-1)） | 〔K〕 | 原論文 |
| H4 | TransformerLens の帰納スコアの操作化（対角のずれ −(n − 1)） | 〔K〕 | TransformerLens の文書 |
| H5 | Olsson らの複写スコアの式（OV 回路の正の固有値の割合、[Framework](https://transformer-circuits.pub/2021/framework/index.html)） | 〔K〕 | 原文 |
| H6 | Wu らの図 7（検索ヘッド対ランダムヘッドのマスク、モデルごと） | 図のみ | PDF の図から読み取る |
| H7 | Wu らの「幻覚時に検索ヘッドが不活性」 | ICLR 版の検索要約経由 | 全文 |
| H8 | Wu らの導入部の「13 ヘッド」対本文の「12 ヘッド」（Llama-2-7B-80K） | 論文内の不整合 | 最新版 |
| H9 | RazorAttention の閾値（echo 上位 1%、帰納上位 14%）、約 70% の削減、会場 | 〔U〕 | 全文、ICLR 2025 のプロシーディング |
| H10 | HeadKV の検索・推論スコアの式 | 未取得 | 全文 |
| H11 | Retrieval Heads are Dynamic の方法・モデル・長さ・失敗例の結果 | 未取得 | 全文 |
| H12 | 2410.22316（[PDF](https://arxiv.org/pdf/2410.22316)）、SEAL（[PDF](https://arxiv.org/pdf/2501.15225)）、RoPE の次元非効率（[PDF](https://arxiv.org/pdf/2502.11276)）、2601.11020（[PDF](https://arxiv.org/pdf/2601.11020)）、RoPE と検索ヘッド（[PDF](https://arxiv.org/pdf/2606.21249)）、BOSCH（[PDF](https://arxiv.org/pdf/2604.05942)） | 〔U〕 | 各全文 |
| H13 | argmax 型の検索の失敗が局在化の人工物だったという報告（2606.21249） | 〔U〕 | 全文 |
| H14 | Barbero らの会場（COLM 2025） | 〔U〕 | プロシーディング |
| H15 | Massive Activations の約 10⁴ 倍とコード（[GitHub](https://github.com/locuslab/massive-activations)） | 〔K〕 | 原論文 |
| H16 | Lost in the Middle の課題設定（10 / 20 / 30 文書、75 / 140 / 300 の KV、[arXiv](https://arxiv.org/abs/2307.03172)） | 〔K〕 | 原論文 |
| H17 | Found in the Middle の著者一覧と Findings の扱い | 〔K〕 | Anthology |
| H18 | RULER の 13 課題、雛形、実効長の定義 85.6% | 〔K〕 | 原論文・リポジトリ |
| H19 | passkey の雛形（Landmark）、MQAR の既定値、Repeat After Me の電話帳の形式 | 〔K〕 | 原論文・リポジトリ |
| H20 | SSMax の「事前学習済みモデルへの差し替えでも改善」 | 要旨と二次要約で確認 | 全文 |
| H21 | Scale-invariant attention（[PDF](https://arxiv.org/pdf/2505.17083)）の内容と NeurIPS 2025 | 〔U〕 | 全文 |
| H22 | Attention Entropy is a Key Factor（2412.16545）、From Rigid to Dynamic（2606.09508）、Cracks in the Foundation（2608.10296） | 〔U〕（検索断片のみ） | 全文（新規性への脅威として優先） |
| H23 | Insights into LLM Long-Context Failures（2406.14673）、Attention Sorting（2310.01427） | 〔U〕 | 全文 |
| H24 | TransformerLens・nnsight の機能と 16K・24 GB での限界 | 〔K〕、限界の記述なし | 文書と自前の計測 |
| H25 | FlashAttention の LSE で 16K 以上のエントロピーを出す保守されたツール | 見つからず | 自前実装 |
| H26 | 16K の attention 行列のメモリ算術（1 ヘッド 1.07 GB / 0.54 GB、層 34 / 17 GB） | ノートの算術 | P10 で実測 |
| H27 | 7 桁の数のトークン化が Llama-2 / Llama-3 / Qwen で異なる | ノートの推論 | 各トークナイザで確認 |
| H28 | トークナイザによる NIAH の分散を定量した研究 | 見つからず | — |
| H29 | 長さ別のヘッド指標の AUROC で NIAH の正答を予測した研究 | 見つからず（空白の根拠） | 被引用走査で再確認 |
| **M. SSM・線形・ハイブリッドのモデルとチェックポイント** | | | |
| M1 | Hidden Attention of Mamba（2403.01590）と Unified Implicit Attention（2405.16504）の最終会場（ECCV / ICLR?） | 未確認 | プロシーディング |
| M2 | Ali らの「約 100N 倍」対「三桁」の食い違い | 論文内の不整合 | 著者 |
| M3 | DeciMamba の ICLR 2025、平均距離の数表の不在 | 会場は未確認 | プロシーディング・コード |
| M4 | LongMamba（2504.16053）の局所・大域チャネルと受容野の縮小 | 〔U〕（断片） | 全文 |
| M5 | GLA の暗黙の写像 q_iᵀ diag(∏ ゲート) k_j、DeltaNet / GDN の WY / UT 逆行列の形 | ノートの調査者の導出 | fla の参照実装と照合 |
| M6 | fla に `fla/ops/*/naive.py` の参照関数があるか、attention 相当の行列を返す API があるか | 未確認 | リポジトリ |
| M7 | GLA / DeltaNet / GDN の暗黙の attention の抽出論文、実効ランク・sink 質量の測定 | 見つからず | 追加調査 |
| M8 | Bick らの ICML 2025、Zamba2 の 64.3 対 65.1 の食い違い | 会場は未確認、論文内の不整合 | プロシーディング |
| M9 | Jet-Nemotron-4B の全対 attention の位置（本文と表 9 が食い違う）、学習文脈長 | 論文内の不整合、記載なし | 著者・コード |
| M10 | Waleffe らの attention 層の位置 11 / 23 / 34 / 46 | ノートの調査者が導出 | config |
| M11 | Stuffed Mamba の法則の単位 | 本文で明示なし | 全文 |
| M12 | Samba / Falcon-H1 / Nemotron-H の層ごとの検索の局在の研究、Mamba の帰納ヘッドの機構論文、RWKV-7 / GDN の記憶解析 | 見つからず | 追加調査 |
| M13 | Pythia の約 300B トークン・GPT-NeoX トークナイザ・2048 文脈・同一データ順（[arXiv](https://arxiv.org/abs/2304.01373)） | 〔K〕 | 原論文 |
| M14 | state-spaces の Pile モデル同士、および Pythia との同一データ順 | 未確認 | 著者・README |
| M15 | state-spaces の学習文脈長 2K（Mamba-130M 以外）、`mamba2attn-2.7b` の層パターン | ノートの推論、config 未読 | config |
| M16 | `mamba2-2.7b` のヘッド数 80（headdim 64 / expand 2 の仮定） | 仮定 | config |
| M17 | `simarora/gated-deltanet-{d256,k4}-p1b-t50b-pile-re0` と Hazy 1B-50B の一致 | 未確認（カード未読） | カード |
| M18 | Hazy 1B-50B の三つのライセンス、形式、学習文脈長 | 表示なし・記載なし | カード・著者 |
| M19 | m-a-p の Transformer 基準線のライセンス・データ・トークン・文脈長、公開重みのライセンス、340M の正確な ID の末尾 | 未確認 | カード・コレクション |
| M20 | m-a-p の文脈長（論文 4,096 対 config 2048） | 矛盾 | 著者 |
| M21 | m-a-p の「nearly doubles」と表の不一致 | 論文内の不整合 | — |
| M22 | fla-hub の学習文脈長（config は 2048、GLA / DeltaNet の論文は未確認）、`delta_net-1.3B-100B` のカードが空（ライセンス・データなし）、`delta_net-1.3B-8K-100B` の文脈長 | 未確認 | 論文・著者 |
| M23 | Falcon-H1 0.5B / 1.5B の文脈長（16K / 128K?）、Hymba の 8K、RecurrentGemma の 8K 系列・2K 局所窓 | 未確認 | カード・技術報告 |
| M24 | Nemotron-H の基盤チェックポイントの所在 | HF 検索で見つからず | NVIDIA のコレクション |
| M25 | `nvidia/gpt3-8b-multi-3.5t-base` / `mamba2-hybrid-8b-*` の HF 変換と transformers での読み込み、Megatron の 4bit 変換の道具 | 未確認 | 変換スクリプトを試す |
| M26 | HF transformers の Mamba-2 が `mamba_ssm` カーネルの無いとき torch 経路に落ちるか（`is_fast_path_available` が見当たらない） | 未確認 | コード・実行 |
| M27 | 16K でのメモリ見積もり（Mamba-2 の層あたり約 86 GB、1.3B / 2.7B / 8B の重み、16K の推論の可否） | ノートの見積もり | P10 で実測 |
| M28 | fla-hub 1.3B / m-a-p モデルの 1K〜16K の S-NIAH / passkey の公表値、Hazy のチェックポイントでの MQAR | 見つからず | P10 で自分で測る |
| M29 | Repeat After Me の電話帳の長さ別の数値 | 図のみ | 図から読み取る |
| M30 | 2.7B 以下で 4bit 量子化が attention の統計を乱すという主張 | 本報告の推論 | P10 で bf16 と比べる |
| M31 | state-spaces の小型 Mamba / Mamba-2 のライセンス | tag を確認したのは 2.7B の三つのみ | 各カード |

---

## 結論

疎な attention は、効率的 attention の三系統の中で、**失敗を一つの上流の決定に帰着できる唯一の系統**である。線形 attention と SSM の損失は、写像・ランク・状態容量に分散していて、どれが効いたかを外から切り分けにくい。疎な attention は支持の上で softmax を厳密に保つので、損失は「正しいキーを選んだか」にほぼ集約され、支持再現率という一つの数で直接測れる。このことは研究の設計に好都合である。「構造は鋭さの事前分布として働く」という H2 の主張は、疎の列では「同じ k の予算の下で、構造で選んだ支持は学習で選んだ支持より高い再現率を持つか」という、ほとんど機械的に検定できる形になる。SASA の ablation は学習選択の側に傾いているが、それは入力ごとに採点しない、コーパス統計による選択との比較だった。入力ごとのルータに対して同じ k で問い直した研究はまだ無い。

実在モデルの分析については、一致したチェックポイントが全て 2K〜4K で学習されているという事実が、H5 の読み方を変える。長さの軸の大半は外挿であり、外挿での softmax の失敗は位置符号化によるもの、SSM の失敗は新近性への崩壊によるものとして、向きの違う現象になる。「エントロピーが長さとともに上がる」という単一の予言は、少なくとも一方の族について誤りになりうる。その代わり、向きに依らない「指標が正答を予測する力（AUROC）の長さに伴う変化」を、分布内と外挿に分けて測れば、族をまたいで同じ問いを立てられる。この分析は文献に無く、統制実験の鋭さ指標が実モデルの失敗を説明するかどうかを判定する、H1〜H4 の外部妥当性の試金石になる。
