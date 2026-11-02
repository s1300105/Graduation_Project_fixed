# 効率的 attention は「容量」で鋭さを失い、構造はその安い回復候補の一つにとどまる

作成: 2026-11-02。第 2 回文献調査の報告書。調査ノートは `research_notes/効率的 attention の鋭さと構造 文献調査/`（5 本）。対象の設計書は `docs/DESIGN_structural_prior_efficient_attention_ja.md`（v0.3）、第 1 期計画は `docs/PHASE1_PLAN_survey_and_pilot_ja.md`。第 1 回報告書 `reports/LLM GNN 融合 attention 先行研究.md` の内容は再掲せず、新しい証拠が結論を変える箇所だけ参照する。本文の出典は全てノートで URL を開いたもの。ノートが「unverified」とした項目は本文で断定せず、§8 の一覧に全て引き継いだ。

**要旨。** 線形・カーネル化 attention と SSM が softmax に対して失う「鋭さ」とは、N ≫ d 個のキーの中から少数を argmax 的に選び出す読み出し能力であり、その喪失は五つの機構（カーネル分母による勾配の非有界性、非負特徴写像による分布の一様化、attention 行列のランク ≤ d、固定サイズ状態による検索容量の Ω(N) ビット下界、加法更新による消去不能）として記述されている。このうち**形式的な定理があるのは勾配・ランク・容量**で、長さに伴う分散（エントロピーが log n に近づく）の定理は **softmax に対してのみ**存在し（Veličković らの Lemma 2.1: 有界ロジットなら全ての重みが Θ(1/n)）、線形 attention については存在しない。ノートはその類似（カーネル値が [m, M] なら最大重み ≤ (M/m)/n）を導出しているが、これは推論であって文献の主張ではない。回復手段として最も強い証拠を持つのは**構造を使わない**ものである: 局所 softmax 窓 64〜128 トークンが検索精度の 90.8% を取り戻し（Based）、Llama 3 8B の線形化では窓の有無だけで MMLU が 23.8 → 52.8 に動き（LoLCATs）、データ依存ゲートと delta 則が状態側の鋭さを回復し、製品の線形 attention LLM は**例外なく** softmax 層を 1/4〜1/8 残すハイブリッドである（Qwen3-Next 3:1、Kimi Linear 3:1、MiniMax-01 7:1、Jamba 7:1、Nemotron-H 8%）。したがって計画研究の中心仮説「構造は最も安い回復手段か」は、窓・ゲート・周期的 softmax 層という安く強い対照に対して立てなければならず、H4 の対照水準の組み替えが最優先の設計変更になる。一方、**言語の線形 attention / SSM に構文・グラフ構造を入れた研究は依然として見つからず**（3 クエリとも空振り、TreeGPT は「Mamba に木構造機構はない」と明言）、作用素型の雛形は Chimera（TMLR 2025、LM なし）が用意している。新規性は保たれるが、主張を「最も安い」から「窓・ゲート・周期 softmax と同コストで比較したときの回復量と、その大きさ・相補性依存」に限定するのが安全である。H3 は「長さに伴うエントロピー法則」ではなく「状態容量に対する検索失敗」として書き直す。

---

## 1. 線形 attention の LLM 系譜: 失われるのは argmax 的読み出しで、窓とゲートが大半を取り戻す

### 1.1 系譜と、失われる機構の五つの記述

系譜は、softmax を近似する特徴写像（Linear Transformer の elu+1、Performer の正の直交ランダム特徴、cosFormer の ReLU ＋ cos 再重み付け）→ 失敗の診断（The Devil in Linear Transformer、Hedgehog、Zoology / Based）→ データ依存の減衰・ゲート（RetNet → GLA / HGRN2 / RWKV-6）→ delta 則の状態更新（DeltaNet → Gated DeltaNet → RWKV-7 → Kimi Delta Attention → Gated DeltaNet-2）→ softmax からの蒸留（Hedgehog、SUPRA、Mamba-in-the-Llama、LoLCATs、Liger、RADLADS）と進んだ。初期の特徴写像の損失は大きい: 6 層 512 次元の WikiText-103 で 1+elu は 32.63/34.25 ppl、Performer は 75.29/77.65 ppl に対し vanilla は 29.63/31.01、GLUE 平均は 70.00 / 63.41 対 78.79 である（[Devil in Linear Transformer](https://ar5iv.labs.arxiv.org/html/2210.10340)）。

失われる「鋭さ」の記述は 5 種類あり、互いに補完的である。**(A) 勾配の非有界性**: softmax は ∂p_ij/∂s_ik が 1/4 で抑えられるがカーネル線形 attention は 1/(4 \|s_ik\|) で、Proposition 3.1 は任意の M について勾配が M を超える q, k の存在を示す（[Devil](https://ar5iv.labs.arxiv.org/html/2210.10340)）。**(B) 一様化（attention dilution）**: 線形 Transformer は「初期層でも系列全体に自明に分散する」。Hedgehog はこれを「低エントロピー（spiky）」と「内積単調性」の欠如と定式化し、合成連想想起で 1+ELU / Performer / cosFormer が 17.0% に対し softmax・Taylor・Hedgehog が 100% で、「t=2 で spikiness を上げるだけで課題が解ける」ことを示した（[Hedgehog](https://arxiv.org/html/2402.04347)）。PolaFormer は唯一の形式的主張として、非負特徴に対し g′>0 かつ g″>0 の要素写像のみが正系列エントロピー PSE を下げ、ReLU と ELU+1 はこの条件を満たさないことを証明した（[PolaFormer](https://arxiv.org/html/2501.15061v2)）。**(C) 低ランク**: rank(φ(Q)φ(K)ᵀ) ≤ min{N, d}。DeiT-Tiny で softmax は 196/196、線形は 54/196（[FLatten](https://arxiv.org/html/2308.00442v2)）。MHLA は「N に依らず d で頭打ち」と「N が増えると各トークンの寄与が無視できるほど小さくなり一様に近づく」を合わせて global context collapse と呼ぶが、後者は非形式的である（[MHLA](https://arxiv.org/html/2601.07832v2)）。**(D) 固定状態の検索容量**: Zoology は gated-conv と attention の Pile ppl 差の 82% が全トークンの 6.4% に過ぎない連想想起トークンに集中することを示し、70M の attention が 1.4B の gated-conv に連想想起で勝つ（[Zoology](https://arxiv.org/html/2312.04927)）。Based の Theorem 3.1 は、因果的な再帰モデルが MQAR を解くには Ω(N) ビットの状態が必要と述べる（[Based](https://arxiv.org/html/2402.18668v2)）。DeltaNet は加法更新では系列長 L が d を超えるとキーが「衝突」し消去できないと診断する（[DeltaNet](https://arxiv.org/html/2406.06484v6)）。**(E) 消去不能とゲートの粗さ**: Mamba2 のゲートは全ての連想を一様に減衰させ、DeltaNet は一度に 1 対しか書き換えられない（[Gated DeltaNet](https://arxiv.org/html/2412.06464v3)）。Bick らは SSM の検索ヘッドが「滑らかな attention パターン」を示し、Transformer の aggregate ヘッドが「argmax 的」であること、Llama-3.1-8B で 1 つの G&A ヘッドを切ると MMLU が 66% → 25% に落ちることを示した（[Gather-and-Aggregate](https://arxiv.org/html/2504.18574v2)）。

これら五つは一つの現象の別の面である: **d×d（または d_k×d_v）の固定状態は、N ≫ d のキーに対して argmax 的な読み出しを作れない**。鋭さを「エントロピー」だけで定義すると、この容量側の機構を取り逃がす。

### 1.2 構造なしの回復手段と、その効き目（論文自身の ablation）

| 手段 | 代表 | 回復の大きさ（同一条件の ablation） | 出典 |
|---|---|---|---|
| 特徴写像の鋭化（Taylor 2 次、Hedgehog の exp-MLP、FLatten の冪、PolaFormer の学習冪） | Based、Hedgehog、FLatten、PolaFormer | Based 362M の同一窓・畳み込み下で Performer 写像は FDA 0.36 / SWDE 8.10、Taylor d′=16 は 11.71 / 29.16。Hedgehog は WT-103 125M の差の 68.6%、GLUE 変換で 99.3% を回復。perplexity はほぼ回復するが検索は部分的 | [Based](https://arxiv.org/html/2402.18668v2)、[Hedgehog](https://arxiv.org/html/2402.04347) |
| 正規化（NormAttention）＋ 初期層の局所ブロック | TransNormer | WT-103 29.57/31.01（vanilla 29.63/31.01）、GLUE 79.38。ただし前半 6 層はブロック 64 の局所 attention で、純粋に構造なしではない | [Devil](https://ar5iv.labs.arxiv.org/html/2210.10340) |
| データ依存ゲート | GLA、HGRN2、Mamba | 学習 ppl: 固定減衰 16.55 → スカラーゲート 15.56 → 細粒度ゲート 14.77。1.3B の FDA は GLA 19.9 対 RetNet 14.3 対 Transformer++ 27.4 | [GLA](https://arxiv.org/html/2312.06635v6) |
| delta 則 ＋ ゲート | DeltaNet、GDN、KDA | 1.3B で GDN が Wiki ppl 16.42 / 平均 55.32、Transformer++ 18.53 / 52.25 を上回る。S-NIAH-2 4K で GDN 92.2 対 Mamba2 56.2 対 DeltaNet 18.6 | [Gated DeltaNet](https://arxiv.org/html/2412.06464v3) |
| **局所 softmax 窓 ＋ 線形大域** | Based（窓 64〜128）、LoLCATs（窓 64）、Samba / GDN-H1（SWA 2K） | Based: 窓 64〜128 が「softmax の検索精度の 90.8% を 1e-5 倍の遅延で」回復。LoLCATs Llama 3 8B: 窓なし MMLU 23.80 → 窓あり 52.77（元 66.6）；窓の寄与が attention 蒸留の寄与を上回る。DeltaNet 340M: SWA を 1 層おきに入れると SWDE 26.4 → 39.3 | [Based](https://arxiv.org/html/2402.18668v2)、[LoLCATs](https://arxiv.org/html/2410.10254v3)、[DeltaNet](https://arxiv.org/html/2406.06484v6) |
| **周期的な全対 softmax 層** | DeltaNet ＋ 2 層、Kimi 3:1、Qwen3-Next 3:1、MiniMax-01 7:1 | DeltaNet 1.3B に全対 attention を 2 層: SWDE 49.5 → 71.0（Transformer++ 66.6 を超える）。Kimi の比率 ablation は 3:1 が 5.65 で純 softmax 0:1 の 5.77 より良い | [DeltaNet](https://arxiv.org/html/2406.06484v6)、[Kimi Linear](https://arxiv.org/html/2510.26692) |
| softmax からの蒸留 | LoLCATs、Mamba-in-the-Llama、Liger | 5-shot MMLU の差の 77.8〜78.1% を回復（8B〜405B）が、窓を伴うときのみ。Mamba-in-the-Llama は attention 残存率 50/25/12.5/0% で ppl 2.09/2.20/—/3.36 と単調に劣化 | [LoLCATs](https://arxiv.org/html/2410.10254v3)、[Mamba in the Llama](https://arxiv.org/html/2408.15237v4) |

ablation の順序は一貫している: **局所 softmax 窓が単独で最大の検索回復**を与え、次にゲート＋delta 則、特徴写像の鋭化は perplexity を戻すが検索の差は部分的にしか埋めない。Jet-Nemotron は凍結した Qwen2.5-1.5B の MLP の下で attention の配置を探索し、MMLU には 2 層、検索には 2〜3 層だけが決定的で、しかも層が異なることを見つけた（[Jet-Nemotron](https://arxiv.org/html/2508.15884v3)）。MiniMax は 1:7 ハイブリッドを 456B 規模で訓練したが、M2 ではリーダーボードで同等に見えたハイブリッドが「複雑な多段推論で明確な欠損」を示し、SWA ハイブリッドは「文脈長が伸びると顕著に劣化」したため全層 softmax に戻った。線形 attention は「数値精度に遥かに敏感」で、検索パターンは「事前学習の早期に確立され」追加学習で直らなかったとも述べている（[MiniMax M2 blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)）。

### 1.3 長さに伴う悪化は「検索容量」として支持され、「エントロピー法則」としては支持されない

長さ依存の証拠は全て検索課題である: GDN の S-NIAH-2 は 1K/2K/4K/8K で 100.0/99.8/92.2/29.6（[Gated DeltaNet](https://arxiv.org/html/2412.06464v3)）、TNL は 16K の NIAH で 1B 以上でも最大約 10%（LLaMA 約 50%）（[Scaling Laws for Linear Complexity LMs](https://arxiv.org/html/2406.16690v1)）、Griffin は電話帳課題を局所窓 1024 までしか解けない（[Griffin](https://arxiv.org/html/2402.19427v1)）、Repeat After Me は Corollary 2.8 で mem(S) < L log D − 1 の GSSM は長さ L の複写で誤り > 1/2 と述べ、Pythia-410M が 2.8B の GSSM を L ≥ 70 の電話帳で上回る（[Repeat After Me](https://arxiv.org/html/2402.01032v2)）。Stuffed Mamba は正確な passkey 検索の最大長が状態サイズの指数関数で伸びることを示す（[Stuffed Mamba](https://arxiv.org/abs/2410.07145)）。一方で行エントロピーの長さ依存を測った曲線は言語モデルに存在せず、Hedgehog は softmax との KL が 256〜4096 トークンで「一定」と報告する（[Hedgehog](https://arxiv.org/html/2402.04347)）。Devil の dilution 指標は近傍を系列長の**割合** r で定義しており、長さの関数としての解析はない。

### 1.4 線形 attention に構文・グラフ構造を入れた研究は無い

線形 attention の内部に置かれた構造的事前分布は 1 次元の局所性・減衰だけである: cosFormer の cos 再重み付け、RetNet の γ^{n−m}、TransNormerLLM の LRPE ＋ 指数減衰（「attention dilution を避けるため」と明記）、Devil のブロック対角初期層、Log-Linear Attention の対数個の多尺度状態（[cosFormer](https://ar5iv.labs.arxiv.org/html/2202.08791)、[RetNet](https://arxiv.org/html/2307.08621v4)、[TransNormerLLM](https://arxiv.org/abs/2307.14995)、[Log-Linear Attention](https://arxiv.org/abs/2506.04761)）。Mamba-2 の SSD は選択的 SSM を「1-半分離マスク L を要素積した線形 attention」として書き、Polyline Path Masked Attention はこれを画像の隣接に置き換えた（[Mamba-2](https://arxiv.org/html/2405.21060v1)、[arXiv 2506.15940](https://arxiv.org/html/2506.15940v2)）。構文で attention を誘導する研究は全て二次 softmax（SG-Net、Syntax-BERT、Dependency Transformer Grammars、StructFormer）で、2026 年の規模拡大研究は標準因果 attention の構文 LM が構造マスク版と同等以上と報告する（[arXiv 2610.16837](https://arxiv.org/html/2610.16837)）。「syntax-guided linear attention」等の 3 クエリは線形 attention ＋ 構文の論文を返さなかった。

### 設計への含意（§4.2、§4.4、§3.2、§10）

1. **§4.4 因子 C（H4 の対照）を組み替える。** 文献の ablation が強い順に、(a) 局所 softmax 窓 w ∈ {64, 128} ＋ 線形大域（Based / LoLCATs 形、全融合層）、(b) データ依存ゲート ＋ delta 則（GLA / GDN 形、状態側の回復）、(c) 融合層 F 層のうち 1 層を全対 softmax にする（F = 4 なら 3:1、Kimi / Qwen3-Next 形）、(d) 特徴写像の鋭化（Hedgehog または Taylor 2 次、特徴数を揃える）。設計書の「ランク回復ゲート（GraphTARIF 形）」は GraphTARIF では**隣接を使う GAT 分岐**なので構造なしではなく、S = φ の外の一水準として移す。構造なしのランク回復は FLatten の深さ方向畳み込みか MHLA のトークン分割（[FLatten](https://arxiv.org/html/2308.00442v2)、[MHLA](https://arxiv.org/html/2601.07832v2)）。Gumbel 疎化は新ノートに支持証拠が無いので任意水準に落とす。
2. **§4.2 の softmax を「精度の上限」と呼ばない。** Kimi の 3:1 が 0:1 を上回り、GDN 1.3B が Transformer++ を上回る。softmax は「コストの基準」であり、ハイブリッド対照が softmax を超える可能性を事前登録に書く。
3. **H4 の「同コスト」は壁時計と KV / 状態のメモリで揃える。** 窓は O(Nw) の FLOPs と KV を足し、Jet-Nemotron は「KV キャッシュの大きさが長文脈スループットの最重要因子」と述べる。パラメータ一致だけでは窓の対照が有利に見える。
4. **§10 に「線形 attention は数値精度に敏感」（MiniMax、RWKV-7 の WKV7 カーネル）を追加**し、融合層の線形 φ は fp32 累積で実装する。

---

## 2. SSM / Mamba: 限界は検索容量と状態追跡、製品はすべてハイブリッド、作用素への構造注入は Chimera が雛形

### 2.1 理論の限界

三つの限界に定理がある。**状態追跡**: S4・Mamba・（Transformer も）TC⁰ に入り、S₅ の置換合成を解けない（[Illusion of State](https://arxiv.org/abs/2404.08819v3)）。非負の遷移固有値（exp / sigmoid ゲート）ではパリティも不可能で、固有値域を [−1, 1] に広げると DeltaNet の剰余算術が 31.4% → 97.1% になる（[Sarrof et al.](https://arxiv.org/abs/2405.17394v1)、[Grazzi et al.](https://arxiv.org/abs/2411.12537)）。**複写・文脈内検索**: 2 層 Transformer は指数長の文字列を複写できるが固定状態モデルは状態サイズで制限される（[Repeat After Me](https://proceedings.mlr.press/v235/jelassi24a.html)）。RNN ＋ CoT でもボトルネックは文脈内検索で、**Transformer 層を 1 層**足すか RAG を付ければ多項式時間の全問題が解ける（[Wen et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/79dc391a2c1067e9ac2b764e31a60377-Abstract-Conference.html)）。**長さ一般化**: 訓練長を超えると少数チャネルが爆発する state collapse が起き、学習不要の緩和で 1M トークンを超えられる（[Stuffed Mamba](https://arxiv.org/abs/2410.07145v1)）。逆に SSM が得意なのは star-free 言語と有界 Dyck の厳密・長さ一般化する解で、Transformer はこれを近似しかできない（[Sarrof et al.](https://arxiv.org/abs/2405.17394v1)）。Jamba の 1.3B 純 Mamba は IMDB / QuAC / NarrativeQA で「要求された回答形式を無視」し、著者は attention 層に induction 的ヘッドを見つけた。また「RoPE なしの Jamba が RoPE ありと同等」で、Mamba 層が暗黙の位置情報を与える（[Jamba](https://arxiv.org/html/2403.19887v2)）。Chimera はこれを一般化し、SSM のマスク L = (I − A)⁻¹ が「位相を符号化するので位置埋め込みは不要」と述べる（[Chimera](https://arxiv.org/html/2510.12111)）。

### 2.2 製品・公開モデルにおける attention : 線形/SSM の比率

| モデル（年） | attention : 線形/SSM | 配置 | 著者の理由 / ablation | 出典 |
|---|---|---|---|---|
| Jamba（2024） | 1 : 7（層） | 8 層ブロック内に 1 層、MoE 2 層ごと | 1.3B の ablation で 1:3 と 1:7 が同等、効率で 1:7 を採用。ハイブリッドが純 attention・純 Mamba の両方に勝つ | [Jamba](https://arxiv.org/html/2403.19887v2) |
| Griffin / RecurrentGemma（2024） | 1 : 2（ブロック） | 局所 MQA（窓 1024、RecurrentGemma は 2K と記載） | Hawk（再帰のみ）は電話帳で失敗、Griffin は窓長まで完全。選択複写は「局所 attention 1 層で」Transformer に並ぶ | [Griffin](https://arxiv.org/html/2402.19427v1)、[Google blog](https://developers.googleblog.com/en/gemma-explained-recurrentgemma-architecture/) |
| Samba（ICLR 2025） | 1 : 1（Mamba と SWA を層ごとに交互、MLP 併置） | SWA 窓 2048 | SWA が「Mamba の再帰状態が明瞭に捉えない中短期の詳細信号を取り出す」。4K 訓練で 256K passkey 完全 | [Samba](https://arxiv.org/html/2406.07522v2) |
| Mamba-2-Hybrid 8B（NVIDIA 2024） | 7% attention / 43% Mamba-2 / 50% MLP | 分散配置 | 純 SSM は 5-shot MMLU・電話帳・長文脈推論で劣る。ハイブリッドは Transformer に平均 +2.65 | [Waleffe et al.](https://arxiv.org/abs/2406.07887) |
| Nemotron-H（2025） | ≈ 8%（8B: 4/52、56B: 10/118） | 均等分散 | （理由の ablation は未抽出） | [Nemotron-H](https://arxiv.org/pdf/2504.03624) |
| Hymba-1.5B（2024） | attention : Mamba パラメータ 1 : 2.12 で飽和 | 同一層内で並列ヘッド | 「attention ヘッドは高解像度の想起、SSM ヘッドは効率的な文脈要約」。比率を上げると改善が飽和 | [Hymba](https://arxiv.org/pdf/2411.13676) |
| Falcon-H1（2025） | チャネル SSM : attention : MLP = 2 : 1 : 5 | 同一ブロック内で半並列 | 「attention チャネルを増やすと性能が劣化」 | [Falcon-H1](https://arxiv.org/html/2507.22448v1) |
| MiniMax-01（2025） | 1 : 7（層） | 7 transnormer ブロックごとに softmax 1 層 | 純線形は「NIAH を除き同等」だが検索不能で LLM に不適。M2 では全層 softmax に回帰 | [MiniMax-01](https://arxiv.org/html/2501.08313v1)、[M2 blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) |
| Qwen3-Next-80B-A3B（2025） | 1 : 3（12 attention ＋ 36 Gated DeltaNet） | 4 層周期 | 「超長文脈の効率的なモデリング」 | [HF model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| Kimi Linear 48B-A3B（2025） | 1 : 3（MLA : KDA） | 4 層周期、MLA は NoPE | 比率 ablation（検証 ppl）: 3:1 5.65、1:1 5.66、7:1 5.70、0:1 5.77、15:1 5.82。「長文脈検索が純線形の主要ボトルネック」 | [Kimi Linear](https://arxiv.org/html/2510.26692) |
| Jet-Nemotron-2B（NeurIPS 2025） | 全対 2 層 ＋ SWA 2 層 / 28 | 探索で決定（層 15, 20 全対、21, 22 SWA） | MMLU は 2 層、検索は 2〜3 層だけが決定的 | [Jet-Nemotron](https://arxiv.org/html/2508.15884v3) |
| DeltaNet ハイブリッド（2024） | SWA を 1 層おき、または全対 2 層 | — | 全対 2 層で SWDE 71.0 > Transformer++ 66.6 | [DeltaNet](https://arxiv.org/html/2406.06484v6) |
| Gated DeltaNet H1 / H2（ICLR 2025） | GDN ＋ SWA 2K（H2 は Mamba2 も） | — | 平均 56.40 / 56.18 対 純 GDN 55.32 | [Gated DeltaNet](https://arxiv.org/html/2412.06464v3) |
| Based / LoLCATs | 全層で窓 64〜128 ＋ 線形 | 層内 | 窓 64〜128 で検索の 90.8%；窓が MMLU 23.8 → 52.8 | [Based](https://arxiv.org/html/2402.18668v2)、[LoLCATs](https://arxiv.org/html/2410.10254v3) |
| Mamba-in-the-Llama（NeurIPS 2024） | attention 残存 50 / 25 / 12.5 / 0% | 漸進置換 | MT-Bench 7.35 / 6.86 / 6.46 / 5.64、ppl 2.09 / 2.20 / — / 3.36 | [Mamba in the Llama](https://arxiv.org/html/2408.15237v4) |
| Zamba（2024）、Granite 4.0（2025） | 共有 attention ブロックを約 6 Mamba ブロックごと；9 : 1 | — | 二次情報（§8） | [Zamba](https://arxiv.org/abs/2405.16712v1)、[Raschka gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/) |

**対照腕への結論。** 比率は定数ではなく、曲線は早く平坦になる（Jamba 1:3 ≈ 1:7、Hymba の飽和、Falcon-H1 の「増やすと悪化」、Kimi の 3:1 > 1:1）。したがって融合層 F 層の構造なしハイブリッド対照は、**(a) 全層に局所 softmax 窓 64〜128（Based 形）**と **(b) F 層のうち 1 層だけ全対 softmax（F = 4 で 3:1、文献の最頻比）**の 2 腕で文献を代表できる。製品の全例が示す「検索のために softmax を残す」という事実は、H4 の帰無仮説「構造は鋭さの回復に必要ではない」を強く予想させる。

### 2.3 公開 Mamba 系 LLM と微調整

24 GB で現実的な候補は、純 SSM の Falcon-Mamba-7B と Codestral Mamba 7B（4 bit ＋ LoRA）、ハイブリッドの Falcon-H1 0.5B〜7B、Hymba-1.5B、Zamba2-7B である。Jamba の LoRA はモデルカード自身が「約 120 GB、2×A100 80GB」を要求する（[Jamba discussion #49](https://huggingface.co/ai21labs/Jamba-v0.1/discussions/49)、[Falcon-H1](https://github.com/tiiuae/Falcon-H1)、[Hymba-1.5B](https://huggingface.co/nvidia/Hymba-1.5B-Base)、[Zamba2-7B](https://huggingface.co/Zyphra/Zamba2-7B)、[HF Mamba2 docs](https://huggingface.co/docs/transformers/en/model_doc/mamba2)）。LoRA は**線形射影（in_proj / out_proj / x_proj / embeddings）に効き、SSM パラメータ自体にはほとんど効かず**、prefix tuning は無効である（[Galim et al.](https://arxiv.org/abs/2410.09016v1)、[HF Mamba2 docs](https://huggingface.co/docs/transformers/en/model_doc/mamba2)）。flash-linear-attention は HF 互換で、config の `attn` で任意層に標準 attention を差し込めるため、自作ハイブリッドの最短経路になる（[FLA README](https://github.com/fla-org/flash-linear-attention)）。

### 2.4 再帰の作用素に構造を入れる三つの方法

**Chimera（TMLR 2025）**: SMA 形 Y = (L ∘ CBᵀ)V において系列の L = (I − A)⁻¹ = ΣAⁱ（A は有向線グラフの重み付き隣接）であることを使い、A を任意グラフから作る。A_ij = exp(−(Δ_i + Δ_j + Δ′_ij)/3) で両端ノードの選択性と辺項を入れ、行正規化で ‖A‖ < 1 を保つ。**DAG では隣接が冪零なので和が有限になり、トポロジカル順の 1 パス O(\|V\|+\|E\|)** で h_i = Σ_{j∈parents(i)} A_ij h_j − B̄_i v_i。無向・巡回グラフは 2 本の有向線グラフや 4 本の格子 DAG に分解するか、直径で打ち切った Neumann 級数（最悪二次、厳密は三次）で近似する。GLUE 83.93（BERT-Base 83.2）、ImageNet-1k 81.4（ViT-B 78.8）、22M の ablation で 2D-DAG 77.8 対 双方向 1-D 76.5 対 前方 1-D 73.8、LRGB では近似版が GraphGPS ＋ Transformer と同等、現実装は Transformer より約 1.5 倍遅い（[Chimera](https://arxiv.org/html/2510.12111)）。**木スキャン**: GrootVL / MambaTree は入力依存の全域木上で動的計画法により線形時間で状態を伝播する（[GrootVL](https://arxiv.org/abs/2406.02395)）。**pLSTM**: DAG の線グラフ上で Source / Transition / Mark ゲートを置き多次元 associative scan で並列化する（[pLSTM](https://arxiv.org/html/2506.11997v1)）。双方向の 1-D 作用素としては Hydra の擬分離行列がある（[Hydra](https://arxiv.org/abs/2407.09941v1)）。多方向ラスタースキャン（VMamba 等）は 1-D パスの合成で層あたり k 倍のコストを払い、ベースラインに留める（[VMamba](https://arxiv.org/html/2401.10166v3)）。**構文木 / AST 上で木構造 SSM を走らせた論文は見つからず**、TreeGPT は「Mamba は逐次処理が中心で木構造に必要な機構を欠く」と明言して TreeFFN を作っている（[TreeGPT](https://arxiv.org/html/2509.05550v2)）。

### 2.5 カーネルと消費者向け GPU

Mamba-2 SSD は A100 でコア層のみ 2K で FA2 と交差し 16K で 6 倍、ただし著者自身が「2K 程度の短い系列ではモデル全体は Transformer より遅い可能性」と述べる（[Mamba-2](https://arxiv.org/pdf/2405.21060)）。FLA の GB200 表では B=4, T=2048 で FA2 0.346 ms 対 GDN 0.753 ms と **FA2 が勝ち**、B=1, T=8192 で逆転する（[FLA README](https://github.com/fla-org/flash-linear-attention)）。RTX 3090 / 4090 での Mamba 対 FA2 の一次資料は無い。mamba-ssm は Linux ＋ CUDA（ROCm はパッチ）で Windows は非公式、FLA は Triton で NVIDIA / AMD / Intel 対応（[mamba README](https://raw.githubusercontent.com/state-spaces/mamba/main/README.md)）。

### 設計への含意（§4.2、§4.3 の SSM 列、§4.7、§13 項目 5）

1. **SSM 融合層は Chimera 形で確定する。** 結合列 [トークン ; ノード] を一つの DAG にする: テキストは有向線グラフ、AST は親→子（＋兄弟、次トークン）の有向辺で自然に DAG、整合辺はトークン→ノードの有向辺。これで**作用素型 SSM が線形時間の 1 パス**になり、「構造なし」水準は系列順の線グラフ、「φ の内」水準は隣接を A に入れた形、と §4.3 の 3 水準が同一実装の A の違いになる。一般グラフ（TAG、合成グラフ）は辺を 2 方向の DAG に分解するか直径打ち切りの近似（二次）を使い、その二次コストを §6.2 で測る。
2. **SSM の「φ の外」水準は Chimera の Δ′_ij（辺項）**、「厳密マスク」の対照は同じ隣接上のマスク付き softmax（FlexAttention）と定める。Chimera 自身が SSD の L と等価だと述べるので、比較は「線形マスク＋状態」対「softmax＋明示的対スコア」を分離する。
3. **状態追跡が要る課題（合成グラフのサイクル検出・到達可能性）では固有値域 [−1, 1] を任意水準に加える**（Grazzi）。必須ではないが、否定結果が出たときの説明変数になる。
4. **§4.7 の Mamba 系 LLM は Falcon-H1 1.5B / 3B（ハイブリッド）と Codestral Mamba 7B または Falcon-Mamba-7B（純 SSM、4 bit）を候補とし、LoRA は射影のみ**に当てる。Jamba は除外。
5. **§13 項目 5 は解決**: 「Chimera 形のレゾルベント、DAG は厳密、巡回は直径打ち切り、構造を入れる位置は A の辺重み」。
6. **P8 のマイクロベンチには短い系列（128〜2048）を必ず含める**。文献上そこで線形・SSM は FA2 に負ける。

---

## 3. 鋭さの理論と測定: softmax の分散は定理、線形の分散は推論、指標は支持の大きさで正規化する

### 3.1 形式的結果の一覧

| 結果 | 正確な主張 | 条件 | 対象 | 出典 |
|---|---|---|---|---|
| Veličković et al. Lemma 2.1 | m ≤ e_k ≤ M、δ = M − m、温度 θ のとき全ての k で (1/n)e^{−δ/θ} ≤ α_k^{(n)} ≤ (1/n)e^{δ/θ}、すなわち α_k = Θ(1/n) | ロジットが有界 | softmax | [Softmax is not Enough](https://arxiv.org/html/2410.01104v3) |
| 同 Theorem 2.2 | 有限語彙と連続な層の下で、任意の θ, ε に対し n が存在して全ての softmax_θ(e^{(n)})_k < ε | 有限入力語彙、連続 FF / attention | softmax（全ての大域ヘッド） | 同上 |
| 同 Prop. 3.1 / 3.2 | δ ≤ 2σ_max(Q)σ_max(K)‖y‖max_i‖x_i‖；dH/dβ = −β Var(e)（エントロピーは β = 0 で最大、\|β\| に単調） | — | softmax | 同上 |
| Nakanishi（SSMax） | max_i softmax(z)_i ≤ 1/((n−1)e^{−(z_max−z_min)} + 1) → 0 | ロジット幅固定 | softmax | [SSMax](https://arxiv.org/html/2501.19399v1) |
| Vasylenko et al.（ASEntmax） | 任意の有界ロジットで H(softmax)/log n → 1（complete dispersion）。α-entmax のエントロピーは O(log s)、支持 s = O(n^β) なら正規化エントロピー ≤ β < 1 | 固定温度 | softmax 対 α-entmax | [ASEntmax](https://arxiv.org/html/2506.16640v4) |
| Chiang & Cholak Prop. 3 | ロジットを log n 倍すると FIRST を任意の交差エントロピー η 以下で認識できる（Hahn: スケールなしでは文長とともに 1 bit に近づく） | 単一記号依存の言語 | softmax | [ACL 2022](https://aclanthology.org/2022.acl-long.527) |
| Zhai et al. Theorem 3.1 | σ̄ = ‖W_K W_Qᵀ‖₂‖XXᵀ‖₂、β = exp(−σ̄√(T/(T−1))) として Ent(A_i) ≥ log(1 + (T−1)β) + σ̄√(T(T−1))β/(1 + (T−1)β)（tight） | 重みノルム有界 | softmax（下界、崩壊側） | [Entropy Collapse](https://arxiv.org/html/2303.06296v2) |
| Barbero et al. 2024 Thm 4.2 / Lemma B.1 | 長さ n と n+1（末尾反復）の最終表現は n → ∞ で任意に近づく；付加トークンの softmax 重みは 0 に向かう。**速度は与えられず「1/n を支持するものは無い」** | 有界表現、距離で減衰する PE | softmax | [Transformers need glasses](https://arxiv.org/html/2406.04267v2) |
| Dong et al. Theorem 2.3 | ‖res(SAN(X))‖_{1,∞} ≤ (4γβH/√d_qk)^{(3^L−1)/2}‖res(X)‖_{1,∞}^{3^L}（深さで二重指数） | スキップなし、4γβH < √d_qk | softmax、**深さ** | [Rank Collapse](https://arxiv.org/html/2103.03404v2) |
| Wu et al. 2023 Theorem 1 | attention 型 GNN / グラフ Transformer のノード表現は深さに指数的に収束 | 仮定 A1〜A4 | グラフ attention、**深さ** | [Oversmoothing](https://arxiv.org/abs/2305.16102) |
| FLatten / MHLA | rank(φ(Q)φ(K)ᵀ) ≤ min{N, d}；「N に依らず d で上界」 | 非負特徴写像 | **線形** | [FLatten](https://arxiv.org/html/2308.00442v2)、[MHLA](https://arxiv.org/html/2601.07832v2) |
| GraphTARIF Theorem 1 / 2 | E[tr S_B(MX)] ≤ C·r（attention 作用素のランク r がクラス間散布を上界）；低ランク平滑化の合成は E[tr S_B] を狭義に減らす。**エントロピーとランクを結ぶ定理は無い** | — | **線形**グラフ Transformer | [GraphTARIF](https://arxiv.org/html/2510.10631v2) |
| PolaFormer Theorem 1 | 非負特徴に対し g′ > 0 かつ g″ > 0 の要素写像は PSE を下げる；ReLU と ELU+1 は条件を満たさない | — | **線形** | [PolaFormer](https://arxiv.org/html/2501.15061v2) |
| Devil Prop. 3.1 | 任意の M > 0 に対し勾配の大きさが M を超える q, k が存在（softmax は 1/4 で有界） | カーネル分母あり | **線形** | [Devil](https://ar5iv.labs.arxiv.org/html/2210.10340) |
| Zoology Claim 1 / Prop. 4.3 | 2 層 gated-conv は AR を解くのにモデル次元が系列長に少なくとも線形で必要；attention は softmax なしでも O(c²) パラメータで MQAR を解く | — | 容量 | [Zoology](https://arxiv.org/html/2312.04927) |
| Based Theorem 3.1 | 因果的に入力に依存する再帰モデルは MQAR に Ω(N) ビットの状態を要する | 状態要素が有界 | 容量（線形・SSM） | [Based](https://arxiv.org/html/2402.18668v2) |
| Repeat After Me Thm 2.3 / 2.7 / Cor. 2.8 | 深さ 2・次元 O(n log D) の Transformer は誤り < L²D^{−n} で複写；GSSM の誤り > 1 − \|S\|/D^L；mem(S) < L log D − 1 なら誤り > 1/2 | 一様文字列 | 容量（SSM） | [Repeat After Me](https://arxiv.org/html/2402.01032v2) |
| **線形 attention の長さ依存分散** | **定理は存在しない。** Devil（近傍を割合で定義、長さ解析なし）、Hedgehog（KL は 256〜4096 で一定）、MHLA（非形式的な「N が増えると各トークンの寄与が無視できる」） | — | **線形** | 上記 |
| ノートの推論（文献の主張ではない） | 非負 φ でカーネル値が [m, M]（m > 0）なら a_j ∈ [m/(nM), M/(nm)]、max_j a_j ≤ (M/m)/n = Θ(1/n)。softmax は δ を重みノルムで log n に比例して伸ばせる（指数比 e^{δ/θ}）が、elu+1 / ReLU 型は比 M/m が特徴ノルムの**多項式**でしか伸びない | — | **線形** | `attention_sharpness_entropy_theory.md` KQ1 |

要点は三つ。第一に、softmax の分散定理は全て「ロジット幅 δ が有界なら重みは Θ(1/n)」という一つの機構に帰着し、修正は log n スケーリング（Chiang & Cholak、SSMax、ASEntmax、Su の entropy-invariance）か疎な正規化（entmax、Softpick、sigmoid）に収束している。第二に、崩壊側（Zhai、Hong & Lee）と分散側は同じレバー δ で制御される。第三に、**線形 attention について「エントロピーが長さで増える」ことは示されておらず、計画研究がそれを前提に書くと文献上の根拠を失う**。

### 3.2 sink と「中身のない鋭さ」

sink は softmax が和を 1 にしなければならないために余剰質量を常に見える初期トークンに捨てる現象で、Pythia-14M にも存在し、十分なデータ（50〜200M トークン）の後に現れる。sigmoid attention では 1B で sink 率 2.46% 対 softmax 45.11%（[Gu et al.](https://arxiv.org/html/2410.10781v2)）。Gemma 7B の ⟨bos⟩ を外すと attention map が滑らかになり ARC-e 80.77 → 28.49、RULER-4096 82.57 → 0.00 と崩れ、sink 率は事前学習の文脈長とともに上がる（128 では「本質的に形成されない」）（[Barbero et al. 2025](https://arxiv.org/html/2504.02732v4)）。BERT では層 6〜10 の attention の半分以上が [SEP] に向く（[Clark et al.](https://arxiv.org/pdf/1906.04341)）。**低エントロピーは sink によっても達成されるので、行エントロピーだけでは「内容を選ぶ鋭さ」と「捨て場への集中」を区別できない**。Hong & Lee が指摘するとおり sink 指標は列方向、エントロピーは行方向である（[Hong & Lee](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.421.pdf)）。

### 3.3 グラフでの同じ現象

GAT の係数は困難領域で「ほぼ一様」になり GAT ≈ GCN となる（[Fountoulakis et al.](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/JMLR/2023/graph_attention_retrospective__072abd0f.txt)）。ノードごとの最大エントロピーは log deg(v) なので生のエントロピーはノード間で比較できない（[DGL GAT blog](https://www.dgl.ai/blog/2019/02/17/gat.html)）。大域 attention は「遠いノードに過剰に集中」する over-globalizing を起こす（[Less is More](https://arxiv.org/abs/2405.01102v2)）。softmax グラフ Transformer の正規化エントロピーをグラフサイズ N の関数として測った論文は無い。

### 3.4 測定手順（ノートの統合。引用された確立手順ではない）

| 指標 | 定義 | 長さ・次数をまたぐ比較 | 用途 |
|---|---|---|---|
| 正規化エントロピー Ĥ | H(A_i)/log(支持の大きさ)。支持 = 因果なら i、グラフなら deg(v)、全対なら n。Oh & Schuler の NAE、Vasylenko の dispersion ratio と同形 | 可 | 主指標（行方向） |
| 実効支持比 | exp(H)/支持。「実質的に見ているキーの割合」。Θ(1/n) の重みなら定数に収束 | 可 | Ĥ の直感的な対 |
| 最大重み / top-k 質量 | max_j A_ij、固定 k の上位質量（Voita の confidence） | 可（自明下界 1/n、k/n を併記） | 行動指標との橋渡し |
| sink 質量 | 指定キー（BOS / CLS / 仮想ノード）への平均質量と、Gu らの Sink_k^ε（ε = 0.3、T と系列数を明記） | ε と T を固定すれば可 | 列方向、「中身のない鋭さ」の除外 |
| 実効ランク | erank(A) = exp(H(σ/Σσ))（Roy & Vetterli）を attention 行列**と層出力**の両方に | 行列サイズに依存するので同 N で比較 | GraphTARIF Thm 1 と Dong らの残差に対応 |
| 構造質量 | 1-hop / 2-hop / 整合辺上の質量の割合（設計書 §3.1 既存） | 近傍サイズを併記 | 構造が読まれたか |
| 線形 attention での計算 | 暗黙重み φ(q)ᵀφ(k_j)/Σ_i φ(q)ᵀφ(k_i)（PSE と同一）に同じ統計を適用 | 同 n で softmax と比較 | 正規化子の効果を長さ効果から分離 |
| 行動指標 | 合成検索課題の正答率を長さ別に（MQAR / S-NIAH 相当。本研究では「属性 X を持つノードはどれか」型） | 可 | 鋭さと精度の連鎖の証拠 |

### 設計への含意（§3.1、H3、§8 P4、§10）

1. **§3.1 を上の表で置き換える。** 支持の定義を層・水準ごとに明記し、生エントロピーは報告しない。実効ランクは attention 行列と層出力の両方。sink 質量は必ず別途報告し、結合列の仮想ノードや CLS があれば除外した上で Ĥ を再計算する。
2. **H3 を書き直す。** 「線形 attention の一様化は入力が長いほど悪化する」は文献に無い。支持されるのは「固定状態の検索は L が状態容量を超えると失敗する（Ω(N) ビット、Cor. 2.8、S-NIAH の 4K → 8K の崩落）」である。H3 は「構造は実効支持を deg(v) に縮めて状態に要求される容量を下げるので、回復量は L_t と N に増える」と述べ、鋭さ指標は長さ別に層別し、**同じ長さで合成検索課題の正答率を併記**する。反証条件は「回復量が L に平坦で、かつ検索正答率の長さ依存も無い」。
3. **P4 に長さ層別（例: 256 / 1K / 4K / 8K トークン、N = 50 / 500 / 5000）を入れ、softmax 対線形の Ĥ 差が長さで開くかを「測る」**。これが H3 の前提の実測になり、文献の空白（線形 attention の行エントロピー対 N の曲線が無い）をそのまま埋める。
4. **§10 に「sink / 仮想ノードによる偽の鋭さ」を追加。** 結合列に CLS 型の読み出しトークンを置くなら、それが sink になることを前提に指標から外す。

---

## 4. GRF・位置符号化・整合: 重要度重みなしの GRF は無マスクと同じ、既定の PE は RRWP、整合は GraphCodeBERT 形

### 4.1 GRF の仕組みと失敗条件

GRF は正規化隣接 W の冪級数 K_α = Σ_k α_k W^k の不偏ランダムウォーク推定量で、各ノードから m 本の打ち切り歩みを出し、訪問ノードの特徴に（辺重みの積）× f_len / p(ω) を加える。ICLR 2025 の位相マスクでは学習可能な Taylor 係数 α_k を f の自己畳み込みに分解し、結合特徴 vec(φ(q_i) ⊗ φ_G(v_i))（長さ mN）の内積が K_LR,ij · M_α,ij になる（[Topological Masking](https://arxiv.org/html/2410.03462v2)）。同論文の濃度不等式 Pr(\|M̂ − M\| > t) ≤ 2exp(−t²n³/(2(2n−1)²c⁴)) は N に依存せず、次数・辺重み・p_halt は c = Σ_k \|f_k\|(max\|W_ij\| d_i/(1−p_halt))^k を通じてのみ入る。**直径の関数としての分散上界はどの論文にも無い**。

実務上の失敗条件は明確である。**(i) 重要度重みなしでは無効**: ImageNet ViT で GRF マスク 0.730、無マスク線形 0.693、1/p(ω) と辺重みを落とした「ad-hoc」マスク 0.689、同次元の学習可能グラフ特徴 0.696（[Topological Masking](https://arxiv.org/html/2410.03462v2)）。**(ii) 歩みが交わらない対はゼロ**: 距離 r を覆う確率は p_next^{(r−2)} で減り、GRFs++ の縫合（l 対の短い歩みの特徴行列の積）は l = ⌈r/2⌉ で p_next に依らない下界を与える。長距離実験は直径 159 の Peptides で行われた（[GRFs++](https://arxiv.org/html/2510.07716v1)）。**(iii) 分散低減**: 反対称打ち切り（q-GRF、p ≤ 1/2 で同時停止なし、拡散 MSE が ER で 0.0179 → 0.0085）と反発歩みは不偏のまま差し込める（[q-GRFs](https://arxiv.org/html/2305.12470)、[Repelling Random Walks](https://arxiv.org/abs/2310.04854)）。**(iv) 厳密計算が勝つ**: 5,985 頂点メッシュで厳密 0.9194 > GRF++ 0.9154 > GRF 0.9091（[GRFs++](https://arxiv.org/html/2510.07716v1)）。報告された設定は、ICLR 2025 のスクラッチ学習で **n = 100 本、p_halt = 0.1、最大長 100**、事前学習済みの微調整で n = 20、p_halt = 0.1、最大長 10、点群で 3 ホップ反発歩み n = 27。**公開コードは ICLR 2025 の位相マスクも GRFs++ も見つからず**（「受理後に公開」のまま）、著者リポジトリにあるのは general_graph_random_features、antithetic_termination、repelling_random_walks の 3 つである。

### 4.2 位置符号化の理論と費用

Black らは APE-GT と RPE-GT の識別力が相互変換可能であること（Thm 3.8 / 3.10）、ただし RPE → APE はノード特徴があると狭義に弱くなり得ること（Example 1）、正規化隣接・Laplacian・熱核の**冪の積み重ねはスペクトル核と少なくとも同等**で追加費用が無いこと（Thm 4.6 / 4.8 / 4.9）を示した（[Black et al.](https://arxiv.org/html/2402.14202v4)）。GRIT の RRWP は (D⁻¹A)^k の積み重ねで SPD を狭義に上回り（Prop. 3.2）、PPR と熱核を特殊例に含み、K の感度は ZINC MAE で K=2 0.147、7 0.063、21 0.059、42 0.061 と 7 以上で平坦（[GRIT](https://arxiv.org/pdf/2305.17589)）。GDT は AST 由来の OGB-Code2 で NoPE 19.27、LPE 19.48、SPE 19.35、RWSE 19.39、RRWP 19.42 と **F1 の差 0.2 に対し RRWP のステップ時間 0.219 s（NoPE 0.092 s）、メモリ 19.2 GB（9.7 GB）**で「PE の効率は大きく変わるが予測性能の差は小さい」（[GDT](https://arxiv.org/html/2511.08028v1)）。OpenGT の PubMed（19.7k ノード）前処理はランダムウォーク統計 10 分、スペクトル 1 時間、全対最短路 2 時間（[OpenGT](https://arxiv.org/html/2506.04765v1)）。SPE は符号・基底の曖昧さを安定に解くが固有分解と n×n×m テンソル上の GIN を要する（[SPE](https://arxiv.org/html/2310.02579v3)）。木固有の符号化（Shiv & Quirk の n·k 次元アフィン符号化、Code Transformer の祖先・兄弟・SPD・PPR の 32 ビン、TPTrans の bi-GRU パス）は RRWP と直接比較されていない（[Shiv & Quirk](https://proceedings.nips.cc/paper_files/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf)、[Code Transformer](https://arxiv.org/pdf/2103.11318)、[TPTrans](https://papers.nips.cc/paper_files/paper/2021/file/4e0223a87610176ef0d24ef6d2dcde3a-Paper.pdf)）。

### 4.3 トークンとノードの整合

三つの再利用可能な規則がある。**葉ノード ↔ サブワード区間**: GraphCodeBERT は `tree_to_token_index` で葉の区間を取り、`ori2cur_pos` でサブワード区間に写し、ノードを列の末尾に追加して 0/−∞ マスクで自分の区間とデータフロー前任者にだけ注意させる（ノード位置埋め込みは 0、Node Alignment 事前学習を外すと MRR 0.713 → 0.703〜0.707）（[GraphCodeBERT](https://arxiv.org/pdf/2009.08366)、[run.py](https://raw.githubusercontent.com/microsoft/CodeBERT/master/GraphCodeBERT/codesearch/run.py)）。**内部ノード ↔ 最小包含範囲**: Code Transformer は各トークンに「ソース範囲が最短で含む AST ノード」を割り当て、範囲が重なる場合は短い子へ貪欲に降りる（[Code Transformer](https://arxiv.org/pdf/2103.11318)）。**葉のみ ＋ 開始トークン**: GREAT は内部ノードの辺を区間の開始トークンの葉に下ろし、ノード数を 2〜3 分の 1 にする（[GREAT](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2020/global_relational_models_of_source_code__ae1fb05d.txt)）。NLP では Sachan らが依存辺を先頭サブワード → 残りサブワードに拡張し、**金の構文木では SRL +3.5 F1 だが Stanza の予測木では利得がほぼ消える**（[Sachan et al.](https://arxiv.org/pdf/2008.09084)）。プーリングは「先頭サブワード」が形態論で最悪、attention / LSTM プーリングが最良である（[Ács et al.](https://arxiv.org/abs/2102.10864)）。tree-sitter の C/C++ 文法は前処理後のコードを対象とし、マクロと `#if` 分岐は ERROR ノードや誤解析（クラスが関数として解析される）を生む（[tree-sitter-cpp issue #293](https://github.com/tree-sitter/tree-sitter-cpp/issues/293)、[bug-gnu-emacs](https://lists.nongnu.org/archive/html/bug-gnu-emacs/2022-11/msg01966.html)）。GraphCodeBERT の `DFG.py` はループ本体を 2 回処理する近似で、`run.py` の素の `except` は失敗を空のデータフローとして黙って通す（[DFG.py](https://raw.githubusercontent.com/microsoft/CodeBERT/master/GraphCodeBERT/codesearch/parser/DFG.py)）。

### 設計への含意（§4.3 の φ 内、§10、§12、P1、P5）

1. **φ 内の既定設定**: 対称正規化 w_ij = 1/√(d_i d_j)（c を有限に保つ）、重要度重み必須、**n = 100、p_halt = 0.1、最大長 100** を出発点、反対称打ち切り（p ≤ 0.5）を既定で入れ、AST の深さ 10〜50 に対しては**縫合 l = 2〜⌈r/2⌉** を P5 で掃引する。GRFs++ の報告値は l = 2、p_halt = 0.1、m ∈ {16, 256}。
2. **N ≤ 10k では厳密な打ち切り冪級数 Σ_{k<K} α_k W^k（K 本の疎行列積、O(K\|V\|\|E\|)）を φ 内の「厳密版」として必ず置く。** GRF は N×N マスクを実体化できないときの近似であり、ICLR 2025 自身が 30k ノード超で動機づけている。AST では厳密版の方が安く、しかも精度が上回る可能性が高い（メッシュの 0.9194 対 0.9154）。H1 の「φ の内」は厳密版と GRF 版の 2 つに分け、GRF の分散が回復量を食う量を直接報告する。
3. **位置符号化の既定は RRWP（K = 21〜32、関係バイアスとして）を木と一般グラフの両方に固定し、ノード特徴だけで済む所は RWSE（対角）**。理由: 符号・基底の曖昧さが無い、Black らの定理で冪の積み重ねはスペクトル核と同等、GRF マスクが推定する対象 Σ θ_k W^k と同一物なので厳密対 GRF の ablation が綺麗、GDT で AST 上の PE 差は 0.2 F1。LapPE / SPE は副次水準。RRWP の N×N×K メモリ（N = 10k、K = 32 で fp32 12.8 GB）は GDT と同じく実行時計算にする。Black らの Example 1 に従い、構造信号は**ノード PE をトークンに足すだけでなく関係バイアス / マスクとして**入れる。
4. **整合規則**: 葉 ↔ サブワード区間（GraphCodeBERT 形）、内部ノード ↔ 最小包含範囲（Code Transformer 形）、ノード特徴は区間の**平均または attention プーリング**で先頭サブワードは使わない。ノードを列として入れ区間マスクで学習させる無プーリング案を P1 の比較対象にする。
5. **§12 `ast_tree_sitter.py` の要件**: `root_node.has_error()` と ERROR / MISSING ノード数を manifest に記録し閾値で除外、C/C++ は既知マクロをスタブしてから解析、失敗は空グラフにせずログ。
6. **§10 に追加**: 「位相マスクと GRFs++ の公開実装が無い」。再現対象 3 件のうち ICLR 2025 は自前実装の再現になるので、ViT の ImageNet 数値ではなく 1-D 格子の FLOPs 実験（n = 4、p_halt = 0.5、d = 8、m = 8）と小さな格子グラフでの厳密対 GRF 誤差を再現の定義にする。

---

## 5. 評価方法論: 交差点は自分で測る、学習曲線は幾何格子、seed 5 は小差の主張に足りない

### 5.1 カーネルのベンチ

標準手順は FlashAttention-2 / 3 が定めた: 1 GPU、**batch × seqlen = 16k を固定**し 512 → 16k を倍々、head dim 64 / 128、因果・非因果、fp16 / bf16、FLOPs は 4·B·L²·H·d（因果で半分、逆伝播 2.5 倍）、FA3 は **100 回反復の平均、GPU クロック 1830 MHz 固定、ライブラリ版固定**、数値誤差は FP64 参照に対する RMSE（[FA2](https://ar5iv.labs.arxiv.org/html/2307.08691)、[FA3](https://arxiv.org/html/2407.08608v2)、[benchmark script](https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py)）。計時は `triton.testing.do_bench`（L2 相当のバッファを毎回クリア、CUDA イベント、warmup / rep はミリ秒指定、分位点を返せる）か `torch.utils.benchmark.Timer`（自動同期、`timeit` はカーネル起動時間しか測らない）（[do_bench](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)、[PyTorch recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)）。FlexAttention は A100 で FA2 の順伝播 90%、逆伝播 85%、**要素ごとのマスク（score_mod）は 15〜20% の劣化、`create_block_mask` は数百 µs でキャッシュ必須、速度向上はブロック（128×128）単位の疎性に比例**するので、ブロックを丸ごと 0 にしないグラフマスクには恩恵が無い（[FlexAttention blog](https://pytorch.org/blog/flexattention/)、[FlexAttention paper](https://arxiv.org/html/2412.05496v1)）。GLA の著者自身が「多くの chunkwise 実装は I/O を意識しておらず 2K〜4K では FlashAttention より遅い」と述べ、FLA の GB200 表でも T ≤ 2048 では FA2 が速い（[GLA](https://arxiv.org/html/2312.06635v6)、[FLA README](https://github.com/fla-org/flash-linear-attention)）。Lightning Attention-2 は 0.4B で 32K において TNL 37,364 対 LLaMA-FA2 9,715 tokens/GPU/s と報告するが自己申告である（[Lightning Attention-2](https://ar5iv.labs.arxiv.org/html/2401.04658)）。**RTX 3090 / 4090 での交差点の一次資料は無い。**

### 5.2 前線の報告

強い論文の揃え方は、同一データ・同一順序・同一トークナイザ、パラメータ ±2%、トークン一致、FLOPs（Kaplan / Chinchilla 流）**と**実測壁時計・メモリの両方、アーキテクチャごとの学習率掃引（Zoology は対数 4 点の最良）である（[Zoology](https://ar5iv.labs.arxiv.org/html/2312.04927)、[Based](https://arxiv.org/html/2402.18668v2)）。Based は再帰状態のバイト数対 MQAR 精度、Griffin / Mamba-2 は損失対 FLOPs と対壁時計の両方を描く（Griffin は RG-LRU が memory-bound で Pallas カーネルが 3 倍速く、7B 短系列では MQA より遅い）（[Griffin](https://ar5iv.labs.arxiv.org/html/2402.19427)、[Mamba-2](https://arxiv.org/html/2405.21060v1)）。批判側は、LRA が「完全に公平な評価はほぼ不可能」と認め、Amos らは**スクラッチ学習が構造差を過大評価し**、課題データでの denoising 事前学習だけで Transformer が LRA で S4 に並ぶことを示し、Tay らは「最良モデルは規模で入れ替わる」と述べる（[LRA](https://ar5iv.labs.arxiv.org/html/2011.04006)、[Never Train from Scratch](https://arxiv.org/abs/2310.02980)、[Scaling Laws vs Architectures](https://arxiv.org/abs/2207.10551)）。グラフでは GraphGPS の Performer が小グラフで遅く精度も低く（ZINC 18 s 対 14 s / epoch）、Exphormer の利点は「速度ではなくメモリ」、Tönshoff らは LRGB の GT 対 MPNN の差がベースライン調整で消えた（Peptides-func GCN 0.5930 → 0.6860）（[GraphGPS](https://ar5iv.labs.arxiv.org/html/2205.12454)、[Exphormer](https://arxiv.org/html/2303.06147v2)、[Tönshoff et al.](https://arxiv.org/html/2309.00367v2)）。

### 5.3 学習曲線

Viering & Loog は幾何（倍々）格子、サイズごとの再チューニング、複数のランダム抽出、log-log 表示、オフセット付き冪 POW3 a·n^{−b} + c の当てはめと保留点での検証を推奨し、**「曲線下面積は全サンプルサイズを等確率に扱い交差を隠す」**と警告する（[Viering & Loog](https://ar5iv.labs.arxiv.org/html/2103.10948)）。Hestness は 2 倍シャードごとにモデル探索し冪則からの乖離を再チューニングの合図に使う（[Hestness et al.](https://ar5iv.labs.arxiv.org/html/1712.00409)）。Chinchilla は学習率スケジュールをトークン地平に合わせ、25% 以上の超過が最終損失を悪化させる（[Hoffmann et al.](https://ar5iv.labs.arxiv.org/html/2203.15556)）。

### 5.4 統計と事前登録

Bouthillier らは**データ分割の再抽出が最大の分散源**（重み初期化はその半分以下）で、推奨検定は対応ある分割上の P(A > B) の percentile bootstrap 区間（下限 > 0.5 で有意、上限 > γ = 0.75 で意味あり）、単一実行の比較は偽陽性約 10%・偽陰性約 75% だと示す（[Bouthillier et al.](https://ar5iv.labs.arxiv.org/html/2103.03098)）。Henderson らは**同じ設定の 5 seed と別の 5 seed が t = −9.09、p = 0.0016 で異なる**ことを示し、Agarwal らは 3 回では bootstrap 区間が過小被覆で IQM の区間は約 10 回から挙動が良いと述べる（[Henderson et al.](https://ar5iv.labs.arxiv.org/html/1709.06560)、[Agarwal et al.](https://ar5iv.labs.arxiv.org/html/2108.13264)）。Demšar は複数データセット上の対 t 検定を「概念的に不適切で統計的に危険」とし Wilcoxon / Friedman ＋ 事後検定を推奨、Dror らは F1 等の非平均指標には対応ある bootstrap / 置換検定を推奨する（[Demšar](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)、[Dror et al.](https://aclanthology.org/P18-1128/)）。Dodge らの期待検証性能対予算曲線は追加学習なしで描ける（[Dodge et al.](https://ar5iv.labs.arxiv.org/html/1909.03004)）。事前登録ワークショップの形式は「確証実験なしの提案 ≤ 5 頁 → 採択後に結果を追加」（[preregister.science](https://preregister.science/)）。グラフ側では Shchur らの 100 分割 × 20 初期化で Planetoid 分割の順位が別分割で逆転し、Errica らの 10-fold 外側 ＋ 内側ホールドアウトが標準、Platonov らは squirrel / chameleon の重複ノード（5,201 中 2,978）による漏洩を示した（[Shchur et al.](https://ar5iv.labs.arxiv.org/html/1811.05868)、[Errica et al.](https://ar5iv.labs.arxiv.org/html/1912.09893)、[Platonov et al.](https://arxiv.org/html/2302.11640v2)）。

### 5.5 生成器とデータ

| 資源 | 大きさと制御 | 正解の出所 | 注意 | 出典 |
|---|---|---|---|---|
| GraphQA | ER / BA / SFN / SBM / star / path / complete、5〜20 ノード、7 課題 | NetworkX | 公開は「受理後」で未確認 | [GraphQA](https://arxiv.org/html/2310.04560v1) |
| NLGraph | n と p を難度層で制御、5〜35 ノード、8 課題 | 実装 | 連結性の精度が一般 74.67 対 鎖 51.67 と**疑似相関**、語の言い換えで変動 | [NLGraph](https://ar5iv.labs.arxiv.org/html/2305.10037) |
| GraphArena | 実部分グラフ（RW with restart）、4〜50 ノード、NP 課題含む 10 課題 | 厳密アルゴリズム / 全探索 | 結果を missing / hallucinatory / suboptimal / correct に分類 | [GraphArena](https://arxiv.org/html/2407.00379v2) |
| GraphInstruct（Luo et al.） | ER / BA / small-world、5〜35 ノード、21 課題、**中間手順もアルゴリズム生成** | コード | p 等の生成パラメータ未記載 | [GraphInstruct](https://arxiv.org/html/2403.04483v4) |
| GraphWiz | ER、≤100 ノード、9 課題 | NetworkX | 推論経路は GPT-4 生成なので正解として使わない | [GraphWiz](https://arxiv.org/html/2402.16029v5) |
| CLRS-30 | 30 アルゴリズム、16 ノード訓練 / 64 ノード試験 | 教科書擬似コード実装 | ヒント（中間状態）つき | [CLRS](https://ar5iv.labs.arxiv.org/html/2205.15659) |
| TAPE | Cora / PubMed / ogbn-arxiv / arxiv-2023 の**生テキスト** | — | arxiv-2023 は GPT-3.5 の cutoff 後 | [TAPE](https://ar5iv.labs.arxiv.org/html/2305.19523) |
| CS-TAG | Amazon / GoodReads / DBLP 等の生テキスト CSV | — | MIT、HF 配布 | [TAG-Benchmark](https://github.com/sktsherlock/TAG-Benchmark) |
| Platonov 異類性 | roman-empire（同類性 0.05）、amazon-ratings 等 | — | **fastText 特徴のみ、生テキスト無し** | [Platonov et al.](https://arxiv.org/html/2302.11640v2) |
| POJ-104 / CodeNet | 問題 ID による厳密ラベル、CodeNet は正誤・実行時間つき、C++1000 は 50 万件 | 実行 | 問題単位の分割 | [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104)、[CodeNet](https://ar5iv.labs.arxiv.org/html/2105.12655) |
| 脆弱性データ | — | — | ラベル精度 Devign 24〜80%、BigVul 25〜54%、D2A 29%；漏洩 BigVul 12.7%、CVEFixes 18.9% → 除外 | [Croft et al.](https://ar5iv.labs.arxiv.org/html/2301.05456)、[PrimeVul](https://arxiv.org/html/2403.18624v2) |

### 設計への含意（§5、§6.1〜6.3、§7、§8 P8 / P9）

1. **§6.2 のマイクロベンチ手順を固定する**: `do_bench`（warmup ≥ 100 ms、中央値と 20/80 分位、L2 クリア）、1 点を `Timer` で検算、ライブラリ版・ドライバ・電力上限・温度状態を記録、クロック固定が可能なら 1830 MHz 相当を試す；格子は FA2 の 16k 固定トークン格子**＋ 128〜2048 の短系列帯**；FLOPs は FA2 式で因果半減の有無を明記；ピークメモリは `reset_peak_memory_stats` 後の `max_memory_allocated`；FlexAttention は BlockMask 構築コスト・キャッシュ・ブロック疎性・compile のウォームを記録し、**FlexAttention-dense（何もしないマスク）腕を置いて**カーネル差とマスク費用を分離；線形 / SSM は状態サイズとチャンクサイズを記録。交差点は文献を引かず自分で測り、FLA README を「交差点は世代と GPU で動く」証拠として引く。
2. **§6.1 の前線は 2 枚組**: 精度対 log(状態サイズまたはピークメモリ)（Based 形）と、精度対 FLOPs ＋ 精度対壁時計（Mamba-2 / Griffin 形）。二者の乖離自体を結果にする。同一バックエンド比較（FlexAttention 同士）と異バックエンド比較（FA2 対 Triton）を両方載せ、実装成熟度の偏りを宣言する。
3. **§6.3 の 5 / 10 / 25 / 50 / 100% を幾何格子 1/16, 1/8, 1/4, 1/2, 1 に変え**、POW3 を当てて「全データ精度の 90% に要するデータ量」を bootstrap 区間つきで読む。AUC は単独では出さず曲線と併記。最小サイズで調整し冪則から外れたら再調整（Hestness）。
4. **§7 の seed 5 を見直す**: 主比較（H1 の 線形＋φ内 − 線形なし、H4 の 対 窓）は **10 回の対応ある反復で、各反復で分割と初期化を再抽出**し、決定則は Bouthillier の P(A > B)（γ = 0.75）か Wilcoxon ＋ Holm、項目単位の指標は対応ある bootstrap（Dror）。副比較は 5 seed のままで良いが「小差は見分けられない」と事前登録に書く。Dodge 曲線で各腕の調整予算を公開する。
5. **§5 のデータ**: roman-empire / amazon-ratings は生テキストが無いので「相補 TAG」から外すか構造のみの実験に回し、相補型の TAG は CS-TAG の Amazon / GoodReads か TAPE の題名のみ版で作る。合成グラフは GraphArena 級の NP 課題を含めつつ **N = 50〜5,000 は自前生成器**（既存生成器は ≤ 100 ノード）で、生成族（ER / BA / 鎖 / クリーク）で層別して報告する（NLGraph の疑似相関）。CLRS の 16 → 64 ノード OOD 形式を「大きさ」軸の既製形として借りる。
6. **§10 に Amos らの警告を追加**: 融合層をスクラッチ学習すると構造差を過大評価し得る。凍結 LM のキャッシュ上で融合層だけを学習する本設計はこの批判に部分的に答えるが、融合層に課題データでの短い denoising 事前学習を入れた対照を 1 本置く。

---

## 6. 新規性の判定: 「構文・グラフ構造を線形 attention / SSM の作用素に入れて LM と結ぶ」

**空白は維持されている。** (1) 言語の線形 attention に 1 次元の局所性・減衰以外の構造を入れた研究は無く、構文誘導 attention は全て二次 softmax である（§1.4）。(2) 木構造・DAG 構造の SSM は Chimera・GrootVL・pLSTM で存在するが、**いずれも LM と結ばれておらず、AST / 構文木に適用した例も無い**（TreeGPT が空白を明言）。(3) 第 1 回報告書の「構造つき線形 attention と LM の接続はゼロ」「グラフ SSM と LLM の融合は HSA-Net 1 件」は、今回の 5 本のノートでも覆らなかった。

**ただし「最も安い回復手段」という主張は脅かされている。** 構造なしの回復手段は 2024〜2026 年に急速に成熟し、窓 64〜128 で検索の 90.8%、ゲート＋delta 則で Transformer++ 超え、周期 softmax 1/4 で純 softmax 超えという数字を持つ（§1.2）。したがって新規性の中心は「構造を入れること」ではなく、**同じ壁時計・メモリの下で、グラフ構造（辺）と系列構造（窓・減衰）のどちらが鋭さを安く戻すかを、相補性の型と大きさで層別して測ること**に置くべきである。これは第 1 回報告書の結論（「新規性は因子分解より相補性の操作化と効率の測定枠組み」）を強める。

**第 1 回報告書の結論が変わる箇所。** (a) 第 1 回は線形 attention の損失原因を「一様化と低ランク化」（GraphGPS、GraphTARIF）としたが、LLM 系譜では**検索容量**が主因と診断されており、設計書 §1 の第 1 行は「一様化・低ランク化・検索容量の三つ」に改める。(b) 第 1 回は SSM 融合を「作用素型（GSSC、Chimera、MP-SSM）」と総称したが、LM との結合で線形時間が成り立つのは **DAG 上の Chimera 形だけ**で、GSSC の固有分解と MP-SSM は別物として扱う。(c) 第 1 回が挙げた GraphTARIF の「ランク回復ゲート」は構造を使うので H4 の構造なし対照には使えない。(d) 2026 年の構文 LM 研究（標準 attention が構造マスクと同等）は、H2 の softmax 行「導出可能型で利得なし」を言語側からも支持する。

---

## 7. 設計書への変更提案

| 節 | 現行 | 変更案 | 根拠 |
|---|---|---|---|
| §0 / 一文 | 「最も安く取り戻す手段か」 | 「系列の局所性・ゲート・周期 softmax と同コストで比べたとき、グラフ構造はどれだけ鋭さを戻すか。それはグラフの大きさと相補性にどう依存するか」 | §1.2、§2.2 |
| §1 事実表 | 原因は一様化と低ランク化 | 「一様化・ランク ≤ d・固定状態の検索容量 Ω(N) ビット」の三つ。追加行: 「softmax の分散は定理（Θ(1/n)）、線形の分散は定理なし」「製品の線形 LLM は全てハイブリッド」「窓 64〜128 が検索の 90.8%」「GRF は重要度重みなしで無効、公開実装なし」 | §1、§3.1、§4.1 |
| §2 H3 | 「一様化は入力が長いほど悪化する」 | 「固定状態の検索は L が容量を超えると失敗する。構造は実効支持を縮めるので回復量は L_t・N に増える」。反証: 回復量が L に平坦**かつ**検索正答率に長さ依存なし。鋭さ指標は長さ層別、同長で合成検索課題を併記 | §1.3、§3.1 |
| §2 H4 | 対照 = 局所窓、Gumbel、ランク回復ゲート、特徴写像 | 対照 = (a) 窓 64 / 128、(b) ゲート＋delta 則、(c) 1/F 周期 softmax、(d) Hedgehog / Taylor 写像。GraphTARIF 形ゲートは S = φ外 に移動、Gumbel は任意。反証条件に「(a)〜(c) のいずれかが同コストで同等以上」 | §1.2 |
| §2 H1 | 回復順 厳密マスク ≥ φ内 > φ外 > なし | φ内 を「厳密冪級数」と「GRF 近似」に分け、順序を 厳密マスク ≥ φ内(厳密) ≥ φ内(GRF) > φ外 > なし と予測 | §4.1 |
| §3.1 鋭さ | 正規化エントロピー、実効ランク、構造質量 | §3.4 の表に置換: 支持の定義、実効支持比、max / top-k、sink 質量（別報告・除外再計算）、attention と層出力の実効ランク、線形は暗黙重み、長さ層別、行動指標 | §3.4 |
| §4.2 因子 A | softmax = 「精度の上限」 | 「コストの基準」。ハイブリッド対照が softmax を上回る可能性を明記。線形 φ は fp32 累積 | §1.2、§2.2 |
| §4.3 SSM 列 | 「レゾルベント型（Chimera / GSSC に準ずる）」 | Chimera 形に確定: 結合列を DAG（テキスト線グラフ ＋ AST 有向辺 ＋ 整合辺）として 1 パス、φ外 = 辺項 Δ′_ij、φ内 = 隣接を A に、厳密マスクの対照 = 同隣接のマスク付き softmax。一般グラフは 2 方向 DAG か直径打ち切り（二次）。固有値域 [−1, 1] を任意水準 | §2.4 |
| §4.3 φ内（線形） | GRF テンソル積 | 既定: 対称正規化、重要度重み、n = 100 / p_halt = 0.1 / 最大長 100、反対称打ち切り、AST は縫合 l を掃引。N ≤ 10k は厳密冪級数を併置 | §4.1 |
| §4.4 | 同上 | 揃え方を「壁時計・ピークメモリ・KV / 状態バイト」に（パラメータ一致だけでは窓が有利） | §1 含意 3 |
| §4.7 | Mamba 系 LLM は Falcon-Mamba、Mamba-2 系 | Falcon-H1 1.5B / 3B、Codestral Mamba 7B または Falcon-Mamba-7B（4 bit）。LoRA は in_proj / out_proj / x_proj。Jamba 除外 | §2.3 |
| §5 データ | roman-empire / amazon-ratings を相補 TAG に | 生テキストが無いので除外または構造のみ。相補 TAG は CS-TAG / TAPE 題名のみ版。合成は自前生成器で N = 50〜5,000、生成族で層別、CLRS 形の OOD サイズ | §5.5 |
| §6.1 | 前線 1 枚 | 精度対状態 / メモリ、精度対 FLOPs、精度対壁時計の 3 枚。同一バックエンド比較を併記 | §5.2 |
| §6.2 | 「100 回計測」 | §5 含意 1 の手順。短系列帯 128〜2048 を追加。FlexAttention-dense 腕。ブロック疎性を記録 | §5.1 |
| §6.3 | 5 / 10 / 25 / 50 / 100%、AUC と 90% 到達率 | 幾何格子、POW3 当てはめ、90% 到達データ量の bootstrap 区間、AUC は併記のみ | §5.3 |
| §7 | seed 5、二段 bootstrap、Holm | 主比較は 10 反復（分割＋初期化を再抽出）、P(A > B) γ = 0.75 または Wilcoxon ＋ Holm、項目指標は対応 bootstrap；Dodge 予算曲線；事前登録は「提案 → 結果」の二段形式 | §5.4 |
| §8 P4 | 鋭さ指標の妥当性 | 長さ層別（256 / 1K / 4K / 8K × N = 50 / 500 / 5,000）で softmax 対線形の Ĥ 差を測り、H3 の前提を実測で判定。合成検索課題を併設 | §3 含意 3 |
| §8 P5 | GRF の分散 | 厳密冪級数との誤差、ゼロ推定対の割合、縫合 l、反対称打ち切りの効果、AST 深さ別。ICLR 2025 の再現は 1-D 格子 FLOPs 実験と定義 | §4 含意 2・6 |
| §8 P7 | 4 水準 | 新 (a)〜(d) の 4 水準。同コスト化は壁時計とメモリ | §1 含意 1 |
| §10 リスク | — | 追加: 線形 attention の数値精度感度；GRF 実装非公開；FlexAttention はブロック疎性が無いと速くならない；sink / 仮想ノードの偽の鋭さ；tree-sitter の C/C++ マクロ誤解析；スクラッチ学習の構造差過大評価（Amos）；短系列では線形 / SSM が FA2 に負ける（交差点が実データ外になる） | 各節 |
| §11 位置づけ | — | 追加行: 「線形 attention の LLM 系譜と構造なし回復」= Devil、Hedgehog、Based、GLA、DeltaNet / GDN、LoLCATs、Kimi Linear；「鋭さの理論」= Veličković、Nakanishi、Vasylenko、Zhai、Barbero；「作用素への構造」= Chimera、GrootVL、pLSTM、Hydra | — |
| §13 項目 5 | 未決 | 解決（Chimera 形、§2 含意 1・5） | §2.4 |

---

## 8. 未検証一覧（ノートから引き継ぎ。事前登録前に解消する）

| # | 項目 | 状態 | 解消方法 |
|---|---|---|---|
| **A. 線形 attention の系譜** | | | |
| A1 | Based の ICML 2024、GLA の ICML 2024、DeltaNet の NeurIPS 2024、RWKV-4 の EMNLP 2023 Findings、ReBased の ACL 2024、RNNs-are-not-Transformers の ICLR 2025、Repeat-After-Me の ICML 2024 の会場 | arXiv ページに会場表示なし | 会議プロシーディングで確認 |
| A2 | ReBased の具体的機構（学習可能アフィン / 正規化） | 要旨のみ | 全文 |
| A3 | Log-Linear Attention の Fenwick 木バケット構成と数値 | 要旨のみ | 全文 |
| A4 | Qwen3.5 が同じハイブリッドを維持 | 二次情報 | 公式カード |
| A5 | Mamba-2 の「attention 約 10% が最良」という ablation、MQAR 対状態サイズの数値 | 記憶由来 | Mamba-2 本文 §9 |
| A6 | Hedgehog のエントロピー実数値（Fig. 2 / 4）と系列長別 KL（Table 5） | HTML から抽出不能 | PDF |
| A7 | Deng–Song–Yuan–Zhou「Why Softmax Attention Outperforms Linear Attention」の形式的主張 | 要旨のみ | 全文 |
| A8 | MHLA の付録 F.4（エントロピー対 N の掃引） | 未閲覧 | 全文 |
| A9 | Based Table 6 の PosELU / ReLU / Square / Identity 行と窓 16 / 256 | 未取得 | 全文 |
| A10 | 「構文 ＋ 線形 attention」の Semantic Scholar 被引用走査（Qin 2022、Hedgehog） | 未実施 | API キー取得後に実施（第 1 期計画 §2.4） |
| A11 | Griffin / RecurrentGemma の局所窓（論文 1024、ブログ 2K）の食い違い | 両資料で異なる | 両方を確認し注記 |
| **B. SSM / Mamba** | | | |
| B1 | S4 / H3 / Hyena / Griffin の詳細と arXiv ID | 著者知識、再取得せず | 各ページ |
| B2 | Mamba-3（arXiv 2603.15569）の内容とカーネル要件 | 未読 | 全文 |
| B3 | DeltaProduct の会場 | 未確認 | — |
| B4 | Samba の窓 2048 と層数 | SSM ノートでは未確認（線形ノートは全文から 2048 と記録） | 線形ノートの記録で解消済みとみなすが論文表で再確認 |
| B5 | Nemotron-H の 8% を選んだ理由の ablation | 未抽出 | 全文 |
| B6 | Granite 4.0 の 9:1、Nemotron 3 Nano の構成 | 二次情報 | 公式カード |
| B7 | Hymba のメタトークン・層間 KV 共有・SWA の詳細 | 著者知識 | 全文 |
| B8 | Falcon-Mamba の論文 ID（2410.05355）、ライセンス（TII 2.0 対 Apache-2.0 の食い違い）、HF クラス名 | 未確認 | 現行カード |
| B9 | Codestral Mamba のライセンスと 256k 試験 | 記憶由来 | Mistral 告知 |
| B10 | Jamba 1.5 / 1.6 / 1.7、Zamba2 1.2B / 2.7B、Nemotron-H の HF 公開・ライセンス・文脈長、Bamba-9B の詳細 | 未取得 | 各カード |
| B11 | MambaPEFT、State-offset Tuning の arXiv ID | 未確認 | Galim et al. の参照 |
| B12 | 「Priming: Hybrid SSMs from Pre-trained Transformers」（2605.08301） | 未読 | 全文 |
| B13 | Sarrof Thm 2、Merrill の定理番号 | 要旨から | 全文 |
| B14 | GrootVL の DP 再帰式とテキスト課題の数値 | 要旨のみ | 全文 |
| B15 | Chimera のコード公開とカーネル状況、LRGB 数値の独立再現 | 未確認 | リポジトリ検索 |
| B16 | causal-conv1d の最小 compute capability | コミュニティ wheel から推定 6.0〜9.0 | README |
| B17 | Mamba-2 対 FA2 の RTX 3090 / 4090 数値 | 一次資料なし | P8 で実測 |
| **C. 鋭さの理論** | | | |
| C1 | Hahn 2020 の正確な主張 | 記憶由来 | 全文 |
| C2 | Su Jianlin の entropy-invariance の定数 κ | kexue.fm が 403 | 別経路で取得 |
| C3 | YaRN の温度式 1/√t = 0.1 ln(s) + 1 | 記憶由来 | 全文 |
| C4 | Noci 2022、Shi 2022、Wu 2023 の定理文 | 要旨・スニペット | 全文 |
| C5 | Wang et al. 2024（NoPE 長さ一般化）の ACL Findings 会場 | 未確認 | — |
| C6 | Vasylenko et al. の ICLR 2026 会場、Round and Round の ICLR 2025 会場、「Why do LLMs attend to the first token?」の会場 | arXiv に表示なし | — |
| C7 | sigmoid attention の −log n バイアス | 記憶由来 | 本文 |
| C8 | 「catch, tag, and release」という語句の出典 | 見つからず | 使用しない |
| C9 | sparsemax の再取得、α-entmax の閉形式 | 記憶由来 | — |
| C10 | Adaptive Attention Span の内容 | 記憶由来 | 全文 |
| C11 | Shi et al. のトークン余弦類似度指標 | 記憶由来 | 全文 |
| C12 | Vig & Belinkov の層別エントロピーと先頭トークン注意の数値 | 要旨のみ | 本文 |
| C13 | ALiBi の傾き 2^{−8i/n_heads} | 記憶由来 | 本文 |
| C14 | Men et al.（RoPE base）の機構 | 要旨のみ | 本文 |
| C15 | RLGM 2019 ワークショップ論文（paper 62）の題名 | 未確認 | PDF |
| C16 | Oh & Schuler 2022 の題名・会場 | PDF から推定 | ACL Anthology |
| **D. GRF・PE・整合** | | | |
| D1 | ICLR 2025 位相マスクの公開コード、google-research の GRF リポジトリ | 見つからず | 著者に照会 / 再検索 |
| D2 | GRFs++ の ICML 2026 camera-ready と濃度結果 | PMLR ページのみ | 全文 |
| D3 | ViT 実験の線形 attention 特徴次元 m（クロネッカー幅 mN） | 本文に記載なし | 著者照会 |
| D4 | GraphGPS の「分子は RWSE、他は LapPE」の実践則 | 要旨のみ | 全文 |
| D5 | SignNet / BasisNet の ICLR 2023 会場 | 未確認 | — |
| D6 | SPE の前処理実行時間（付録 B.6）、Black et al. のコード URL | 未読 | 全文 |
| D7 | Syntax-BERT のマスク族、SG-Net の SDOI マスク式 | 要旨のみ | 全文 |
| D8 | Sachan et al. の GitHub URL | 抽出で切断 | 論文 PDF |
| D9 | CodeXGLUE 各課題が GraphCodeBERT の parser を再利用しているか | 未確認 | リポジトリ |
| D10 | 木に特化した GRF の数値、直径依存の分散上界 | 文献に無い | P5 で測る |
| **E. 評価方法論** | | | |
| E1 | FA2、FA3、FlexAttention（MLSys 2025）、GLA、Based、Zoology（ICLR 2024）、LRA（ICLR 2021）、Amos（ICLR 2024）、Efficient Transformers（CSUR）、GraphGPS（NeurIPS 2022）、Exphormer（ICML 2023）、Viering & Loog（TPAMI）、Hoffmann（NeurIPS 2022）、Dodge（EMNLP）、Tönshoff（LoG 2023）、Platonov（ICLR 2023）、GraphQA（ICLR 2024）、GraphWiz（KDD 2024）、CodeNet（NeurIPS D&B）、Shchur（ワークショップ）の会場 | 取得ページに表示なし | プロシーディング |
| E2 | `do_bench` の L2 クリア用バッファが 256 MB か | 未確認 | ソース |
| E3 | FlexAttention 論文の計時法・反復数 | 本文に記載なし | 著者 / ブログ |
| E4 | Mamba-2 §9.3 の GPU・バッチ・ヘッド設定 | HTML が途中で切断 | PDF |
| E5 | 「GraphQA-Hardened」というデータセット | 見つからず（学生プロジェクトのみ） | 名称を使わない |
| E6 | GraphQA の公式コード・データ公開 | 未確認 | リポジトリ検索 |
| E7 | Platonov et al. の生テキスト公開 | 論文・README に記載なし | 著者照会 |
| E8 | CS-TAG のノード・辺数とデータ一覧、論文の arXiv ID | README に無し | 論文 |
| E9 | GraphArena のソルバライブラリ名 | 「厳密アルゴリズム」とのみ | コード |
| E10 | Demšar の Holm 対 Bonferroni-Dunn の手順文 | 部分抽出 | 全文 |

---

## 結論

今回の調査で最も重要な理解の変化は、「線形時間の attention が失う鋭さ」が分布の形（エントロピー）の問題であるより先に**状態容量**の問題だという点である。softmax について分散の定理が揃っている一方、線形 attention については定理が無く、長さ効果は検索課題でのみ測られてきた。この非対称は、計画研究にとって弱点ではなく機会になる: 線形 attention の正規化エントロピーを長さとグラフサイズの関数として測り、同じ条件で合成検索課題の正答率を併記するだけで、文献に無い曲線を提供できる。H3 をエントロピー法則から容量仮説へ書き直すことは、主張を弱めるのではなく、反証可能で文献と整合する形に直すことである。

第二の変化は、競争相手が変わったことである。設計書は softmax を上限、構造を回復手段として置いたが、LLM 系譜の 2024〜2026 年は、窓・ゲート・delta 則・周期 softmax という安い部品で softmax に追いつき、時に追い越した。したがって研究の価値は「構造が効くか」ではなく、「辺という 2 次元の構造は、系列の局所性という 1 次元の構造に対して、同じコストで何を上乗せするか、それはグラフが大きく、構造がテキストと相補的なときに限られるか」を、統制された条件で初めて測ることにある。Chimera が示したとおり、SSM のマスクはグラフの隣接で置き換えられ、DAG なら線形時間で動く。作用素型 SSM と LM の接続、構文木上の木構造 SSM、線形 attention への構文注入はいずれも未着手のままであり、GRF の公開実装が無いという実装上の空白も含めて、第 1 期の予備調査がそのまま先行研究の空白を埋める構成になっている。
