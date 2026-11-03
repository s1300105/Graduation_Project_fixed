# Methods for Analysing Attention Inside Pretrained Transformer LMs (2019–2026): Functional Heads, Concentration Metrics, and Links to Long-Context Retrieval Failure

Verification legend used below:
- **[V]** = verified in this session against the primary page (arXiv abstract/HTML, proceedings, or repo).
- **[K]** = well-known detail from the original paper, URL given, but the specific number or formula was NOT re-fetched in this session. The thesis author should check it against the PDF before quoting.
- **[U]** = unverified / secondary-source only.

---

## Q1. Per-paper protocol records: how functionally important heads are identified, scored, causally tested, and how this depends on length

### Takeaway
The field identifies heads in three main ways. (a) Behavioural attention scores on synthetic probes: prefix-matching or induction scores on repeated random tokens, echo scores, and Wu et al.'s copy-paste retrieval score on Needle-in-a-Haystack (NIAH). (b) Gradient or gate-based importance: Michel's gate gradient, Voita's L0 gates, DuoAttention's optimised gates. (c) Attention-mass statistics: Voita's "confidence", the sink rate of Gu et al. and Barbero et al., and query-focused attention mass in QRHead. The causal evidence is consistent. A small fraction of heads (about 1–5% for retrieval or echo heads) carries long-context retrieval. Masking them collapses NIAH performance, while masking random heads does not.

### Cited Findings

#### 2019–2022 foundations (brief)

**What Does BERT Look At? An Analysis of BERT's Attention.** Clark, Khandelwal, Levy, Manning. BlackboxNLP@ACL 2019. https://arxiv.org/abs/1906.04341
- Phenomena: broad vs. focused heads, heads attending to [SEP]/[CLS]/punctuation (a precursor of the "sink" idea), positional heads (previous or next token), and syntactic and coreference heads. [K] — [Source](https://arxiv.org/abs/1906.04341)
- Metrics: per-head **attention entropy** averaged over tokens (they report that some early-layer heads have very broad attention, with entropy near uniform), and the fraction of attention on special tokens per layer. They used gradient-based importance to argue that [SEP] attention acts as a "no-op". Syntactic heads are scored as accuracy of the most-attended token matching a dependency relation. Model: BERT-base. [K] — [Source](https://arxiv.org/abs/1906.04341)
- Code: https://github.com/clarkkev/attention-analysis [K] — [Source](https://github.com/clarkkev/attention-analysis)

**Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.** Voita, Talbot, Moiseev, Sennrich, Titov. ACL 2019. https://aclanthology.org/P19-1580/ (arXiv https://arxiv.org/abs/1905.09418)
- **Confidence** of a head = the average, over tokens, of the maximum attention weight of that head (excluding the end-of-sentence token). This is a direct precursor of a "top-1 mass" statistic. Importance was also measured with layer-wise relevance propagation (LRP). Head roles: positional (attends to an adjacent token in more than 90% of cases), syntactic, and rare-word. [K] — [Source](https://arxiv.org/abs/1905.09418)
- Causal test: heads are pruned with stochastic Hard-Concrete gates and an L0 penalty. On WMT En–Ru, 38 of 48 encoder heads were pruned for a drop of 0.15 BLEU (Transformer-base NMT encoder). [K] — [Source](https://aclanthology.org/P19-1580/)
- Code: https://github.com/lena-voita/the-story-of-heads [K]

**Are Sixteen Heads Really Better than One?** Michel, Levy, Neubig. NeurIPS 2019. https://arxiv.org/abs/1905.10650
- Head importance score: a gate ξ_h multiplies each head's output, and I_h = E_x |∂L(x)/∂ξ_h| = E_x |Att_h(x)^T ∂L(x)/∂Att_h(x)| (a first-order Taylor or gradient criterion), normalised per layer. [K] — [Source](https://arxiv.org/abs/1905.10650)
- Causal test: masking individual heads at test time. Many layers of WMT Transformer and BERT can be reduced to a single head with little loss. Iterative pruning removes about 20% (WMT) to about 40% (BERT) of heads before large drops. Encoder–decoder attention is the most sensitive to pruning. [K] — [Source](https://arxiv.org/abs/1905.10650)
- Code: https://github.com/pmichel31415/are-16-heads-really-better-than-1 [K]

**In-context Learning and Induction Heads.** Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, Mann, Askell, Bai, Chen, Conerly, Drain, Ganguli, Hatfield-Dodds, Hernandez, Johnston, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, Olah. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Induction head = a head that, on a **repeated sequence of random tokens**, shows (1) **prefix matching**: it attends to the earlier token that was followed by the current token, and (2) **copying**: its output raises the logit of the attended token. Measurements form a continuum, but a clear subset is far above chance on both. No numeric threshold is given. [V] — [Source](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- **In-context learning score** = loss at token 500 minus the average loss at token 50 (contexts of length 512), averaged over examples. [V] — [Source](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Scale: 34 transformers studied across training, with more than 50,000 head ablations. A phase change (a bump in the loss curve) appears only in models with more than one layer and coincides with induction-head formation. Knocking out induction heads at test time greatly reduces in-context learning in small attention-only models. [V] — [Source](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Operationalisation commonly used in practice (TransformerLens tutorials): feed `[BOS] + r + r` with r a random token block of length n. The induction score of a head is the mean attention on the diagonal offset by −(n−1), i.e. from position i in the second copy to position i−n+1. [K] — [Source](https://transformerlensorg.github.io/TransformerLens/)
- Gap: the exact Olsson copying-score formula (the positive-eigenvalue fraction of the OV circuit, from "A Mathematical Framework for Transformer Circuits", https://transformer-circuits.pub/2021/framework/index.html) was not re-fetched. Treat it as [K].

#### 2023–2026: retrieval heads and head-level long-context work

**Retrieval Head Mechanistically Explains Long-Context Factuality.** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. arXiv April 2024, published at ICLR 2025 (in the ICLR 2025 proceedings). https://arxiv.org/abs/2404.15574 ; proceedings PDF https://proceedings.iclr.cc/paper_files/paper/2025/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Conference.pdf
- **Retrieval score** of head h = |g_h ∩ k| / |k|. Here k is the needle's token set and g_h is the set of needle tokens that head h "copy-pastes". A copy-paste event at a decoding step requires two conditions: (1) the generated token w is in the needle, and (2) the input position that receives head h's **argmax attention** holds that same token and lies inside the needle span. This is a token-level recall. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Threshold: a head with retrieval score at least 0.1 is a retrieval head (it copy-pastes in 10% or more of cases). [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Detection data: 3 needle sets (question q, needle k, haystack x). Each is tested at **20 lengths uniformly spaced from 1K to 50K** and **10 depths**, for about 600 instances per model. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Models: Llama-2-7B, Llama-2-7B-80K, Llama-2-13B (60K/64K), Mistral-7B-v0.2, Mistral-7B-Instruct-v0.2, Mixtral-8x7B-v0.1, Yi-6B, Yi-6B-200K, Yi-34B-200K, Qwen1.5-14B, Qwen1.5-14B-Chat. That is 4 families and 6B–34B parameters plus 8×7B MoE. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Sparsity: about 3–6% of heads score above 0.1, and fewer than 5% are active more than 50% of the time. The abstract says "less than 5%". A small set is "always on": 12 heads in Llama-2-7B-80K (the introduction says 13, an internal inconsistency) and 36 in Yi-6B-200K. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Universality: retrieval heads exist in base short-context models and are reused after long-context continued pretraining, chat tuning, or MoE upcycling. Retrieval scores correlate with Pearson r > 0.8 within a family and r < 0.1 across families. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Causal test: masking the **top-K = 50** retrieval heads (about 5% of heads) drops NIAH below 50 for all models. Masking random non-retrieval heads has a much smaller effect (exact values are in Fig. 7 only). On extractive QA, masking retrieval heads reduces F1 by 9.2% and 23.1%, while masking random heads has no significant effect. For CoT on MMLU, MuSiQue and GSM8K (Mistral-7B-Instruct-v0.2), masking retrieval heads hurts CoT substantially but barely affects direct answers. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- Failure mechanism: when the model hallucinates on NIAH, retrieval heads are only partially activated or inactive. Weaker heads recover only part of the needle. [V, via search summary of the ICLR version] — [Source](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Conference.pdf)
- Code: https://github.com/nightdessert/Retrieval_Head [V for the repo name and existence; the HTML page's link was garbled] — [Source](https://arxiv.org/html/2404.15574v1)

**DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.** Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, Song Han. arXiv October 2024, published at ICLR 2025 (proceedings PDF). https://arxiv.org/abs/2410.10819 ; https://proceedings.iclr.cc/paper_files/paper/2025/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf
- Identification is optimisation-based rather than score-based. Each KV head gets a gate α ∈ [0,1], initialised to 1, with LLM weights frozen. Head output = α·FullAttn + (1−α)·StreamingAttn, where streaming attention keeps only the sink and recent tokens. Loss = L2 distance between the final hidden states of the full and mixed models, over the last l passkey tokens only, plus λ·L1(α) with λ = 0.05. Training: AdamW for 2,000 steps, learning rate 0.02 with warm-up and decay to 0.002. [V] — [Source](https://arxiv.org/html/2410.10819)
- Synthetic data: ten 32-token random passkeys inserted at random positions in BookSum text. Lengths span 50 intervals from 1K to the model maximum. The window used during identification is 128 sink tokens plus 256 recent tokens. [V] — [Source](https://arxiv.org/html/2410.10819)
- Heads are binarised by a quantile threshold τ. Retrieval-head ratio: 25% for Llama-2-7B-32K-Instruct (MHA) and 50% for Llama-3-8B-Instruct-Gradient-1048k (GQA). At these ratios NIAH matches full attention up to 1,048K tokens. H2O, TOVA, StreamingLLM and FastGen fail at various depths. [V] — [Source](https://arxiv.org/html/2410.10819)
- Ablation: an optimisation-based selection beats attention-profiling selection (FastGen/RazorAttention style). The authors argue that attention scores ignore the end-to-end effect of removing a head's KV. They do **not** benchmark Wu et al.'s retrieval score directly. [V] — [Source](https://arxiv.org/html/2410.10819)
- Efficiency: memory reduced up to 2.55× (MHA) and 1.67× (GQA). Decoding up to 2.18× and 1.50× faster. With quantisation, Llama-3-8B reaches 3.3M tokens on one A100. Code: https://github.com/mit-han-lab/duo-attention [V] — [Source](https://arxiv.org/abs/2410.10819)

**RazorAttention: Efficient KV Cache Compression Through Retrieval Heads.** Hanlin Tang et al. arXiv July 2024. https://arxiv.org/abs/2407.15891
- Defines two retrieval-head types. **Echo heads** attend to the previous occurrence of the current token. **Induction heads** attend to the token after the previous occurrence. Both are scored on a probe of random tokens repeated 4 times. Per a secondary summary, the top 1% of heads by echo score and the top 14% by induction score keep a full cache. The other heads keep only sink and recent tokens plus a "compensation token" summarising dropped tokens. Reported KV reduction is up to about 70%, and echo heads (about 1%) are described as essential. [U for thresholds; V for the method outline via secondary summary] — [Source](https://huggingface.co/papers/2407.15891); [summary](https://www.emergentmind.com/topics/razorattention)
- Venue: conflicting secondary info (some pages imply ICLR; not confirmed). Mark the venue as **unverified**; ICLR 2025 acceptance is plausible but was not confirmed in this session. — [Source](https://arxiv.org/abs/2407.15891)

**Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning (HeadKV / HeadKV-R2).** Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, Wen Xiao. ICLR 2025. https://arxiv.org/abs/2410.19258
- Per-head importance is estimated on contextual QA examples that need both retrieval and reasoning. It extends the Wu et al. retrieval score into a "retrieval-reasoning" score. Head-level KV budgets are allocated by importance. Models: Llama-3-8B-Instruct and Mistral-7B-Instruct. Benchmarks: LongBench and LooGLE. Retains 1.5% of the KV cache with 97% of full-KV performance on contextual QA. Strongest gains are at KV sizes 64 and 128. Code: https://github.com/FYYFU/HeadKV [V]; the exact score formula was not retrieved [gap]. — [Source](https://arxiv.org/abs/2410.19258)

**Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking (QRHead).** Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen, Xi Ye. EMNLP 2025. https://arxiv.org/abs/2506.09944 ; https://aclanthology.org/2025.emnlp-main.1214.pdf
- Heads are identified by aggregating **attention mass from query tokens to the gold context**, using a handful of real-task examples, instead of NIAH copy-paste. The accumulated attention mass of the selected heads then serves as a retrieval score (QRRetriever). Reports more than 10% gains over full context on LongMemEval and CLIPPER and strong zero-shot BEIR re-ranking versus RankGPT. Both query-context scoring and task selection matter. The authors note that copy-paste scoring misses semantic or paraphrase retrieval heads. Code: https://github.com/princeton-pli/QRHead [V] — [Source](https://arxiv.org/abs/2506.09944)

**Retrieval Heads are Dynamic.** Yuping Lin, Zitao Li, Yue Xing, Pengfei He, Yingqian Cui, Yaliang Li, Bolin Ding, Jingren Zhou, Jiliang Tang. ACL 2026, arXiv 2602.11162 (v1 January 2026). https://arxiv.org/abs/2602.11162
- Argues that the set of retrieval heads changes across generation steps rather than being static. Validated on NIAH and multi-hop QA, and used within a dynamic RAG framework. Method details, models and lengths were not retrieved. [V for metadata; method gap] — [Source](https://arxiv.org/abs/2602.11162)

**Other 2024–2026 retrieval-head follow-ups (from search; metadata only, unverified detail):**
- "Understanding Synthetic Context Extension via Retrieval Heads" (arXiv 2410.22316). Links retrieval heads learned from synthetic fine-tuning data to downstream long-context performance. [U] — [Source](https://arxiv.org/pdf/2410.22316)
- "SEAL: Scaling to Emphasize Attention for Long-Context Retrieval" (arXiv 2501.15225). Learns head- and channel-wise scaling to boost retrieval. [U] — [Source](https://arxiv.org/pdf/2501.15225)
- "The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval" (arXiv 2502.11276). [U] — [Source](https://arxiv.org/pdf/2502.11276)
- "From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models" (arXiv 2601.11020). [U] — [Source](https://arxiv.org/pdf/2601.11020)
- "Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis Across Model Families" (arXiv 2606.21249). Per the search summary, zeroing low-frequency RoPE dimensions inside retrieval heads sharply reduces recall, while zeroing random dimensions does not. It is a single-author preprint and also reports that an argmax-based "failure" was a localisation artefact (a relevant caveat for argmax-based retrieval scores). [U] — [Source](https://arxiv.org/pdf/2606.21249)
- "BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs" (arXiv 2604.05942). Another head-selection method. [U] — [Source](https://arxiv.org/pdf/2604.05942)

#### Attention sinks and massive activations

**Efficient Streaming Language Models with Attention Sinks (StreamingLLM).** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. ICLR 2024. https://arxiv.org/abs/2309.17453
- Window attention collapses once the initial tokens are evicted. Keeping about 4 initial "sink" tokens plus a rolling window restores perplexity, and streaming up to 4M tokens was shown on Llama-2, MPT, Falcon and Pythia. A dedicated learnable sink token used in pretraining is also proposed. Code: https://github.com/mit-han-lab/streaming-llm [K] — [Source](https://arxiv.org/abs/2309.17453)

**When Attention Sink Emerges in Language Models: An Empirical View.** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. ICLR 2025 (Spotlight). https://arxiv.org/abs/2410.10781
- **Formula.** Importance of token k in head (l,h): α_k^{l,h} = (1/(T−k+1)) Σ_{i=k}^{T} A_{i,k}^{l,h}. **Sink rate** Sink_k^ε = (1/L) Σ_l (1/H) Σ_h 𝟙(α_k^{l,h} > ε), with **ε = 0.3**. The threshold was chosen as strict and relatively insensitive to T. For their own pretraining runs they used 100 sequences, T = 64, no BOS. [V] — [Source](https://arxiv.org/html/2410.10781v2)
- Sink₁ (%) for natural / random-token / repeated-token inputs: GPT2-XL 77.00/70.29/62.28; Mistral-7B 97.49/75.21/0.00; LLaMA2-7B base 92.47/90.13/0.00; LLaMA3-8B base 99.02/91.23/0.00. Chat models are similar (LLaMA3-8B chat 98.85; Mistral-7B chat 88.34; LLaMA2-13B 91.69 base / 90.94 chat). Sinks appear even in Pythia-14M. [V] — [Source](https://arxiv.org/html/2410.10781v2)
- Mechanism: the first token has a massive hidden-state norm but small key and value norms. Its key lies in a distinct region, giving high cosine similarity with queries. The sink "acts like key biases" that store excess attention without contributing values. Learnable KV biases absorb the sink. Sigmoid attention without normalisation prevents sinks in models up to 1B. Code: https://github.com/sail-sg/Attention-Sink [V] — [Source](https://arxiv.org/html/2410.10781v2)

**Why do LLMs attend to the first token?** Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu. arXiv 2504.02732 (v4 August 2025). The venue is not stated on arXiv; treat it as unverified (it is reportedly a COLM 2025 paper [U]). https://arxiv.org/abs/2504.02732
- Uses Gu et al.'s sink rate, with ε = 0.3 by default and ε = 0.8 for LLaMA 3.1. Sink metric rises with scale: 45.97 (8B, 32 layers, 1,024 heads), 73.49 (70B), 78.29 (405B). Middle layers are the most active. [V] — [Source](https://arxiv.org/html/2504.02732v4)
- **Length-dependent finding:** models pretrained on longer contexts develop much stronger sinks. Sinks are almost absent at training context 128. On a fixed budget of 5B tokens, sink formation increases with training context length until it saturates (about 120M-parameter LLaMA2-style models trained from scratch). [V] — [Source](https://arxiv.org/html/2504.02732v4)
- Causal test: removing BOS from Gemma 7B at inference drops ARC-e from 80.77 to 28.49 and **RULER-4K from 82.57 to 0.00**. In models pretrained with a fixed BOS, removing it drops the sink metric from 90.84% to 0.05% and raises validation loss from 2.69 to 7.56. [V] — [Source](https://arxiv.org/html/2504.02732v4)
- Theory: sinks let heads act as near no-ops (attending to a low-norm value), which limits over-mixing and over-squashing. The bounds depend on depth, head count and context length. Code: adapted from sail-sg/Attention-Sink; no separate repository. [V] — [Source](https://arxiv.org/html/2504.02732v4)

**Massive Activations in Large Language Models.** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. COLM 2024. https://arxiv.org/abs/2402.17762
- A few hidden-state activations are about 10^4 times larger than the median. They are concentrated on the starting token or delimiter tokens in specific feature dimensions. They act as fixed biases and are tied to attention concentrating on those tokens (sinks). Setting them to zero destroys performance. Code: https://github.com/locuslab/massive-activations [K] — [Source](https://arxiv.org/abs/2402.17762)
- Gu et al. confirm the first token's large ℓ2 norm (massive activation) but small key and value norms. [V] — [Source](https://arxiv.org/html/2410.10781v2)

#### Positional bias and length-controlled behaviour

**Lost in the Middle: How Language Models Use Long Contexts.** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. TACL 2024. https://arxiv.org/abs/2307.03172 ; https://aclanthology.org/2024.tacl-1.9/
- Tasks: multi-document QA (NaturalQuestions-Open with 10, 20 or 30 retrieved documents and the gold document moved through positions) and synthetic key–value retrieval (JSON of 75, 140 or 300 UUID key–value pairs). Result: a U-shaped accuracy curve, best when relevant information is at the start or end and worst in the middle. The paper is behavioural only; it reports no attention analysis. Code and data: https://github.com/nelson-liu/lost-in-the-middle [K] — [Source](https://arxiv.org/abs/2307.03172)

**Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization.** Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long T. Le, Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, Tomas Pfister. Findings of ACL 2024. https://arxiv.org/abs/2406.16008
- Links lost-in-the-middle to an intrinsic **U-shaped attention bias**. Documents at the beginning and end receive more attention regardless of relevance. Calibration: model the observed attention to a document as relevance plus a position bias; estimate the bias with a dummy or irrelevant document at each position; subtract it to get calibrated attention. Calibrated attention ranks the gold document better, and attention intervention improves RAG by up to 15 percentage points. [V for the main claims via abstract and secondary pages; the author list and the exact "Findings" status are K] — [Source](https://arxiv.org/html/2406.16008v2)

**RULER: What's the Real Context Size of Your Long-Context Language Models?** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA). COLM 2024. https://arxiv.org/abs/2404.06654
- 13 tasks in 4 categories: retrieval (S-NIAH ×3, MK-NIAH ×3, MV-NIAH, MQ-NIAH), multi-hop tracing (variable tracking), aggregation (CWE, FWE), and QA (SQuAD and HotpotQA with distractor paragraphs). Lengths are configurable, with 4K–128K standard. "Effective length" = the largest length at which the score exceeds Llama-2-7B's 4K score (85.6%). Code: https://github.com/NVIDIA/RULER [K] — [Source](https://arxiv.org/abs/2404.06654)

### Inferences
- For a thesis, the cleanest per-head "functional label" pipeline is: (i) Olsson/TransformerLens induction score and a RazorAttention-style echo score on repeated random tokens (a cheap, length-free probe); (ii) Wu et al.'s retrieval score on S-NIAH across the 1K–16K grid; (iii) the Gu et al. sink importance α₁^{l,h} with ε = 0.3. All three need only argmax attention or the attention column on one key, so none requires full attention matrices (see Q2).
- Optimisation-based (DuoAttention) and query-mass-based (QRHead) selections disagree with argmax copy-paste selection by design. Reporting the overlap between the head sets (Jaccard at matched K) is itself a contribution.
- The Barbero result that sink strength grows with training context length, together with the RULER-4K collapse without BOS, suggests that sink mass must be modelled as a covariate when relating entropy to failure. Entropy that is high because mass goes to the sink is different from entropy that is high because mass is dispersed over haystack tokens.

### Gaps
- Exact Figure-7 masking numbers in Wu et al. (random vs. retrieval heads per model) are in figures only.
- The HeadKV retrieval-reasoning score formula and the RazorAttention exact thresholds and venue were not verified from primary text.
- The Olsson copying-score formula and the exact Voita and Michel numbers are [K] (from memory of the papers, not re-fetched).

---

## Q2. Measuring per-head concentration / sharpness (entropy, top-k mass, sink mass, effective rank) at long context without materialising full attention; tooling and 24 GB GPU limits

### Takeaway
No standard library computes per-head entropy at 16K out of the box. Practical options: (1) HF eager attention with `output_attentions=True`, which materialises B×H×N×N and is infeasible at 16K for a 7B model on 24 GB; (2) hooks that capture Q and K (after RoPE) and recompute the softmax over **query chunks or sampled queries**; (3) a two-pass streaming computation that reuses FlashAttention's per-row log-sum-exp (`softmax_lse`). The theory literature motivates *length-normalised* entropy, since softmax entropy tends to grow toward log N as N grows (attention fading or dispersion).

### Cited Findings
- The FlashAttention interface returns the output and `softmax_lse` (the per-query log-sum-exp normaliser). The full softmax matrix is returned only via `return_attn_probs`/`return_softmax`, which is intended for testing (in some paths, honoured only with dropout). [V] — [Source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
- The log-sum-exp alone cannot recover entropy. Entropy requires an extra accumulation of Σ p·s per row (a custom kernel or second pass). [V; reasoning reported by search tool, consistent with the math below] — [Source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
- Theory of dispersion. Veličković, Perivolaropoulos, Barbero, Pascanu, "softmax is not enough (for sharp out-of-distribution)" (arXiv 2410.01104; v3 retitled "Softmax is not Enough (for Sharp Size Generalisation)"): for any softmax circuit with bounded logits, attention must disperse as the number of items grows at test time, even for max-finding. They propose an **adaptive temperature** at inference as a mitigation. [V] — [Source](https://arxiv.org/abs/2410.01104v1); [v3](https://arxiv.org/html/2410.01104v3)
- Nakanishi, "Scalable-Softmax Is Superior for Attention" (arXiv 2501.19399, January 2025): the maximum softmax output tends to 0 as input size grows ("attention fading"). SSMax multiplies logits by s·log n. It improves pretraining loss, long-context performance and key-information retrieval, and gives some gain even when swapped into already-pretrained models. [V via abstract and secondary summary] — [Source](https://arxiv.org/html/2501.19399v1)
- Vasylenko, Pitorro, Martins, Treviso, "Long-Context Generalization with Sparse Attention" (ICLR 2026, arXiv 2506.16640): with softmax, non-informative tokens accumulate mass as length grows, causing dispersion and representational collapse. α-entmax gives exact zeros. ASEntmax (learnable, length-scaled temperature) reaches up to 1000× length extrapolation on synthetic tasks and better retrieval at 8× training length than softmax, SSMax and fixed α-entmax. [V] — [Source](https://arxiv.org/abs/2506.16640)
- Anson, "Scale-invariant attention" (arXiv 2505.17083; reported NeurIPS 2025 [U]): attention becomes increasingly diffuse (high entropy) as context grows, and the paper proposes scale-invariant logits. [U] — [Source](https://arxiv.org/pdf/2505.17083)
- "Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models" (arXiv 2412.16545): higher attention entropy at the query tokens under parallel encoding is tied to performance drops. Sinks and selective attention reduce it. [U, from search snippet] — [Source](https://arxiv.org/html/2412.16545)
- "From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs" (arXiv 2606.09508): heads split into "rigid" and "dynamic" entropy behaviour. [U] — [Source](https://arxiv.org/pdf/2606.09508)
- "Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long Context Extension" (arXiv 2608.10296): attention entropy and sink strength cluster by presence of QK-norm, and sinks correlate with better long-context extension. [U] — [Source](https://arxiv.org/pdf/2608.10296)
- Clark et al. (2019) already used per-head average attention entropy as a head descriptor, and Voita et al. (2019) used mean max-weight "confidence" (= top-1 mass). [K] — [Clark](https://arxiv.org/abs/1906.04341); [Voita](https://arxiv.org/abs/1905.09418)
- Tooling (documentation not re-fetched in this session):
  - TransformerLens: `hook_pattern` and `hook_attn_scores` per layer expose B×H×N×N tensors. It is designed for models up to about 7B, and its caching of all activations is memory-heavy. [K] — [Source](https://transformerlensorg.github.io/TransformerLens/)
  - nnsight: wraps HF models and allows saving or intervening on any module output, e.g. `q_proj` and `k_proj`. [K] — [Source](https://nnsight.net/)
  - The Wu et al. and DuoAttention repositories provide reference HF-based detection code. [V for repo existence] — [Retrieval_Head](https://github.com/nightdessert/Retrieval_Head); [duo-attention](https://github.com/mit-han-lab/duo-attention)

### Inferences (protocol proposals; derived, not from a single source)
- **Memory arithmetic.** One N×N attention map at N = 16,384 has 2.68×10^8 entries: about 1.07 GB in fp32 or 0.54 GB in bf16, *per head*. A 32-head layer is about 34 GB in fp32 or 17 GB in bf16. So `output_attentions=True` for even one layer at 16K does not fit on a 24 GB GPU alongside a 7B bf16 model (about 14–15 GB of weights). At 4K it is 16× smaller (about 1 GB per layer in bf16), which is feasible layer by layer.
- **Recommended exact computation.** Hook each attention module to capture post-RoPE Q (B×H×N×d) and K/V (B×H_kv×N×d; expand for GQA). For a query block of size b (e.g. 256–1,024), compute S = QKᵀ/√d with the causal mask in fp32 and p = softmax(S). Then accumulate:
  - normalised entropy H̃_i = H(p_i)/log(i) (normalise by the causal row length i, not N);
  - top-k mass Σ_{top-k} p_i;
  - sink mass p_{i,1} (or the mass on the first s tokens);
  - needle mass Σ_{j∈needle} p_{i,j}.
  Memory is b×N×H×4 bytes, e.g. 512×16K×32×4 ≈ 1.07 GB.
- **Streaming entropy identity.** With logits s_{ij} and L_i = logsumexp_j s_{ij} (available as FlashAttention's `softmax_lse`), H_i = L_i − Σ_j p_{ij} s_{ij}, where p_{ij} = exp(s_{ij} − L_i). So a second pass over key blocks only needs to accumulate Σ exp(s−L)·s, which is O(N·d) memory. Top-k mass needs a running block-wise top-k merge.
- **Query sampling.** For retrieval failure analysis, the decisive queries are the final question tokens and the first answer-generation steps. Computing statistics only for those (plus a random sample of about 256 haystack queries for "background" entropy) reduces cost by about N/(sample size). Wu et al.'s score itself uses only decode-step queries. [Wu protocol: V] — [Source](https://arxiv.org/html/2404.15574v1)
- **Effective rank.** The usual definition is exp of the Shannon entropy of the normalised singular values (Roy & Vetterli, 2007; not fetched). For a head this can be computed on the b×N attention block or on the head-output matrix for sampled queries via randomised SVD, which avoids the N×N matrix.
- **GQA caveat.** DuoAttention operates on *KV heads* (gates per KV head), while scores such as Wu's are per *query head*. Results should be reported at both granularities. [V for DuoAttention KV-head gating] — [Source](https://arxiv.org/html/2410.10819)

### Gaps
- I found no published, maintained tool that computes per-head entropy or top-k at 16K+ via FlashAttention's LSE; this would need to be implemented.
- TransformerLens and nnsight memory limits at 16K on 24 GB were not documented in sources fetched; the figures above are arithmetic estimates.
- The empirical claims in 2412.16545, 2505.17083, 2606.09508 and 2608.10296 were seen only as search snippets.

---

## Q3. Papers that condition attention statistics on task success vs failure, and whether concentration on the target predicts correctness (and how this changes with length)

### Takeaway
Direct correct-vs-incorrect comparisons of attention on the needle are rare. The strongest evidence:
- Wu et al. find that hallucinated NIAH outputs coincide with retrieval heads being partially or fully inactive.
- Found in the Middle shows positional attention bias swamping relevance, with calibrated attention to the gold document tracking correct use.
- QRHead shows that query-to-gold attention mass in selected heads is a good predictor of relevance (used as a retriever).
- The dispersion and SSMax line shows theoretically and empirically that peak attention decays with N.

A per-length, success-conditioned attention-entropy analysis on public models appears to be an open niche.

### Cited Findings
- Wu et al.: when the model fails NIAH (hallucinates), retrieval heads are only partially activated or not activated. Weaker retrieval heads hold partial needle information, and masking the strongest leaves only partial recovery. The detection grid covers 1K–50K and 10 depths. [V via ICLR version search summary + HTML] — [Source](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Conference.pdf); [HTML](https://arxiv.org/html/2404.15574v1)
- Found in the Middle: attention to documents at the beginning and end is high regardless of relevance (U-shape). After subtracting the positional-bias estimate, attention ranks the gold document correctly more often, and intervening on attention yields up to +15 pp on RAG. The implication is that raw attention is a biased but informative signal of what the model will use. [V] — [Source](https://arxiv.org/html/2406.16008v2)
- QRHead: the accumulated attention mass from query to candidate documents in the top query-focused heads works as a zero-shot retriever and re-ranker (beats RankGPT on BEIR; more than 10% over full context on LongMemEval and CLIPPER). Attention concentration on the right span in specific heads therefore carries relevance information. [V] — [Source](https://arxiv.org/abs/2506.09944)
- Barbero et al.: removing the sink (BOS) drops Gemma-7B RULER-4K from 82.57 to 0. Sink mass is causally necessary for retrieval success, so "concentration" on non-target tokens is not simply harmful. [V] — [Source](https://arxiv.org/html/2504.02732v4)
- SSMax: as n grows, the maximum softmax probability decays ("attention fading"). Length-scaled logits improve key-information retrieval, which is consistent with dispersion being a cause of length-dependent retrieval failure. [V via abstract/summary] — [Source](https://arxiv.org/html/2501.19399v1)
- ASEntmax (ICLR 2026): dispersion grows with length under softmax. Sparse, length-adaptive attention retains retrieval accuracy at 8× training length. [V] — [Source](https://arxiv.org/abs/2506.16640)
- Other works close to "success/failure conditioning", not verified in this session:
  - "Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell" (Lu et al., 2024; arXiv 2406.14673). Probing shows hidden states encode the target position even when the answer is wrong. [U] — [Source](https://arxiv.org/abs/2406.14673)
  - "Attention Sorting Combats Recency Bias in Long Context Language Models" (Peysakhovich & Lerer, 2023; arXiv 2310.01427). Reorders documents by the attention they receive and improves QA. [U] — [Source](https://arxiv.org/abs/2310.01427)
- A search specifically for "attention mass on needle in correct vs incorrect cases" returned no dedicated 2025 paper beyond the retrieval-head line. [V: negative search result] — [Source](https://arxiv.org/pdf/2602.11162)

### Inferences
- Proposed thesis analysis. For each (model, length L ∈ {1K, 2K, 4K, 8K, 16K}, depth d) instance, label success or failure. Then compare, per (layer, head):
  - needle mass at the first answer step;
  - normalised entropy of the last question token's attention;
  - top-k mass;
  - sink mass.

  Use the effect size (Cohen's d or AUROC of the statistic as a correctness predictor) and its slope against log L. The literature predicts the following, all of which are testable:
  - AUROC is highest in retrieval heads.
  - Needle mass decays with L, and failure begins when the retrieval heads' needle mass falls below a threshold.
  - Sink mass is roughly stable or grows.
- Control for position (U-shaped bias, per Found in the Middle) by stratifying by depth, or by subtracting a dummy-needle baseline as in their calibration.
- Causal follow-up: mask the top-K retrieval heads (Wu et al.: K = 50) versus K random heads at each L, or apply SSMax/temperature scaling only to retrieval heads (a SEAL-like intervention). Then test whether the decline of correctness with L flattens.

### Gaps
- No verified paper reports per-length AUROC of per-head attention entropy or needle mass for predicting NIAH correctness. The claim that "concentration predicts correctness, and this weakens with length" is plausible but not established in the sources found.
- The "Retrieval Heads are Dynamic" method and its results on failure cases were not retrieved.

---

## Q4. Synthetic, length-controllable retrieval benchmarks suitable for small models; generation procedures and pitfalls

### Takeaway
Use RULER's S-NIAH and MK-NIAH generators (the most configurable and widely cited) and passkey retrieval as the simplest probe. Use MQAR (Zoology) or the phonebook and copy tasks (Repeat After Me) for small or from-scratch models. Wu et al.'s and DuoAttention's own detection sets are effectively S-NIAH/passkey variants, so reusing them aligns head detection with evaluation. Main pitfalls: tokenisation of needle values, needle depth (U-shaped bias), distractor similarity, and the haystack type (repeated filler vs. natural essays).

### Cited Findings
- **RULER S-NIAH / MK-NIAH / MV-NIAH / MQ-NIAH**: needle template "One of the special magic {type_needle_k} for {key} is: {value}." Keys are words; values are 7-digit numbers, words or UUIDs. Haystack is either a repeated noise sentence ("The grass is green. The sky is blue. ...") or Paul Graham essays. MK-NIAH adds distractor needles with different keys (the hardest variant fills the haystack with distractors). Lengths are set by the tokenizer-specific token count. Variable tracking and CWE/FWE give non-retrieval controls. Code: https://github.com/NVIDIA/RULER [K] — [Source](https://arxiv.org/abs/2404.06654)
- **Passkey retrieval** (Mohtashami & Jaggi, "Landmark Attention", 2023, arXiv 2305.16300): a random 5-digit passkey ("The pass key is 71432. Remember it.") is embedded in repeated filler ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.") and queried at the end. [K] — [Source](https://arxiv.org/abs/2305.16300)
- **DuoAttention passkey variant**: ten 32-token passkeys in BookSum text, lengths across 50 intervals from 1K to the maximum, loss on the passkeys only. [V] — [Source](https://arxiv.org/html/2410.10819)
- **Wu et al. retrieval-head NIAH**: 3 (question, needle, haystack) triples × 20 lengths (1K–50K) × 10 depths. Detection requires that the needle tokens be copied verbatim, so needles should not be predictable from parametric knowledge. [V] — [Source](https://arxiv.org/html/2404.15574v1)
- **MQAR (Zoology)**: Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré, "Zoology: Measuring and Improving Recall in Efficient Language Models" (ICLR 2024, arXiv 2312.04927). Sequences of key–value token pairs are followed by multiple queries of keys, with power-law-distributed query positions. Vocabulary size, sequence length and the number of KV pairs are controllable. Designed for small models (e.g. 2-layer, d_model 64–512) trained from scratch. Code: https://github.com/HazyResearch/zoology [K] — [Source](https://arxiv.org/abs/2312.04927)
- **Repeat After Me** (Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers are Better than State Space Models at Copying", ICML 2024, arXiv 2402.01032): copying random strings, and phone-book lookup (a name–number list plus a query name) at controllable list sizes. Pretrained transformers (Pythia) outperform SSMs (Mamba) as list length grows. [K] — [Source](https://arxiv.org/abs/2402.01032)
- **Lost in the Middle KV retrieval**: JSON objects of 75, 140 or 300 UUID key–value pairs, with the queried key at a controlled position. Useful for depth-controlled analysis. [K] — [Source](https://arxiv.org/abs/2307.03172)
- **Pitfall: positional bias.** A U-shaped attention bias favours the beginning and end, so needle depth must be balanced or stratified. [V] — [Source](https://arxiv.org/html/2406.16008v2)
- **Pitfall: semantic vs. verbatim retrieval.** Copy-paste NIAH heads transfer poorly to paraphrased or semantic retrieval. QRHead finds that task selection for head detection matters. [V] — [Source](https://arxiv.org/abs/2506.09944)
- **Pitfall: BOS/sink handling.** Dropping BOS (e.g. when concatenating segments or using a tokenizer without `add_bos_token`) can collapse retrieval: Gemma-7B RULER-4K fell from 82.57 to 0. [V] — [Source](https://arxiv.org/html/2504.02732v4)
- **Pitfall: argmax-based scoring artefacts.** One 2026 preprint reports that an apparent argmax retrieval "failure" was a localisation artefact, which argues for also measuring needle-span mass, not only argmax. [U] — [Source](https://arxiv.org/pdf/2606.21249)

### Inferences
- Tokeniser effects: 7-digit numbers tokenise very differently across Llama-2 (digit-by-digit), Llama-3 (up to 3-digit chunks) and Qwen (single digits). The needle token span |k| in the Wu score therefore differs across models. Generate by *token* length, record the needle token span per tokenizer, and prefer word or UUID values when comparing across families. (This is inferred from known tokenizer behaviour; not verified against a source in this session.)
- Distractor design: report three versions, with no distractors, with same-format different-key distractors (MK-NIAH), and with near-duplicate keys. Attention-mass-on-needle should be computed over the true needle *and* distractors to separate "dispersion" failures from "wrong-target" failures.
- For a 24 GB budget, lengths of 1K–16K on 1B–8B public models (e.g. Llama-3.2-1B/3B, Qwen2.5-1.5B/7B, Mistral-7B) are feasible with the chunked-query protocol in Q2. Wu et al.'s grid (20 lengths × 10 depths × 3 needles) can be reduced to 5 lengths × 10 depths × at least 20 needles to get enough failures per cell for AUROC.

### Gaps
- Exact RULER template strings, Zoology MQAR defaults and the Repeat After Me phonebook format are [K] (not re-fetched). Verify them against the repositories before citing.
- No source found that quantifies tokenizer-induced variance in NIAH or retrieval-score results.
