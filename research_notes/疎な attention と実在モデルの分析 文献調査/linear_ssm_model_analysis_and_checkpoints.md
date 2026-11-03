# Attention-like behaviour inside pretrained linear-attention, SSM (Mamba / Mamba-2) and hybrid LMs, and matched-softmax checkpoints for a 24 GB-GPU analysis (2023–2026)

Research date: 2026-11-03. All HuggingFace (HF) IDs below were checked against the HF API (`huggingface.co/api/models?author=…`) on that date unless marked otherwise. Paper metadata (authors, dates, venue comments) was checked against the arXiv API. "Unverified" means I could not confirm it from a primary source in this session.

## Q1. Implicit attention in SSMs and linear RNNs: extraction, normalisation, and length-dependent concentration

### Takeaway
Two papers set up "implicit attention" for selective SSMs and gated linear RNNs. Ali–Zimerman–Wolf (S6 only) give α̃ᵢⱼ = Cᵢ(∏ₖ₌ⱼ₊₁ⁱ Āₖ)B̄ⱼ. Zimerman–Ali–Wolf extend this to a whole-layer operator H, which folds in conv1d, gates and norms, for Mamba, Mamba-2, Griffin, RWKV, RetNet and HGRN. Mamba-2's SSD form gives the same object per head as a masked kernel L ⊙ CBᵀ. None of these maps sums to 1 or is guaranteed non-negative. Published work therefore takes absolute values and then either min-max scales them (for pictures) or L1-normalises each row (DeciMamba, which yields a distribution usable for entropy). DeciMamba is the only published, length-resolved measurement of these maps I found (2K→16K). No 2025–2026 paper reports a normalised-entropy / top-k / effective-rank curve of implicit maps versus length 1K–16K, so the thesis would be filling a gap.

### Cited Findings
**"The Hidden Attention of Mamba Models"** (Ameen Ali, Itamar Zimerman, Lior Wolf; Tel Aviv University). arXiv 2403.01590, submitted 2024-03-03, v2 2024-03-31. No venue is given on the arXiv page; I did not verify whether it was later published at a conference. Code: https://github.com/AmeenAli/HiddenMambaAttn — [arXiv](https://arxiv.org/abs/2403.01590)
- Exact implicit attention (Eq. 12): α̃ᵢ,ⱼ = Cᵢ (∏ₖ₌ⱼ₊₁ⁱ Āₖ) B̄ⱼ. Expanded with the discretisation (Eq. 16) it is S_C(xᵢ)·exp(Σₖ₌ⱼ₊₁ⁱ softplus(S_Δ(xₖ)))·A·softplus(S_Δ(xⱼ))·S_B(xⱼ). A simplified Q̃ᵢH̃ᵢⱼK̃ⱼ form (Eqs. 18–19) maps it onto Q/K language — [HTML](https://arxiv.org/html/2403.01590v2)
- Structure: each channel d has N inner matrices, one per state coordinate, which are summed into one L×L matrix per channel (Eq. 14). That gives D matrices per layer. The authors estimate this as about D·N/H ≈ 100N times more attention matrices than a Transformer layer, e.g. D=384 versus H=3 for ViM-Tiny versus DeiT-Tiny. The introduction instead says "three orders of magnitude", and the paper does not reconcile the two figures — [HTML](https://arxiv.org/html/2403.01590v2)
- Aggregation and normalisation:
  - Rollout uses α̃^λ = I + E_d(α̃^{λ,d}), i.e. the mean over channels plus the identity, and layers are multiplied together.
  - No softmax is applied. The paper names the per-row softmax as the main difference from self-attention.
  - For visualisation, Transformer rows get a softmax, while Mamba maps get **min-max normalisation of absolute values**.
  - The attribution variant uses a positive-part operator (·)⁺.
  - No claim is made that the weights are non-negative or sum to 1.
  — [HTML](https://arxiv.org/html/2403.01590v2)
- Checkpoints: NLP visualisation compares Mamba-130m with Pythia-160m, both trained on the Pile. IMDb experiments use `trinhxuankhai/mamba_text_classification`. Vision experiments use ViM-T/S versus DeiT-T/S — [HTML](https://arxiv.org/html/2403.01590v2)
- Numbers (ImageNet perturbation AUC; Pos ↓, Neg ↑):

  | Method | Pos Mamba | Pos ViT | Neg Mamba | Neg ViT |
  |---|---|---|---|---|
  | Raw-Attention | 17.268 | 20.687 | 34.025 | 40.766 |
  | Attribution | 16.619 | 15.351 | 39.632 | 48.089 |

  The paper reports no rank or effective-rank measurements — [HTML](https://arxiv.org/html/2403.01590v2)

**"Explaining Modern Gated-Linear RNNs via a Unified Implicit Attention Formulation"** (Itamar Zimerman, Ameen Ali, Lior Wolf). arXiv 2405.16504, 2024-05-26, v2 2024-10-18. The page I read did not confirm the venue; I believe it is ICLR 2025, but that is unverified. Code: https://github.com/Itamarzimm/UnifiedImplicitAttnRepr — [arXiv](https://arxiv.org/abs/2405.16504)
- The layer is written as Y = H·X. Conv1D becomes a causal Toeplitz matrix M, and gates and activations become diagonal matrices. Per-model forms:
  - Mamba: H = W′·α̂·Z′·M, with W′ = diag(SiLU(Linear₂x)) and Z′ = diag(σ(Conv1D x)).
  - Mamba-2: H = ℕ·W′·α̂·Z′·M, where ℕ is the GroupRMSNorm turned into a diagonal 1/σ_h matrix.
  - Griffin/RG-LRU: H = diag(GeLU(·))·α̃·M.
  - RWKV: diag(σ(r))·α̂ (WKV matrix). Token shift on values is ignored and W_o is dropped.
  - RetNet: G·R.
  - HGRN: G·α_r·G_ACT.
  — [HTML](https://arxiv.org/html/2405.16504v2)
- Normalisation: as before, Transformers get a row softmax and non-Transformer maps get **min-max of absolute values**. Sign handling is not discussed further — [HTML](https://arxiv.org/html/2405.16504v2)
- Checkpoints for the NLP attention maps: RecurrentGemma-2B, RWKV-430M (Pile; `BlinkDL/rwkv-2-pile-430m`), and Mamba 2.8B (Pile). ARC-E perturbation uses Mamba-1.3B and 2.8B — [HTML](https://arxiv.org/html/2405.16504v2)
- Numbers:
  - ImageNet perturbation, attribution: Pos AUC 11.350 versus 16.619 for Ali et al. 2024 (lower is better); Neg AUC 51.310 versus 39.632. ViT reference: 15.351 / 48.089.
  - Segmentation mIoU: 62.51 versus 54.24 (Ali et al.) versus 60.63 (DeiT-S Transformer-Attr).
  - Ablation mIoU: without Conv1D 50.64; without gate 55.78; S6 only 53.19. So the non-S6 components matter for faithfulness.
  — [HTML](https://arxiv.org/html/2405.16504v2)

**"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"** (Tri Dao, Albert Gu). ICLR is not involved; the arXiv comment says ICML 2024. arXiv 2405.21060, 2024-05-31. This paper introduces the SSD framework, semiseparable-matrix duality, and Mamba-2 (2–8× faster than Mamba's selective SSM) — [arXiv](https://arxiv.org/abs/2405.21060)
- In HF `transformers`, the pure-PyTorch Mamba-2 path materialises the intra-chunk decay mask explicitly: `L = torch.exp(segment_sum(A))` (modeling_mamba2.py, around line 301). It is computed over chunks of `chunk_size`, with inter-chunk decay `decay_chunk = exp(segment_sum(...))`. This is the L of the SSD quadratic form L ⊙ (CBᵀ) — [modeling_mamba2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)

**Length-resolved analysis of implicit maps — "DeciMamba: Exploring the Length Extrapolation Potential of Mamba"** (Assaf Ben-Kish, Itamar Zimerman, Shady Abu-Hussein, Nadav Cohen, Amir Globerson, Lior Wolf, Raja Giryes). arXiv 2406.14528, 2024-06-20, v3 2025-04-09. I believe it was published at ICLR 2025, but that is unverified. Code: https://github.com/assafbk/DeciMamba — [arXiv](https://arxiv.org/abs/2406.14528)
- Normalisation (Eq. 7): N(x)ⱼ = |xⱼ| / Σₖ|xₖ|. Each absolute attention row is **L1-normalised into a distribution**. This is the closest published precedent for an entropy-like statistic on implicit maps — [HTML](https://arxiv.org/html/2406.14528v3)
- "Mamba Mean Distance" (Eq. 8) is the expected distance from the last token, Σⱼ N(α)_{L,j}·(L−j). It is computed per channel and averaged over channels and layers, on 100 WikiText-103 examples — [HTML](https://arxiv.org/html/2406.14528v3)
- Qualitative length findings:
  - The hidden attention is dense at 2K but sparse at 16K (8× extrapolation), with entries vanishing toward the lower rows.
  - Raw mean distance grows with length, but normalised by length, context utilisation "drops dramatically".
  - Σₖ Δₖ grows exponentially in layers 16–17, which are the most global layers.
  - The paper gives no numeric mean-distance table.
  — [HTML](https://arxiv.org/html/2406.14528v3)
- Models: `state-spaces/mamba-130m`, `-370m`, `-790m`, `-1.4b`, `-2.8b`, and `xiuyul/mamba-2.8b-zephyr`, with Pythia as the Transformer comparison — [HTML](https://arxiv.org/html/2406.14528v3)

**Other 2024–2026 items, partly verified**
- "Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing" (Peihao Wang et al.; ICLR 2025, arXiv 2501.00658). It proves SSMs have strong recency bias, and that depth leads to over-smoothing. It proposes "polarising" two state-transition channels to 0 and 1 to improve long-range associative recall — [arXiv](https://arxiv.org/abs/2501.00658)
- "MambaLRP: Explaining Selective State Space Sequence Models" (Rezaei Jafari, Montavon, Müller, Eberle; arXiv 2406.07592). An LRP-based attribution alternative to implicit attention, which the authors say is more faithful. It is also used to analyse long-range capability claims — [arXiv](https://arxiv.org/abs/2406.07592)
- "LongMamba" / Mamba long-context channels (ICLR 2025, arXiv 2504.16053). From the search snippet only: Mamba channels split into local and global ones, and the global channels' receptive field shrinks beyond the training length. Unverified in detail — [arXiv PDF](https://arxiv.org/pdf/2504.16053)
- I found no 2025–2026 paper that reports normalised entropy, top-k mass, sink mass or effective rank of implicit SSM/linear-attention maps as a function of input length. A targeted search returned only DeciMamba-style qualitative and mean-distance analyses — [search result context: DeciMamba](https://arxiv.org/html/2406.14528)

### Inferences
- **Normalisation choices for an entropy-like statistic.** Implicit weights are signed and unnormalised. In Mamba-1, Cᵢ and B̄ⱼ are input-dependent and can have either sign, while Āₖ ∈ (0,1). The thesis should therefore pre-register one primary normalisation and report the others as robustness checks:
  - (a) |α| / Σ|α| per row, which matches DeciMamba's Eq. 7. This is the primary candidate.
  - (b) ReLU(α) / Σ ReLU(α), which matches the positive-part operator in the Ali et al. attribution.
  - (c) α² / Σα², an "energy" distribution.
- **Extra statistics only implicit maps have.** Because no softmax is applied, report:
  - the raw row L1 mass Σⱼ|αᵢⱼ|, which can grow or shrink with length. A softmax row always sums to 1.
  - the negative-mass fraction Σ|α⁻| / Σ|α|.
  For normalised entropy, divide H(row i) by log(i) (number of visible keys), so values are comparable across 1K–16K.
- **Granularity mismatch.** Choose the unit before comparing with Transformer heads:
  - Mamba-1 gives one map per channel: D = 2·d_model, e.g. 1536 for 130M and 5120 for 2.8B.
  - Mamba-2 gives one map per SSD head, shared across the head dimension P. It has a scalar decay per head, so M = L ⊙ (C Bᵀ) scaled by Δⱼ.
  - Comparison options: per Mamba-2 head (closest to an attention head), the channel mean (Ali rollout), or the full-layer operator H (Zimerman et al.). The Zimerman ablation shows that S6-only maps lose faithfulness (mIoU 53.19 versus 62.51), so whole-layer H is preferable when relating maps to retrieval.
- **"Sink mass" for SSMs.** It is well defined only after normalisation. A natural definition is the share of normalised |α| on token 0, or the first k tokens. Because Āₖ products decay geometrically, a large first-token weight in an SSM needs near-1 retention. Stuffed Mamba reports exactly this: first-token retention above 0.997 at t=8K in some heads (see Q2).

### Gaps
- I could not confirm the final venues of 2403.01590 and 2405.16504 (possibly ECCV/ICLR); the arXiv pages show none.
- I found no published effective-rank analysis of implicit SSM maps. Effective rank (entropy of normalised singular values) would be a new measurement.
- I found no published sink-mass measurement for Mamba, GLA or DeltaNet implicit maps. "Hybrid Architectures for Language Models: Systematic Analysis and Design Insights" (arXiv 2510.04800) uses a sink + SWA baseline but, from the snippet, does not measure SSM sinks — [arXiv](https://arxiv.org/html/2510.04800v1)
- For GLA, DeltaNet and Gated DeltaNet I did not verify a published implicit-attention extraction paper.
  - GLA: the implicit map should be qᵢᵀ diag(∏ gates) kⱼ.
  - DeltaNet / Gated DeltaNet: the implicit map involves the WY / UT-transform inverse (I + tril(diag(β)KKᵀ))⁻¹ ([DeltaNet paper, arXiv 2406.06484](https://arxiv.org/abs/2406.06484), not re-read).
  - Both are my derivations and should be checked against the fla `naive` reference implementations.

## Q2. Mechanistic findings in SSM / hybrid LMs (G&A heads, retrieval layers, state collapse, copying, Jet-Nemotron)

### Takeaway
Retrieval in both Transformers and SSMs runs through a handful of Gather-and-Aggregate heads. In hybrids, those heads sit in the few attention layers: Zamba2-7B has nine attention heads, and ablating them drops MMLU from 64.3% to 34.9%. Pure SSMs show smoother, less sharp G&A maps. Jet-Nemotron independently finds that only 2–3 full-attention layers matter for retrieval. Waleffe et al. show 4 attention layers out of 56 are enough for 8B-scale phonebook and RULER parity or better. Copying and phonebook gaps for pure SSMs appear within hundreds to about 1K tokens. Mamba-2 fails to forget beyond its training length, and this depends on state size.

### Cited Findings
**"Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism"** (Aviv Bick, Eric Xing, Albert Gu; CMU and others). arXiv 2504.18574, v1 2025-04-22, v2 2025-06-11. The venue is not stated on arXiv; I believe it appeared at ICML 2025, but that is unverified. Code: https://github.com/goombalab/Gather-and-Aggregate — [arXiv](https://arxiv.org/abs/2504.18574)
- Models: Llama-3.1-8B-Instruct, Falcon-Mamba-7B-Instruct (Mamba-1, no MLPs), Llamba-8B-Untied (Mamba-2 distilled from Llama-3.1-8B), Zamba2-7B-Instruct (6:1 Mamba-2:attention, two shared attention blocks), and the 3B counterparts — [HTML](https://arxiv.org/html/2504.18574v2)
- Llama-3.1-8B pruned "minimal" model:
  - With L16H22 (Gather) and L17H24 (Aggregate): MMLU 66.32%.
  - Removing either head gives 24.36% or 25.59%, i.e. chance level.
  - KV-retrieval with 20 pairs: 85.6% → 76% without L16H22, and → 62% without L17H24.
  — [HTML](https://arxiv.org/html/2504.18574v2)
- Head locations:
  - Llama-3.1-8B: layers 16–17 of 32.
  - Llamba-8B: the same layers L16–L17 as its teacher.
  - Falcon-Mamba-7B: layers 35–36 of 64. Four Mamba-1 channels alone reach 52.26% MMLU.
  - Zamba2-7B: nine attention heads, L47H{2,4,6,8,18,21,25} and L59H{17,21}.
  — [HTML](https://arxiv.org/html/2504.18574v2)
- Hybrids put Aggregate work in their attention layers. In Zamba2-7B, ablating the nine attention heads drops MMLU from 64.3% to 34.9%, while knowledge accuracy stays about 70%. Swapping Llama's L17 into Llamba-8B raises MMLU from 33% to 50% — [HTML](https://arxiv.org/html/2504.18574v2)
- Table 2 (head removal):

  | Model | Heads removed | MMLU before → after | Change |
  |---|---|---|---|
  | Llama-3.1-8B | 20 | 68.1 → 38.1 | −44.0% |
  | Llamba-8B | 30 | 60.7 → 58.1 | −4.3% |
  | Zamba2-7B | 60 | 65.1 → 36.2 | −44.4% |

  The paper gives Zamba2's baseline as 64.3 in Section 5.3 and 65.1 in Table 2, an internal inconsistency — [HTML](https://arxiv.org/html/2504.18574v2)
- SSM G&A maps are "smoother", with leakage across segment boundaries. This rests on visual inspection only (Fig. 5); **no entropy or sharpness statistic is reported**. That leaves a direct opening for the thesis's quantitative measurement — [HTML](https://arxiv.org/html/2504.18574v2)

**"Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search"** (Yuxian Gu, Qinghao Hu, Shang Yang, Haocheng Xi, Junyu Chen, Song Han, Han Cai; NVIDIA / MIT). NeurIPS 2025, arXiv 2508.15884. Code: https://github.com/NVlabs/Jet-Nemotron. HF: `jet-ai/Jet-Nemotron-2B`, `jet-ai/Jet-Nemotron-4B`; licence "other", name `ncslv1` (non-commercial) — [arXiv](https://arxiv.org/abs/2508.15884), [HF card](https://huggingface.co/jet-ai/Jet-Nemotron-2B)
- PostNAS method:
  - Start from Qwen2.5-1.5B (2B model) or Qwen2.5-3B (4B model) and freeze the MLPs.
  - Build a once-for-all super-network with linear-attention paths and train it with feature distillation.
  - Beam-search full-attention placement under a budget, e.g. 2 layers.
  — [HTML](https://arxiv.org/html/2508.15884v3)
- Key findings:
  - "Not all attention layers contribute equally": about 2 layers matter for MMLU and 2–3 for retrieval.
  - The layers critical for MMLU differ from those critical for retrieval.
  - The top MMLU and retrieval layers cover most math-critical layers.
  - The analysis was done on Qwen2.5-1.5B.
  — [HTML](https://arxiv.org/html/2508.15884v3)
- Jet-Nemotron-2B keeps full attention only in layers 15 and 20, plus sliding-window attention in layers 21 and 22. Everything else is JetBlock (Gated-DeltaNet time-mixing with dynamic causal convolution on V). For the 4B model, the text and Table 9 disagree on the full-attention indices — [HTML](https://arxiv.org/html/2508.15884v3)
- Training: 50B tokens in stage 1 (Nemotron-CC + Redstone-QA, MLPs frozen) plus 350B tokens in stage 2. The training context length is not stated — [HTML](https://arxiv.org/html/2508.15884v3)
- Linear-block comparison (Table 1, MMLU / math / retrieval / commonsense):

  | Block | MMLU | Math | Retrieval | Commonsense |
  |---|---|---|---|---|
  | Gated DeltaNet | 55.6 | 32.3 | 69.3 | 58.7 |
  | Mamba2 | 51.5 | 26.0 | 68.9 | 57.5 |
  | GLA | 55.8 | 31.2 | 66.6 | 58.5 |
  | RetNet | 53.6 | 29.9 | 63.7 | 58.1 |
  | DeltaNet | 48.9 | 27.4 | 67.9 | 56.6 |
  | JetBlock | 56.3 | 32.8 | 69.9 | 58.5 |

  Retrieval average (FDA/SWDE/SQuAD): Jet-Nemotron-2B 74.2 versus 76.1 for Qwen3-1.7B-Base — [HTML](https://arxiv.org/html/2508.15884v3)

**"An Empirical Study of Mamba-based Language Models"** (Roger Waleffe, Wonmin Byeon, …, Tri Dao, Albert Gu, …, Bryan Catanzaro; NVIDIA). arXiv 2406.07887, 2024-06-12 — [arXiv](https://arxiv.org/abs/2406.07887)
- The 8B Mamba-2-Hybrid has 56 layers: 24 Mamba-2, 4 self-attention and 28 MLP (7.1% attention). The attention layers sit at 1-based positions 11, 23, 34 and 46, as derived by the reader from the paper's pattern — [HTML](https://arxiv.org/html/2406.07887v1)
- Phonebook:
  - The 4K Transformer is about 100% up to 4K.
  - Mamba and Mamba-2 at 1.1T tokens start failing around 500 tokens; Mamba-2 at 3.5T fails beyond about 1K.
  - The 4K hybrid is 100% up to about 5.5K. The 16K and 32K hybrids are perfect past their training length, and the 128K hybrid past 128K.
  — [HTML](https://arxiv.org/html/2406.07887v1)
- MMLU 5-shot at 3.5T tokens: Transformer 50.07, Mamba-2 48.7, Hybrid 53.60. At 1.1T: Transformer 46.28, Mamba 28.00, Mamba-2 29.19 — [HTML](https://arxiv.org/html/2406.07887v1)
- The paper does **not** analyse which individual layers retrieve — [HTML](https://arxiv.org/html/2406.07887v1)

**"Stuffed Mamba: Oversized States Lead to the Inability to Forget"** (Yingfa Chen, Xinrong Zhang, Shengding Hu, Xu Han, Zhiyuan Liu, Maosong Sun; Tsinghua). COLM 2025, arXiv 2410.07145. This version links no code repository — [arXiv](https://arxiv.org/abs/2410.07145)
- Models:
  - `state-spaces/mamba2-130m` … `mamba2-2.7b`
  - `state-spaces/mamba-130m` … `mamba-2.8b`
  - `RWKV/rwkv-5-world-all-pth`, `RWKV/v6-Finch-1B6-HF`, `RWKV/v6-Finch-3B-HF`
  - HGRN-2 1.3B (appendix)
  — [HTML](https://arxiv.org/html/2410.07145v4)
- Passkey: Mamba-2 is near-perfect within 8K and "poor or even zero" beyond 16K for every size except 130M. Larger models generalise worse in length. With continued pretraining on longer sequences, Mamba-2-370M reaches near-perfect retrieval at 256K — [HTML](https://arxiv.org/html/2410.07145v4)
- Fitted laws (R² > 0.999; units not explicit in the text I read):
  - Minimum training length to learn forgetting: T_forget = 5.172·N_S − 4.469 (linear in state size).
  - Passkey recall length: T_recall = 4.756·(1.365^{N_S} − 1) − 0.742 (exponential in state size).
  — [HTML](https://arxiv.org/html/2410.07145v4)
- Mechanism:
  - Memory retention αₜ = exp(−Δₜ·exp(A)).
  - In layer 38, three heads keep cumulative first-token retention above 0.997 at t=8K, i.e. the model barely forgets.
  - In Mamba-2-370M L38H2 at t=20K, the state variance explodes, driven by a few outlier channels.
  - Training-free fix ("RRI"): multiply αₜ by 0.9999 and Bₜ by 0.75.
  — [HTML](https://arxiv.org/html/2410.07145v4)

**"Repeat After Me: Transformers are Better than State Space Models at Copying"** (Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach; Harvard / Kempner). ICML 2024, arXiv 2402.01032. Code: https://github.com/sjelassi/transformers_ssm_copy — [arXiv](https://arxiv.org/abs/2402.01032)
- Pretrained comparison: Pythia 410M/1.4B/2.8B versus Mamba 360M/1.4B/2.8B (all Pile). The 410M Pythia beats the 2.8B Mamba on phonebook lookup once the book has at least 70 entries. The largest Mamba gets 0 accuracy copying 300-token shuffled strings. Exact figures appear only in the plots — [HTML](https://arxiv.org/html/2402.01032v2)
- Synthetic results:
  - A Transformer trained on strings of at most 50 tokens generalises to length 1000 with Hard-ALiBi.
  - GSSMs need about 100× more samples in-distribution at length ≤300.
  — [HTML](https://arxiv.org/html/2402.01032v2)

**"Locating and Editing Factual Associations in Mamba"** (Arnab Sen Sharma, David Atkinson, David Bau). COLM 2024, arXiv 2404.03646.
- Causal tracing in Mamba finds mid-layer components at the last subject token, matching the pattern in Transformers.
- ROME-style rank-one editing works.
- Attention-knockout is adapted to Mamba and compared directly with a similar-size Transformer.
— [arXiv](https://arxiv.org/abs/2404.03646)

**"Hymba: A Hybrid-head Architecture for Small Language Models"** (Xin Dong et al.; NVIDIA). arXiv 2411.13676.
- Attention heads give "high-resolution recall" and SSM heads do summarisation.
- Learnable meta tokens are motivated by relieving the "forced-to-attend" (sink) burden. This makes Hymba directly relevant to sink-mass analysis.
— [arXiv](https://arxiv.org/abs/2411.13676)

### Inferences
- Bick et al. and Jet-Nemotron together predict what the thesis should see in hybrids: a sharp, low-entropy distribution concentrated in 1–3 attention layers (or heads), carrying most of the retrieval-relevant mass, and diffuse SSM / linear maps elsewhere. A per-layer sharpness profile correlated with head-ablation retrieval drops would turn Bick's visual "smoother" claim into numbers.
- Stuffed Mamba's first-token retention above 0.997 suggests a sink-like effect can emerge from near-unit decay in SSMs, without any softmax. Measure sink mass on normalised implicit maps against length, especially beyond the 2K training length of the state-spaces checkpoints.

### Gaps
- I did not verify any published per-layer retrieval-localisation study for Samba, Falcon-H1 or Nemotron-H; searches returned none.
- I did not verify any published "induction head in Mamba" mechanistic paper with numbers, beyond the Mamba paper's synthetic induction task (not re-read).
- I did not re-read any RWKV-7 or Gated-DeltaNet memory-analysis paper, so I have no numbers for them.

## Q3. Matched checkpoints: same data, same size, softmax versus linear / SSM

### Takeaway
Four public suites give strict matching, in order of cleanliness:
- **state-spaces 2.7B trio** (Mamba-2 / Transformer++ / Mamba-2-attention hybrid): Pile, 300B tokens, same GPT-NeoX-size vocab (50277), Apache-2.0.
- **HazyResearch 1B / 360M Based / Mamba / attention trios**: Pile, "exact same data order" for the 1B-50B set.
- **fla-hub 1.3B / 2.7B-100B** (Transformer++, GLA, DeltaNet, RetNet, HGRN, HGRN2, GSA, Mamba): SlimPajama, 32K vocab.
- **m-a-p 340M-20B / 1.3B-100B** (6 linear families × pure + 4 hybrid ratios, plus Transformer baselines): FineWeb-Edu. This is the only one with Gated DeltaNet, and it includes hybrid ratios.

NVIDIA's 8B Mamba-2 / Mamba-2-Hybrid / Transformer trio is public and matched (3.5T tokens). It is Megatron format and 4K-trained (hybrid also 32K and 128K), and at 8B it is tight on 24 GB at fp16. Falcon-H1, Hymba, RecurrentGemma, Zamba2 and Jet-Nemotron have no same-data softmax twin. Jet-Nemotron's closest reference is its parent, Qwen2.5-1.5B.

### Cited Findings
**A. state-spaces (Mamba / Mamba-2 / Transformer++ / hybrid)**
- HF IDs:
  - `state-spaces/mamba-130m`, `-370m`, `-790m`, `-1.4b`, `-2.8b`, plus `-hf` variants and `mamba-390m-hf`
  - `state-spaces/mamba2-130m`, `-370m`, `-780m`, `-1.3b`, `-2.7b`
  - `state-spaces/transformerpp-2.7b`, `state-spaces/mamba2attn-2.7b`, `state-spaces/mamba-2.8b-slimpj`
  - Also listed and not yet analysed: `mamba3-siso-{187m,443m,893m,1.5b}`, `mamba3-mimo-{187m,444m,894m,1.5b}`
  — [HF API listing](https://huggingface.co/state-spaces)
- The README says all Mamba-1 / Mamba-2 sizes, `transformerpp-2.7b` and `mamba2attn-2.7b` were "trained on 300B tokens on the Pile". `mamba-2.8b-slimpj` used 600B SlimPajama tokens. They are base models without instruction tuning — [state-spaces/mamba README](https://github.com/state-spaces/mamba)
- Licence: Apache-2.0 (HF tags on `transformerpp-2.7b`, `mamba2-2.7b`, `mamba2attn-2.7b`) — [HF](https://huggingface.co/state-spaces/transformerpp-2.7b)
- Configs:
  - `mamba2-2.7b`: d_model 2560, 64 layers, vocab 50277.
  - `transformerpp-2.7b`: d_model 2560, 32 layers, 32 heads × head_dim 80, rotary_emb_dim 40, d_intermediate 6912, vocab 50277.
  - So the Transformer++ and Mamba-2 share the tokenizer vocabulary size and width.
  — [mamba2-2.7b config](https://huggingface.co/state-spaces/mamba2-2.7b/blob/main/config.json), [transformerpp-2.7b config](https://huggingface.co/state-spaces/transformerpp-2.7b/blob/main/config.json)
- Pythia is also Pile-trained. The README's eval and benchmark commands use `EleutherAI/pythia-160m` and `EleutherAI/pythia-2.8b` as comparators — [README](https://github.com/state-spaces/mamba)
- The Pythia details (about 300B tokens, GPT-NeoX-20B tokenizer, 2048 context, same data order across sizes) come from my prior knowledge of the [Pythia paper, arXiv 2304.01373](https://arxiv.org/abs/2304.01373) and were **not re-verified** in this session. Two caveats against "exact" matching with Mamba:
  - Data order: Pythia uses its own fixed order (standard and deduplicated variants); Mamba's data order is not documented as identical. Unverified.
  - Recipe: optimiser and LR schedule differ.
  - The Mamba vocab of 50277 equals the GPT-NeoX tokenizer size. That is an inference from the configs.

**B. fla-hub (flash-linear-attention)**
- HF IDs:
  - 1.3B, 100B tokens: `fla-hub/transformer-1.3B-100B`, `gla-1.3B-100B`, `delta_net-1.3B-100B`, `delta_net-1.3B-8K-100B`, `gsa-1.3B-100B`, `hgrn-1.3B-100B`, `hgrn2-1.3B-100B`, `retnet-1.3B-100B`, `mamba-1.3B-100B`
  - 2.7B, 100B tokens: `transformer-`, `gla-`, `delta_net-`, `gsa-`, `hgrn-`, `hgrn2-`, `retnet-`, `mamba-2.7B-100B`
  - 340M (mismatched token counts): `transformer-340M-10B`, `gla-340M-15B`
  - Also: `rwkv7-{168M,421M,1.47B}-pile`, `rwkv6-*`, `rwkv7-*-world/g1*`
  - There is **no Gated DeltaNet checkpoint in fla-hub**.
  — [HF fla-hub](https://huggingface.co/fla-hub)
- `transformer-1.3B-100B` and `gla-1.3B-100B` carry the dataset tag `cerebras/SlimPajama-627B` and an MIT licence. The `delta_net-1.3B-100B` and `transformer-340M-10B` cards are auto-generated blanks with no licence or dataset — [HF transformer-1.3B-100B](https://huggingface.co/fla-hub/transformer-1.3B-100B), [HF gla-1.3B-100B](https://huggingface.co/fla-hub/gla-1.3B-100B), [HF delta_net-1.3B-100B](https://huggingface.co/fla-hub/delta_net-1.3B-100B)
- Both 1.3B configs have hidden 2048, 24 layers, `max_position_embeddings` 2048 and vocab 32000. GLA has 4 heads with expand_k 0.5; the Transformer has 32 heads with rope_theta 1e4 — [config](https://huggingface.co/fla-hub/gla-1.3B-100B/blob/main/config.json), [config](https://huggingface.co/fla-hub/transformer-1.3B-100B/blob/main/config.json)
- Models load with `trust_remote_code` / `import fla` through `AutoModelForCausalLM` — [fla README](https://github.com/fla-org/flash-linear-attention)

**C. m-a-p hybrid-linear-attention suite** — "A Systematic Analysis of Hybrid Linear Attention" (Dustin Wang, Rui-Jie Zhu, Steven Abreu, …, Jason Eshraghian; UC Santa Cruz, ByteDance Seed, Groningen, CASIA, PolyU, M-A-P). arXiv 2507.06457, 2025-07-08 — [arXiv](https://arxiv.org/abs/2507.06457)
- Setup:
  - 72 models: 36 at 340M parameters (20B tokens) and 36 at 1.3B (100B tokens).
  - Families: DeltaNet, GatedDeltaNet, GLA, HGRN, HGRN2, RetNet.
  - Ratios: pure, 24:1, 12:1, 6:1, 3:1 (linear:full).
  - Data FineWeb-Edu; AdamW with cosine schedule; trained with fla.
  - The Limitations section mentions a 4,096-token context window.
  — [HTML](https://arxiv.org/html/2507.06457v2)
- HF collection: https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e (62 items). Example IDs:
  - `m-a-p/1.3B-100B-GatedDeltaNet-pure`, `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-{3,6,12,24}-1`, same pattern for GLA, DeltaNet, HGRN, HGRN2, RetNet
  - `m-a-p/340M-20B-*-pure(-baseline)`, `m-a-p/340M-20B-*-hybrid-*-1`
  - Matched softmax baselines: `m-a-p/transformer_1.3B_baseline`, `m-a-p/transformer_340M_baseline`
  — [HF API collection](https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e)
- Config of `1.3B-100B-GatedDeltaNet-pure`: hidden 2048, 24 layers, 8 heads × head_dim 256, qk_norm l2, `max_position_embeddings` 2048. This conflicts with the paper's 4,096 mention — [config](https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure/blob/main/config.json)
- RULER recall averages:
  - By ratio: pure 0.256, 24:1 0.338, 12:1 0.340, 6:1 0.390, 3:1 0.397.
  - Transformer baseline: 0.423.
  - Best hybrids: GatedDeltaNet 3:1 0.440, HGRN2 6:1 0.434, DeltaNet 3:1 0.426.
  - Pure models: GatedDeltaNet 0.356, DeltaNet 0.320, HGRN2 0.203, GLA 0.144.
  - The paper's "nearly doubles" wording does not match its own table (+17% from 24:1 to 3:1).
  — [HTML](https://arxiv.org/html/2507.06457v2)
- Licence of the released weights is not stated in the paper — [HTML](https://arxiv.org/html/2507.06457v2)

**D. HazyResearch Zoology / Based**
- HF IDs:
  - `hazyresearch/based-360m`, `based-360M-30B`, `based-1b`, `based-1b-50b`, `based-1.3b`
  - `hazyresearch/attn-360m`, `attn-360M-30B`, `attn-1b`, `attn-1b-50bn`
  - `hazyresearch/mamba-360m`, `mamba-360M-30B`, `mamba-1b`, `mamba-1b-50b`, `mamba-1.3b`
  - `hazyresearch/transformer-pp-1.3b`
  - Also `JRT-360M-30B`, `JRT-1B-50B`
  — [HF hazyresearch](https://huggingface.co/hazyresearch)
- The `based-1b-50b`, `attn-1b-50bn` (Llama-architecture: RoPE, SwiGLU, RMSNorm) and `mamba-1b-50b` cards state that all three are "pretrained on 50Bn tokens of the Pile in the exact same data order". There is no licence tag on these three; `based-360M-30B` and `transformer-pp-1.3b` are Apache-2.0 — [HF based-1b-50b](https://huggingface.co/hazyresearch/based-1b-50b), [HF attn-1b-50bn](https://huggingface.co/hazyresearch/attn-1b-50bn), [HF mamba-1b-50b](https://huggingface.co/hazyresearch/mamba-1b-50b)
- Papers:
  - "Simple linear attention language models balance the recall-throughput tradeoff" (Arora, Eyuboglu, Zhang, et al.; arXiv 2402.18668; ICML 2024 per my prior knowledge, unverified). Based beats Mamba on recall-intensive tasks by 6.22 points at matched perplexity, up to 1.3B — [arXiv](https://arxiv.org/abs/2402.18668)
  - "Zoology: Measuring and Improving Recall in Efficient Language Models" (Arora et al.; arXiv 2312.04927). 17 attention and gated-convolution models pretrained on the Pile; 82% of the perplexity gap is explained by associative recall; a 70M attention model beats a 1.4B gated-convolution model on AR. Introduces MQAR — [arXiv](https://arxiv.org/abs/2312.04927)
- Pile-50B Gated-DeltaNet checkpoints `simarora/gated-deltanet-{d256,k4}-p1b-t50b-pile-re0` exist on HF. They may match the Hazy 1B-50B set, but that is **unverified** (no card read) — [HF search](https://huggingface.co/models?search=gated-deltanet)

**E. NVIDIA Mamba-2 / Mamba-2-Hybrid / Transformer 8B** (Waleffe et al. 2024)
- Collection `nvidia/ssms-666a362c5c3bb7e4a6bcfb9c`: `nvidia/mamba2-8b-3t-4k`, `nvidia/mamba2-hybrid-8b-3t-4k`, `nvidia/mamba2-hybrid-8b-3t-32k`, `nvidia/mamba2-hybrid-8b-3t-128k`, and the matched Transformer `nvidia/gpt3-8b-multi-3.5t-base`. The Transformer's card says "Reference 8B-parameter base Transformer model trained on 3.5T tokens with 4K sequence length". Licence Apache-2.0, `library_name: Megatron-LM` — [HF collection](https://huggingface.co/collections/nvidia/ssms-666a362c5c3bb7e4a6bcfb9c), [HF gpt3-8b](https://huggingface.co/nvidia/gpt3-8b-multi-3.5t-base), [HF hybrid card](https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k)
- The hybrid repo holds only `release/mp_rank_00/model_optim_rng.pt` plus a SentencePiece `.model` file. That is a Megatron checkpoint, not HF-transformers format — [HF API](https://huggingface.co/api/models/nvidia/mamba2-hybrid-8b-3t-4k)
- Training data: 70% English, 15% non-English, 15% code; SentencePiece tokenizer with 256K vocab; 4K pretraining. Long-context extensions to 16K, 32K and 128K used +50B tokens; only the 32K and 128K are on HF — [paper HTML](https://arxiv.org/html/2406.07887v1)

**F. Hybrids without a same-data softmax twin**
- **Falcon-H1** (TII). IDs `tiiuae/Falcon-H1-0.5B-Base`, `-1.5B-Base`, `-1.5B-Deep-Base`, `-3B-Base`, `-7B-Base`, `-34B-Base`, plus Instruct, GPTQ-Int4/Int8 and GGUF variants. Parallel attention + Mamba-2 hybrid; licence `falcon-llm-license` (Falcon-LLM License). Context lengths were not verified in this session — [HF](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base), [tech report arXiv 2507.22448](https://arxiv.org/abs/2507.22448)
- **Hymba-1.5B** (NVIDIA). `nvidia/Hymba-1.5B-Base`, `nvidia/Hymba-1.5B-Instruct`. NVIDIA Open Model License. 32 layers, embedding 1600, 25 attention heads; Mamba and attention heads run in parallel; meta tokens; 90% of attention layers are SWA; cross-layer KV sharing — [HF](https://huggingface.co/nvidia/Hymba-1.5B-Base)
- **RecurrentGemma** (Google DeepMind; Griffin = RG-LRU + local attention). `google/recurrentgemma-2b`, `-2b-it`, `-9b`, `-9b-it`. Gemma licence, gated (manual approval). The report says it reaches "comparable performance to similarly-sized Gemma baselines despite being trained on fewer tokens", so it is **not** data-matched to Gemma — [HF](https://huggingface.co/google/recurrentgemma-2b), [arXiv 2404.07839](https://arxiv.org/abs/2404.07839)
- **Zamba2** (Zyphra). `Zyphra/Zamba2-1.2B`, `-2.7B`, `-7B`, plus instruct variants. G&A heads were analysed in Bick et al. — [HF](https://huggingface.co/Zyphra/Zamba2-2.7B)
- **Jet-Nemotron**. `jet-ai/Jet-Nemotron-2B`, `-4B`; non-commercial `ncslv1` licence. The card pins `transformers<=4.53.0`, `torch<=2.7.1` and flash-attn — [HF](https://huggingface.co/jet-ai/Jet-Nemotron-2B)
- **Nemotron-H**: my HF search with this query did not surface Nemotron-H base checkpoints, so this is unverified here.

### Inferences
- **Best primary design for a 24 GB GPU.**
  - (1) state-spaces `mamba2-2.7b` versus `transformerpp-2.7b` versus `mamba2attn-2.7b`: same data, tokens and vocab, Apache-2.0, about 5.4 GB of fp16 weights each.
  - (2) m-a-p 1.3B-100B pure / hybrid-{24,12,6,3}:1 for GatedDeltaNet / GLA / DeltaNet / HGRN2 / RetNet / HGRN, versus `m-a-p/transformer_1.3B_baseline`. This gives a hybrid-ratio axis plus Gated DeltaNet.
  - (3) fla-hub 1.3B-100B (Transformer++ / GLA / DeltaNet / RetNet / HGRN2 / Mamba) as a SlimPajama replication.
  - (4) the Hazy 1B-50B trio (Based / Mamba / Llama-attention, same data order), which also has Hazy's own recall benchmarks (FDA / SWDE / SQuAD).
- **Length caveat.** All of these were trained at 2K, or 4K for m-a-p and NVIDIA. Evaluating at 16K is length extrapolation for both families:
  - The Transformer side relies on RoPE extrapolation and will collapse; Pythia passkey works only at 1K–2K per DeciMamba.
  - The SSM side shows ERF shrinkage and failure to forget (Q1, Q2).
  - The thesis should therefore present 1K–2K (or 4K) as in-distribution and 4K–16K as extrapolation. Alternatively use `nvidia/mamba2-hybrid-8b-3t-32k`, `fla-hub/delta_net-1.3B-8K-100B`, or the Falcon-H1 / Hymba models for in-distribution long context, accepting that they are unmatched.
- **Memory (my estimates).**
  - At fp16, 1.3B ≈ 2.6 GB and 2.7B ≈ 5.4 GB of weights. A no-grad 16K forward fits in 24 GB for SSMs and linear models. For Transformers, use SDPA/flash for the forward pass and recompute attention maps per head separately.
  - 8B at fp16 is about 16 GB of weights. A 16K prefill is likely feasible only with chunking, or with 4-bit quantisation. 4-bit for the Megatron checkpoints requires conversion first (unverified tooling).

### Gaps
- I did not verify whether the state-spaces Pile models used exactly the same data order as each other, or as Pythia.
- I did not verify the licence, the exact data and token counts, or the context length of the m-a-p Transformer baselines (cards not read).
- Training context length of the fla-hub models: the configs say 2048. I did not check the GLA and DeltaNet papers, which I believe used 2K for the 1.3B/100B runs.
- I did not check context lengths for Falcon-H1 0.5B/1.5B (possibly 16K / 128K, unverified), Hymba (possibly 8K, unverified), or RecurrentGemma (8K sequence, 2K local window; prior knowledge, unverified).
- I did not verify the `mamba2attn-2.7b` layer pattern (number and position of attention layers); its config was not read.
- I did not check whether an HF-format conversion of `nvidia/gpt3-8b-multi-3.5t-base` / `mamba2-hybrid-8b` exists.

## Q4. Tooling: extracting states and implicit attention (HF transformers, fla), pitfalls, memory at 16K

### Takeaway
HF `transformers` Mamba and Mamba-2 have pure-PyTorch paths besides the CUDA-kernel path. Mamba has `slow_forward` with optional `use_mambapy`. Mamba-2's torch path computes the SSD chunked form and explicitly builds `L = exp(segment_sum(A))` inside each chunk. Neither exposes an `output_attentions` argument, so implicit maps must be rebuilt from hooked x, Δ, A, B, C, gate and conv activations. The reference repos (HiddenMambaAttn, UnifiedImplicitAttnRepr, DeciMamba) already do this for Mamba-1. At 16K, full maps are too large to materialise per channel or head; compute selected rows, per-head tiles, or streaming statistics instead.

### Cited Findings
- HF `modeling_mamba.py` provides `cuda_kernels_forward` and a slow path, with a `use_mambapy` option (parallel-scan fallback). No `output_attentions` handling appears — [modeling_mamba.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba/modeling_mamba.py)
- HF `modeling_mamba2.py`:
  - Implements `reshape_into_chunks` and `segment_sum` (lower-triangular cumulative-sum masks).
  - Computes `L = torch.exp(segment_sum(A))` (intra-chunk decay) and `decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[...,-1], (1,0))))` (inter-chunk) over `config.chunk_size`.
  - No `output_attentions` handling appears.
  — [modeling_mamba2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)
- Installing `mamba-ssm` without `MAMBA_KEEP_CUDA_BUILD=TRUE` skips `selective_scan_cuda`, so a no-kernel install is possible — [mamba README](https://github.com/state-spaces/mamba)
- The original `state-spaces/*` repos (not `-hf`) use the `mamba_ssm` model format, e.g. `mamba2-2.7b` / `transformerpp-2.7b` configs with `ssm_cfg` and `attn_layer_idx`. They load via `mamba_ssm.models.mixer_seq_simple`, not HF Auto classes. The `-hf` variants exist only for Mamba-1 — [HF listing](https://huggingface.co/state-spaces), [config](https://huggingface.co/state-spaces/transformerpp-2.7b/blob/main/config.json)
- fla models default to `attn_mode: "chunk"` (Triton kernels), as in the `gla-1.3B-100B` and `1.3B-100B-GatedDeltaNet-pure` configs — [config](https://huggingface.co/fla-hub/gla-1.3B-100B/blob/main/config.json)
- Reference implementations for extraction:
  - https://github.com/AmeenAli/HiddenMambaAttn (Mamba-1 / ViM)
  - https://github.com/Itamarzimm/UnifiedImplicitAttnRepr (Mamba, Mamba-2, RWKV, Griffin, RetNet, HGRN)
  - https://github.com/assafbk/DeciMamba (L1-normalised rows plus a mean-distance metric on `state-spaces/mamba-*`)
  — [2403.01590](https://arxiv.org/abs/2403.01590), [2405.16504](https://arxiv.org/abs/2405.16504), [2406.14528](https://arxiv.org/abs/2406.14528)

### Inferences
These are my engineering estimates and should be checked in code.
- **Mamba-2, one head at 16K.** One 16384×16384 fp32 map is about 1.07 GB. `mamba2-2.7b` has d_inner 5120 / headdim 64 = 80 heads per layer (assuming the default `headdim=64` / `expand=2`; unverified for this checkpoint). A whole layer would be about 86 GB, so iterate head by head and keep only statistics: per-row entropy, top-k mass, sink mass, and, for effective rank, an SVD of a subsampled or blocked matrix.
- **Mamba-1.** There are D = 2·d_model channels, e.g. 5120 for 2.8B, each with its own map, so full materialisation is impossible. Compute only the query rows needed (last token, answer token, needle position): each row costs O(L·N) per channel through cumulative sums of Δ.
- **Transformers.** `output_attentions=True` requires eager attention. At 16K, 32 heads × 16K² × 2 bytes ≈ 17 GB per layer. Run the forward pass with SDPA, cache Q and K, then recompute attention one head at a time in blocks.
- **fla Triton chunk kernels** do not return per-token states. Use the naive / recurrent reference paths or recompute from hooked q, k, v, gates and β. GLA maps follow directly; DeltaNet maps need the WY / UT inverse. The existence of naive reference functions in fla (e.g. `fla/ops/*/naive.py`) is unverified.
- **Numerics.** Compute cumulative log-decays in fp32. exp(segment_sum) underflows for long spans, which is benign (true zeros) but distorts log-based entropy, so use the convention 0·log 0 := 0.

### Gaps
- I did not verify whether HF transformers' Mamba-2 implementation automatically falls back to the torch path when `mamba_ssm` kernels are absent, or whether this is configurable. The grep found no `is_fast_path_available` string in the current file, which suggests a refactor.
- I did not verify whether fla exposes a public API to return attention-equivalent matrices.
- I did not verify HF / transformers support for loading `nvidia/*-8b-*` Megatron checkpoints.

## Q5. Published length-controlled retrieval results for these suites (sanity-check targets)

### Takeaway
Published numbers to reproduce, by setting:
- **Mamba-130M passkey (2K-trained, DeciMamba):** perfect to 8K, 0.8 at 16K, 0 at 32K and beyond. Pythia-160M succeeds only at 1K–2K.
- **Mamba-2 (Stuffed Mamba):** near-perfect passkey within 8K, poor or zero beyond 16K, except 130M.
- **Phonebook (Repeat After Me, Waleffe et al.):** Pythia-410M beats Mamba-2.8B from 70 entries. 8B Mamba-2 fails beyond about 1K tokens; the 4K hybrid is perfect to about 5.5K.
- **m-a-p RULER averages:** pure linear 0.256, 3:1 hybrids 0.397, Transformer 0.423.
- **NVIDIA 8B RULER:** Transformer-4K 77.62 versus Hybrid-4K 81.99 versus Mamba-2 52.14.

### Cited Findings
- DeciMamba, passkey success for 2K-trained models:

  | Length | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
  |---|---|---|---|---|---|---|---|---|
  | Mamba-130M | ✓ | ✓ | ✓ | ✓ | 0.8 | ✗ | ✗ | ✗ |
  | DeciMamba-130M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

  Pythia-160M succeeds only at 1K–2K — [HTML](https://arxiv.org/html/2406.14528v3)
- DeciMamba, multi-document retrieval accuracy (Mamba, trained with about 11 docs ≈ 2K tokens; 10→160 docs ≈ 2K→32K tokens): 68.3, 74.0, 64.7, 45.3, 24.7, 5.7, 1.0, 0.3, 0.3 — [HTML](https://arxiv.org/html/2406.14528v3)
- DeciMamba, PG-19 perplexity: Mamba-2.8B is 9.17 at 2K, 11.6 at 4K, and inf from 8K. Mamba-1.4B is 10.31 at 2K, 14.43 at 8K, 18.4 at 10K, and inf from 20K — [HTML](https://arxiv.org/html/2406.14528v3)
- Stuffed Mamba: `state-spaces/mamba2-*` passkey is near-perfect within 8K and "poor or even zero" beyond 16K, except 130M. The recall threshold is the longest context with more than 95% accuracy — [HTML](https://arxiv.org/html/2410.07145v4)
- Repeat After Me: phonebook lookup with Pythia-410M beats Mamba-2.8B once the book has at least 70 entries. The largest Mamba gets 0% copying 300-token shuffled strings. SQuAD F1 for Mamba-2.8B falls faster with paragraph length than Pythia-2.8B — [HTML](https://arxiv.org/html/2402.01032v2)
- Waleffe et al. (8B, 3.5T tokens):
  - Phonebook: Transformer about 100% to 4K; Mamba-2 fails beyond about 1K; Hybrid-4K 100% to about 5.5K.
  - RULER average (13 tasks):

  | Context | Transformer | Mamba-2-Hybrid | Mamba-2 |
  |---|---|---|---|
  | 4K | 77.62 | 81.99 | 52.14 |
  | 16K | 62.33 | 74.19 | — |
  | 32K | 56.49 | 69.93 | — |

  — [HTML](https://arxiv.org/html/2406.07887v1)
- m-a-p (340M and 1.3B, FineWeb-Edu): RULER averages are pure 0.256, 24:1 0.338, 12:1 0.340, 6:1 0.390, 3:1 0.397, Transformer 0.423. The best hybrid is GatedDeltaNet 3:1 at 0.440 — [HTML](https://arxiv.org/html/2507.06457v2)
- Zoology: MQAR is defined there. A 70M attention model beats a 1.4B gated-convolution model on associative recall (Pile-trained suite) — [arXiv](https://arxiv.org/abs/2312.04927)
- Based: Hazy releases the recall benchmarks `hazyresearch/based-fda`, `based-swde` and `based-squad`, linked from the 1B-50B model cards. Based beats Mamba by 6.22 points on recall-intensive tasks — [HF card](https://huggingface.co/hazyresearch/based-1b-50b), [arXiv](https://arxiv.org/abs/2402.18668)
- Jet-Nemotron, FDA / SWDE / SQuAD retrieval average: Jet-Nemotron-2B 74.2 versus 76.1 for Qwen3-1.7B-Base. Table 1 retrieval scores by block: Gated DeltaNet 69.3, Mamba2 68.9, GLA 66.6, RetNet 63.7, DeltaNet 67.9 — [HTML](https://arxiv.org/html/2508.15884v3)

### Inferences
- A cheap first validation: on `state-spaces/mamba-130m`, reproduce DeciMamba's passkey curve (0.8 at 16K, 0 at 32K) and its densification at 2K versus sparsification at 16K. Then test whether the thesis's normalised-entropy / top-k statistics on the L1-normalised implicit maps track the passkey breakdown between 8K and 16K.
- For the m-a-p suite, RULER improves 0.256 → 0.397 from pure to 3:1. Per-layer sharpness should show the attention layers carrying the low-entropy, high-top-k mass, and the effect should grow with the number of attention layers.

### Gaps
- I found no published S-NIAH / passkey sweep at 1K–16K for the fla-hub 1.3B-100B models or the m-a-p models specifically. The GLA, DeltaNet and Gated DeltaNet papers (not re-read) report S-NIAH for their own runs, which may not be the released checkpoints.
- I found no published MQAR results on the released pretrained Hazy checkpoints; Zoology's MQAR uses synthetic-trained models.
- Exact per-length phonebook numbers for Repeat After Me are available only in figures, and were not extracted.
