# Sparse Attention Lineage (2019–2026): selection mechanisms, what is preserved/lost vs dense softmax, real-kernel cost, and structure-guided support

Verification legend used throughout:
- **[V]** = verified this session against the primary source (arXiv abstract/HTML, ACL Anthology, official blog).
- **[K]** = from the researcher's prior knowledge of the primary paper; URL is the primary source but the specific number/wording was NOT re-fetched this session. Treat as "verify before quoting".
- **[S]** = secondary/aggregator source only; treat as unverified pointer.

---

## Q1. The fixed-pattern line (Sparse Transformer, Longformer, BigBird, ETC, Mistral SWA, StreamingLLM sinks)

### Takeaway
Fixed-pattern sparse attention (local window + strided/dilated + a few global tokens + optionally random links) was the 2019–2020 answer to O(n²); its surviving descendants in production LLMs are (i) sliding-window attention interleaved with full-attention layers (Mistral, Gemma, gpt-oss) and (ii) "sink + window" KV retention (StreamingLLM). These keep exact softmax over a content-independent support, are trivially block-sparse on GPUs, but cannot retrieve arbitrary out-of-window tokens in a single layer.

### Cited Findings

**Sparse Transformer** — "Generating Long Sequences with Sparse Transformers", Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever; arXiv preprint 2019 (OpenAI).
- Selection: fixed factorized patterns, "strided" (local + every-l-th) and "fixed" (block + summary columns); token-level patterns implemented as block-sparse; complexity O(n√n); custom GPU block-sparse kernels (OpenAI blocksparse); natively trained. [K] — https://arxiv.org/abs/1904.10509

**Longformer** — "Longformer: The Long-Document Transformer", Iz Beltagy, Matthew E. Peters, Arman Cohan; arXiv 2020 (AllenAI).
- Selection: sliding window (local) + dilated window (in upper layers, char-LM) + task-specific global tokens (e.g., [CLS], question tokens) that attend everywhere and are attended by all; linear in n; custom CUDA kernel via TVM for banded matmul; natively trained / continued-pretrained from RoBERTa to 4,096 tokens. [K] — https://arxiv.org/abs/2004.05150

**BigBird** — "Big Bird: Transformers for Longer Sequences", Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed; NeurIPS 2020.
- Selection: block-level window + global + random blocks (block size 64 in the paper's main configs; roughly 3 window blocks, 2 global blocks, 3 random blocks — exact per-task values in paper appendix) [K]; ITC (global tokens are existing tokens) vs ETC (extra global tokens) variants; extends to 4,096 tokens (8× BERT). [K] — https://arxiv.org/abs/2007.14062
- Theory: sparse attention with O(1) global tokens is a universal approximator of sequence-to-sequence functions and is Turing complete (preserving full-attention's properties); also a negative result: there is a task solvable by full attention in O(1) layers that requires Ω̃(n) layers for any sparse attention with Õ(n) edges. [K] — https://arxiv.org/abs/2007.14062

**ETC** — "ETC: Encoding Long and Structured Inputs in Transformers", Joshua Ainslie, Santiago Ontañón, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, Li Yang; EMNLP 2020.
- Selection: global–local attention (a small set of extra global tokens + local window over the long input), relative position encodings, and CPC pretraining; global tokens can be tied to document structure (sentences/paragraphs) — an early instance of structure-defined global support. [K] — https://arxiv.org/abs/2004.08483

**Mistral 7B sliding-window attention** — "Mistral 7B", Albert Q. Jiang et al.; arXiv 2023.
- Selection: per-layer causal window W=4096; with 32 layers the theoretical receptive field is ~131K tokens (information propagates W per layer); rolling-buffer KV cache of fixed size W (paper: 8× cache reduction at 32K); paper reports ~2× speed at 16K with W=4096 using FlashAttention + xFormers. Natively trained. [K] — https://arxiv.org/abs/2310.06825
- FlexAttention blog uses Mistral-style SWA (window 1024) as a canonical BlockMask example and reports it is "significantly faster than FA2 with a causal mask as this mask has significantly more sparsity". [V] — https://pytorch.org/blog/flexattention/

**StreamingLLM (attention sinks)** — "Efficient Streaming Language Models with Attention Sinks", Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis; ICLR 2024.
- Selection: keep the first few tokens (4 "sink" tokens) + a recent window; inference-time (post hoc) KV policy, also proposes pre-training with a dedicated learnable sink token; reports stable LM up to 4M tokens and up to 22.2× speedup over sliding-window-with-recomputation. Failure mode: evicted middle tokens are unrecoverable, so it does not extend effective context for retrieval. [K] — https://arxiv.org/abs/2309.17453

**Hybrid local/global in modern open models** (production fixed-pattern sparsity):
- Gemma 3: 5 local (sliding window 1024) : 1 global layer ratio to shrink KV cache. [K] — https://arxiv.org/abs/2503.19786
- gpt-oss: alternating dense and locally banded sparse attention layers (bandwidth 128). [K] — https://arxiv.org/abs/2508.10925

### Inferences
- The fixed-pattern line converged on two primitives that are easy to make hardware-friendly: a band (window) and a handful of "always visible" columns (global/sink tokens). For a thesis "sparse" level, a window+sink baseline is the natural *structure-free, content-free* control below the learned top-k arm.
- BigBird's random-block component has largely disappeared from production LLMs; production uses window + periodic full layers instead, which supplies global connectivity via depth rather than via random edges.

### Gaps
- Exact BigBird per-task block counts and Longformer per-layer window schedules were not re-fetched; check paper appendices before quoting.
- No primary source found this session quantifying Mistral SWA vs full attention quality on retrieval benchmarks at matched size.

---

## Q2. Content-based / learned selection line (Reformer → Routing → … → NSA, MoBA, DSA, SeerAttention) and inference-time sparse methods

### Takeaway
Content-based selection moved from hashing/clustering at token level (Reformer, Routing Transformer, 2020) to **block-level top-k over compressed/pooled keys with exact softmax over the selected support** (Landmark 2023 → MoBA, NSA 2025), and finally to **token-level top-k via a cheap learned indexer** (DeepSeek DSA, 2025). The dominant modern recipe is: score blocks (or tokens) with a cheap proxy → select top-k → run exact FlashAttention-style softmax over selected KV, plus a sliding window and/or always-on initial/local blocks. Training-time (native) methods (NSA, MoBA, DSA, SeerAttention-gate) now match or beat dense baselines on reported benchmarks; inference-time methods (H2O, SnapKV, Quest, MInference, DuoAttention, RetrievalAttention) trade small accuracy losses for KV/latency savings.

### Cited Findings

#### 2020–2023 foundations (brief)
- **Reformer** — "Reformer: The Efficient Transformer", Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya; ICLR 2020. Selection: LSH (random-rotation hashing of shared QK) into buckets, sort and attend within chunks; O(L log L); plus reversible residuals and chunked FFN; natively trained; token-level. [K] — https://arxiv.org/abs/2001.04451
- **Routing Transformer** — "Efficient Content-Based Sparse Attention with Routing Transformers", Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier; TACL 2021. Selection: online k-means clustering of queries/keys, attend within cluster; O(n^1.5 d); abstract reports WikiText-103 15.8 vs 18.3 ppl for comparable sparse models and PG-19 33.2 ppl. [K] — https://arxiv.org/abs/2003.05997
- **Sinkhorn attention** — "Sparse Sinkhorn Attention", Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, Da-Cheng Juan; ICML 2020. Selection: learned differentiable block sorting (Sinkhorn-normalized permutation) then local block attention; block-level, trainable. [K] — https://arxiv.org/abs/2002.11296
- **Combiner** — "Combiner: Full Attention Transformer with Sparse Computation Cost", Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, Bo Dai; NeurIPS 2021. Selection: factorized "full" attention where distant regions are summarized (structured factorization; each token reaches all others directly or via a summary); O(L log L) or O(L√L). [K] — https://arxiv.org/abs/2107.05768
- **H-Transformer-1D** — "H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences", Zhenhai Zhu, Radu Soricut; ACL 2021. Hierarchical (H-matrix-inspired) multi-resolution attention, linear complexity. [K] — https://arxiv.org/abs/2107.11906
- **Memorizing Transformers** — Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy; ICLR 2022. kNN lookup into a non-differentiable external memory of past (k,v) pairs, top-k exact retrieval. [K] — https://arxiv.org/abs/2203.08913
- **Landmark Attention** — "Random-Access Infinite Context Length for Transformers" (Landmark Attention), Amirkeivan Mohtashami, Martin Jaggi; NeurIPS 2023. Selection: input split into blocks, each with a trained landmark token; attention to the landmark gates retrieval of whole blocks; top-k block retrieval; trainable; extended LLaMA-7B to 32K context in fine-tuning. A direct ancestor of MoBA/NSA block top-k. [K] — https://arxiv.org/abs/2305.16300
- **Unlimiformer** — "Unlimiformer: Long-Range Transformers with Unlimited Length Input", Amanda Bertsch, Uri Alon, Graham Neubig, Matthew R. Gormley; NeurIPS 2023. Selection: offload cross-attention over encoder states to a kNN index; each head retrieves top-k keys; can be applied to pretrained encoder–decoders without training (also with training). [K] — https://arxiv.org/abs/2305.01625

#### Native Sparse Attention (NSA), DeepSeek, 2025
- Title/authors: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention", Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng; arXiv 2502.11089 (Feb 2025). [V] — https://arxiv.org/abs/2502.11089
- Venue: ACL 2025 (long), Anthology ID 2025.acl-long.1126, pp. 23078–23097; Anthology metadata lists a **Best Paper** award. [V via search of Anthology entry] — https://aclanthology.org/2025.acl-long.1126 ; note a Hacker News thread claimed the ACL awards page initially did not list it (unresolved) [S] — https://hnews-reader.vercel.app/jobs/44761548
- Mechanism: three parallel branches combined by learned gates — (1) compressed coarse tokens: compression block length l=32, stride d=16; (2) selected fine blocks: block size l'=64, n=16 selected blocks including 1 initial block and 2 local blocks, chosen by top-n over block importance derived from the compression branch's attention scores; (3) sliding window w=512. Exact softmax within each branch. [V] — https://arxiv.org/html/2502.11089v2
- GQA alignment: all query heads in a GQA group must select the same KV blocks (importance scores summed over heads in the group), so the kernel loads all queries of a group together and streams the shared sparse KV blocks sequentially into SRAM; outer loop over grid in Triton. This avoids FlashAttention-style query-block tiling, which would cause scattered loads under sparse selection. [V] — https://arxiv.org/html/2502.11089v2
- Training: natively trained from scratch; 27B-total / 3B-active MoE (72 routed + 2 shared experts, top-6), 30 layers, hidden 2560, 64 heads, 4 GQA groups; 270B tokens at 8K then 32K continued training with YaRN (introduction says 260B — internal inconsistency). [V] — https://arxiv.org/html/2502.11089v2
- Speed (Triton, 8×A100): up to 9.0× forward and 6.0× backward vs Triton FlashAttention-2 at 64K; decoding memory access per step NSA 2048/2560/3584/5632 tokens vs full 8192/16384/32768/65536 at 8K/16K/32K/64K → *expected* (memory-volume-derived, not measured) decode speedups 4×/6.4×/9.1×/11.6×. [V] — https://arxiv.org/html/2502.11089v2
- Quality: general benchmark average 0.456 (NSA) vs 0.443 (full), wins 7/9 (MMLU, MBPP slightly lower); LongBench avg 0.469 NSA vs 0.437 full, 0.423 Exact-Top, 0.392 Quest, 0.383 InfLLM, 0.303 H2O (baselines at 2,560-token budget incl. 128 leading + 512 local); perfect 64K needle-in-a-haystack; AIME-24 after R1-distill SFT: 0.121 vs 0.046 (8K gen limit), 0.146 vs 0.092 (16K). [V] — https://arxiv.org/html/2502.11089v2
- Limitations noted: may not exploit efficiency at short sequences; decode speedups projected; no linear-attention or SSM baseline; single-setting needle/AIME evaluations; argues prior trainable-selection approaches (ClusterKV, MagicPIG, HashAttention, Quest-style heuristic selection with aux loss) have training-efficiency or non-differentiability problems. [V] — https://arxiv.org/html/2502.11089v2

#### MoBA (Moonshot AI, 2025)
- "MoBA: Mixture of Block Attention for Long-Context LLMs", Enzhe Lu, Zhejun Jiang, Jingyuan Liu, + 22 co-authors (Jiezhong Qiu submitter); arXiv 2502.13189 (Feb 2025); no peer-reviewed venue listed on arXiv page; code github.com/MoonshotAI/MoBA; "deployed to support Kimi's long-context requests"; "less structure" principle. [V] — https://arxiv.org/abs/2502.13189
- Mechanism: blocks of the KV sequence; gate score = ⟨q, mean-pooled keys of block⟩; each query takes top-k blocks; never routes to future blocks; current block always included with causal mask; implemented with FlashAttention varlen + MoE-style routing; current-block and selected-history attention merged with online softmax. Exact softmax over selected support. [V] — https://arxiv.org/html/2502.13189v1
- Settings: scaling-law runs 568M–2.1B, 8K seq, block 512, top-3; Llama-3.1-8B continual pretraining 128K→1M, 100B tokens, block 4096, top-12, last 3 layers kept full attention. Granularity ablation at 75% sparsity (1.5B, 32K): coarsest 2-of-8 blocks ~1e-2 worse loss than finer. [V] — https://arxiv.org/html/2502.13189v1
- Results: LM loss within ~1e-3 of full at 8K; trailing-token loss gap at 32K slightly larger but shrinking with scale; RULER@128K 0.7818 vs 0.7849 (≈62.5% sparsity); LongBench@32K 0.4828 vs 0.4821; NIAH "satisfactory" to 1M; MoBA used for prefill only, full attention in decode. Speed: up to 6.5× attention speedup at 1M prefill; 16× attention-time reduction at 10M with 95.31% sparsity vs FlashAttention. GPU type not specified in the extracted text. [V] — https://arxiv.org/html/2502.13189v1
- Failure modes: degraded SFT due to sparse gradients from prompt-loss masking — fixed with layer-wise hybrid (last layers full); hybrid 90% MoBA → 10% full training recovers full-attention loss; limited evidence on complex reasoning. [V] — https://arxiv.org/html/2502.13189v1

#### FlashMoBA / "Optimizing Mixture of Block Attention" (MIT/NVIDIA, Nov 2025)
- Guangxuan Xiao, Junxian Guo, Kasra Mazaheri, Song Han; arXiv 2511.11571. Signal-to-noise model of the router: retrieval depends on distinguishing relevant vs noise blocks from q·(mean key); design principle is the head-dim-to-block-size ratio d/B (smaller blocks better); short convolution on keys clusters relevant signals; FlashMoBA CUDA kernel up to 14.7× over FlashAttention-2 for small blocks; improved MoBA matches dense baselines when trained from scratch. Reported SNR ∝ √(d/2B) formula is from a secondary summary [S]. [V for the rest] — https://arxiv.org/abs/2511.11571 ; code https://github.com/mit-han-lab/flash-moba

#### DeepSeek Sparse Attention (DSA), DeepSeek-V3.2-Exp / V3.2 (Sep–Dec 2025)
- "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models" (DeepSeek-AI), arXiv 2512.02556; V3.2 keeps V3.2-Exp's architecture, differing from V3.1-Terminus only by DSA added via continued training. [V] — https://arxiv.org/abs/2512.02556 ; HTML https://arxiv.org/html/2512.02556v1
- Mechanism: lightning indexer I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s) (ReLU "for throughput"), few heads, FP8-capable; each query attends to keys in Top-k(I_{t,:}) with **token-level** k = 2048; instantiated on the MQA mode of MLA so each latent KV entry is shared by all query heads (kernel efficiency). [V] — https://arxiv.org/html/2512.02556v1 ; index_topk=2048 also in vLLM blog [V-secondary] — https://vllm.ai/blog/2025-09-29-deepseek-v3-2
- Training (post hoc conversion of a dense model, but trained): dense warm-up with all but indexer frozen, KL(p_main ‖ softmax(I)) where p_main = head-summed, L1-normalized main attention; lr 1e-3, 1,000 steps × 16 × 128K = 2.1B tokens; then sparse stage, all params trained, KL restricted to selected set; lr 7.3e-6, 15,000 steps × 480 × 128K = 943.7B tokens; indexer input detached (indexer gets only KL loss; main model only LM loss). [V] — https://arxiv.org/html/2512.02556v1
- Cost: core attention O(L²) → O(Lk); indexer remains O(L²) but cheap; Figure 3 shows per-token prefill/decode cost vs position on H800 at $2/GPU-hour (no numbers extracted). Quality: no substantial degradation vs V3.1-Terminus; ChatbotArena Elo closely matched; AA-LCR +4 points in reasoning mode. [V] — https://arxiv.org/html/2512.02556v1
- Number of indexer heads not stated in extracted text (gap).

#### SeerAttention (Microsoft, 2024–2025)
- "SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs", Yizhao Gao et al.; arXiv 2410.13276. Learnable AttnGate on pooled Q and K selects important blocks of the attention map; gate trained by self-distillation (from block-pooled dense attention map) while the base LLM is frozen → "post hoc but trained" class; 90% sparsity at 32K with YaRN long-context fine-tuning with minimal perplexity loss, 5.67× speedup over FlashAttention-2 (block-sparse FlashAttention kernel). [V via abstract snippet] — https://arxiv.org/abs/2410.13276 ; code https://github.com/microsoft/SeerAttention
- A review site reports 7.3× at 90% sparsity at 128K for Llama-3.1-8B-Instruct [S] — https://www.themoonlight.io/en/review/seerattention-learning-intrinsic-sparse-attention-in-your-llms

#### Inference-time (post hoc, training-free) sparse methods
- **H2O** — "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", Zhenyu Zhang et al.; NeurIPS 2023. Keeps recent + "heavy-hitter" tokens by accumulated attention; KV eviction (irreversible); with 20% heavy hitters reports throughput up to 29× over DeepSpeed Zero-Inference / HF Accelerate and 3× over FlexGen. [K] — https://arxiv.org/abs/2306.14048
- **SnapKV** — Yuhong Li et al.; NeurIPS 2024. Observation window at the end of the prompt votes (with pooling) on which prefix positions to keep per head; one-shot compression after prefill; 3.6× generation speed and 8.2× memory efficiency at 16K; up to 380K tokens on one A100-80GB. [K] — https://arxiv.org/abs/2404.14469
- **Quest** — "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", Jiaming Tang et al.; ICML 2024. Page-level (KV page) selection using per-page elementwise min/max key upper bound on q·k; top-k pages per query; KV kept (not evicted); up to 7.03× self-attention speedup, 2.23× end-to-end latency reduction. [K] — https://arxiv.org/abs/2406.10774
- **MInference 1.0** — Huiqiang Jiang et al.; NeurIPS 2024. Per-head offline pattern assignment (A-shape, vertical-slash, block-sparse) with online index estimation for prefill; up to 10× prefill speedup at 1M on a single A100. [K] — https://arxiv.org/abs/2407.02490
- **DuoAttention** — Guangxuan Xiao et al.; ICLR 2025. Identifies "retrieval heads" (full KV) vs "streaming heads" (sink + window) via a lightweight optimization; memory up to 2.55× (MHA)/1.67× (GQA) lower, decode up to 2.18×/1.50× faster; 3.3M context on one A100 with quantization. [K] — https://arxiv.org/abs/2410.10819
- **RetrievalAttention** — Di Liu et al.; arXiv 2024. ANN vector index over KV on CPU, attention-aware to handle Q/K distribution mismatch; accesses only 1–3% of keys; serves 128K context for an 8B model at 0.188 s/token on a single **RTX 4090 (24 GB)**. [K] — https://arxiv.org/abs/2409.10516
- **InfLLM** — Chaojun Xiao et al.; NeurIPS 2024. Training-free block-level memory units with representative tokens, top-k block retrieval + sink + local window. [K] — https://arxiv.org/abs/2402.04617
- **Block Sparse Flash Attention (BSFA)** — arXiv 2512.07011 (Dec 2025): training-free drop-in for FlashAttention, exact block scoring with per-layer/per-head calibrated thresholds; ~50% compute/memory-transfer saving; up to 1.10× on reasoning and 1.24× on NIAH for Llama-3.1-8B at >99% of baseline accuracy; GPU not identified in search snippet. [V via search abstract] — https://arxiv.org/abs/2512.07011
- **BLASST** — arXiv 2512.12087 (Dec 2025): dynamic block pruning inside FlashAttention's online softmax with threshold λ=a/L; 1.62× prefill at 74.7% sparsity, 1.48× decode at 73.2% sparsity on H200/B200. [V via search abstract] — https://arxiv.org/abs/2512.12087

#### Catalogue table (compact)

| Method | Year/venue | Selection | Granularity / budget | Native vs post hoc | Exact softmax over support? | Kernel | Headline cost | URL |
|---|---|---|---|---|---|---|---|---|
| Sparse Transformer | 2019 arXiv | fixed strided/fixed | token pattern, block-sparse impl., O(n√n) | native | yes | custom CUDA blocksparse | — | https://arxiv.org/abs/1904.10509 |
| Longformer | 2020 arXiv | window+dilated+global | token, w≈512 | native/continued | yes | TVM CUDA | linear | https://arxiv.org/abs/2004.05150 |
| BigBird | NeurIPS 2020 | window+global+random | block 64 | native | yes | blocked dense gathers | linear | https://arxiv.org/abs/2007.14062 |
| ETC | EMNLP 2020 | global-local (extra global tokens) | token | native | yes | — | linear | https://arxiv.org/abs/2004.08483 |
| Reformer | ICLR 2020 | LSH buckets | token chunks | native | yes (within bucket) | — | O(n log n) | https://arxiv.org/abs/2001.04451 |
| Routing Transformer | TACL 2021 | online k-means | token clusters | native | yes | — | O(n^1.5) | https://arxiv.org/abs/2003.05997 |
| Sinkhorn | ICML 2020 | learned block sort | block | native | yes | — | — | https://arxiv.org/abs/2002.11296 |
| Mistral SWA | 2023 | window W=4096 | token band | native | yes | FA + xFormers | 2× at 16K | https://arxiv.org/abs/2310.06825 |
| StreamingLLM | ICLR 2024 | 4 sinks + window | token | post hoc (or sink pretrain) | yes | — | 22.2× vs recompute | https://arxiv.org/abs/2309.17453 |
| Landmark | NeurIPS 2023 | landmark-gated top-k blocks | block | trained (fine-tune) | yes | — | — | https://arxiv.org/abs/2305.16300 |
| H2O | NeurIPS 2023 | heavy hitters (accum. attn) + recent | token, eviction | post hoc | yes | — | ≤29× thrpt | https://arxiv.org/abs/2306.14048 |
| Quest | ICML 2024 | min/max-bound page top-k | page | post hoc | yes | CUDA | 7.03× attn | https://arxiv.org/abs/2406.10774 |
| MInference | NeurIPS 2024 | per-head A-shape/vertical-slash/block | mixed | post hoc | yes | Triton/CUDA | 10× prefill @1M A100 | https://arxiv.org/abs/2407.02490 |
| DuoAttention | ICLR 2025 | retrieval vs streaming heads | head-level | post hoc (light opt.) | yes | — | 2.55×/1.67× mem | https://arxiv.org/abs/2410.10819 |
| RetrievalAttention | 2024 arXiv | ANN over KV | token | post hoc | yes | CPU+GPU | 0.188 s/tok RTX 4090 | https://arxiv.org/abs/2409.10516 |
| SeerAttention | 2024/25 arXiv | learned AttnGate block top-k/threshold | block | gate trained, base frozen | yes | block-sparse FA | 5.67× vs FA2 @32K, 90% | https://arxiv.org/abs/2410.13276 |
| NSA | ACL 2025 (Best Paper per Anthology) | compress (l=32,d=16) + top-16 blocks (64) + window 512 | block 64, GQA-shared | native | yes per branch, gated sum | Triton | 9.0×/6.0× fwd/bwd @64K A100 | https://arxiv.org/abs/2502.11089 |
| MoBA | 2025 arXiv | mean-pooled key gate, top-k blocks | block 512 (top-3) / 4096 (top-12) | native or continued | yes | FA varlen | 6.5× @1M; 16× @10M | https://arxiv.org/abs/2502.13189 |
| FlashMoBA | 2025 arXiv | MoBA + key conv, small blocks | small B | native | yes | CUDA | ≤14.7× vs FA2 | https://arxiv.org/abs/2511.11571 |
| DSA (V3.2) | 2025 tech rpt | lightning indexer top-k | token, k=2048 | continued training of dense model | yes | custom (FP8 indexer) | O(L²)→O(Lk) core | https://arxiv.org/abs/2512.02556 |
| BSFA | 2025 arXiv | calibrated threshold on block max | block | post hoc | yes | CUDA | ≤1.24× | https://arxiv.org/abs/2512.07011 |
| BLASST | 2025 arXiv | in-softmax block skip, λ=a/L | block | post hoc | approx. (skips small blocks) | CUDA | 1.62× prefill H200/B200 | https://arxiv.org/abs/2512.12087 |

### Inferences
- The "learned top-k block selection" control arm requested by the thesis is best instantiated as **MoBA-style** (block mean-pooled key gate, top-k, current block forced, causal) or as **NSA's selection branch alone** (block 64, top-n incl. initial + local blocks). Both keep exact softmax over the support, are structure-free (no parser), and have public kernels (MoBA: FA varlen; NSA: Triton re-implementations; FlashMoBA: CUDA). MoBA is the cleaner single-mechanism control; NSA confounds selection with compression and window branches.
- The field split into "selection score from compressed attention" (NSA, MoBA, Quest, Landmark) vs "selection score from a separate learned scorer" (DSA lightning indexer, SeerAttention AttnGate). The second family decouples scorer training (KL distillation to dense attention) from the main model, which makes it closer to "post hoc but trained".
- Token-level selection (DSA, k=2048) became viable only with MQA-style shared KV (MLA) so that one gather serves all heads; block-level (NSA/MoBA) is the choice for GQA/MHA models.

### Gaps
- MoBA GPU type and kernel-level memory numbers not found in extracted text.
- DSA indexer head count and Figure 3 numeric costs not extractable.
- InfLLM-v2 (MiniCPM4, trainable dense–sparse switchable attention, arXiv 2506.07900) and SeerAttention-R (arXiv 2506.08889) are known 2025 follow-ups but were not fetched this session — unverified details: https://arxiv.org/abs/2506.07900 , https://arxiv.org/abs/2506.08889
- "Flash Sparse Attention" (alternative NSA kernel addressing small GQA group counts, arXiv 2508.18224) recalled but not verified: https://arxiv.org/abs/2508.18224
- Semantic Scholar cited-by sweep of NSA/MoBA not performed (tool-call budget); 2026 follow-ups beyond NAtS-L and Sparse Frontier camera-ready are under-covered.

---

## Q3. What sparse attention preserves or loses vs dense softmax; sharpness; direct sparse-vs-linear-vs-SSM comparisons

### Takeaway
Sparse attention keeps the softmax *on its support* exact, so it preserves argmax-like, sharp lookups **whenever the right key is selected** — its failure mode is *selection error* (a recall miss is total), not the blurring/forgetting of a fixed-size state that limits linear attention/SSMs. The few matched comparisons available (He & Garner 2025, 0.4B/1.4B) show pure NSA roughly on par with or below a full Transformer on NIAH and recall, above pure Gated DeltaNet at 0.4B but below it on S-NIAH at 1.4B; hybrids (linear + sparse, linear + full) are best. Training-free sparsity is task-fragile: even moderate sparsity can significantly hurt at least one task (Sparse Frontier).

### Cited Findings
- **Matched sparse vs linear vs hybrid (0.4B/10B tok and 1.4B/30B tok, FineWeb-Edu)** — "Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction", Mutian He, Philip N. Garner; arXiv 2510.20787 (Oct 2025; no venue listed). [V] — https://arxiv.org/html/2510.20787
  - RULER S-NIAH avg (0.4B / 1.4B): GDN 51.7 / 88.9; GDN+SWA 54.7 / 62.4; laNSA (GDN+NSA hybrid) 74.8 / 77.2; GDN+full-Attn 80.4 / 86.8; **NSA 72.8 / 74.8**; Transformer++ 75.1 / 91.4.
  - Recall-intensive EVAPORATE avg (FDA, SWDE, NQ, SQuAD, TriviaQA, DROP) (0.4B / 1.4B): GDN 19.47 / 29.50; GDN+SWA 22.81 / 32.42; laNSA 25.58 / 33.80; GDN+Attn 27.19 / 37.85; **NSA 24.18 / 32.38**; Transformer++ 27.01 / 34.19.
  - Commonsense avg (0.4B / 1.4B): NSA 48.8 / 54.7 vs Transformer++ 49.3 / 54.0 vs GDN 48.7 / 56.2.
  - Authors' framing: linear attention compresses history into a fixed-size state ("a variable-length history can never be losslessly compressed into a fixed-size state"); query-aware sparse methods like NSA avoid this but require O(N) KV cache.
- **NAtS-L** — "Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models", arXiv 2602.03681 (2026): reports that NSA, MoBA and GDN+SWA(1024) sparse variants achieve similar retrieval performance to full attention while NAtS-L variants are stronger. [S — snippet only] — https://arxiv.org/pdf/2602.03681
- **Gated DeltaNet** paper: linear recurrent models still have a gap vs Transformers on retrieval; hybrids with attention outperform pure attention on retrieval. [S — snippet] — https://arxiv.org/abs/2412.06464
- **The Sparse Frontier** — "The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs", Piotr Nawrot et al.; arXiv 2504.17768; **Findings of ACL 2026**. Largest study of training-free sparse attention (six methods, up to 128K, sparsity up to 0.95, nine tasks): at very long sequences larger highly-sparse models beat smaller dense ones at equal compute (isoFLOP); decode tolerates higher sparsity than prefill; longer sequences tolerate more sparsity; **even moderate sparsity significantly degrades at least one task** — sparse attention is not a universal fix; fixed-budget production methods may be suboptimal. [V via search abstract] — https://arxiv.org/abs/2504.17768 ; Anthology preview https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1926/
- **NSA vs inference-time sparse (same model, LongBench)**: natively trained NSA 0.469 > full 0.437 > Exact-Top 0.423 > Quest 0.392 > InfLLM 0.383 > H2O 0.303 — eviction-based heavy-hitter selection is worst; exact top-k post hoc loses ~1.4 points vs dense. [V] — https://arxiv.org/html/2502.11089v2
- **MoBA granularity**: at fixed 75% sparsity, coarse blocks (2 of 8) ≈1e-2 worse LM loss than finer granularity; trailing-token (long-range) loss gap larger than average loss gap. [V] — https://arxiv.org/html/2502.13189v1
- **FlashMoBA SNR analysis**: block-router retrieval accuracy degrades as block size grows relative to head dimension, because mean-pooled keys dilute a single relevant key among B−1 noise keys — a precise statement of *why* block-level selection can miss sharp, single-token targets. [V for qualitative; formula S] — https://arxiv.org/abs/2511.11571
- **Fixed-window failure**: StreamingLLM's sink+window evicts middle tokens irrecoverably (no long-range retrieval) [K] — https://arxiv.org/abs/2309.17453 ; GDN+SWA in He & Garner falls to 62.4 S-NIAH at 1.4B vs 88.9 for plain GDN [V] — https://arxiv.org/html/2510.20787
- **SFT fragility of sparse**: MoBA underperforms in SFT due to sparse gradients under prompt-loss masking; mitigated by full-attention last layers. [V] — https://arxiv.org/html/2502.13189v1
- Zoology/Based-style recall benchmarks (MQAR) include sliding-window attention baselines; sliding windows only recall within the window, motivating Based's linear+window hybrid (Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff", ICML 2024). [K] — https://arxiv.org/abs/2402.18668 ; Zoology https://arxiv.org/abs/2312.04927 (see sibling notes in `効率的 attention の鋭さと構造 文献調査/linear_attention_lineage_llm.md`).

### Inferences
- On "sharpness": sparse attention is strictly better than linear attention at preserving argmax-like selection *conditional on correct support selection*, because the normalized weights on the support are the dense softmax renormalized (exact for NSA/MoBA/DSA/Quest). The quality risk moves upstream to the scorer (mean-pooled block keys, compressed tokens, indexer). Thus, a thesis "sparse" level should report both (i) recall@support (did the selected blocks contain the gold key?) and (ii) downstream accuracy — this separation is the methodological contribution the literature mostly lacks.
- The He & Garner numbers suggest that at small scale (≤1.4B), pure NSA is not a clear winner over a well-tuned delta-rule linear model on S-NIAH, but is consistently better than linear+SWA; hybrids dominate. This is the closest available "sparse vs linear at matched size/tokens" evidence; it is *matched in parameters and tokens, not in wall-clock*.
- Training-free sparsity numbers (Sparse Frontier) should not be used to predict natively trained sparse models; NSA/MoBA show native training closes or reverses the gap.

### Gaps
- No primary source found that compares sparse vs linear vs SSM **at matched wall-clock and memory** on the same kernel stack — this is an open gap the thesis can fill.
- NAtS-L numbers not extracted (only snippet).
- No direct entropy/sharpness measurement (e.g., attention entropy on selected support vs dense) found for NSA/MoBA.

---

## Q4. Kernels and cost: block size, GQA, FlexAttention BlockMask, sparsity thresholds, consumer GPUs

### Takeaway
Block sizes cluster at 64–128 (NSA 64; FlexAttention default 128; BigBird 64) because they match tensor-core tiles and SRAM tiles; larger blocks (MoBA 512–4096) are kernel-cheap but hurt routing precision, and FlashMoBA shows small blocks become fast only with a dedicated kernel. FlexAttention gives block-sparse masks at ~90%/85% of FlashAttention-2 throughput (fwd/bwd, A100) for dense masks and skips fully masked 128×128 blocks, so speedup ≈ fraction of skipped blocks minus overhead; data-dependent per-layer masks pay BlockMask construction cost. Speedups over FA2 are large only at long contexts (≥32K) and high sparsity (≥~75–90%); at short contexts, NSA itself notes limited gains. Consumer-GPU measurements are scarce.

### Cited Findings
- **NSA block/GQA design**: l'=64 selection blocks; GQA group-shared selection so queries of a group are loaded together and KV blocks streamed contiguously; Triton; 8×A100; up to 9.0×/6.0× fwd/bwd vs Triton FA2 at 64K; gains smaller at shorter lengths (only figure). [V] — https://arxiv.org/html/2502.11089v2
- **FlexAttention (blog, PyTorch team; updated 2025-05-30)**: `score_mod` (per-score modification) vs `mask_mod` → `create_block_mask` → `BlockMask`; default BLOCK_SIZE=128; fully masked blocks skipped, "full" blocks skip mask application, partial blocks apply mask; per-element masking costs ~15–20% performance if not block-skipped; causal via BlockMask ~2× faster than score_mod masking at same TFLOPs; sliding window faster than FA2 causal; BlockMask metadata is [B, H, Q_LEN/128, KV_LEN/128] (≈60 MB at 1M tokens; <1 MB with BLOCK_SIZE=1024); recomputing a BlockMask costs "hundreds of microseconds" vs seconds for recompilation; `create_block_mask(_compile=True)` reduces runtime/memory "often an order of magnitude"; data-dependent per-layer masks are "the hardest setting" since construction can't be amortized; performance 90% of FA2 forward, 85% backward on A100 (causal); limitations at time of writing: not FA3-level on H100, sequence length multiple of 128, GQA support pending. [V] — https://pytorch.org/blog/flexattention/
- **FlexAttention paper**: "FlexAttention: A Programming Model for Generating Optimized Attention Kernels" (Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, Horace He; MLSys 2025) — details not re-fetched. [K] — https://arxiv.org/abs/2412.05496 ; examples repo https://github.com/pytorch-labs/attention-gym
- **MoBA cost**: implemented as FlashAttention varlen over gathered blocks; 6.5× at 1M prefill; 16× at 10M with 95.31% sparsity. [V] — https://arxiv.org/html/2502.13189v1
- **FlashMoBA**: small-block MoBA is theoretically better but GPU-inefficient without a fused kernel; up to 14.7× over FA2 for small blocks. [V] — https://arxiv.org/abs/2511.11571
- **SeerAttention**: 5.67× over FA2 at 90% block sparsity, 32K. [V] — https://arxiv.org/abs/2410.13276
- **Sparsity → speed at modest sparsity**: BLASST 1.62× prefill at 74.7% sparsity, 1.48× decode at 73.2% (H200/B200); BSFA ≈50% compute skipped → only 1.10–1.24× end-to-end. This illustrates that ~50–75% block sparsity yields far less than the ideal 2–4× once scoring and irregular access are paid. [V via search abstracts] — https://arxiv.org/abs/2512.12087 ; https://arxiv.org/abs/2512.07011
- **DSA cost model**: core attention O(Lk) with k=2048 but indexer still O(L²) (cheap: few heads, ReLU, FP8). [V] — https://arxiv.org/html/2512.02556v1
- **Block-sparse FlashAttention (original)**: FlashAttention paper introduced block-sparse FlashAttention with IO complexity proportional to the fraction of nonzero blocks. [K] — https://arxiv.org/abs/2205.14135
- **Consumer GPU**: RetrievalAttention runs 128K context for an 8B model on a single RTX 4090 (24 GB) at 0.188 s/token (CPU-offloaded ANN over KV). [K] — https://arxiv.org/abs/2409.10516 . No RTX 3090/4090 block-sparse-vs-FA2 kernel benchmarks found in this session's searches. [gap]

### Inferences
- For a wall-clock/memory-matched thesis arm on consumer GPUs, the practical path is FlexAttention BlockMask (block 128, or 64 with custom construction) for structure-defined masks *and* for learned top-k block selection (build `num_blocks`/`indices` tensors directly from top-k indices to avoid `create_block_mask` cost). Because the learned mask changes per layer and per input, construction overhead must be measured and included in the wall-clock budget — FlexAttention explicitly flags this case.
- Speedup ≈ 1/(1−sparsity) is an upper bound; observed numbers (BLASST, BSFA) suggest ~0.5–0.65 efficiency at 50–75% sparsity on datacenter GPUs. At thesis-scale sequence lengths (≤8K–16K) and block 64–128, the learned top-k arm may not beat FA2 at all; matching against linear attention should therefore match both FLOPs and measured time, reporting both.
- Memory: training-time block-sparse attention does not reduce KV memory (all KV retained, as He & Garner note O(N) KV); it reduces activation traffic and compute. Linear attention reduces state to O(d²) per head. A "matched memory" comparison must state whether KV cache or peak activation memory is matched.

### Gaps
- No measured FlexAttention numbers on RTX 3090/4090 found; FA3 is Hopper-only, so consumer-GPU baselines should be FA2.
- No primary source giving the exact sparsity threshold at which FlexAttention block-sparse beats FA2 dense causal on a given GPU.

---

## Q5. Structure-guided sparsity: graph / tree / syntax / AST-defined sparse support

### Takeaway
Structure-defined sparse support exists mainly in (a) graph transformers (Exphormer: expander + local + virtual global; GraphGPS's BigBird/Performer options) and (b) a few code models (SASA: AST-adjacency blocks + learned/statistical top-k blocks + window + global; LongCoder: bridge + "memory" tokens chosen by code statements such as imports/definitions). Syntax-guided NLP models (SG-Net, dependency-mask transformers) use dense masking, not block-sparse kernels. **No paper found compares learned top-k selection against structure-defined selection at matched k as separately trained arms**; SASA's ablation (removing AST blocks vs removing top-k blocks, both k=3 block-level in the same model) is the closest.

### Cited Findings
- **SASA** — "Understanding Long Programming Languages with Structure-Aware Sparse Attention", Tingting Liu, Chengyu Wang, Cen Chen, Ming Gao, Aoying Zhou; SIGIR 2022 (short). [V] — https://arxiv.org/abs/2205.13730 ; venue record https://pure.ecnu.edu.cn/en/publications/understanding-long-programming-languages-with-structure-aware-spa/
  - Four components, BigBird-style block level, block size b=32: sliding window w=3 blocks; 2 global key blocks; **top-k block attention (k=3)** selected from a pre-computed attention-frequency table on CodeSearchNet (statistical, not per-input learned); **AST-based block attention** from a tree-sitter AST converted to adjacency matrix, blocked, top-k blocks by summed AST score. Paper does not specify which AST relations count, and does not say whether a fused block-sparse kernel is used. [V] — https://ar5iv.labs.arxiv.org/html/2205.13730
  - Backbone CodeBERT at length 1024; memory saving 23–33%; >7 GB saved at batch 8 on 32 GB V100 (CodeBERT OOM above batch 10); no latency numbers. [V] — https://ar5iv.labs.arxiv.org/html/2205.13730
  - Long-code CodeXGLUE (BCB F1 / Defect Acc / Search MRR / Summ BLEU-4): SASA 0.911 / 57.44 / 46.36 / 12.88; BigBird 0.895 / 56.79 / 41.89 / 12.42; Longformer 0.820 / 56.90 / 44.09 / 12.20; GraphCodeBERT 0.871 / 55.39 / 43.13 / 12.93; CodeBERT 0.789 / 55.71 / 38.37 / 12.26. [V] — https://ar5iv.labs.arxiv.org/html/2205.13730
  - Ablation: w/o top-k: BCB F1 0.911→0.828 (−0.083), Defect −0.6; w/o AST: BCB 0.911→0.874 (−0.037), Defect −0.22 → the data-driven top-k blocks matter more than AST blocks in that setup. [V] — https://ar5iv.labs.arxiv.org/html/2205.13730
- **LongCoder** — "LongCoder: A Long-Range Pre-trained Language Model for Code Completion", Daya Guo, Canwen Xu, Nan Duan, Jian Yin, Julian McAuley; ICML 2023. Sliding-window attention + "bridge tokens" (global tokens inserted at intervals) + "memory tokens" placed on statements with long-range importance (e.g., package imports, class/function definitions) — a syntax-informed choice of global support; introduces LCC (long code completion) benchmark. [K] — https://arxiv.org/abs/2306.14893
- **Exphormer** — "Exphormer: Sparse Transformers for Graphs", Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J. Sutherland, Ali Kemal Sinop; ICML 2023. Support = local graph edges + an expander graph (random d-regular) + virtual global nodes; cost linear in nodes/edges; shown to be a universal approximator under these components; integrated into GraphGPS. [K] — https://arxiv.org/abs/2303.06147
- **GraphGPS** — "Recipe for a General, Powerful, Scalable Graph Transformer", Ladislav Rampášek et al.; NeurIPS 2022. Modular MPNN + global attention with options including Performer (linear) and BigBird (sparse). [K] — https://arxiv.org/abs/2205.12454
- **Syntax-guided masks in NLP (dense-masked)**: SG-Net restricts each token's attention to its dependency-tree ancestors (parser run in advance) [V-snippet] — https://arxiv.org/abs/2012.13915 ; Syntax-Guided Transformers (multimodal grounding) restrict attention to tree-connected tokens [V-snippet] — https://arxiv.org/abs/2311.04364 ; Dependency Transformer Grammars use masks following dependency transition systems [V-snippet] — https://arxiv.org/abs/2407.17406 ; Tree Transformer induces constituents and restricts attention within them, later shown "an ineffective model of syntactic constituency" [V-snippet] — https://arxiv.org/html/2411.16993v1
- **GraphCodeBERT** uses a graph-guided masked attention over a data-flow graph (dense mask at 512 tokens). [K] — https://arxiv.org/abs/2009.08366 (see sibling notes `LLM GNN 融合 attention 先行研究/structure_aware_code_lms.md`).
- ETC can tie global tokens to document structure (sentences/paragraphs), an early structure-defined global support. [K] — https://arxiv.org/abs/2004.08483

### Inferences
- **The "learned selection vs structure-defined selection at the same k" comparison has not been run** as a controlled experiment in any work found: SASA mixes both in one model with k fixed (3 blocks each) and ablates; LongCoder compares against other sparse baselines, not against a learned top-k with the same number of global tokens; Exphormer compares expander vs other graph transformers, not learned routing. This is a clean, defensible novelty claim for the thesis (with the caveat that the Semantic Scholar sweep was not exhaustive).
- SASA's ablation hints that data-driven top-k is more valuable than AST adjacency at equal k for clone/defect tasks — the thesis should anticipate this result and design the comparison to test it fairly (per-input learned scoring rather than SASA's corpus-frequency table; matched block size; matched FLOPs).
- AST/dependency adjacency tends to be local in token order (parent/child spans overlap), so blocked AST masks may overlap heavily with a sliding window; the thesis should measure the *incremental* support beyond the window to avoid attributing window effects to structure.

### Gaps
- No block-sparse fused kernel using AST/dependency structure in a decoder LLM found.
- SASA's precise AST relation definition is not given in the paper text extracted.
- No 2024–2026 structure-guided sparse attention for code LLMs at scale was found in this session (searches returned only older syntax-mask work).

---

## Q6. Theory: expressivity of sparse attention, connectivity/expansion requirements

### Takeaway
Sparse attention with O(n) connections is a universal approximator of continuous sequence-to-sequence functions provided the pattern (stacked over layers) connects every pair of positions — i.e., global reachability within a bounded number of layers (Yun et al. 2020; BigBird; Exphormer for graphs via expanders + virtual nodes). The flip side: sparse patterns can require Ω̃(n) layers for tasks solved by full attention in O(1) layers (BigBird), so expressivity guarantees are about depth-amplified connectivity, not single-layer retrieval.

### Cited Findings
- **Yun et al. 2020** — "O(n) Connections are Expressive Enough: Universal Approximability of Sparse Transformers", Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, Sanjiv Kumar; NeurIPS 2020. Unifying framework covering Sparse Transformer / BigBird / Star-Transformer-style patterns; sufficient conditions: each token attends to itself, the patterns contain a Hamiltonian-path-like chain, and all tokens become reachable after p layers; then sparse Transformers with O(n) connections per layer are universal approximators. [K] — https://arxiv.org/abs/2006.04862
- **BigBird theory** — universal approximation and Turing completeness with global tokens; lower bound requiring Ω̃(n) layers for some tasks under Õ(n)-edge sparse attention. [K] — https://arxiv.org/abs/2007.14062
- **Exphormer** — expander graphs provide spectral approximation of complete-graph attention with linear edges; with virtual global nodes, universal approximation results for graph transformers. [K] — https://arxiv.org/abs/2303.06147
- **FlashMoBA SNR theory** — statistical model of when block-level routing retrieves the relevant block; links block size B and head dim d to retrieval accuracy. [V qualitative] — https://arxiv.org/abs/2511.11571

### Inferences
- For the thesis's "structure inside the sparse support" vs "learned top-k" design, the theory implies both arms need guaranteed connectivity (window + a few global/sink tokens) to retain universal-approximation guarantees; the variable is then *which* extra edges are added, which is exactly where structure vs learned selection differ.
- The theory does not address *content-dependent* (learned top-k) supports, whose pattern varies by input; universal-approximation proofs assume fixed patterns. Learned selection's guarantees are empirical only (gap).

### Gaps
- No formal expressivity result found for content-based top-k sparse attention (NSA/MoBA/DSA).
- A dedicated "On the expressive power of sparse attention" paper beyond Yun et al. and BigBird was not located in this session.
