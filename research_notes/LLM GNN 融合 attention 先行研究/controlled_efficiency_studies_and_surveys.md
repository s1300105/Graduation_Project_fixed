# Controlled / systematic studies of text+graph (LLM+GNN) fusion with EFFICIENCY as the dependent variable, and surveys / taxonomies of the LLM+graph field (2023–2026)

Purpose: establish whether the following study has already been done:
"a factorial decomposition of which attention component (score bias, attention support/mask, normalization such as softmax vs kernel/linear, value transform such as relation-specific W_V, positional/structural encodings) carries the benefit of structural fusion, run across datasets of different structure–text complementarity (structure necessary / complementary / derivable from text), measuring data efficiency (learning curves vs training fraction), scale efficiency (small LLM + graph vs larger LLM alone), and context/token efficiency."

Research date: 2026-11-02. Sources: arXiv (abstract + HTML full texts), NeurIPS/ICML/ICLR virtual pages, Google Research blog, Microsoft Research pages, Snap Research page. Everything below was verified against at least the arXiv abstract page unless marked "unverified".

---

## Master table: papers recorded (one row per paper)

| # | Title (original) | Authors (first/last) | Venue, year | What was systematically varied | Domains / datasets | Params matched across conditions? | Key findings (incl. negative) | URL |
|---|---|---|---|---|---|---|---|---|
| 1 | Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs | Z. Chen, H. Mao, … J. Tang | SIGKDD Explorations (arXiv v1 Jul 2023, v4 Jan 2024) | Text encoder family/size (Deberta vs Sentence-BERT/e5/ada-002 vs LLaMA-7B), label rate (low vs high), prompt-structure injection (2-hop neighbor summary), pipeline (enhancer vs predictor) | Node classification: Cora, Citeseer, Pubmed, ogbn-arxiv, ogbn-products | No (different encoders; LLM predictors tested on 200 sampled nodes due to API cost) | "Simply enlarging the model size of LLMs may not help"; fine-tuned LMs collapse at low label rate (Deberta→GCN 59% low vs 86% high on Cora); 2-hop neighbor summaries help Cora (67.00→71.75) and arxiv but hurt Citeseer (65.5→62.0) and Pubmed (90.75→88.0) ("LLMs with structure prompts may suffer from heterophilous neighboring nodes"); time/memory table (GLEM 68,071 s / 18.22 GB vs Sentence-BERT 239 s / 1.30 GB on ogbn-arxiv) | https://arxiv.org/abs/2307.03393 |
| 2 | Large Language Models on Graphs: A Comprehensive Survey | B. Jin, … J. Han | IEEE TKDE 2024 (arXiv 2312.02783) | Survey (no experiments) | pure / text-attributed / text-paired graphs | n/a | Taxonomy LLM-as-Predictor (Graph-as-Sequence, Graph-Empowered LLM, Graph-Aware Finetuning) / Encoder / Aligner; future direction #3 "Efficient LLMs on Graphs" (graph linearization inflates sequence length; "lack of discussion on graph-aware LLM efficient tuning methods") | https://arxiv.org/abs/2312.02783 |
| 3 | A Survey of Graph Meets Large Language Model: Progress and Future Directions | Y. Li, Z. Li, … J. X. Yu | IJCAI 2024 survey track (arXiv 2311.12399) | Survey | — | n/a | Taxonomy enhancer (explanation-/embedding-based) / predictor (flatten-/GNN-based) / GNN-LLM alignment (symmetric/asymmetric); future work mentions time/space inefficiency, PEFT, per-node API cost; no claim about which fusion component matters | https://arxiv.org/abs/2311.12399 |
| 4 | A Survey of Large Language Models for Graphs | X. Ren, … C. Huang (HKUDS) | KDD 2024 survey (arXiv 2405.08011) | Survey | — | n/a | Framework-design taxonomy: GNNs-as-Prefix / LLMs-as-Prefix / LLMs-Graphs Integration / LLMs-only; scalability flagged (NodeFormer, DIFFormer) | https://arxiv.org/abs/2405.08011 |
| 5 | Graph Foundation Models: A Comprehensive Survey | Z. Wang + 18 | arXiv May 2025 (93 pp., 438 refs) | Survey | — | n/a | Modular backbone/pretraining/adaptation taxonomy; open question 10.1: GFMs lack a scaling law; efficiency only as sub-theme of scalability | https://arxiv.org/abs/2505.15116 |
| 6 | Towards Graph Foundation Models: A Survey and Beyond (a.k.a. GFM: Concepts, Opportunities and Challenges) | J. Liu, … | arXiv 2310.11829 (2023–24) | Survey | — | n/a | GNN-based / LLM-based / GNN+LLM-based GFMs | https://arxiv.org/abs/2310.11829 |
| 7 | Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications | G. Su, … J. Pei | arXiv Oct 2025 | Survey | TAGs | n/a | "LLM for TAG" vs "TAG for LLM"; orchestration sequential/parallel/multi-module; open challenges not detailed in abstract | https://arxiv.org/abs/2510.21131 |
| 8 | GLBench: A Comprehensive Benchmark for Graph with Large Language Models | Y. Li, P. Wang, … J. Li | NeurIPS 2024 D&B (arXiv 2407.07457) | 12 GraphLLM methods under one protocol; LLM backbone size; GNN depth/width; supervised vs zero-shot | 7 TAGs: Cora, Citeseer, Pubmed, ogbn-arxiv, WikiCS, Reddit, Instagram | No (methods as published) | Enhancers most robust; predictors unstable/invalid outputs; "no clear scaling law" (larger LLM/GNN does not consistently help); time & memory on one A800 (GraphLLM > GNN cost; GLEM/ENGINE big gains at big cost); training-free "Emb w/ NA" beats OFA/GraphGPT zero-shot | https://arxiv.org/abs/2407.07457 |
| 9 | When Do LLMs Help With Node Classification? A Comprehensive Analysis (v1 title: A Comprehensive Analysis on LLM-based Node Classification Algorithms; LLMNodeBed) | X. Wu, Y. Shen, … H. Cheng | ICML 2025 (arXiv 2502.00829) | Learning paradigm (semi-sup / sup / zero-shot); LLM backbone family; LLM size (Qwen 3B→32B inside LLaGA); encoder LM vs LLM; prompt type; homophily (10 homophilic + 4 heterophilic); explainer strength; >2,700 models | Cora, Citeseer, Pubmed, arXiv, WikiCS, Instagram, Reddit, Books, Photo, Computer, Cornell, Texas, Wisconsin, Washington (+ogbn-products appendix) | No | LLM methods win clearly only in semi-supervised; marginal in supervised; LLM encoders beat LMs mainly on heterophilic graphs (≈+10–12% Texas); predictor gains "marginal beyond Qwen-7B/14B", Qwen-32B ≈5× slower/sample; stronger explainer helps more with less supervision; time/memory reported (TAPE ≈5.9 h Pubmed; LLMIT >30 h arXiv), no token counts | https://arxiv.org/abs/2502.00829 ; https://llmnodebed.github.io |
| 10 | When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected | H. Xu, Y. You, T. Ma | arXiv Nov 2025 (preprint) | Template (LLaGA ND with Laplacian PE vs HN-k vs CO text-only); adapter (GCN/GAT/GIN vs MLP, GraphToken-style); LLM scale (LLaMA2 7B vs 13B); text richness (titles-only); dataset size (Products, ArXiv) | Cora, Citeseer, PubMed, School, Roman-Empire, Amazon-Ratings; BACE/BBBP/HIV; Davis DTI | Partially (components/training held constant; parameter counts not reported; encoder dims 300 vs 312) | Text-only CO avg 75.29% > ND 69.48%; MLP adapter 76.26% > GCN 67.50/GAT 67.07/GIN 68.24; gap persists at 13B; large datasets ND≈HN (p=0.20/0.49); no attention-component ablation | https://arxiv.org/abs/2511.16767 |
| 11 | Can LLMs Effectively Leverage Graph Structural Information through Prompts in Text-Attributed Graphs, and Why? | J. Huang, X. Zhang, Q. Mei, J. Ma | arXiv 2309.16595 (v4 Jun 2024; OpenReview-reviewed) | Prompt structure form (k-hop titles / +labels / attention-ranked), linearized vs structured, ego-graph rewiring, neighbor drop/add by label, text richness | arxiv-2023 (leakage-free), ogbn-arxiv, cora, pubmed, ogbn-products | n/a (prompting) | Structure gains larger when text scarce (+4.5 pts vs +1.1 rich on ogbn-arxiv); rewiring hurts LLM 0.25% vs MPNN 5.55% → LLM reads prompts "as paragraphs, not graphs"; gain tracks local homophily; benefit attributed to label-relevant phrases, not topology; no leakage evidence | https://arxiv.org/abs/2309.16595 |
| 12 | Let Your Graph Do the Talking: Encoding Structured Data for LLMs (GraphToken) | B. Perozzi, B. Fatemi, … J. Halcrow | arXiv Feb 2024 (Google) | Encoder type (GCN/GIN/MPNN/HGT/MHA/Node-Set/Edge-Set); feature type (learned IDX vs Laplacian PE) vs text prompting baselines | GraphQA (9 synthetic tasks) | No (encoder bodies 0–199k params; projection ≈11M vs frozen PaLM 2 S) | GraphToken best on all 9 tasks (up to +73 pts); no encoder dominates; linear set encoders win counting tasks; attention encoders generalize best; token counts / inference cost NOT measured | https://arxiv.org/abs/2402.05862 |
| 13 | Talk like a Graph: Encoding Graphs for Large Language Models | B. Fatemi, J. Halcrow, B. Perozzi | ICLR 2024 (arXiv 2310.04560) | Text serialization (node naming × edge phrasing), task, graph generator, LLM size | GraphQA | n/a | Encoding choice swings accuracy 4.8–61.8%; bigger models better; graph structure itself affects accuracy | https://arxiv.org/abs/2310.04560 |
| 14 | Less is More: Learning Graph Tasks with Just LLMs | S. Shirai, … S. Sohrabi (IBM) | arXiv Aug 2025 | Injection: Graph Tokens (400 fixed soft tokens) vs Graph Tokens+Text vs P-Tuning vs LoRA on text; ID vs OOD size; few-shot (30 examples) new tasks; cross-architecture transfer | GraphQA-style synthetic (8 tasks); PrOntoQA; LC-QuAD 2.0 | Partially (P-Tuning uses 400 virtual tokens to match Graph Tokens) | LoRA on text best and most robust (ND OOD 1.00 vs Graph Tokens .158); Graph Tokens "particularly brittle" OOD and untrainable from 30 examples; Graph Tokens' remaining value = lower input token count; Phi4-Mini 4B, Granite 2B/8B | https://arxiv.org/abs/2508.10115 |
| 15 | \<SOG_k\>: One LLM Token for Explicit Graph Structural Understanding | J. Wu, … C. Zhou (SJTU) | arXiv Feb 2026 (search snippets say ICLR 2026; arXiv page does not state venue — unverified) | Single discrete structural token vs graph-to-text vs soft prompts; codebook size; anchor strategy; LLaMA3-3B vs LLaMA2-7B vs 70B+ zero/few-shot | BBBP, Tox21, ClinTox, HIV, BACE; Cora, PubMed | No (3B/7B tuned vs 70B+ untuned) | 3B model Cora 91.58 vs GPT-4o 68.62; soft prompts 8.6–24.2 pts below zero-shot at 3B; claims "single token" cost-efficiency but reports NO token/latency/memory numbers | https://arxiv.org/abs/2602.01771 |
| 16 | Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning (GTLM) | D. Vajda | arXiv May 2026 (v2 Oct 2026) | Three additive attention biases (SPD, RRWP, Magnetic Laplacian) ablated one-at-a-time; base model 1B vs 8B; RoPE architecture (BLOOM/ALiBi, Gemma) | GraphQA; Cora, PubMed, OGBN-Arxiv, Reddit; BACE/BBBP/HIV; ChEBI-20; WebQSP/CWQ; needle-in-a-graph 1k–65k tokens | Bias params 173k (0.015%) + LoRA ≈2%; not matched vs 7B baselines | 1B GTLM ≈ 7B InstructMol on molecules; removing Magnetic bias largest drop (Edge Count 57.2→8.9), SPD removal no effect; needle-in-graph 98.0% vs flat text 8.5% at ≈65k tokens; training step ≈2× FlashAttention cost; no data-fraction curves | https://arxiv.org/abs/2605.10247 |
| 17 | Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning (GaLA) | D. Loveland, P. Trivedi, A. Weinstein, E. W. Huang, D. Koutra | KDD 2026 (arXiv 2606.15633) | Inference-time additive graph-distance bias (early layers), training-free; LLM size (Qwen2.5-3B reported) | TAG benchmarks (Cora, PubMed, Arxiv per snippets) | n/a (training-free) | RoPE suppresses attention between graph-adjacent but sequence-distant nodes; GaLA +4.7–13.5% over GraphICL on Qwen2.5-3B; mixed on Arxiv; no mask-vs-bias factorial | https://arxiv.org/abs/2606.15633 |
| 18 | Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data | Z. Guan, L. Wu, H. Zhao, M. He, J. Fan | ICML 2025 (arXiv 2505.02130) | Attention window/scope (fully connected vs fixed connectivity vs intermediate), train vs inference scope | (not in abstract) | not stated | LLM inter-node attention does not match ideal structural patterns; intermediate attention windows at training + fully connected at inference best | https://arxiv.org/abs/2505.02130 |
| 19 | What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA | J. Petersen, … R. Maggioni (Forgis / ETH) | arXiv Feb 2026 (v4 Jun 2026) | Sparse adjacency mask vs edge-type attention bias vs relation-specific query scaling vs relation-specific value gating; variants Full / Mask-only / Bias-only / vanilla; degree PE held constant | MetaQA 1/2/3-hop, WebQSP, CWQ (KGQA, DistilBERT question encoder) | YES (67.1M Rasa = 67.1M vanilla ≈ 67.0M R-GCN) | Mask alone gives most of the gain (+72.5pp MetaQA-3hop, +45.5 WebQSP, +53.9 CWQ over vanilla); relation params add +3.3–8.3pp; Bias-only without mask ≈ vanilla or worse (CWQ 2.1% vs 2.7%); latency 6× vanilla; NOT factorial (components 2–4 never isolated); 1k/5k sample sweep only in appendix; no LLM | https://arxiv.org/abs/2602.02834 |
| 20 | GL-Fusion: Rethinking the Combination of Graph Neural Network and Large Language model | H. Yang, X. Wang, … M. Zhang | arXiv Dec 2024 | Leave-one-out ablation: cross-attention, gate, multi-aggregator, GNN predictor, text predictor, LoRA rank | ogbn-arxiv, Cora, CS-TAG; FB15k-237-ind; CommonsenseQA; ogbg-code2 | No (≈10% of Llama-3-8B trained) | Every component removal hurts; cross-attention reduces complexity O((nLn+Lt)²)→O(nLn·Lt) but no measured runtime/tokens | https://arxiv.org/abs/2412.06849 |
| 21 | Are Large Language Models In-Context Graph Learners? | J. Li et al. | arXiv Feb 2025 | QueryRAG / LabelRAG / FewShotRAG retrieval of neighbor text/labels into prompt | Cora, PubMed, … | n/a | Plain LLMs lag GNNs; retrieval of neighbor labels lets LLM match/exceed supervised GNNs; assumes homophily | https://arxiv.org/abs/2502.13562 |
| 22 | How to Make LMs Strong Node Classifiers? | Z. Xu, … H. Tong | arXiv Oct 2024 (v3 Jan 2026) | Input augmentation (topological + semantic retrieval) + GNN candidate pruning, unchanged Flan-T5 | "real-world datasets" (not in abstract) | not stated | Off-the-shelf LM + augmentation ≈ SOTA GNN; still uses a lightweight GNN | https://arxiv.org/abs/2410.02296 |
| 23 | Rethinking Message Passing as Retrieval for Text-Attributed Graph Learning (RTA) | J. Li, … R. Ji | arXiv Aug 2026 | MLP + label-aware retrieval replaces message passing | multiple TAGs | not stated | Matches/outperforms GNN and graph-LLM baselines; efficiency claimed, not quantified in abstract | https://arxiv.org/abs/2608.26732 |
| 24 | Label-free Node Classification on Graphs with Large Language Models (LLM-GNN) | Z. Chen, … J. Tang | ICLR 2024 (arXiv 2310.04668) | LLM annotates subset → GNN; node selection heuristics | ogbn-products etc. | n/a | 74.9% on ogbn-products for < $1; third-party summary cites 2,124× cheaper than direct LLM inference (unverified from abstract) | https://arxiv.org/abs/2310.04668 |
| 25 | GraphArena: Evaluating and Exploring Large Language Models on Graph Computation | J. Tang et al. | ICLR 2025 (arXiv 2407.00379) | >10 LLMs; 4 P + 6 NP tasks; graph size; CoT/IT/code/test-time compute | synthetic + real graphs | n/a | Hallucination worse on larger graphs and smaller models; CoT insufficient | https://arxiv.org/abs/2407.00379 |
| 26 | GraphInfer-Bench: Benchmarking LLM's Inference Capability on Graphs | Z. Peng, … Y. Yi | arXiv Jun 2026 | Graph-token alignment models vs zero-shot frontier LLMs vs Graph2Text SFT vs plain GNNs | 42k samples, 6 TAGs; outlier/community/masked-node/description tasks | No | "Across every task, plain GNNs match or beat the strongest LLM-based row" (largest gap: community detection); no efficiency numbers | https://arxiv.org/abs/2606.11562 |
| 27 | Agentic Graph Token Reasoning | Z. Peng, Y. Yang | arXiv Aug 2026 | On-demand graph tokenization (node / k-hop / cluster) during reasoning; 3B & 7B backbones | 7 graph domains incl. OGBN-Products | not stated | Accuracy-vs-token-consumption Pareto plotted per backbone size (Fig. 4 per snippet); 3B/7B at high-accuracy end with bounded token budget; AgentGL uses most context but trails | https://arxiv.org/abs/2608.00542 |
| 28 | Improving LLM Reasoning with Homophily-aware Structural and Semantic TAG Compression (HS2C) | Z. Di, … X. Wang | arXiv Jan 2026 (v3 Jun 2026) | Neighborhood compression via structural-entropy communities; LLMs of varying size | 10 node-level + 7 graph-level benchmarks | not stated | Claims higher compression rate and accuracy; no numbers in abstract | https://arxiv.org/abs/2601.08187 |
| 29 | GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph | J. Yang et al. | NeurIPS 2021 | Progressive learning, unidirectional simplification, layer-wise GNN sharing | textual graphs (product/academic) | — | Nested GNN inside LM layers; ablations are training-recipe, not attention-component | https://arxiv.org/abs/2105.02605 |
| 30 | Edgeformers / Heterformer | B. Jin et al. | ICLR 2023 / KDD 2022 | Virtual token types (center/neighbor/neighbor-text) | textual-edge / heterogeneous text-rich networks | — | Each virtual token contributes; no mask-vs-bias ablation | https://arxiv.org/abs/2302.11050 |
| 31 | (MDPI Applied Sciences 15(5):2418, 2025) SLM + GCN+graph-transformer encoder | (not retrieved; HTTP 403) | Applied Sciences 2025 | SLM prompt-tuning vs fine-tuning with/without graph encoder vs an LLM | not retrieved | unverified | Snippet: graph encoder lets SLMs "closely approach" LLM performance — UNVERIFIED | https://www.mdpi.com/2076-3417/15/5/2418 |

---

## Key Question 1: Surveys and taxonomies — what taxonomy do they use, and do they name "which attention component" or "efficiency" as open problems?

### Takeaway
All major surveys (Chen 2023, Jin/TKDE 2024, Li/IJCAI 2024, Ren/KDD 2024, Wang GFM 2025, Su 2025) organize the field by the *role* of the LLM (enhancer/encoder, predictor, aligner) or by *pipeline topology* (prefix / integration), not by attention-level mechanism. Every one of them lists efficiency as an open problem, but only at the level of "LLM inference/tuning is expensive" or "linearization inflates sequence length"; none poses the question of which attention component carries the fusion benefit, and none frames data-, scale- or token-efficiency as a dependent variable to be measured under matched conditions.

### Cited Findings
- Chen et al. 2023 taxonomy: "LLMs-as-Enhancers" vs "LLMs-as-Predictors", node classification on TAGs; to appear in SIGKDD Explorations — [arXiv 2307.03393](https://arxiv.org/abs/2307.03393)
- Chen et al. 2023 future directions: "developing more efficient strategies to utilize LLMs is currently a challenge"; API cost scales with N nodes ("querying LLMs' APIs at least N times for a graph with N nodes"); aligning graph and LLM feature spaces via frozen LLM + linear projection suggested; long-range tasks do not fit context windows — [arXiv 2307.03393 HTML](https://arxiv.org/html/2307.03393v4)
- Jin et al. (TKDE 2024) taxonomy: scenarios pure/text-attributed/text-paired graphs × roles Predictor (Graph-as-Sequence, Graph-Empowered LLM, Graph-Aware LLM Finetuning) / Encoder / Aligner — [arXiv 2312.02783](https://arxiv.org/abs/2312.02783)
- Jin et al. six future directions: better benchmarks; broader task space; **Efficient LLMs on Graphs** ("they suffer from inefficiency in graph linearization and model optimization"; "the length of the transferred sequence will increase significantly as the size of the graph increases"; "there is a lack of discussion on graph-aware LLM efficient tuning methods"); generalizable/robust LLMs; multi-modal foundation models; LLMs as dynamic agents. The section "does not discuss fusion components such as attention or positional encoding" — [arXiv 2312.02783 HTML §VIII](https://arxiv.org/html/2312.02783v4)
- Jin et al. describe "Graph-Empowered LLMs" as methods that modify attention and positional encoding to jointly encode text and structure, but "does not claim which architectural component of fusion matters most" — [arXiv 2312.02783 HTML](https://arxiv.org/html/2312.02783v4)
- Li et al. (IJCAI 2024) taxonomy: LLM-as-Enhancer (explanation-based / embedding-based), LLM-as-Predictor (flatten-based / GNN-based), GNN-LLM Alignment (symmetric / asymmetric); future work: LLMs "may face inefficiencies in terms of time and space", PEFT (LoRA, prefix tuning) suggested, "more efficient methods may unlock more power"; explanation-based enhancers need one API call per node; "the page does not compare which fusion component matters most" — [arXiv 2311.12399](https://arxiv.org/html/2311.12399)
- Ren et al. "A Survey of LLMs for Graphs" taxonomy by framework design: GNNs-as-Prefix, LLMs-as-Prefix, LLMs-Graphs Integration, LLMs-only; scalability flagged (NodeFormer, DIFFormer) — [arXiv 2405.08011](https://arxiv.org/html/2405.08011v3)
- Wang et al. 2025 GFM survey (93 pp., 438 refs): modular taxonomy backbone/pretraining/adaptation; open question 10.1 is that GFMs lack a scaling law unlike LLMs; graph transformers limited by quadratic complexity; efficiency appears only as sub-theme of scalability — [arXiv 2505.15116](https://arxiv.org/abs/2505.15116)
- Liu et al. GFM survey: GNN-based / LLM-based / GNN+LLM-based GFMs — [arXiv 2310.11829](https://arxiv.org/html/2310.11829v3)
- Su et al. Oct 2025 TAG survey: "LLM for TAG" vs "TAG for LLM", orchestration sequential/parallel/multi-module, TAG-specific pretraining/prompting/PEFT; open challenges not itemized in abstract — [arXiv 2510.21131](https://arxiv.org/abs/2510.21131)
- Mao et al. position paper "Graph Foundation Models are Already Here" (ICML 2024): flattening graphs into prompts gives suboptimal results — [arXiv 2402.02216](https://arxiv.org/html/2402.02216v3)
- Awesome lists enumerating the field: [yhLeeee/Awesome-LLMs-in-Graph-tasks](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks), [HKUDS/Awesome-LLM4Graph-Papers](https://github.com/HKUDS/Awesome-LLM4Graph-Papers), [Zehong-Wang/Awesome-Foundation-Models-on-Graphs](https://github.com/Zehong-Wang/Awesome-Foundation-Models-on-Graphs) (found via search; repos not crawled)

### Inferences
- The field's taxonomic axis is *where the LLM sits in the pipeline*, not *which attention primitive carries structure*. A mechanism-level decomposition would be orthogonal to every existing survey taxonomy.
- "Efficiency" in surveys means engineering cost of running LLMs (API calls, GPU memory, sequence length), never a controlled dependent variable with matched parameter budgets.

### Gaps
- Did not retrieve a 2026 survey; searches for "2026 survey LLM graph" returned only 2024–2025 surveys. Awesome-list repos were not crawled for 2026 entries (GitHub fetch not attempted).

---

## Key Question 2: Benchmarks comparing many LLM+graph methods under one protocol — when does GNN-side structure help vs text-only LLMs, and what about efficiency?

### Takeaway
GLBench (NeurIPS 2024) and LLMNodeBed (ICML 2025) are the two multi-method, single-protocol benchmarks for TAG node classification. Both report wall-clock time and memory, both find no clean scaling with LLM size, and LLMNodeBed shows the structural/LLM advantage is concentrated in semi-supervised (low-label) and heterophilic regimes. Neither matches parameter counts across conditions, neither reports token counts, and neither varies the *mechanism* of structure injection inside attention. GraphInfer-Bench (2026) adds that plain GNNs still match or beat every LLM-based method on structure-heavy inference tasks.

### Cited Findings
- GLBench: 12 GraphLLM methods (enhancer: GIANT, TAPE, OFA, ENGINE, ZeroG; predictor: InstructGLM, GraphText, GraphAdapter, GraphGPT, LLaGA; aligner: GLEM, Patton) + GNN/LM baselines on 7 TAGs; enhancers most consistently strong; predictors often underperform and produce invalid outputs — [arXiv 2407.07457](https://arxiv.org/html/2407.07457v4); [NeurIPS 2024 poster](https://nips.cc/virtual/2024/poster/97881)
- GLBench scaling: "no clear scaling law"; performance does not consistently improve with model size; scaling GNN depth/width shows no trend; OFA with RoBERTa-large only 58.3% on Cora — [arXiv 2407.07457](https://arxiv.org/html/2407.07457v4)
- GLBench efficiency: time and memory on one NVIDIA A800 for Cora/Citeseer/WikiCS/Instagram; GraphLLM methods cost more than GNNs; enhancers give good gains at modest cost; GLEM and ENGINE large gains at substantially higher cost; future work: "improve efficiency so GraphLLM methods can scale" — [arXiv 2407.07457](https://arxiv.org/html/2407.07457v4)
- GLBench zero-shot: training-free "Emb w/ NA" beats OFA and GraphGPT; both structure and semantics matter for transfer — [arXiv 2407.07457](https://arxiv.org/abs/2407.07457v4)
- LLMNodeBed: 14 datasets (10 homophilic, 4 heterophilic), 8 LLM algorithms + 8 classic baselines, >2,700 trained models; taxonomy LLM-as-Encoder / Explainer / Predictor + zero-shot Direct Inference / GFMs — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2); [Microsoft Research page, ICML 2025](https://www.microsoft.com/en-us/research/publication/when-do-llms-help-with-node-classification-a-comprehensive-analysis/)
- LLMNodeBed: LLM-based methods "can significantly outperform traditional methods in a semi-supervised setting, while the advantage is marginal in a supervised setting" — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- LLMNodeBed: LLM encoders beat LM encoders mainly on heterophilic graphs (gaps up to ≈10%, ≈12% on Texas); on homophilic graphs ≤≈4% (Reddit) — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- LLMNodeBed scale: larger predictor backbones help but "gains are marginal beyond Qwen-7B and 14B"; Qwen-32B ≈5× time per sample vs Qwen-7B; stronger explainer LLM helps more with less supervision (up to ≈4% Pubmed), gap narrows as labels grow — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- LLMNodeBed cost: GNNs converge in seconds; LM fine-tuning minutes; TAPE ≈5.9 h on Pubmed; GCN+LLM-emb ≈13.4 min; LLaGA ≈25.6 min; LLMIT >30 h/run on arXiv; memory of LLM encoders ≈ GNNs; "reports time and memory, not token counts" — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- LLMNodeBed structure prompts: structure-enriched prompts give 3–10% on homophilic datasets (Cora, Photo); CoT/ToT/ReAct only minor gains — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- GraphInfer-Bench (Jun 2026): 42k samples, 6 TAGs; "Across every task, plain GNNs match or beat the strongest LLM-based row", largest margin on community detection; no efficiency numbers — [arXiv 2606.11562](https://arxiv.org/abs/2606.11562)
- GraphArena (ICLR 2025): >10 LLMs on 4 P + 6 NP graph problems; hallucination worse on larger graphs and smaller models; CoT insufficient — [arXiv 2407.00379](https://arxiv.org/abs/2407.00379); [ICLR poster](https://iclr.cc/virtual/2025/poster/29278)
- Talk like a Graph (ICLR 2024): serialization choice changes accuracy 4.8–61.8%; bigger models better; graph structure (generator) itself matters — [arXiv 2310.04560](https://arxiv.org/abs/2310.04560); [Google blog](https://research.google/blog/talk-like-a-graph-encoding-graphs-for-large-language-models/)

### Inferences
- Benchmarks operationalize "data fraction" only as two or three discrete regimes (semi-supervised vs supervised vs zero-shot), not as learning curves.
- "Scale efficiency" exists only as "bigger LLM gives diminishing returns" within one method family (LLaGA/Qwen); no benchmark pairs a small LLM+structure against a larger text-only LLM at matched total compute.
- Efficiency is reported descriptively (time/memory of existing methods as published), never as a controlled outcome of an architectural factor.

### Gaps
- Could not retrieve the tail (last ≈66k chars) of LLMNodeBed v2, so any appendix with label-fraction sweeps is unconfirmed.
- GraphInfer-Bench method list was not retrieved beyond family names.

---

## Key Question 3: Data-efficiency studies — few-shot / low-label curves for GNN-augmented LLMs vs LLM-only; small-LLM-with-structure vs large-LLM-alone claims

### Takeaway
Evidence on data efficiency is fragmentary: Chen 2023 and LLMNodeBed contrast "low" vs "high" label regimes, and a few method papers show 1/5/10% or k-shot points, but I found no paper plotting full learning curves (accuracy vs training fraction) for a GNN-augmented LLM against a parameter- or compute-matched text-only LLM. Small-with-structure vs large-without claims exist (GTLM 1B ≈ 7B InstructMol; SOG_k 3B > GPT-4o on Cora) but are never parameter- or compute-matched and are confounded with fine-tuning vs zero-shot.

### Cited Findings
- Chen 2023: fine-tuned Deberta→GCN 59% (low label rate) vs 86% (high) on Cora; deep sentence embeddings competitive in both; GLEM best with plentiful labels (ogbn-arxiv/products); TAPE explanations help most at low label rate — [arXiv 2307.03393 HTML](https://arxiv.org/html/2307.03393v4)
- LLMNodeBed: LLM advantage significant in semi-supervised, marginal in supervised; explainer-strength gap narrows as labeled data grows — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- LLMNodeBed scale: Qwen 3B→32B inside LLaGA, marginal gains beyond 7B/14B at ≈5× latency for 32B — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- GTLM: "classification saturates at 1B, while generation tasks benefit from 8B"; 1B GTLM molecular ROC-AUC BACE 82.0 / BBBP 70.6 / HIV 76.9 "on par with the 7B InstructMol models" (mixed: InstructMol-G 84.3 on BACE); "the visible text includes no training-data-fraction experiments" — [arXiv 2605.10247 HTML](https://arxiv.org/html/2605.10247)
- SOG_k: LLaMA3-3B fine-tuned reaches Cora 91.58 vs GPT-4o zero-shot 68.62; 7B sometimes worse than 3B (Cora F1 71.28 vs 78.62); "no efficiency measurements or parameter-matched comparisons" — [arXiv 2602.01771](https://arxiv.org/html/2602.01771)
- Less is More: with 30 examples per new task, LoRA-on-text reaches EE .966 / SP .720 / CC .994 while Graph Tokens from scratch fail (EE .246) — "30 samples were not enough to train its encoder from scratch" (i.e., the graph encoder is *less* data-efficient than text LoRA on synthetic tasks) — [arXiv 2508.10115](https://arxiv.org/html/2508.10115v1)
- "When Structure Doesn't Help": LLaMA2-7B vs 13B on School: ND 70.98→69.30 vs HN-1 92.09→89.45 (structure template gap persists with scale) — [arXiv 2511.16767](https://arxiv.org/html/2511.16767v1)
- LLM4NG (node generation for few-shot GNNs): 1-shot ogbn-arxiv +76% over baseline; GCN 23%→52% from 1- to 10-shot (GNN-side augmentation, no LLM-only baseline) — [arXiv 2310.09872](https://arxiv.org/html/2310.09872v2)
- AskGNN: labeled-data settings 1%, 5%, 10%; few-shot prompting can underperform zero-shot (Llama3-70B on ogbn-arxiv) — [arXiv 2310.04668 overview via alphaxiv](https://alphaxiv.org/overview/2310.04668v3) (secondary; the AskGNN paper itself was not fetched — unverified)
- LLM-GNN (ICLR 2024): 74.9% on ogbn-products at < $1 — [arXiv 2310.04668](https://arxiv.org/abs/2310.04668)
- 2026 LLM-GNN co-teaching preprint: at 3-shot, +7.86 pp Cora / +7.73 pp ogbn-arxiv over "GNN-as-Judge" (hybrid vs hybrid, not vs LLM-only) — [arXiv 2606.11583](https://arxiv.org/html/2606.11583) (snippet only; unverified)
- TU Delft 2026 thesis "GNN-LLM Hybrids for Node Classification": compares MPNN GNNs, ENGINE, LLaGA and a lightweight hybrid; "hybrid's benefit is most pronounced in low-supervision regimes" — [repository record](https://repository.tudelft.nl/record/uuid:6611851d-79ea-4c8a-aeb9-122149cc5aa6) (page returned 503 on fetch; snippet only — unverified)
- MDPI Applied Sciences 2025: graph encoder (GCN+graph transformer) "significantly improves the performance of SLMs in both prompt tuning and fine-tuning scenarios, achieving results which closely approach the performance of a large language model" — [MDPI 15(5):2418](https://mdpi.com/2076-3417/15/5/2418) (HTTP 403 on fetch; snippet only — unverified)

### Inferences
- The closest thing to a "scale-efficiency" result is GTLM (1B+bias ≈ 7B without), but it compares against *different* published baselines, not the same base LLM scaled up without structure.
- Data-efficiency evidence points two ways depending on task family: on TAG node classification, structure/LLM help is largest with few labels (LLMNodeBed, Chen 2023); on synthetic GraphQA tasks, graph-token encoders are the *least* sample-efficient component (Less is More).

### Gaps
- No paper found that plots accuracy vs training fraction (e.g., 1%–100%) for structure-fused vs text-only models with the same LLM backbone. Queries tried: "learning curve training fraction GNN-augmented LLM", "label ratio ablation LLaGA GraphGPT", "few-shot low-label curve GNN LLM ogbn-arxiv".
- MDPI 2025 and TU Delft 2026 items could not be opened.

---

## Key Question 4: Token / context efficiency — tokens needed when structure is serialized vs encoded, cost-accuracy frontiers

### Takeaway
Token efficiency is asserted far more often than measured. GraphToken, SOG_k, GL-Fusion and HS2C all *claim* token or complexity savings without reporting token counts; the only works that quantify tokens are "Less is More" (fixed 400 soft tokens vs edge-list text; token advantage acknowledged but accuracy worse), GTLM (needle-in-a-graph up to ≈65k tokens: 98.0% vs 8.5% flat text), and the 2026 "Agentic Graph Token Reasoning" accuracy-vs-token Pareto plot. No controlled study holds accuracy fixed and measures the token budget needed by each injection mechanism.

### Cited Findings
- GraphToken: encoder bodies 0–199k params, projection ≈11M vs frozen PaLM 2 S; best on all 9 GraphQA tasks (up to +73 pts); "The paper does not analyze token counts or inference cost" — [arXiv 2402.05862](https://arxiv.org/html/2402.05862v1)
- Less is More: Graph Tokens use a fixed 400 tokens (200 node + 200 edge) and omit the edge list, "a major advantage in input token count, especially for graphs with many edges"; test graphs up to 7,800 edges; but LoRA on text is most accurate (ND OOD 1.00 vs .158) — [arXiv 2508.10115](https://arxiv.org/html/2508.10115v1)
- SOG_k: graph-to-text "consumes large numbers of tokens, scatters attention"; soft prompts suffer modality mismatch (t-SNE Fig. 6); claims "only a single token" but "gives no measured latency, memory, or token-count numbers" — [arXiv 2602.01771](https://arxiv.org/html/2602.01771)
- GTLM needle-in-a-graph: N=128, T=512, k=4 (≈65k tokens) GTLM 98.0% vs flat-text 8.5%; attributed to node-order invariance + per-node RoPE reset; training step ≈2× FlashAttention cost at equal peak memory; bias params 173,056 (0.015%) + LoRA ≈2% — [arXiv 2605.10247 HTML](https://arxiv.org/html/2605.10247)
- GL-Fusion: cross-attention reduces complexity from O((nLn+Lt)²) to O(nLn·Lt); "No measured runtime or token-count results are reported" — [ar5iv 2412.06849](https://ar5iv.labs.arxiv.org/html/2412.06849)
- Agentic Graph Token Reasoning (Aug 2026): Fig. 4 plots accuracy vs token consumption on OGBN-Products with a Pareto frontier per backbone size; 3B/7B at high-accuracy end with bounded token budget; retrieval-based AgentGL uses most context but trails — [arXiv 2608.00542](https://arxiv.org/abs/2608.00542) (Pareto detail from search snippet; abstract page confirms method only)
- HS2C: compresses neighborhoods into community summaries, claims higher compression rate and accuracy on 10 node-level + 7 graph-level benchmarks; "does not report specific numbers" for token reduction in abstract — [arXiv 2601.08187](https://arxiv.org/abs/2601.08187)
- Chen 2023: neighbor-summary prompts limited to 5 neighbors "due to context limits"; API cost forced evaluation on 200 nodes — [arXiv 2307.03393 HTML](https://arxiv.org/html/2307.03393v4)
- Huang et al.: up to 20 one-hop + 5 two-hop neighbors per prompt; 1,000-node samples due to API cost; "no token counts or cost figures" — [arXiv 2309.16595](https://arxiv.org/html/2309.16595v4)
- LLM-GNN: 74.9% ogbn-products under $1; a secondary summary cites 2,124× cheaper than direct LLM inference — [arXiv 2310.04668](https://arxiv.org/abs/2310.04668); [alphaxiv overview](https://alphaxiv.org/overview/2310.04668v3)
- Jin survey: "the length of the transferred sequence will increase significantly as the size of the graph increases" — [arXiv 2312.02783 HTML](https://arxiv.org/html/2312.02783v4)

### Inferences
- Where tokens are counted, the result is a trade: soft tokens save context but lose accuracy/robustness (Less is More), whereas attention-bias injection (GTLM) keeps text uncompressed and wins on long contexts at ≈2× per-step compute. No study compares these two on the same backbone with equal token budget.
- A "tokens-to-reach-accuracy-X" or "accuracy at fixed context budget" curve across injection mechanisms does not exist in the retrieved literature.

### Gaps
- Agentic Graph Token Reasoning's full Pareto numbers were not retrieved (abstract only).
- No GPU-hour-based cross-method frontier found (query: "Pareto cost accuracy frontier LLM graph GPU-hours").

---

## Key Question 5: Negative results and critiques — GNN fusion adds little, benefits vanish with scale, gains come from information rather than inductive bias

### Takeaway
A consistent 2023–2026 line of negative results shows that for TAG node classification with strong text encoders, explicit structure (GNN adapters, structural templates, Laplacian PEs) often adds nothing or hurts; the gains that do appear track homophily and label-relevant neighbor *text* (information), not topology (inductive bias). The notable counter-signal is structure-heavy tasks (community detection, KGQA, long-context needle tasks), where structure-aware attention or plain GNNs still dominate.

### Cited Findings
- "When Structure Doesn't Help" (Nov 2025): text-only CO template 75.29% avg vs LLaGA-style ND (with Laplacian PE) 69.48%; MLP adapter 76.26% vs GCN 67.50 / GAT 67.07 / GIN 68.24 avg; on School ND 66.43 vs CO 91.13; gap persists at LLaMA2-13B; on Products/ArXiv ND≈HN-1 (Welch p=0.20, 0.49); conclusion: "explicit structural priors may be redundant or even detrimental when strong language models are used" — [arXiv 2511.16767](https://arxiv.org/html/2511.16767v1); [project page](https://graphcore-research.github.io/structure-does-not-help)
- Same paper: on molecules MLP ≈ GNN adapters on BACE/HIV, GIN leads BBBP (60.29 vs 54.57) — structure helps where topology is the label signal — [arXiv 2511.16767](https://arxiv.org/html/2511.16767v1)
- Huang et al.: linearizing structured prompts changes accuracy negligibly; rewiring ego-graphs hurts LLM 0.25% vs MPNN 5.55%; gains correlate with local homophily; attributed to "relevant phrases in homophilous neighborhoods, not to graph topology"; no leakage evidence — [arXiv 2309.16595](https://arxiv.org/html/2309.16595v4)
- Chen 2023: neighbor summaries hurt Citeseer and Pubmed; "LLMs with structure prompts may suffer from heterophilous neighboring nodes"; "Simply enlarging the model size of LLMs may not help" — [arXiv 2307.03393 HTML](https://arxiv.org/html/2307.03393v4)
- GLBench: no clear scaling law; training-free Emb w/ NA beats OFA/GraphGPT zero-shot — [arXiv 2407.07457](https://arxiv.org/html/2407.07457v4)
- LLMNodeBed: LLM advantage marginal in supervised regime; LLMIT trails; GFMs < GPT-4o zero-shot — [arXiv 2502.00829](https://arxiv.org/html/2502.00829v2)
- Less is More: graph-token encoders brittle OOD; "LLMs can be trained to reason on graphs, and show good generalization without the need for special graph encoding models" — [arXiv 2508.10115](https://arxiv.org/html/2508.10115v1)
- Attention Mechanisms Perspective (ICML 2025): LLM node-to-node attention "does not match ideal structural patterns"; neither fully-connected nor fixed-connectivity attention optimal — [arXiv 2505.02130](https://arxiv.org/abs/2505.02130)
- GaLA (KDD 2026): RoPE suppresses attention between graph-adjacent but sequence-distant nodes — the failure is positional, fixable at inference without training — [arXiv 2606.15633](https://arxiv.org/abs/2606.15633); [Snap Research](https://research.snap.com/publications/formalizing-and-mitigating-structural-distortion-in-llm-attention-for-zero-shot-graph-reasoning.html)
- Counter-evidence: GraphInfer-Bench — plain GNNs match/beat all LLM-based rows on every structural inference task — [arXiv 2606.11562](https://arxiv.org/abs/2606.11562)
- Counter-evidence: Tabula RASA — adjacency masking gives +45–73 pp on KGQA over a vanilla transformer at matched 67.1M params — [arXiv 2602.02834](https://arxiv.org/html/2602.02834)
- Counter-evidence: GTLM needle-in-a-graph 98.0% vs 8.5% flat text at ≈65k tokens — [arXiv 2605.10247 HTML](https://arxiv.org/html/2605.10247)
- RTA (Aug 2026): message passing re-interpreted as retrieval; MLP + label-aware retrieval matches/outperforms GNN and graph-LLM baselines — [arXiv 2608.26732](https://arxiv.org/abs/2608.26732)
- "Are LLMs In-Context Graph Learners?": plain LLMs lag GNNs; retrieving neighbor labels into the prompt (LabelRAG) matches/exceeds supervised GNNs; assumes homophily — [arXiv 2502.13562](https://arxiv.org/html/2502.13562v1)
- Searches for exact titles "Language Models are Graph Learners" and "Can GNNs still help in the LLM era" returned no such paper (the former appears only as a sotaverified.org listing, not fetched) — [search evidence](https://sotaverified.org/papers/language-models-are-graph-learners) (unverified)

### Inferences
- The "structure necessary / complementary / derivable" split is *implicit* in the literature: derivable (homophilic citation TAGs: Cora/PubMed/arXiv) → structure redundant; complementary (heterophilic / sparse-text) → structure helps when it brings new text; necessary (community detection, KGQA multi-hop, needle-in-graph, BBBP) → structure-aware attention or GNNs win. No single paper constructs datasets along this axis on purpose.
- The information-vs-inductive-bias question is answered only indirectly (Huang's rewiring/linearization controls); no study isolates it inside a trained fusion model.

### Gaps
- No peer-reviewed replication of "When Structure Doesn't Help" found.

---

## Key Question 6: Does any paper do a factorial ablation of attention components for text+graph fusion specifically?

### Takeaway
No. After the queries listed below, the closest works are (a) Tabula RASA (2026, KGQA with a DistilBERT question encoder, parameter-matched, mask vs bias vs relation-specific query/value — but explicitly *not* factorial, no LLM, no TAGs, no efficiency axes), (b) GTLM (2026, leave-one-out over three *additive-bias* types only, no mask/normalization/W_V factor, one base model, no data-fraction curves), and (c) GaLA / Attention-Mechanisms-Perspective (single-factor studies of bias or attention window). No paper varies score bias × support/mask × softmax-vs-linear normalization × relation-specific value transform × positional encoding, and none crosses such factors with datasets chosen for structure–text complementarity or with efficiency outcomes.

### Cited Findings
- Tabula RASA: components = sparse adjacency mask, edge-type attention bias, relation-specific query scaling, relation-specific value gating; degree PE held constant; design = Full / Mask-only / Bias-only / vanilla, "Not factorial or leave-one-out"; "Components 2–4 are not ablated individually in the main body; the authors defer this to future work"; matched 67.1M params; mask alone +72.5 pp (MetaQA-3hop) / +45.5 (WebQSP) / +53.9 (CWQ); relation params +3.3–8.3 pp; Bias-only ≈ vanilla on CWQ (2.1 vs 2.7%); bias is "predominantly topological, not relational"; latency 49.0 vs 7.9 ms; 1k/5k-sample sweep in appendix only; "Does not compete with LLM-augmented methods" — [arXiv 2602.02834](https://arxiv.org/html/2602.02834); [ICML 2026 listing](https://icml.cc/virtual/2026/67549)
- GTLM: three additive per-head biases (SPD lookup, RRWP MLP, Magnetic-Laplacian Deep-Set MLP) added pre-softmax; PrefixLM full-visibility mask on node prefix (fixed, not ablated); RoPE reset per node (fixed); ablation removes one bias at a time: Magnetic removal largest drop (Edge Count 57.2→8.9; Triangle 31.5→21.0), RRWP modest, SPD none; "The page does not report ablations of the bias, mask, or positional encodings" beyond this — [arXiv 2605.10247 HTML](https://arxiv.org/html/2605.10247)
- GaLA: single inference-time additive graph-distance bias in early layers; no mask-vs-bias comparison — [arXiv 2606.15633](https://arxiv.org/abs/2606.15633)
- Attention Mechanisms Perspective (ICML 2025): varies attention window/scope (full vs fixed vs intermediate; train vs inference) only — [arXiv 2505.02130](https://arxiv.org/abs/2505.02130)
- GL-Fusion: leave-one-out over cross-attention / gate / aggregators / predictors / LoRA rank; graph-aware mask and shared PE never ablated separately — [ar5iv 2412.06849](https://ar5iv.labs.arxiv.org/html/2412.06849)
- GraphToken: ablates encoder *type* and *feature* (IDX vs LPE), not attention primitives — [arXiv 2402.05862](https://arxiv.org/html/2402.05862v1)
- GraphFormers / Edgeformers / Heterformer: ablations cover training recipe and virtual-token types, not mask vs bias — [arXiv 2105.02605](https://arxiv.org/pdf/2105.02605); [arXiv 2302.11050](https://arxiv.org/pdf/2302.11050)
- Softmax vs linear attention has been ablated only in *pure* graph transformers: SGFormer (softmax worst due to "over-normalizing") vs GraphTARIF (linear attention low-rank/over-smoothing) — [arXiv 2409.09007](https://arxiv.org/pdf/2409.09007); [arXiv 2510.10631](https://arxiv.org/pdf/2510.10631); "I didn't find an ablation that compares linear and softmax attention specifically inside an LLM–graph fusion architecture"
- Closest mask-bias-vs-prefix head-to-head with matched parameter counts is in *vision*-language: "The Gate Always Closes" (attention-mask bias vs gated prefix tokens vs random-embedding soft-prompt control) — [arXiv 2607.23335](https://arxiv.org/pdf/2607.23335) (snippet; not graph)
- "From Noise to Diversity" notes soft prompting can only shift attention outputs in a fixed direction without changing relative attention among content tokens, and that prior work hasn't separated injection effect from optimization effect — [arXiv 2605.11936](https://arxiv.org/pdf/2605.11936) (snippet)

### Search queries run for the "does this exist" question (all 2026-11-02, WebSearch standard mode)
1. "ablation attention bias vs mask vs relation-specific value graph transformer text-attributed LLM fusion"
2. "GTLM graph-aware attention bias transformer layers 0.015% parameters text-attributed graph"
3. "ablation \"attention mask\" versus \"attention bias\" structure injection language model text-attributed graph node classification"
4. "linear attention vs softmax graph transformer text node features ablation kernel attention normalization LLM graph fusion"
5. "\"soft prompt\" vs \"attention bias\" vs \"cross-attention\" graph injection LLM systematic comparison injection point ablation 2025 2026"
6. "GraphFormers OR Edgeformers OR Heterformer nested transformer text graph ablation \"attention\" structure injection language model"
7. "GaLA graph-aligned language attention zero-shot graph reasoning KDD 2026 Snap"
8. "dataset categorization structure necessary vs complementary vs derivable from text graph LLM benchmark text-structure complementarity"
9. "token cost accuracy trade-off graph serialization versus graph encoder LLM soft tokens context length benchmark"
10. "Pareto cost accuracy frontier LLM graph node classification inference tokens GPU-hours comparison GNN LLM hybrid 2025"
11. "small LLM with graph encoder matches larger LLM alone node classification scale efficiency"
12. "LLM graph few-shot low-label learning curve GNN-augmented LLM vs LLM-only ogbn-arxiv data efficiency"
13. "LLaGA GraphGPT label ratio ablation training data fraction learning curve text-attributed graph LLM predictor low-resource"
14. "GNN LLM text-attributed graph negative result \"does not help\" text-only baseline 2025"
15. "\"Language Models are Graph Learners\" OR \"Can GNNs still help\" LLM era text-attributed graphs"
16. "graph structure \"redundant\" with text homophily LLM text-attributed graph analysis when does structure matter node classification 2025 arXiv"
Plus survey/benchmark queries (GLBench, LLMNodeBed, GraphArena, Talk like a Graph, GFM survey, Awesome lists, ICML 2025 position paper).

### Inferences
- The two mechanism-level studies that exist (Tabula RASA, GTLM) each hold fixed exactly the factors the other varies (RASA fixes PE and has no LLM/text-rich nodes; GTLM fixes mask and PE and varies only bias type), and neither touches normalization (softmax vs kernel) or a relation-specific W_V inside an LLM.
- Tabula RASA's own authors list individual ablation of bias / query-scaling / value-gating as future work, which indicates the factorial design is recognized as open even within KGQA.

### Gaps
- Semantic Scholar "cited by" traversal on Chen 2023 and GraphToken was not performed with the API (only web search); a 2026 paper indexed after mid-2026 could be missed.
- OpenReview was not queried directly.

---

## Key Question 7: Coverage assessment — which parts of the proposed study are already done, which are not

### Takeaway
Each *axis* of the proposed study has partial precedent, but no paper (and no pair of papers) combines them; the attention-component factorial on text-rich graphs with an LLM, crossed with complementarity-stratified datasets and efficiency outcomes, is not in the literature as of Nov 2026.

### Cited Findings (coverage matrix)
| Proposed component | Already covered? | Evidence |
|---|---|---|
| Score bias as a factor | Partially — single-factor or bias-type-only | GTLM leave-one-out over 3 bias types [2605.10247](https://arxiv.org/html/2605.10247); GaLA single bias [2606.15633](https://arxiv.org/abs/2606.15633); RASA Bias-only variant (KGQA, no LLM) [2602.02834](https://arxiv.org/html/2602.02834) |
| Attention support / mask as a factor | Partially — RASA Mask-only (KGQA, 67M, matched) ; attention window study (ICML 2025) | [2602.02834](https://arxiv.org/html/2602.02834); [2505.02130](https://arxiv.org/abs/2505.02130) |
| Mask × bias interaction | Only 3 cells (Full/Mask/Bias) in RASA; not in any LLM/TAG work | [2602.02834](https://arxiv.org/html/2602.02834) |
| Normalization (softmax vs kernel/linear) as a factor in text+graph fusion | **Not covered**; only in pure graph transformers (SGFormer, GraphTARIF) | [2409.09007](https://arxiv.org/pdf/2409.09007); [2510.10631](https://arxiv.org/pdf/2510.10631) |
| Relation-specific value transform (W_V) as a factor | Only RASA "value gating", never isolated (bundled with bias+query scaling) | [2602.02834](https://arxiv.org/html/2602.02834) |
| Positional/structural encodings as a factor | GraphToken IDX vs LPE (synthetic GraphQA); "Structure Doesn't Help" removes Laplacian PE with the ND template (confounded); GTLM/RASA hold PE fixed | [2402.05862](https://arxiv.org/html/2402.05862v1); [2511.16767](https://arxiv.org/html/2511.16767v1) |
| Full factorial across ≥4 attention components | **Not covered** (RASA explicitly non-factorial) | [2602.02834](https://arxiv.org/html/2602.02834) |
| Datasets stratified by structure–text complementarity (necessary / complementary / derivable) | **Not covered as a design principle**; implicit in homophilic vs heterophilic splits (LLMNodeBed, Structure-Doesn't-Help) and text-rich vs titles-only (Huang; Structure-Doesn't-Help) | [2502.00829](https://arxiv.org/html/2502.00829v2); [2511.16767](https://arxiv.org/html/2511.16767v1); [2309.16595](https://arxiv.org/html/2309.16595v4) |
| Data efficiency (learning curves vs training fraction) | Partially — 2–3 regimes (semi-/supervised; low/high label rate; 30-shot); no full curves for fused vs text-only at same backbone | [2502.00829](https://arxiv.org/html/2502.00829v2); [2307.03393](https://arxiv.org/html/2307.03393v4); [2508.10115](https://arxiv.org/html/2508.10115v1) |
| Scale efficiency (small LLM + graph vs larger LLM alone) | Partially — unmatched cross-paper comparisons (GTLM 1B vs 7B InstructMol; SOG_k 3B vs GPT-4o); within-family diminishing returns (LLMNodeBed Qwen 3B–32B; Structure-Doesn't-Help 7B vs 13B) | [2605.10247](https://arxiv.org/html/2605.10247); [2602.01771](https://arxiv.org/html/2602.01771); [2502.00829](https://arxiv.org/html/2502.00829v2); [2511.16767](https://arxiv.org/html/2511.16767v1) |
| Context/token efficiency | Partially — fixed-400-token soft prompts vs text (Less is More); needle-in-graph to 65k tokens (GTLM); token Pareto for one agentic method (2026); most papers claim but do not measure | [2508.10115](https://arxiv.org/html/2508.10115v1); [2605.10247](https://arxiv.org/html/2605.10247); [2608.00542](https://arxiv.org/abs/2608.00542) |
| Parameter-matched conditions | Only RASA (67.1M across variants) and Less-is-More (400 virtual tokens for P-Tuning); benchmarks (GLBench, LLMNodeBed) use methods as published | [2602.02834](https://arxiv.org/html/2602.02834); [2508.10115](https://arxiv.org/html/2508.10115v1); [2407.07457](https://arxiv.org/html/2407.07457v4) |
| Efficiency as the *dependent variable* of an attention-component factor | **Not covered** anywhere | — |

### Inferences
- Already done: (i) proof that attention-level structure injection into a pretrained LLM is cheap (0.015% params) and effective on structure-heavy/long-context tasks (GTLM, GaLA); (ii) proof that on homophilic text-rich TAGs structure is largely redundant (Structure-Doesn't-Help, Huang, LLMNodeBed); (iii) in KGQA, mask > bias at matched params (RASA).
- Not done: the factorial itself (especially normalization and W_V factors), the deliberate complementarity stratification, learning curves, compute/parameter-matched scale comparison, and any token-budget-at-fixed-accuracy measurement across injection mechanisms. The proposed study would be the first to put efficiency on the y-axis against attention-component factors.
- Risk flagged by prior work for the proposed design: GTLM found SPD bias contributes nothing while Magnetic-Laplacian bias dominates, and RASA found bias without mask can be *worse* than vanilla — so factor levels must be chosen carefully (which bias, which mask) or main effects may wash out.

### Gaps
- Could not confirm venue status of SOG_k (ICLR 2026 per snippets, not on arXiv page), "Structure Doesn't Help" (no venue), or RASA (ICML 2026 listing exists but not cross-checked).
- MDPI 2025 SLM+graph paper and TU Delft 2026 thesis not opened (403 / 503).
