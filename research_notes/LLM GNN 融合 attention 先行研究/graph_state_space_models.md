# State-Space Models (SSMs / Mamba) on Graphs, and Whether They Have Been Fused with Pretrained Language Models (2024–2026)

Notes compiled 2026-11-02. All facts below carry a URL; items I could not verify at a URL are marked **unverified**. Titles are kept in their original (English) form. Numbers are the papers' own self-reported numbers unless stated otherwise.

**One-paragraph verdict.** Graph SSMs split into two families: (A) "sequence-ify the graph then run Mamba" (Graph-Mamba, GMN, HeteGraph-Mamba, TopoMamSurv, HSA-Net's projector), where structure enters only through the *input order* plus local GNN/PE features, and the selection gates are plain Mamba gates; and (B) "rewrite the SSM so topology enters the operator itself" (GSSC's distance kernels, GraphSSM's Laplacian-regularised recurrence, Chimera's resolvent (I−A)⁻¹, MP-SSM's recurrence over message-passing depth, GRAMA's graph-ARMA), which are permutation-equivariant by construction. Family A is linear-time and beats GraphGPS/Exphormer on LRGB but is criticised (GSSC, MP-SSM, GRAMA, GSM++) for breaking permutation equivariance and for ordering sensitivity; Family B matches or beats it while staying equivariant. **Fusion with a pretrained LM:** I found exactly one verified system that places a Graph-Mamba module between a molecular graph encoder and a pretrained LLM (HSA-Net, 2025, LLaMA-2-7B-chat / Galactica-1.3B), one KGQA pipeline where a "Graph-based Mamba" generates answers that an LLM then rewrites (AdaptQA, KBS 2025), and one fMRI paper where Mamba runs over *time* (not over graph nodes) before a LLaMA-3-8B LoRA head (NeuroMambaLLM, 2026). I found **no** paper that fuses a graph SSM with an LLM on text-attributed graphs, knowledge-graph completion, or code/AST tasks, and none that uses a Mamba-family *LLM* (Mamba, Jamba, Falcon-Mamba) to read a graph.

---

## Key Question 1 — Catalogue: what graph-SSM papers exist, how do they turn a graph into a sequence, and where does structure enter?

### Takeaway
Roughly 20 verifiable graph-SSM papers exist from 2024–2026. The two originals (Graph-Mamba, GMN) sequence nodes by degree / hop-distance and rely on random permutations plus bidirectional scans to blunt ordering effects; a second wave (GSSC 2024, GraphSSM NeurIPS 2024, GSM++ ICML 2025, MP-SSM NeurIPS 2025, GRAMA 2025, Chimera TMLR 2025) explicitly diagnoses the ordering/permutation problem and re-derives the SSM so topology enters the state transition rather than the token order.

### Cited Findings

#### Quick catalogue table

| # | Paper (year, venue) | How the graph becomes a sequence | Does structure enter SSM parameters? | Complexity claim | Domains | Text/LM? | URL |
|---|---|---|---|---|---|---|---|
| 1 | Graph-Mamba (Wang, Tsepa, Ma, B. Wang; 2024; arXiv only) | Nodes sorted by degree (+ random noise to shuffle ties); m permutations averaged at inference; GMB replaces attention inside GraphGPS, GatedGCN in parallel | No — only via input order + GatedGCN features; selection gates are standard Mamba B,C,Δ | O(L) vs O(L²) | LRGB ×5, GNN-Benchmark ×5 | None | https://arxiv.org/abs/2402.00789 |
| 2 | Graph Mamba Networks / GMN (Behrouz & Hashemi; KDD 2024) | Random-walk-sampled m-hop subgraph tokens ordered by hop (own node last); for m=0, nodes ordered by degree (PPR/k-core alternatives); bidirectional Mamba with separate weights | No — order + local encoder (GatedGCN) + optional PE/SE | linear in input size | LRGB, GNN-Benchmark, heterophilic ×4, ogbn-arxiv | None | https://arxiv.org/abs/2402.08678 ; ACM DL https://dl.acm.org/doi/10.1145/3637528.3672044 |
| 3 | GSSC — "What Can We Learn from State Space Models for Machine Learning on Graphs?" (Huang, Miao, Li; 2024) | No sequence: causal sum replaced by global permutation-equivariant sum; SSM kernel Ā^(i−j) replaced by factorisable kernel of relative graph distance via Laplacian eigenvectors | **Yes** — distance kernel K(u,v)=p_uᵀ(φ(Λ)⊙p_v) is the SSM kernel; optional data-dependent "selection" PE | O(n·m·d) | counting, LRGB, GNN-Bench, ZINC, molhiv | None | https://arxiv.org/abs/2406.05815 |
| 4 | GraphSSM — "State Space Models on Temporal Graphs: A First-Principles Study" (J. Li et al.; NeurIPS 2024) | Temporal graph snapshots as the sequence; Laplacian regulariser added to the SSM online-approximation objective; S4/S5/S6 variants | **Yes** — Laplacian term inside the SSM objective; piecewise system with mixed discretisation | — | 6 temporal-graph benchmarks | None | https://arxiv.org/abs/2406.00943 ; https://nips.cc/virtual/2024/poster/94959 |
| 5 | GSM / GSM++ — "Best of Both Worlds: Advantages of Hybrid Graph Sequence Models" (Behrouz, Parviz, Karami, Sanford, Perozzi, Mirrokni; ICML 2025, PMLR 267) | Unifying 3-stage framework (tokenise → local encode → global encode); GSM++ orders nodes by BFS/DFS over a Hierarchical Affinity Clustering tree; Mixture-of-Tokenisation router | No (ordering + hierarchical PE); hybrid Mamba→Transformer global encoder | — | synthetic graph tasks, GNN-Benchmark | None | https://arxiv.org/abs/2411.15671 ; https://proceedings.mlr.press/v267/behrouz25a.html |
| 6 | Chimera: State Space Models Beyond Sequences (Lahoti, Marwah, Puduppully, Gu; TMLR 2025) | No ordering: SSM mask L generalised to resolvent (I−A)⁻¹ of a learned, selectivity-weighted adjacency; DAGs give a linear recurrence | **Yes** — A_ij = exp(−(Δ_i+Δ_j+Δ′_ij)/3) built from node/edge selectivity; topology is the transition | linear O(\|V\|+\|E\|) on DAGs; quadratic (approx.) / cubic (exact) on general graphs | LRGB ×4, also GLUE, ImageNet | BERT-style LM on GLUE (not graph+LM fusion) | https://arxiv.org/abs/2510.12111 |
| 7 | MP-SSM — "Message-Passing State-Space Models: Improving Graph Learning with Modern Sequence Modeling" (Ceni, Gravina, Gallicchio, Bacciu, Schönlieb, Eliasof; NeurIPS 2025) | Recurrence over *message-passing depth*, not over nodes: X_{t+1}=ÂX_tW+U_{t+1}B; linear diffusion, nonlinear MLP after each block of k steps | **Yes** — Â is the transition operator; permutation-equivariant by construction | parallel closed form; "up to 1000× speed-up over GCN" claimed | property prediction, heterophilic ×5, spatio-temporal ×5, LRGB (appendix) | None | https://arxiv.org/abs/2505.18728 ; https://neurips.cc/virtual/2025/127198 |
| 8 | GRAMA: Adaptive Graph Autoregressive Moving Average Models (Eliasof, Gravina, Ceni, Gallicchio, Bacciu, Schönlieb; 2025 arXiv) | Converts static graph to a sequential (ARMA) form while preserving permutation equivariance; selective attention learns ARMA coefficients; theoretical link to selective SSMs | **Yes** (learned ARMA coefficients) | not stated on abs page | 14 synthetic + real datasets | None | https://arxiv.org/abs/2501.12732 |
| 9 | MbaGCN — "Mamba-Based Graph Convolutional Networks: Tackling Over-smoothing with Selective State Space" (He et al.; IJCAI 2025) | Message Aggregation → Selective State Space Transition → Node State Prediction layers; mechanism details not on abs page | unclear | — | node classification benchmarks | None | https://arxiv.org/abs/2501.15461 |
| 10 | DG-Mamba (Yuan, Sun et al.; AAAI 2025) | Dynamic graph treated as a state-space system; kernelised dynamic message passing; PRI self-supervised regulariser | partly (kernelised MP operator) | quadratic → linear in space/time | dynamic graphs, adversarial robustness | None | https://arxiv.org/abs/2412.08160 ; https://ojs.aaai.org/index.php/AAAI/article/view/34382 |
| 11 | STG-Mamba (L. Li, H. Wang, W. Zhang, Coster; 2024 arXiv) | Spatio-temporal graph as dynamic system; ST-S3M input-dependent feature selection; Kalman-Filtering GNN updates graph | partly (KFGN adapts structure inside the selective scan) | linear | traffic/weather forecasting ×3 | None | https://arxiv.org/abs/2403.12418 |
| 12 | DyGMamba (Z. Ding et al.; TMLR 05/2025) | Node-level SSM over each node's interaction history + time-level SSM over temporal patterns | No (temporal sequences) | — | continuous-time dynamic link prediction | None | https://arxiv.org/abs/2408.04713 |
| 13 | HeteGraph-Mamba / HGMN (Pan, Jeong, Liu, Liu; 2024 arXiv) | Projects node types to shared space; Mamba within each node type, then over a re-ordered full node set | No (ordering) | — | heterogeneous benchmarks vs 19 baselines | None | https://arxiv.org/abs/2405.13915 |
| 14 | GrassNet: State Space Model Meets Graph Neural Network (G. Zhao et al.; 2024 arXiv) | SSM runs over the *ordered graph spectrum* (eigenvalues), not nodes — a spectral filter | **Yes** (SSM is the spectral filter) | — | 9 node-classification benchmarks | None | https://arxiv.org/abs/2408.08583 |
| 15 | MOL-Mamba (Hu et al.; AAAI 2025) | Atom & Fragment "Mamba-Graph" (GNN layer + Mamba block over PE-augmented atom sequences) + Mamba-Transformer fuser for electronic info | No (ordering + PE) | — | 11 molecular datasets | None (no LM, no SMILES text stated on abs page) | https://arxiv.org/abs/2412.16483 |
| 16 | TopoMamSurv — "Graph Mamba Survival Analysis Based on Topology-Aware Ordering" (Y. Chen et al.; May 2026 arXiv) | Topology-aware ordering (TAO) replacing degree / subgraph-size sorting; bidirectional Mamba + GCN | No (ordering) | — | 5 TCGA WSI cohorts | None | https://arxiv.org/abs/2606.02602 |
| 17 | GraMO — "Graph Mamba Operator: A Latent Simulator for Interacting Particle Systems" (2026 arXiv) | state-space update proven permutation-equivariant (per search snippet) | Yes (claimed) | — | particle simulation | None | https://arxiv.org/abs/2606.09432 (**abs page not fetched; details from search snippet only**) |
| 18 | "A Comparative Study on Dynamic Graph Embedding based on Mamba and Transformers" (Pandey, Varghese, Patil, Xu; 2024/2025 arXiv) | DG-Mamba / GDG-Mamba (GIN edge conv) vs TransformerG2G on temporal snapshots | No | linear vs quadratic | UCI, Bitcoin, Reality Mining, SBM | None | https://arxiv.org/abs/2412.11293 |
| 19 | SMILES-Mamba (2024 arXiv) and IBM SMI-SSED/"MoLMamba" | Pure SMILES *string* Mamba — no graph | n/a | linear | ADMET / MoleculeNet | Yes (chemical LM), but **no graph** | https://arxiv.org/abs/2408.05696 ; https://huggingface.co/ibm/materials.smi_ssed |
| 20 | Spatio-Spectral GNNs / S²GNN (Geisler, Kosmala, Herbst, Günnemann; 2024) | **Not an SSM** — spatial + spectral filters; abs page never mentions S4/Mamba/SSM | n/a | — | LRGB peptides, million-node graphs | None | https://arxiv.org/abs/2405.19121 |

#### Per-paper detail (mechanism, structure entry, ablations)

**1. Graph-Mamba (Wang et al., arXiv 2402.00789, Feb 2024; no peer-reviewed venue listed on arXiv page).**
- Replaces the attention module in each GraphGPS layer with a "Graph-Mamba Block"; a GatedGCN MPNN runs in parallel and the two are merged by an MLP — [arXiv HTML](https://arxiv.org/html/2402.00789v1).
- Node prioritisation: nodes sorted by a heuristic, degree recommended, "more important nodes should have access to more context (i.e., a longer history of prior nodes)"; training adds uniform noise in [0,1) to the degree so equal-degree nodes shuffle; inference averages m permutations — [arXiv HTML](https://arxiv.org/html/2402.00789v1).
- Structure enters only via order + GatedGCN; the selective scan's B, C, Δ depend on input features and are argued to act as a gate filtering irrelevant context — [arXiv HTML](https://arxiv.org/html/2402.00789v1).
- Ordering ablation (Table 2, single seed; Peptides-func AP / PascalVOC F1): no permutation/prioritisation 0.6581 / 0.3105 → node-level permutation 0.6821 / 0.4193 → + degree prioritisation 0.6834 / 0.4314; eigenvector-centrality ordering worse (F1 0.3961); Louvain cluster-level permutation worse than node-level (F1 0.3802) — [arXiv HTML](https://arxiv.org/html/2402.00789v1).
- Optional "binning" of long node sequences for large graphs (Appendix E) — [arXiv HTML](https://arxiv.org/html/2402.00789v1).
- Code: github.com/bowang-lab/Graph-Mamba (link given on arXiv page) — [arXiv abs](https://arxiv.org/abs/2402.00789).

**2. GMN (Behrouz & Hashemi, arXiv 2402.08678; KDD 2024).**
- Venue: listed as "Published in: KDD (2024)" by an index, and the ACM DL entry exists at https://dl.acm.org/doi/10.1145/3637528.3672044 (fetch returned 403; the URL appeared in search results with the paper's title) — [search listing](https://cs.reviewer.ly/app/publication/a3f3e629-0634-4e40-a214-e6183def63b8).
- Recipe: (i) tokenisation by M random walks of each length 0…m, token = induced subgraph on visited nodes (m=0 gives node tokens); (ii) local encoding of each token (GatedGCN or random-walk encoder); (iii) token ordering — for m≥1 reversed hop order with own-node token last and same-length tokens shuffled; for m=0 nodes ordered by degree (centrality/k-core/PPR alternatives); (iv) bidirectional Mamba with separate forward/backward weights summed; (v) optional PE/SE and optional MPNN — [arXiv HTML v2](https://arxiv.org/html/2402.08678v2).
- Theory: Thm 4.1 random-walk sampling strictly more expressive than k-hop; Thm 4.2 universality for continuous permutation-equivariant functions with PE (relies on Wang & Xue 2023 SSM universality); Thm 4.3 with full eigenfunctions stronger than any WL; Thm 4.4 without PE/MPNN, for every k there are graphs GMNs separate that k-WL cannot — authors say power comes from sampling/encoding, "rather than Mamba itself" — [arXiv HTML v2](https://arxiv.org/html/2402.08678v2).
- Ablation (Roman-empire / Amazon-ratings / Minesweeper): full 0.8769/0.5407/0.9101; w/o bidirectional Mamba 0.8327/0.5016/0.8597 (largest drop); w/o MPNN 0.8620/0.5312/0.8983; PPR ordering 0.8612/0.5299/0.8991; w/o PE 0.8591/0.5308/0.9011 — [arXiv HTML v2](https://arxiv.org/html/2402.08678v2).
- A naive "GPS+Mamba" swap is consistently below GMN; GMN⁻ (no Transformer, no MPNN, no PE/SE) remains competitive on LRGB — [arXiv HTML v2](https://arxiv.org/html/2402.08678v2).

**3. GSSC (Huang, Miao, Li, arXiv 2406.05815 v2 Oct 2024).**
- Mechanism: SSM convolution h_i=Σ_j Ā^(i−j)B̄x_j factorises as Ā^i·Ā^(−j); on graphs the causal sum becomes a global permutation-equivariant sum and the kernel becomes a permutation-invariant function of relative graph distance (shortest-path / random-walk / heat / resistance), factorised through Laplacian eigenvectors: K(u,v)=p_uᵀ(φ(Λ)⊙p_v); output = global pooled term + separate self term; cost O(nmd) with d=32 (16 for molecules); Proposition 3.1 shows gradient ‖∂h_u/∂x_v‖ need not decay with SPD; data-dependent "selection" PE (O(nmd²)) used only for cycle counting — [arXiv HTML v2](https://arxiv.org/html/2406.05815v2).
- Explicit critique: "Graph-Mamba-I" (Wang) sorts by degree and needs random permutations, so is not permutation-equivariant to re-indexing; "Graph-Mamba-II" (GMN) runs Mamba over a root-node sequence, which "breaks permutation invariance"; authors state any tokenisation that sequentialises a graph breaks permutation symmetry and hurts generalisation; GMN numbers relegated to appendix because they "could not reproduce them without code" — [arXiv HTML v2](https://arxiv.org/html/2406.05815v2).
- Theory: provably counts at least 4-paths and 4-cycles, stronger than MPNNs — [arXiv HTML v2](https://arxiv.org/html/2406.05815v2).
- Code github.com/Graph-COM/GSSC — [arXiv HTML v2](https://arxiv.org/html/2406.05815v2). Venue: NeurIPS 2024 attribution **unverified** (no OpenReview/NeurIPS page surfaced; one index lists "CoRR 2024") — [search](https://cs.reviewer.ly/app/publication/0c770473-508e-46bc-9bde-97857815086b).

**4. GraphSSM (J. Li et al., NeurIPS 2024).** Extends SSM theory to temporal graphs by integrating structural information "via a Laplacian regularization term" into the online approximation objective; the resulting piecewise continuous-time system needs a "mixed discretization strategy for unobserved graph mutations"; GraphSSM-S4/S5/S6 variants; six temporal-graph benchmarks; claims better scaling than SOTA — [arXiv abs](https://arxiv.org/abs/2406.00943), [NeurIPS poster](https://nips.cc/virtual/2024/poster/94959).

**5. GSM / GSM++ (ICML 2025, PMLR 267:3533–3561).**
- Theory: non-causal Transformers without proper PE cannot count node colours, a recurrent model can if width ≥ #colours; SSMs are more sensitive to nearby tokens than Transformers, and both SSMs and causal Transformers can suffer representational collapse with depth; Transformers are more parameter-efficient for connectivity but recurrent models become efficient when edge ordering has bounded node locality; no tokenizer is universally best ("no free lunch" across 9 sequence models × 6 tokenizers × 7 datasets); hybrids solve some connectivity instances more efficiently than either alone — [arXiv HTML](https://arxiv.org/html/2411.15671).
- GSM++: HAC-tree tokenisation (BFS/DFS traversal), hierarchical PE from cluster-level shortest paths, GatedGCN local encoder, hybrid global encoder = Mamba followed by a Transformer block, Mixture-of-Tokenisation top-2 router; removing HAC tokenisation causes the largest ablation drop; reported better than baselines in 8/10 cases — [arXiv HTML](https://arxiv.org/html/2411.15671), [PMLR](https://proceedings.mlr.press/v267/behrouz25a.html).
- The arXiv abstract states that sorting nodes into a sequence "undermines the crucial inductive bias of permutation equivariance inherent to graphs" and that Transformers' equivariance "limits their capacity" on some graph tasks — [search summary of 2411.15671](https://arxiv.org/pdf/2411.15671).

**6. Chimera (TMLR, Oct 2025; author list includes Albert Gu).** For a directed line graph the SSM mask L equals the resolvent (I−A)⁻¹ (Prop. 3.3); Chimera builds A from edges with A_ij=exp(−(Δ_i+Δ_j+Δ′_ij)/3) (node + edge selectivity) and uses L=(I−A)⁻¹ for any topology (Neumann series sums all paths); learned row-normalisation Ψ keeps ‖A‖<1; truncating the series at the diameter preserves all edges/nodes (Prop. 4.6); complexity: cubic exact, O(log diameter) matmuls (worst-case quadratic) for the approximation, linear O(|V|+|E|) recurrence on DAGs; ~1.5× time vs Transformers; ImageNet ablation shows keeping 2-D DAG topology (77.8%) beats bidirectional 1-D flattening (76.5%) and forward-only flattening (73.8%) — [arXiv HTML](https://arxiv.org/html/2510.12111v1).

**7. MP-SSM (NeurIPS 2025).** Abstract: existing graph SSMs "apply SSM modules to sequences extracted from graphs, often compromising core properties such as permutation equivariance, message-passing compatibility, and computational efficiency"; MP-SSM runs the linear recurrence over message-passing steps with Â (sym-normalised adjacency with self-loops), nonlinearity only in a graph-agnostic MLP after k steps; enables exact sensitivity analysis of over-squashing / vanishing gradients; cites Graph-Mamba as sequencing graphs via random walks or degree sorting, "which often sacrifices permutation equivariance"; describes GraphSSM as GNN diffusion followed by an SSM as post-processing — [arXiv HTML](https://arxiv.org/html/2505.18728v1), [NeurIPS listing](https://neurips.cc/virtual/2025/127198).

**8–17.** See table; sources: GRAMA [arXiv](https://arxiv.org/abs/2501.12732); MbaGCN [arXiv](https://arxiv.org/abs/2501.15461) (IJCAI 2025, v4 Apr 2026; authors say it "may not consistently outperform all existing methods on each dataset"); DG-Mamba [arXiv](https://arxiv.org/abs/2412.08160) / [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34382) (12 baselines, adversarial robustness); STG-Mamba [arXiv](https://arxiv.org/abs/2403.12418) (code LincanLi98/STG-Mamba; "first" selective SSM for STG; lower FLOPs/inference time than attention baselines); DyGMamba [arXiv](https://arxiv.org/abs/2408.04713) (TMLR 05/2025; "even stronger superiority as the dataset requires longer temporal dependency"); HeteGraph-Mamba [arXiv](https://arxiv.org/abs/2405.13915); GrassNet [arXiv](https://arxiv.org/abs/2408.08583); MOL-Mamba [arXiv](https://arxiv.org/abs/2412.16483); TopoMamSurv [arXiv](https://arxiv.org/abs/2606.02602) (abstract: "Mamba … is sensitive to input order"; existing Graph-Mamba sorting "based on node degree or subgraph size, ignore[s] topological connectivity").

**Surveys' own catalogues (for completeness).** The Atitallah et al. survey additionally lists GraphS4mer (CHIL 2023, biosignals), BrainMamba, SpoT-Mamba (traffic), GSSM (wireless traffic), Montagna et al. (simplicial complexes), GraphMamba (Yang et al., hyperspectral), SAMBA (stock returns), GGVMamba, and others — [arXiv HTML](https://arxiv.org/html/2412.18322v1). The Wang et al. SSM survey lists GRED (Behrouz et al., 2023: shortest-distance "skip representation" sequence encoded by a linear RNN) — [arXiv HTML](https://arxiv.org/html/2404.09516).

### Inferences
- There is a clean split between "ordering-based" graph SSMs (structure = input order + local GNN) and "operator-based" graph SSMs (structure = transition/kernel). Only the operator-based family (GSSC, Chimera, MP-SSM, GRAMA, GraphSSM, GrassNet) is permutation-equivariant by construction; the ordering-based family relies on bidirectional scans and random permutations as empirical patches, and GMN's own ablation shows bidirectionality is its single most important component.
- GMN's theory attributes expressivity gains to random-walk subgraph sampling plus encoders, "rather than Mamba itself" — i.e., the SSM is primarily an efficiency device in that line of work.
- No "Graph Mamba 2" / "GMN v2" exists under those names as of Nov 2026; searches for those strings returned only the originals and generic Mamba-2/SSD material — [search](https://arxiv.org/pdf/2402.08678).
- "Spatio-Spectral GNNs" is not an SSM paper and should not be catalogued as one (its abstract never mentions S4/Mamba/SSM) — [arXiv](https://arxiv.org/abs/2405.19121).

### Gaps
- Graph-Mamba (Wang et al.) peer-reviewed venue: none found on arXiv; status appears to be arXiv-only.
- GSSC venue: NeurIPS 2024 not verifiable from any fetched page.
- GMN's ZINC results and GRIT comparison: not present in the fetched portion of the paper.
- MbaGCN, GRAMA, MOL-Mamba, HeteGraph-Mamba, GraMO: mechanism details (exact ordering rule, complexity, tables) not extracted — only abstracts/snippets were fetched.
- "Mamba-GTC" (Knowledge-Based Systems, Feb 2026, heterogeneous-graph contrastive learning with SSMs) appeared only as a search snippet — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125021951); and "FGMMa" (IEEE TFS 2026, fuzzy message passing + graph Mamba) appeared only as a snippet with no URL — **unverified**.

---

## Key Question 2 — Has any SSM-based graph model been fused with a pretrained LLM/LM?

### Takeaway
Partially, and only in narrow domains: HSA-Net (Aug 2025) uses a "Graph-Mamba projector" between a molecular GNN and LLaMA-2-7B-chat / Galactica-1.3B; AdaptQA (KBS 2025) uses a "Graph-based Mamba" answer generator followed by LLM refinement for KGQA; NeuroMambaLLM (2026) runs Mamba over fMRI *time series* (not graph nodes) before a LoRA-tuned LLaMA-3-8B. I found **no** work fusing a graph SSM with an LM on text-attributed graphs, KG completion, or code/AST graphs, and none using a Mamba-family LLM as the reader.

### Cited Findings

**HSA-Net: Hierarchical and Structure-Aware Framework for Efficient and Scalable Molecular Language Modeling (Shao, Lei, Wang, Ye, Liu; arXiv 2508.08334, Aug 2025; no venue listed).**
- Motivation: in deep GNNs features over-smooth; authors find Mamba "preserves global topology from deep layers but loses fine-grained detail in shallow layers"; Hierarchical Adaptive Projector (HAP) switches between a cross-attention projector for shallow GNN layers and "a structure-aware Graph-Mamba projector for deep layers"; Source-Aware Fusion (SAF) picks fusion experts — [arXiv abs](https://arxiv.org/abs/2508.08334).
- Graph-Mamba projector takes each GNN layer's node features H(l); uses "a graph node sorting strategy that serializes atoms by considering both molecular fragments and node degrees"; a "GraphSSM mechanism integrates structural information, such as adjacency and distance matrices" (how they enter the SSM is not spelled out); SAF output token sequence Y is passed to the LLM together with the SMILES string and instruction — [arXiv HTML](https://arxiv.org/html/2508.08334v1).
- LLMs: generalist = LLaMA-2-7B-chat (Stage 1: frozen LLM, align encoder+HAP/SAF on Mol-Instructions description split; Stage 2: LoRA on LLM, encoder frozen); specialist = Galactica-1.3B following the MolCA setup (PubChem324k pretrain, per-task finetune) — [arXiv HTML](https://arxiv.org/html/2508.08334v1).
- Results (generalist, Table 1): BLEU/METEOR 43.5/72.1 (description), 65.4/78.9 (IUPAC), property MAE 0.0049 vs best baseline LLaMo 37.9/67.1/61.1/74.1/0.0061. Specialist (Table 2): PubChem324kV2 37.5/63.5, ChEBI-20 52.2/69.9, IUPAC METEOR 72.8 vs LLaMo(Galactica-1.3B) 35.2/52.1, 49.9/64.8, 70.1 and MolCA 30.3/45.6, 44.2/57.1, 65.4; ChEBI-20 BLEU lead over XMolCap only 52.2 vs 51.1 — [arXiv HTML](https://arxiv.org/html/2508.08334v1).
- Ablation (Table 3, BLEU/METEOR): attention-only projector 37.9/67.1; Mamba-only 37.5/67.9; both 38.8/68.5; attention+SAF 40.0/69.2; Mamba+SAF 39.4/69.9; all 43.5/72.1 — Mamba alone is *not* better than cross-attention alone on BLEU; gating shifts toward Mamba in deeper layers (Fig. 5); advantage over cross-attention grows for molecules with 60+ atoms (Fig. 6); no timing/memory/FLOPs reported; no ablation isolating the node-sorting rule or the adjacency/distance inputs — [arXiv HTML](https://arxiv.org/html/2508.08334v1).

**AdaptQA — "Knowledge-based natural answer generation via effective graph learning" (Z. Liu, J. Li, Y. Huang, N. Cui, L. Pei; Knowledge-Based Systems 316:113288, 2025; DOI 10.1016/j.knosys.2025.113288).** Builds a KG from the knowledge base with adaptive multi-hop retrieval, "generates answers through the Graph-based Mamba module (GBM), effectively filtering out redundant information", then "the answers are optimized using a pre-trained large language model"; reports 94.47% (HotpotQA) and 91.38% (WikiHop); the LLM is a post-hoc refiner, not jointly trained with the graph SSM; LLM identity and GBM internals not given on the repository page — [ECU repository](https://ro.ecu.edu.au/ecuworks2022-2026/6144). **Full text not accessed; architecture details unverified.**

**NeuroMambaLLM (Torabi, Razmara, Ajorlou, Baraeinejad; arXiv 2602.13770, Feb 2026, v3 May 2026).** Dynamic latent graph inferred from raw BOLD (temporal convs + self-attention produce adjacency); "individual Mamba blocks process each ROI's time-series" — i.e., the SSM runs over *time*, not over graph nodes; projection layer maps Mamba embeddings into LLaMA-3-8B (4-bit, frozen) token space using "a fixed set of learnable brain-summary tokens" (count not stated); LoRA r=16, α=32 on attention layers; ABIDE I, 39 ROIs, accuracy 0.7212±0.0098, F1 0.6931±0.0560; authors claim to be the first to integrate latent graph learning with Mamba-LLM alignment — [arXiv abs](https://arxiv.org/abs/2602.13770), [arXiv HTML](https://arxiv.org/html/2602.13770).

**Near-misses (not graph+LM fusion):**
- CrossLLM-Mamba (arXiv 2602.22236): pretrained *biological* LMs for RNA/protein/small molecules fused by bidirectional Mamba + cross-Mamba; sequences, not graphs — [arXiv HTML](https://arxiv.org/html/2602.22236) (**search snippet only**).
- SMILES-Mamba (arXiv 2408.05696) and IBM SMI-SSED "MoLMamba" (91M PubChem SMILES, masked-LM + encoder–decoder pretraining, "linear scalability with variable sequence length") are Mamba *chemical language models* over SMILES strings with no graph input — [arXiv](https://arxiv.org/html/2408.05696v1), [Hugging Face](https://huggingface.co/ibm/materials.smi_ssed).
- MOL-Mamba (AAAI 2025) uses a Mamba-Graph + Mamba-Transformer fuser but no language model or SMILES text is mentioned on its abs page — [arXiv](https://arxiv.org/abs/2412.16483).
- "Towards Understanding What State Space Models Learn About Code" (Wu, Anand, Verma, Mezini; arXiv 2602.06774, Feb 2026): probes an SSM *code LM* (sequence input) with AST-sibling / AST-distance / DFG-edge probing tasks; finds SSMs capture syntactic/semantic structure better than Transformers in pretraining but "forget certain relations" in fine-tuning; +6 MRR on NLCodeSearch after architectural fixes — this is an SSM LM *probed for* graph structure, not an SSM graph encoder fused with an LM — [arXiv abs](https://arxiv.org/abs/2602.06774), [arXiv HTML](https://arxiv.org/html/2602.06774v1).
- Chimera trains a BERT-style LM on GLUE (+0.7 over BERT) with the same SSM-on-graph operator, but graph and language are separate experiments, not a fusion — [arXiv](https://arxiv.org/abs/2510.12111).
- Mol-LLM / M³LLM / LLaMo / MolCA use GIN-style graph encoders + projectors into LLMs (standard recipe) with no SSM — [Mol-LLM](https://arxiv.org/pdf/2502.02810), [M³LLM](https://arxiv.org/pdf/2411.04708).

**Searches run for this question (all WebSearch, 2026-11-02), with outcome:**
1. `Mamba graph "language model" LLM fusion state space graph encoder` → HSA-Net (via follow-up), CrossLLM-Mamba, AdaptQA; no TAG/LLM work.
2. `"state space" "knowledge graph" LLM Mamba` → NeuroMambaLLM only; no KG+Mamba+LLM paper.
3. `Mamba "text-attributed graph" OR "graph tokens" LLM state space 2025` → nothing relevant.
4. `knowledge graph completion Mamba state space model embedding` → no KG-completion Mamba paper (only a Rust library `oxirs-embed` with a "mamba_attention" module, not a paper) — [docs.rs](https://docs.rs/oxirs-embed/0.2.2/src/oxirs_embed/mamba_attention.rs.html).
5. `Mamba graph encoder large language model molecule alignment projector 2025` → HSA-Net.
6. `"Graph Mamba" OR "graph state space" "large language model" text-attributed graph node classification 2026` → only LLM-on-TAG surveys and SSM-graph papers separately; no intersection.
7. `MolMamba molecular Mamba SMILES graph state space` → MOL-Mamba (graph, no LM) and SMILES-Mamba / IBM MoLMamba (LM, no graph).
8. `Mamba code AST state space source code graph` → 2602.06774 (probing only).
9. `AdaptQA adaptive knowledge graph retrieval graph-based Mamba LLM` → KBS 2025 paper.

### Inferences
- The only concrete precedent for "SSM as the fusion layer between graph nodes and a transformer LM's tokens" is HSA-Net, and there the Mamba projector is (a) applied to molecular graphs of ≤ a few hundred atoms, (b) ordering-based (fragment + degree sort), and (c) not better than cross-attention on its own — its value comes from complementing attention at deep GNN layers.
- The design space "graph SSM encoder → soft tokens → LLM on text-attributed graphs / KGs / code graphs" appears unoccupied as of Nov 2026. Any such work would also inherit the permutation-equivariance critique of ordering-based graph SSMs unless it uses an operator-based variant (GSSC/Chimera/MP-SSM).
- No paper was found that uses a Mamba-family *LLM* (Mamba, Mamba-2, Jamba, Falcon-Mamba) as the text reader in a graph-LLM system; all three fusion papers use transformer LLMs (LLaMA-2/3, Galactica).

### Gaps
- AdaptQA full text (GBM architecture, LLM used) could not be accessed (ScienceDirect blocked); only the repository abstract is cited.
- No Google-Scholar "cited-by" traversal was possible with the available tools; coverage of 2026 workshop papers may be incomplete.
- Chinese-language venues were not searched.

---

## Key Question 3 — Accuracy and cost of graph SSMs vs (i) softmax graph transformers and (ii) linear / kernelised graph transformers, in the papers' own tables; where do SSMs underperform and why?

### Takeaway
On LRGB, ordering-based graph SSMs beat GraphGPS(+Transformer) and Exphormer by 2–7 points and beat the linear-attention GraphGPS variants (Performer, BigBird) by larger margins; but they lose to GRIT on CLUSTER/ZINC and to Exphormer on several datasets, and the equivariant SSMs (GSSC, Chimera, MP-SSM) match or beat them. No graph-SSM paper I found benchmarks against NodeFormer / SGFormer / Polynormer.

### Cited Findings

**(i) vs softmax graph transformers — LRGB (consolidated from Chimera's Table 4, which reproduces the others' numbers).**

| Model | Peptides-func AP↑ | Peptides-struct MAE↓ | PascalVOC-SP F1↑ | COCO-SP F1↑ |
|---|---|---|---|---|
| GraphGPS+Transformer (Rampášek) | 0.6575 | 0.2510 | 0.3689 | 0.3774 |
| GraphGPS retuned (Tönshoff) | 0.6534 | 0.2509 | 0.4440 | 0.3884 |
| Exphormer | 0.6527 | 0.2481 | 0.3975 | 0.3430 |
| Graph-Mamba (Wang) | 0.6739 | 0.2478 | 0.4191 | 0.3960 |
| GMN (Behrouz & Hashemi) | 0.7071 | 0.2473 | 0.4393 | 0.3974 |
| Chimera | 0.7021 | 0.2433 | 0.4460 | 0.3977 |
Source: [Chimera arXiv HTML](https://arxiv.org/html/2510.12111v1). Chimera notes NeuralWalker is ahead on most LRGB tasks but treats it as complementary because it resamples subgraphs — same source.

- GSSC's own LRGB/ZINC table: GSSC 0.7081 AP / 0.2459 MAE / 0.4561 F1 (PascalVOC) vs GRIT 0.6988 / 0.2460 / — and Graph-Mamba-I 0.6739 / 0.2478 / 0.4191; ZINC-12k MAE GSSC 0.064 vs GRIT 0.059 vs GraphGPS 0.070; ZINC-full 0.019 vs GRIT 0.023; ogbg-molhiv AUROC 80.35 vs GraphGPS 78.80 — [GSSC arXiv HTML](https://arxiv.org/html/2406.05815v2).
- GNN-Benchmark (GSSC table): CIFAR10 GSSC 77.642 / GRIT 76.468 / Exphormer 74.690 / Graph-Mamba-I 73.700 / GraphGPS 72.298; CLUSTER GRIT 80.026 > GSSC 79.156 > Exphormer 78.070 > GraphGPS 78.016 > Graph-Mamba-I 76.800; MNIST Exphormer 98.550 > GSSC 98.492 > Graph-Mamba-I 98.420; MalNet-Tiny GSSC 94.06 ≈ Exphormer 94.02 > GraphGPS 93.50 > Graph-Mamba-I 93.40 — [GSSC arXiv HTML](https://arxiv.org/html/2406.05815v2).
- GMN's GNN-Benchmark: MNIST Exphormer 0.9855 > GMN 0.9839; CIFAR10 GMN 0.7576 > Exphormer 0.7469 > GPS 0.7226; PATTERN GMN 0.8714 > Exphormer 0.8670; MalNet-Tiny GMN 0.9415 > Exphormer 0.9402 > GPS 0.9298 — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2).
- Graph-Mamba (Wang) short/medium datasets: CIFAR10 0.7370 vs Exphormer 0.7413; PCQM-Contact MRR 0.3395 vs Exphormer 0.3587 vs GPS 0.3442; MalNet-Tiny 0.9340 vs Exphormer 0.9422 (GPS+Transformer OOM) — the abstract's "outperforms SOTA" is not uniform — [Graph-Mamba arXiv HTML](https://arxiv.org/html/2402.00789v1).
- Heterophilic node classification (MP-SSM Table 2 reproducing others): Roman-empire MP-SSM 90.91 / Exphormer 89.03 / GMN 87.69 / GPS+Mamba 83.10 / GPS 82.00 (CO-GNN 91.57 best); Amazon-ratings GMN 54.07 / MP-SSM 53.65 / Exphormer 53.51 / GPS 53.10 / **GPS+Mamba 45.13**; Minesweeper MP-SSM 95.33 / GMN 91.01 / Exphormer 90.74 / GPS 90.63 / GPS+Mamba 89.93; Tolokers MP-SSM 85.26 / GMN 84.52 / Exphormer 83.77 — [MP-SSM arXiv HTML](https://arxiv.org/html/2505.18728v1); GMN's own Table 3 gives the same GMN numbers and Exphormer 0.8903 on Roman-empire — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2).
- Large graph ogbn-arxiv (GMN Fig. 2 table): GMN 0.7248 > Exphormer 0.7228 > GOAT 0.7196 > GatedGCN 0.7141 > NAGphormer 0.7013; GPS OOM — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2). (A third-party aggregator lists "Graph Mamba" at 60.4% on ogbn-arxiv, which conflicts with the paper's own 72.48% — treat aggregator as unreliable — [search](https://arxiv.org/pdf/2402.08678).)

**(ii) vs linear / kernelised attention graph transformers.**
- Within GraphGPS, Graph-Mamba vs Performer / BigBird attention: Peptides-func 0.6739 vs 0.6475 / 0.5854; Peptides-struct 0.2478 vs 0.2558 / 0.2842; PascalVOC 0.4191 vs 0.3724 / 0.2762; COCO 0.3960 vs 0.3761 / 0.2622; MalNet-Tiny 0.9340 vs 0.9264 / 0.9234 — [Graph-Mamba arXiv HTML](https://arxiv.org/html/2402.00789v1).
- GMN vs NAGphormer and GOAT (efficient node-level transformers) on ogbn-arxiv: see above (GMN +2.35 pts over NAGphormer, +0.52 over GOAT) — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2).
- NodeFormer / SGFormer / Polynormer: a targeted search found **no** graph-SSM paper reporting them as baselines; for context SGFormer reports 72.63% on ogbn-arxiv vs 59.90% for NodeFormer in SGFormer's own paper — [SGFormer arXiv](https://arxiv.org/pdf/2306.10759); Polynormer's tables compare to NodeFormer/DIFFormer, not to any SSM — [Polynormer arXiv](https://arxiv.org/pdf/2403.01232).

**Where SSMs underperform and the stated reasons.**
- Naively swapping Mamba into GraphGPS ("GPS+Mamba") is below GMN on every LRGB dataset and *below plain GPS* on Amazon-ratings (45.13 vs 53.10) — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2), [MP-SSM arXiv HTML](https://arxiv.org/html/2505.18728v1).
- GSSC: sequentialising tokenisation "breaks permutation symmetry, which hurts generalization"; Graph-Mamba-I trails GRIT on CLUSTER (76.8 vs 80.0) and CIFAR10 (73.7 vs 76.5) — [GSSC arXiv HTML](https://arxiv.org/html/2406.05815v2).
- GSM++ theory: SSMs are "more sensitive to nearby tokens" and prone to representational collapse with depth; Transformers are more parameter-efficient for connectivity unless the edge ordering has bounded node locality — i.e., ordering quality determines whether the recurrent model is efficient — [GSM++ arXiv HTML](https://arxiv.org/html/2411.15671).
- Graph-Mamba's ablation: without permutation/prioritisation PascalVOC F1 is 0.3105 vs 0.4314 with both — a 12-point swing from ordering alone — [Graph-Mamba arXiv HTML](https://arxiv.org/html/2402.00789v1).
- GMN ablation: removing the backward scan costs 4.4 / 3.9 / 5.0 points on Roman-empire / Amazon-ratings / Minesweeper — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2).
- Chimera's ImageNet ablation quantifies the cost of flattening topology into a 1-D scan: 77.8% (2-D DAG) vs 76.5% (bidirectional 1-D) vs 73.8% (forward 1-D) — [Chimera arXiv HTML](https://arxiv.org/html/2510.12111v1).
- MbaGCN authors concede it "may not consistently outperform all existing methods on each dataset" — [arXiv](https://arxiv.org/abs/2501.15461).

### Inferences
- The accuracy picture: ordering-based SSMs ≳ GraphGPS/Exphormer ≫ Performer/BigBird GraphGPS on LRGB, but GRIT (RRWP-based softmax GT) remains ahead on small molecular / clustering benchmarks, and equivariant SSMs (GSSC, Chimera, MP-SSM) are the current best SSM-side numbers.
- The repeated pattern (GPS+Mamba < GMN; large ordering ablations; Chimera's flattening ablation) supports the critique that most of the gain in family-A models comes from tokenisation/ordering engineering rather than the SSM primitive.
- Because no graph SSM paper benchmarks against SGFormer/Polynormer/NodeFormer, the "SSM vs kernelised GT" question on large node-classification graphs is open; the only indirect evidence is GMN vs NAGphormer/GOAT.

### Gaps
- No single table places graph SSMs, GRIT, Exphormer and SGFormer/Polynormer/NodeFormer under one protocol.
- MP-SSM's LRGB numbers are in an appendix I could not read (an aggregator lists Peptides-struct MAE 0.2458, **unverified**) — [search](https://sotaverified.org/papers/message-passing-state-space-models-improving).
- GSM++'s GNN-Benchmark table values were not extracted (only the "8/10" claim).

---

## Key Question 4 — Efficiency: memory / throughput vs graph size; data- or scale-efficiency

### Takeaway
Papers consistently report linear scaling and large constant-factor savings for Mamba-style graph models — e.g., 74% less GPU memory and 66% fewer FLOPs than GPS-Transformer (Graph-Mamba), 3.85 GB vs 36.18 GB for Exphormer on ogbn-arxiv (GMN) — but equivariant variants pay for eigendecomposition or matrix inverses, and no paper reports data-efficiency or scaling-law results.

### Cited Findings
- Graph-Mamba vs GPS-Transformer on Peptides-func: 74% lower GPU memory, 66% fewer FLOPs; vs Exphormer ≈40% lower in both; supports full graphs with about 2× the nodes GPS-Transformer handles at batch size up to 256; FLOPs/memory scale linearly vs quadratically on subsampled MalNet-Tiny (Fig. 2); binning for very long node sequences — [Graph-Mamba arXiv HTML](https://arxiv.org/html/2402.00789v1).
- GMN ogbn-arxiv: 3.85 GB GPU memory vs Exphormer 36.18 GB (one virtual node), GOAT 8.41 GB, NAGphormer 6.24 GB; ~1.2–1.3 s/epoch vs Exphormer 1.97 s, GOAT 13.09 s (GatedGCN faster at 0.68 s); MalNet-Tiny ~36–41 s/epoch vs GPS 148.99 s and Exphormer 57.24 s; memory grows linearly (Fig. 3); full eigendecomposition PE/SE can be O(n²) and become the bottleneck, hence PE/SE made optional — [GMN arXiv HTML](https://arxiv.org/html/2402.08678v2).
- GSSC synthetic scaling (1k–60k nodes, 48 GB GPU): GSSC and Graph-Mamba-I are the most efficient, others OOM at larger sizes; GRIT's RRWP preprocessing most expensive; GSSC uses top-d LOBPCG eigenpairs on GPU, linear in memory; preprocessing = 0.1% of 6.1 h on ZINC-12k, 7.6% of 1.2 h on PascalVOC-SP — [GSSC arXiv HTML](https://arxiv.org/html/2406.05815v2).
- Chimera: exact resolvent cubic; approximation O(log diameter) matmuls (worst-case quadratic); DAG recurrence linear O(|V|+|E|); ~1.5× time ratio vs Transformer — [Chimera arXiv HTML](https://arxiv.org/html/2510.12111v1).
- MP-SSM: closed-form parallel implementation claimed "up to a 1000× speedup over a standard GCN"; comparable to GCN speed even without it — [MP-SSM arXiv HTML](https://arxiv.org/html/2505.18728v1).
- DG-Mamba: kernelised dynamic message passing reduces quadratic space/time to linear — [arXiv](https://arxiv.org/abs/2412.08160).
- STG-Mamba: lower FLOPs and inference time than attention-based STG models — [arXiv](https://arxiv.org/abs/2403.12418).
- Dynamic-graph comparative study: Mamba variants give "significant computational efficiency gains on longer sequences" and beat Transformers on high-variability datasets (UCI, Bitcoin, Reality Mining) while only "competitive" on stable SBM graphs — [arXiv](https://arxiv.org/abs/2412.11293).
- DyGMamba: superiority grows "as the dataset requires longer temporal dependency" — [arXiv](https://arxiv.org/abs/2408.04713).
- HSA-Net (graph-Mamba projector in an LLM): no timing/memory/FLOPs reported; only a by-molecule-size plot where its advantage over cross-attention grows for 60+ atoms — [arXiv HTML](https://arxiv.org/html/2508.08334v1).
- Xu 2026 survey: Mamba-based spatio-temporal models show lower MAE/RMSE than Transformer-based STGNNs at longer horizons (30–60 min) — [CMES](https://www.techscience.com/CMES/v148n2/68573/html).

### Inferences
- Efficiency claims are strongest for ordering-based models on graph-level tasks (where sequences are a few hundred nodes) and for ogbn-arxiv-scale node tasks; equivariant models trade some of that for spectral preprocessing (GSSC) or inverses (Chimera).
- The cost advantage over *linear-attention* GTs (Performer/BigBird/NAGphormer/GOAT) is smaller and reported only in Graph-Mamba (≈40% vs Exphormer) and GMN (vs GOAT/NAGphormer memory).

### Gaps
- No paper reports data-efficiency (sample-complexity) or parameter-scaling results for graph SSMs.
- No throughput comparison between a graph SSM and SGFormer/Polynormer on million-node graphs.

---

## Key Question 5 — Surveys (2024–2026) covering SSMs on graphs

### Takeaway
Two dedicated graph-Mamba surveys exist (Atitallah et al., Dec 2024 arXiv; Xu, CMES 2026) and two general Mamba/SSM surveys contain graph sections (Wang et al. 2024 §III-D; Qu et al. "A Survey of Mamba" §5.2.2). None discusses LLM/graph-SSM fusion or text-attributed graphs.

### Cited Findings
- **"Exploring Graph Mamba: A Comprehensive Survey on State-Space Models for Graph Learning"** (Ben Atitallah, Ben Rabah, Driss, Boulila, Koubaa; arXiv 2412.18322, 24 Dec 2024). Self-described first comprehensive Graph-Mamba survey; taxonomy by graph type (dynamic / heterogeneous / spatio-temporal), scanning mechanism (graph selective scan, temporal, directed, bidirectional, recurrent, parallel), training strategy, and application domain; ~22 models catalogued; challenges section covers scalability, interpretability, training/convergence but **not** ordering or permutation equivariance; no LLM / text-attributed-graph content in the fetched portion — [arXiv HTML](https://arxiv.org/html/2412.18322v1), [arXiv abs](https://arxiv.org/abs/2412.18322).
- **"Graph-Mamba: A Survey of Selective State Space Models for Graph Learning"** (Guangyu Xu; Computer Modeling in Engineering & Sciences 148(2), 2026; DOI 10.32604/cmes.2026.084644; received 27 Apr 2026, published 28 Aug 2026). PRISMA screening of 117 WoS records to 81 papers; frames Graph-Mamba as a hybrid dynamical system (unified propagation equation linking message passing and SSM state updates); treats node serialisation as a permutation π that changes computational neighbourhoods ("implicit structural prior") and lists traversal / breadth / depth / centrality / spectral / geometry-aware / learned orderings; does not analyse permutation equivariance as a property; no LLM or text-attributed-graph content; KGs named only as an example of heterogeneous graphs — [CMES HTML](https://www.techscience.com/CMES/v148n2/68573/html). (The ScienceDirect mirror https://www.sciencedirect.com/org/science/article/pii/S1526149226003681 returned 403.)
- **"State Space Model for New-Generation Network Alternative to Transformers: A Survey"** (arXiv 2404.09516, 2024): §III-D "Graph" and Table I entries 18–23 cover GraphS4mer (CHIL 2023), GMN, Graph-Mamba, GRED, Gregor et al., STG-Mamba; notes citation inconsistencies — [arXiv HTML](https://arxiv.org/html/2404.09516).
- **"A Survey of Mamba"** (Qu, Ning, An, Fan, Derr, Liu, Xu, Li; arXiv 2408.01129): §5.2.2 "Graph-structured Data" cites Graph-Mamba ("graph flattening … sub-graphs into node sequences"), GMN, GSSC ("distance-based graph convolution kernels into the SSM cell"), plus Liu et al. 2024a and Li et al. 2024f; mentions ordering only as "a particular order" and does not discuss permutation invariance — [arXiv HTML](https://arxiv.org/html/2408.01129).
- Graph-side surveys that *mention* SSMs: "Large Language Models Meet Graph Neural Networks: A Perspective of Graph Mining" notes Graph-Mamba as an emerging direction (snippet only) — [arXiv](https://arxiv.org/pdf/2412.19211); "Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications" (arXiv 2510.21131) surfaced in the TAG search but with no SSM content in the snippet — [arXiv](https://arxiv.org/pdf/2510.21131).

### Inferences
- Survey coverage lags the critique literature: neither dedicated graph-Mamba survey engages with GSSC/MP-SSM/GRAMA's permutation-equivariance argument or GSM++'s theory, so a reader relying on surveys alone would miss the main open problem.
- The absence of any LLM section in all four surveys corroborates the KQ2 finding that graph-SSM × LLM fusion is essentially unexplored.

### Gaps
- "Mamba-360" survey: no results under that name; existence not confirmed.
- Only the first ~100k characters of each survey were read; later sections (e.g., Xu §4.3–6, Atitallah §6–8) may contain comparison tables not captured here.
