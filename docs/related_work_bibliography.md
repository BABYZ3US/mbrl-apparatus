# Related Work — MBRL Apparatus

*A living, cited bibliography of the methods the apparatus descends from, with per-work
analysis, per-area synthesis, and implications for research.*

**Scope.** Model-based RL methods relevant to *this* apparatus: a **latent-imagination**
trainer (actor/critic trained on rollouts imagined in a learned latent space), a
**spectral world model** (spectral features as observation; operator-style latent
dynamics; a **closed-form reward fit** in latent space), a **dual-latent** variant, the
**cf4 stabilizers** (return clipping, value-gradient clipping, non-finite-loss skipping,
clipped-double-value), a **symlog reward target** with return scaling, a **latent-capacity**
guard (a too-wide latent over-resolves the closed-form fit), and an optional **planner**.
The eventual application is **proof-reconstruction RL** (e.g. Metamath proof search).

**Maintenance.** Last refreshed **2026-06-21**. Auto-refreshed weekly by the
`weekly-bibliography-refresh` scheduled task, which searches for new work and appends to
the relevant section. Every entry is a verified citation (real arXiv/DOI/proceedings URL);
unverifiable items are marked `[UNVERIFIED]`.

**Reading guide.** The bibliography is organized by sub-area. A few cornerstone works (the
Dreamer line, TD-MPC, MuZero) legitimately recur across sections because they bear on world
modeling, stabilization, *and* planning at once; each appearance is framed for that
section's concern. The cross-cutting synthesis and the concrete research directions are at
the end.

---

## World Models & Latent-Imagination MBRL

**World Models** (David Ha, Jürgen Schmidhuber, 2018; NeurIPS 2018) — https://arxiv.org/abs/1803.10122
Trains a generative model of an RL environment in two parts: a VAE (V) that compresses observations into a latent code and a recurrent mixture-density RNN (M) that predicts the next latent, then evolves a tiny controller (C) on features from V+M. The agent can be trained *entirely inside its own learned dream* and transferred back. The conceptual seed of the apparatus's latent-imagination trainer.

**PlaNet — Learning Latent Dynamics for Planning from Pixels** (Hafner et al., 2019; ICML 2019) — https://arxiv.org/abs/1811.04551
Introduces the Recurrent State-Space Model (RSSM), splitting the latent into a deterministic recurrent path and a stochastic path, trained with multi-step "latent overshooting." Plans directly in latent space with CEM (no learned policy). The canonical learned world model the apparatus's model + dual-latent variant descend from.

**Dreamer — Dream to Control: Learning Behaviors by Latent Imagination** (Hafner et al., 2020; ICLR 2020) — https://arxiv.org/abs/1912.01603
Replaces PlaNet's online planner with an actor-critic learned purely from imagined latent rollouts, backpropagating analytic value gradients through the differentiable RSSM, with λ-returns to propagate reward beyond the horizon. The direct template for the apparatus: imagine in latent space to train actor + critic.

**DreamerV2 — Mastering Atari with Discrete World Models** (Hafner et al., 2021; ICLR 2021) — https://arxiv.org/abs/2010.02193
Swaps Gaussian latents for *categorical (discrete)* stochastic latents with straight-through gradients plus KL balancing, reaching human-level across the 55-game Atari suite from inside a learned model. Its discrete-latent design and KL-balancing tricks bear on the apparatus's dual-latent variant and representation stability.

**DayDreamer — World Models for Physical Robot Learning** (Wu et al., 2022; CoRL 2022) — https://arxiv.org/abs/2206.14176
Applies Dreamer online to four real robots, learning a quadruped gait in ~1 hour of real interaction without simulators. Evidence that imagination-trained policies hold up under the data scarcity the apparatus targets.

**TD-MPC — Temporal Difference Learning for Model Predictive Control** (Hansen, Wang, Su, 2022; ICML 2022) — https://arxiv.org/abs/2203.04955
Learns a *task-oriented, decoder-free* latent dynamics model and combines short-horizon trajectory optimization (MPPI) with a TD-learned terminal value. The prime reference for the apparatus's *optional planner* fused with a value/critic, and for value-via-TD stabilization.

**IRIS — Transformers are Sample-Efficient World Models** (Micheli, Alonso, Fleuret, 2023; ICLR 2023) — https://arxiv.org/abs/2209.00588
A discrete (VQ) autoencoder tokenizes frames and a GPT-style Transformer models the token/reward/termination sequence, casting world modeling as sequence modeling; >1.0 mean human-normalized Atari-100k without search. The tokenized-Transformer alternative to RSSM.

**TWM — Transformer-based World Models Are Happy With 100k Interactions** (Robine et al., 2023; ICLR 2023) — https://arxiv.org/abs/2303.07109
A Transformer-XL backbone over compact latent states/actions/rewards, exploiting long-range attention for credit assignment while staying compute-efficient. A counterpoint on memory/long-horizon modeling relevant to the apparatus's eventual long-proof rollouts.

**STORM — Efficient Stochastic Transformer based World Models** (Zhang et al., 2023; NeurIPS 2023) — https://arxiv.org/abs/2310.09615
Marries a Transformer sequence model with VAE-style *stochastic* latents (~126.7% mean Atari-100k, no search). Stochasticity + attention is a strong combination — relevant to a dual-/stochastic-latent model where uncertainty must be represented.

**DreamerV3 — Mastering Diverse Domains through World Models** (Hafner et al., 2023; *Nature* 640:647–653, 2025) — arXiv https://arxiv.org/abs/2301.04104 · Nature https://www.nature.com/articles/s41586-025-08744-2
A single fixed-hyperparameter Dreamer mastering 150+ tasks (first to mine diamonds in Minecraft from scratch) via **symlog** encoding/prediction, **twohot symlog** distributional reward/critic targets, **percentile return normalization**, KL balancing, and free bits. The single most load-bearing reference for the apparatus — its symlog reward target and value/return stabilizers are exactly the cf4 components.

**TD-MPC2 — Scalable, Robust World Models for Continuous Control** (Hansen, Su, Wang, 2024; ICLR 2024) — https://arxiv.org/abs/2310.16828
Scales TD-MPC into a single-hyperparameter algorithm over 100+ tasks and an 80-task 317M-parameter multitask agent, with normalization fixes for stability at scale. Reinforces the planner-plus-value design and the importance of return/representation normalization.

**Genie — Generative Interactive Environments** (Bruce, Dennis, Edwards, Parker-Holder et al., 2024; ICML 2024) — https://arxiv.org/abs/2402.15391
An 11B foundation world model trained unsupervised from unlabeled video, with a learned **latent action model** and autoregressive dynamics enabling action-controllable generated worlds. A frontier example of latent-action/dynamics learning when explicit action labels are scarce.

**DIAMOND — Diffusion for World Modeling** (Alonso et al., 2024; NeurIPS 2024 Spotlight) — https://arxiv.org/abs/2405.12399
Trains the agent inside a *diffusion-based* world model predicting frames in continuous space, preserving detail token compression discards (record 1.46 mean Atari-100k for in-world-model agents). The diffusion alternative to RSSM/Transformer dynamics.

**TWISTER — Transformer World Models with Contrastive Predictive Coding** (Burchi, Timofte, 2025; ICLR 2025) — https://arxiv.org/abs/2503.04416
Trains the world model with action-conditioned Contrastive Predictive Coding over long horizons instead of next-state prediction, setting a search-free Atari-100k record (162% mean). Highlights representation objectives beyond reconstruction for accurate long-horizon imagination.

**DALI — Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization** (Röder, Benad, Eppe, Banerjee, 2025; NeurIPS 2025) — https://arxiv.org/abs/2508.20294
Integrates a self-supervised forward-dynamics encoder into the DreamerV3 architecture to infer a *latent context* from interaction history, conditioning both world model and policy so imagined rollouts stay physically consistent (perturbing a gravity-encoding latent dimension changes the dream plausibly), enabling zero-shot generalization across contextual-MDP variations with a proof that the encoder is necessary for efficient context inference. Speaks directly to the apparatus's latent-imagination trainer and its **dual-latent** variant — read as a separately-inferred context latent alongside the dynamics latent — and to keeping imagined rollouts well-behaved under distribution shift.

*Planner-lineage anchor:* **MuZero** (Schrittwieser et al., 2020; Nature 588:604–609) — https://arxiv.org/abs/1911.08265 — learns a value/policy/reward-predictive latent model and plans with MCTS; with **EfficientZero** (Ye et al., 2021) — https://arxiv.org/abs/2111.00210 — defines the search-based branch TD-MPC contrasts against.

### Synthesis
The field has converged on one recipe — **learn a compact latent world model, then improve behavior by imagining rollouts inside it** — pioneered by World Models and PlaNet/RSSM and crystallized by the Dreamer line (stochastic latent dynamics + an actor-critic trained on imagined trajectories). The *form* of the dynamics model remains the active design axis (RSSM, discrete-token Transformers, diffusion, SSM memory). A second convergence is **stabilization machinery**: DreamerV3's symlog targets, twohot critics, KL balancing, and percentile return normalization are now near-standard, echoed from the planning side by TD-MPC2. The planning-vs-imagination split has softened (TD-MPC fuses latent planning with TD bootstrapping). Open frontiers: representation objectives beyond reconstruction, long-horizon memory/credit assignment, controllable dynamics from action-free data, and — most relevant here — porting these vision/control-centric methods to **discrete, combinatorial, sparse-reward symbolic domains** where "imagination" rolls out structured states, not pixels.

---

## Spectral & Koopman Methods for Dynamics and RL

The unifying premise — that strongly nonlinear evolution becomes linear in the right (often spectral) coordinates, where a linear operator advances the state and quantities of interest are read off cheaply — is the same premise underlying the apparatus's spectral world model with operator-style latent dynamics and closed-form latent reward fitting.

### Koopman & operator-theoretic dynamics

**Spectrum of the Koopman Operator, Spectral Expansions in Functional Spaces, and State-Space Geometry** (Igor Mezić, 2020; *J. Nonlinear Science*) — https://arxiv.org/abs/1702.07597
Modern spectral theory of the Koopman operator (Kato-style decomposition; generalized/open eigenfunctions); characterizes stable/unstable/center manifolds as joint zero-level sets of Koopman eigenfunctions. The theoretical backbone for treating a learned latent operator's eigenstructure as the carrier of dynamics — i.e. spectral features as the natural coordinates for a linear-in-latent world model.

**Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition** (Takeishi, Kawahara, Yairi, 2017; NeurIPS) — https://arxiv.org/abs/1710.04340
A data-driven neural method that learns observables (a Koopman-invariant subspace) by minimizing one-step linear-prediction residual; at optimum the observables span a subspace on which dynamics are exactly linear. A template for learning the spectral embedding in which latent dynamics are linear — which a closed-form reward fit then exploits.

**Deep learning for universal linear embeddings of nonlinear dynamics** (Lusch, Kutz, Brunton, 2018; *Nature Communications*) — https://arxiv.org/abs/1712.09707
A parsimonious autoencoder discovers Koopman eigenfunctions that globally linearize dynamics on a low-dimensional manifold, with an auxiliary network for continuous/parametric spectra. The canonical "deep Koopman" recipe for a spectral latent with linear evolution — the architecture a spectral world model adapts, swapping reconstruction for a latent reward fit.

**Deep Variational Koopman Models** (Morton, Witherden, Kochenderfer, 2019; IJCAI) — https://arxiv.org/abs/1902.09742
Learns *distributions* over Koopman observations that propagate linearly, yielding a distribution over linear models for long-horizon prediction, folded into uncertainty-aware control. Shows how a linear latent (Koopman) model supports planning and how to attach uncertainty to operator-style dynamics.

**Linear predictors for nonlinear systems: Koopman operator meets MPC** (Korda, Mezić, 2018; *Automatica*) — https://arxiv.org/abs/1611.03537
Lifts controlled nonlinear dynamics into a space where evolution is approximately linear (finite Koopman approx via EDMD); Koopman-MPC has the cost of linear MPC while beating local/Carleman linearization. Establishes that a linear lifted operator with control inputs suffices for efficient planning — motivating operator-style latent dynamics for action-conditioned MBRL.

**Efficient Dynamics Modeling in Interactive Environments with Koopman Theory** (Mondal, Panigrahi, Rajeswar, Siddiqi, Ravanbakhsh, 2024; ICLR) — https://arxiv.org/abs/2306.11941
A *diagonalized* Koopman operator linearizes dynamics in a high-dim latent, parallelizing long-range action-conditioned rollout via convolution, with stability/gradient analysis; plugs into model-based planning and model-free RL. A recent RL-native operator-style latent world model — almost exactly the apparatus's design point, minus the closed-form reward fit.

**ResKoopNet — Learning Koopman Representations for Complex Dynamics with Spectral Residuals** (Xu, Shao, Logothetis, Shen, 2025; ICML 2025) — https://arxiv.org/abs/2501.00701
Learns neural Koopman observables by directly minimizing the *spectral residual*, recovering a more accurate and complete operator spectrum (including continuous spectra and high-dimensional systems) and overcoming the "spectral inclusion" limitation of residual-DMD, with theoretical guarantees. Sharpens the very tool the apparatus leans on — a faithful latent operator whose eigenstructure carries the dynamics — by improving how completely that spectrum can be learned from data rather than filtered from a precomputed set.

### Spectral representations in reinforcement learning

**Proto-value Functions: A Laplacian Framework** (Mahadevan, Maggioni, 2007; JMLR) — https://jmlr.org/papers/v8/mahadevan07a.html
Task-independent basis functions from the low-order eigenfunctions of the state-transition graph Laplacian, used inside Representation Policy Iteration (Nyström extension to continuous domains). The foundational argument that the transition-operator spectrum yields the right basis for value functions — i.e. rewards/values are cheaply fit in a spectral coordinate system.

**The Laplacian in RL: Learning Representations with Efficient Approximations** (Wu, Tucker, Nachum, 2019; ICLR) — https://arxiv.org/abs/1810.04586
A scalable, model-free method to approximate Laplacian eigenvectors of the policy-induced transition graph as state representations, beyond tabular settings. The practical bridge between operator spectra and deep-RL-sized spectral features.

**Successor Features for Transfer in RL** (Barreto et al., 2017; NeurIPS) — https://arxiv.org/abs/1606.05312
Factorizes value into successor features (expected discounted feature occupancy) × a reward-weight vector, decoupling dynamics from reward; a new reward reduces to fitting the linear weight vector. The cleanest precedent for **closed-form / linear reward fitting** on a fixed dynamics representation — directly analogous to the apparatus's closed-form latent reward fit.

**Deep Successor Reinforcement Learning** (Kulkarni, Saeedi, Gautam, Gershman, 2016) — https://arxiv.org/abs/1606.02396
A deep successor representation splitting value into a successor map × a linear reward predictor, trained end-to-end. The deep instantiation of dynamics-vs-reward factorization with a linear reward head — echoed by closed-form latent reward fitting.

**Spectral Decomposition Representation for RL (SPEDER)** (Ren, Zhang, Lee, Gonzalez, Schuurmans, Dai, 2023; ICLR) — https://arxiv.org/abs/2208.09515
Derives state-action representations from a spectral decomposition of the stochastic transition operator (policy-independent), under which transition probabilities are inner products and reward/value are *linear* in the representation. The most direct modern statement of the apparatus's theoretical core: spectral features in which dynamics factorize and reward is linear.

**Diffusion Spectral Representation for RL (Diff-SR)** (Shribak, Gao, Li, Xiao, Dai, 2024; NeurIPS) — https://arxiv.org/abs/2406.16121
Extracts sufficient spectral representations for value functions in MDPs/POMDPs via the diffusion–energy-based-model link, gaining expressiveness while bypassing slow sampling at inference. Shows expressive generative dynamics can be distilled into a spectral representation with linearly-recoverable value.

**Spectral Representation-based Reinforcement Learning** (Gao, Sun, Li, Schuurmans, Dai, 2025; arXiv) — https://arxiv.org/abs/2512.15036
From the spectral decomposition of the transition operator, derives a single framework unifying *latent-variable* and *energy-based* operator structures into spectral state-action features in which value is linear — each structure implying a concrete extraction algorithm and a corresponding RL method — with a provable extension to POMDPs (validated on 20+ DeepMind Control tasks against model-free and model-based baselines). The most recent and direct successor to SPEDER (shared authors Schuurmans, Dai) and the closest current statement of the apparatus's core bet: spectral features in which dynamics factorize and reward/value are linear, now reaching partial observability.

### Dynamic Mode Decomposition (DMD) and ML variants

**Extending DMD: A Data-Driven Approximation of the Koopman Operator (EDMD)** (Williams, Kevrekidis, Rowley, 2015; *J. Nonlinear Science*) — https://arxiv.org/abs/1408.4408
Regresses dynamics onto a dictionary of nonlinear observables, producing finite Koopman approximations with eigenvalues/eigenfunctions/modes (kernel variant scales via random Fourier features/Nyström). The algorithmic ancestor of learned-dictionary Koopman models — and a reminder the operator can be fit in closed form by linear regression once features are fixed.

**Dynamic Mode Decomposition and Its Variants** (Schmid, 2022; *Annual Review of Fluid Mechanics*) — https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-030121-015835
A consolidated review from snapshot-based DMD through optimized, sparsity-promoting, control-aware (DMDc), and randomized/ML variants, framed via the Koopman connection. Situates DMD as the practical route to a system's spectral modes — the classical counterpart to learned spectral observations.

### Synthesis
A single idea recurs: the operator governing a system — Koopman for deterministic dynamics, the transition/Laplacian operator for MDPs — is linear, and its *spectrum* furnishes coordinates in which dynamics evolve linearly and quantities of interest become linear functionals. DMD/EDMD show the operator is recoverable by closed-form linear regression once observables are fixed; deep-Koopman methods show the observables themselves can be *learned*. In parallel, spectral RL (proto-value functions, the Laplacian representation, successor features/DSR, SPEDER, Diff-SR, and the unified latent-variable/energy-based framework of Gao et al. 2025) establishes the complementary half: in the right spectral representation, value and reward are linear, so reward fitting collapses to solving for a weight vector — recently with a provable extension to partial observability. The apparatus sits exactly at this intersection — spectral features as observations, an operator-style latent transition, and a closed-form latent reward fit. Open: features that are simultaneously good for prediction *and* linear reward recovery; controlling spectral error over long horizons; stochastic/partially-observed dynamics; and extending operator-spectral models from continuous control to **structured graph/sequence objects**, where the spectrum is over discrete combinatorial state spaces.

---

## Value/Return Estimation & Training Stabilization

The apparatus's **cf4 stabilizers** — return clipping, value-gradient clipping, non-finite-loss skipping, clipped-double-value, plus a symlog reward target and return scaling — descend from a small, well-studied set of value-learning failure modes.

**Double Q-learning** (van Hasselt, 2010; NeurIPS) — https://papers.nips.cc/paper/3964-double-q-learning
The max operator used for both selection and evaluation amplifies noise into a positive (overestimation) bias; the fix uses two independent estimators, one to select and one to evaluate. The conceptual ancestor of the apparatus's **clipped-double-value** critic.

**Deep RL with Double Q-learning (Double DQN)** (van Hasselt, Guez, Silver, 2016; AAAI) — https://arxiv.org/abs/1509.06461
Shows DQN overestimates and decouples selection (online net) from evaluation (target net). Motivates the clipped-double-value lineage and a separate evaluation network to keep value targets honest under approximation error.

**Addressing Function Approximation Error in Actor-Critic Methods (TD3)** (Fujimoto, van Hoof, Meger, 2018; ICML) — https://arxiv.org/abs/1802.09477
Introduces *Clipped Double Q-learning* — two critics, take the **minimum** as the bootstrap target — plus delayed updates and target smoothing. The most direct antecedent of the apparatus's **clipped-double-value** critic.

**Soft Actor-Critic (SAC)** (Haarnoja et al., 2018; ICML) — https://arxiv.org/abs/1801.01290
A max-entropy off-policy actor-critic that adopts min-of-two-(target)-critics for the soft Bellman target, establishing it as a default stabilizer. Reinforces the clipped-double-value choice.

**Human-level control through deep RL (DQN)** (Mnih et al., 2015; *Nature*) — https://www.nature.com/articles/nature14236
Introduces the periodically-updated **target network** (fixed bootstrap target between syncs) and experience replay. The target-network idea underlies the slow-moving value targets that stabilize the apparatus's critic.

**On the difficulty of training Recurrent Neural Networks** (Pascanu, Mikolov, Bengio, 2013; ICML) — https://arxiv.org/abs/1211.5063
Proposes the now-standard **gradient-norm clipping** heuristic against exploding gradients. Direct ancestor of the apparatus's **value-gradient clipping** stabilizer.

**Learning values across many orders of magnitude (PopArt)** (van Hasselt, Guez, Hessel, Mnih, Silver, 2016; NeurIPS) — https://arxiv.org/abs/1602.07714
Adaptively normalizes learning targets (running mean/variance, compensated output layer) so value learning is scale-invariant, removing the need for crude reward clipping. The canonical reference for the apparatus's **return scaling** — a principled alternative/complement to hard **return clipping**.

**A Distributional Perspective on RL (C51)** (Bellemare, Dabney, Munos, 2017; ICML) — https://arxiv.org/abs/1707.06887
Models the full return distribution as a softmax over value atoms with a projected categorical Bellman (cross-entropy) loss, empirically stabilizing value learning. The origin of the **two-hot / categorical value representation** that DreamerV3's symexp-twohot critic — and the apparatus's value head — descend from.

**Dream to Control (Dreamer)** (Hafner et al., 2020; ICLR) — https://arxiv.org/abs/1912.01603
Defines the value-estimation setting the apparatus operates in: a critic trained on imagined λ-returns. The base architecture into which the cf4 stabilizers are inserted.

**Mastering Atari with Discrete World Models (DreamerV2)** (Hafner et al., 2021; ICLR) — https://arxiv.org/abs/2010.02193
Refines the critic-on-imagined-returns loop (λ-returns, target critic) at human-level Atari on one GPU. Directly relevant to the clipped-double-value and target-critic machinery applied to imagined returns.

**Mastering Diverse Domains through World Models (DreamerV3)** (Hafner et al., 2023; *Nature* 2025) — https://arxiv.org/abs/2301.04104 · https://www.nature.com/articles/s41586-025-08744-2
A single config across 150+ tasks via **symlog** transforms, a **symexp two-hot** distributional reward/critic loss, **percentile return normalization**, KL balancing, and an EMA-regularized critic. The most direct parent of the apparatus's **symlog reward target**, **return scaling**, and twohot value head — cf4 is essentially a hardened extension of this stabilizer set.

**Stop Regressing: Training Value Functions via Classification (HL-Gauss)** (Farebrother et al., 2024; ICML) — https://arxiv.org/abs/2403.03950
Replacing MSE value regression with categorical cross-entropy (HL-Gauss histogram) markedly improves scalability/stability across Atari, offline RL, robotics, and Transformers, partly via better-behaved gradient norms. Validates the apparatus's two-hot/symlog value head and complements value-gradient clipping at the loss level.

**Deep RL and the Deadly Triad** (van Hasselt et al., 2018) — https://arxiv.org/abs/1812.02648
Dissects how function approximation + bootstrapping + off-policy learning can drive value estimates to diverge to unbounded magnitudes, and which design choices provoke/mitigate it. The diagnostic framing for *why* value targets blow up — direct motivation for the apparatus's defensive **return clipping**, **non-finite-loss skipping**, and gradient clipping.

**Overestimation, Overfitting, and Plasticity in Actor-Critic** (Nauman et al., 2024; ICML) — https://arxiv.org/abs/2403.00514
A large-scale study finding **layer normalization** curbs Q-value overestimation more effectively than several purpose-built methods. Contextualizes the clipped-double-value critic within the modern toolkit and suggests a complementary normalization-based defense worth testing in the apparatus.

**MAD-TD — Model-Augmented Data Stabilizes High Update Ratio RL** (Voelcker, Hussing, Eaton, Farahmand, Gilitschenski, 2025; ICLR 2025) — https://arxiv.org/abs/2410.08896
Traces high-update-ratio instability to value functions failing to generalize to unobserved on-policy actions, and stabilizes training by mixing in a *small* amount of on-policy data generated from a learned world model, markedly curbing value overestimation on hard DeepMind Control tasks. A model-based stabilizer complementary to the cf4 family: where clipped-double-value attacks overestimation at the bootstrap target, MAD-TD attacks its data-coverage cause — and the Farahmand co-authorship ties it back to the value-aware model learning (IterVAML) the apparatus's latent reward fit descends from.

### Synthesis
These works target intertwined failure modes: **overestimation** from maximizing over noisy estimates (Double Q → TD3/SAC), **scale sensitivity / heavy-tailed returns** (PopArt, DreamerV3 percentile norm + symlog), **gradient pathologies** (norm clipping; classification/HL-Gauss reframing), and outright **divergence** from the deadly triad (target networks, EMA critics). The cf4 family is one stabilizer per failure mode: clipped-double-value ← Double-Q → TD3/SAC; symlog target + return scaling ← PopArt + DreamerV3; value-gradient clipping ← Pascanu norm clipping; return clipping + non-finite-loss skipping ← pragmatic deadly-triad-aware guards against unbounded blow-ups and numerical faults when training a critic on imagined returns. In effect, cf4 hardens the Dreamer line by layering the classical overestimation/divergence toolkit on top of DreamerV3's normalization/transform stabilizers, with explicit numerical-robustness guards the upstream literature usually leaves to implementation detail.

---

## Latent Capacity, Regularization & Planning

The through-line for the apparatus: a model that is *too expressive* spends capacity reconstructing irrelevant detail (or over-resolving the reward fit), whereas a model trained to be useful for value-based planning can be deliberately smaller — the argument for regularizing latent dimensionality in a spectral latent and then searching in it.

### Planning with learned models

**MuZero — Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model** (Schrittwieser et al., 2020; *Nature*) — https://arxiv.org/abs/1911.08265
Couples MCTS with a learned latent model whose only outputs are what planning needs — policy, value, reward — never trained to reconstruct observations (a *value-equivalent*, not generative, state). The canonical demonstration that a latent model can be searched without modeling observation dynamics; motivates the apparatus's planner-over-a-learned-latent-model design.

**Sampled MuZero — Learning and Planning in Complex Action Spaces** (Hubert et al., 2021; ICML) — https://arxiv.org/abs/2104.06303
Extends MuZero to large/continuous action spaces by sampling a subset of actions and running corrected sample-based policy iteration/MCTS. Proof states branch into effectively combinatorial action sets, so sampled-action search is the natural planning mechanism for the proof-reconstruction setting.

**EfficientZero — Mastering Atari Games with Limited Data** (Ye et al., 2021; NeurIPS) — https://arxiv.org/abs/2111.00210
Makes MuZero sample-efficient via a self-supervised *consistency* loss on latent dynamics, value-prefix prediction, and off-policy correction (~194% mean human on Atari-100k). The consistency loss is a representation regularizer that keeps the latent well-conditioned under scarce, sparse reward.

**EfficientZero V2** (Wang et al., 2024; ICML Spotlight) — https://arxiv.org/abs/2403.00564
Generalizes the recipe to discrete + continuous actions and visual + low-dim inputs (Gumbel-search exploration; sampling for continuous control), surpassing DreamerV3 on 50/66 limited-data tasks. Evidence that learned-model search generalizes across action types under tight data budgets.

**PETS — Deep RL in a Handful of Trials using Probabilistic Dynamics Models** (Chua et al., 2018; NeurIPS) — https://arxiv.org/abs/1805.12114
An *ensemble* of probabilistic dynamics models (separating epistemic/aleatoric uncertainty) planned with MPC + CEM over particle-propagated trajectories. The reference design for sampling-based planning in a learned model, using uncertainty to keep the planner from exploiting model error.

**TD-MPC** (Hansen, Wang, Su, 2022; ICML) — https://arxiv.org/abs/2203.04955
Plans in a learned *task-oriented latent* (reward/value-shaped, not reconstructive) with MPPI/CEM + a TD-learned terminal value. The closest architectural analogue to the apparatus: a reward/value-shaped latent that is both regularized toward usefulness *and* searched by a sampling planner.

**TD-MPC2** (Hansen, Su, Wang, 2024; ICLR) — https://arxiv.org/abs/2310.16828
Hardens TD-MPC with normalization/regularization so one hyperparameter set works across 104 tasks, scaling to a 317M multitask agent. Evidence latent-space planning is robust and scalable when the representation is properly regularized.

**AlphaProof — Olympiad-level Formal Mathematical Reasoning with Reinforcement Learning** (Hubert, Mehta, Sartran et al., 2025; *Nature*) — https://www.nature.com/articles/s41586-025-09833-y
An AlphaZero-style agent that learns to find *formal* Lean proofs through reinforcement learning over millions of auto-formalized problems, combining proof search with inference-time ("test-time") RL on generated problem variants, as part of a system that reached silver-medal level at the 2024 IMO. The clearest realization to date of the apparatus's application target — RL plus search for proof reconstruction — but with the Lean kernel as a *ground-truth* environment and **no** learned world model, which is exactly the gap a learned spectral world model would fill: imagining and planning over proof states without exhaustive verifier calls (cf. Sampled MuZero's sampled-action search above).

### Representation regularization & value-equivalence

**The Value Equivalence Principle for MBRL** (Grimm, Barreto, Singh, Silver, 2020; NeurIPS) — https://arxiv.org/abs/2011.03506
Two models are equivalent if they induce the same Bellman updates over a set of policies/value functions; limited capacity is better spent on models *useful for planning* than on predicting transitions. The theoretical backbone for the apparatus's claim that a smaller, reward-shaped latent need not lose planning performance — so regularizing latent capacity is *justified*, not merely tolerated.

**Value Prediction Networks (VPN)** (Oh, Singh, Lee, 2017; NeurIPS) — https://arxiv.org/abs/1707.03497
Abstract states trained to make option-conditional *value* predictions rather than reconstruct observations, then unrolled for planning. A precursor showing value-predictive latents both regularize the model and improve planning — the pattern the apparatus formalizes.

**DeepMDP — Learning Continuous Latent Space Models for Representation Learning** (Gelada et al., 2019; ICML) — https://arxiv.org/abs/1906.02736
Learns a latent MDP from just latent reward + latent next-state prediction, proven to bound representation quality and connected to *bisimulation*. Theory for *why* a compact latent suffices and what it must preserve — a lens for choosing the regularizer on a spectral latent.

**Deep Bisimulation for Control (DBC)** (Zhang et al., 2021; ICLR) — https://arxiv.org/abs/2006.10742
Trains an encoder so latent distances match *bisimulation* distances, collapsing task-irrelevant variation without reconstruction. A concrete regularizer that *reduces effective latent capacity* — the explicit antidote to a latent that "over-resolves" irrelevant detail.

**Iterative Value-Aware Model Learning (IterVAML)** (Farahmand, 2018; NeurIPS) — https://papers.nips.cc/paper/8121-iterative-value-aware-model-learning
Replaces maximum-likelihood model fitting with a loss weighting model errors by their effect on the current value function. Decision-aware model learning — precisely the mechanism that prevents a latent from over-fitting the reward/value target on dimensions planning never uses.

**Deep Variational Information Bottleneck (VIB)** (Alemi et al., 2017; ICLR) — https://arxiv.org/abs/1612.00410
A tractable variational IB that compresses the representation while retaining task-predictive information, improving generalization/robustness. The information-theoretic justification and a concrete penalty for bounding latent *capacity* directly — a candidate for bounding the spectral latent's width.

### Capacity, self-prediction & overfitting

**Self-Predictive Representations (SPR)** (Schwarzer et al., 2021; ICLR) — https://arxiv.org/abs/2007.05929
Learns a latent transition model by predicting its *own* future latents (EMA target encoder), no reconstruction or negatives, plus augmentation-consistency (SOTA Atari-100k). A lightweight regularizer toward temporal coherence for a learned latent that is later searched.

**DreamerV3** (Hafner et al., 2023) — https://arxiv.org/abs/2301.04104
Its categorical latents and KL balancing are practical capacity-regularization mechanisms for latent-imagination systems (see also §1, §3).

**Mitigating Planner Overfitting in MBRL** (Arumugam, Abel, Asadi, Gopalan, Grimm, Lee, Lehnert, Littman, 2018) — https://arxiv.org/abs/1812.01129
Names *planner overfitting* — the planner exploiting errors in an imperfect model — and proposes regularizers (on horizon, on plans considered) that reduce it. Directly relevant to the apparatus's optional planner: even a well-regularized latent can be exploited by an over-strong search, so planner-side regularization complements latent-capacity control.

### Synthesis
Model **capacity**, **value-equivalence**, and **planning** pull against one another. Reconstructive world models spend capacity modeling everything (including control-irrelevant detail), which wastes budget and — when searched — invites *planner overfitting* and exploitation of hallucinated dynamics. The value-equivalence line (Grimm et al.; VPN; IterVAML; DeepMDP/DBC; TD-MPC) resolves this by training the latent to preserve only what Bellman updates need, shrinking the model and giving a principled knob on latent dimensionality — exactly the apparatus's claim that an over-wide latent over-resolves the closed-form reward fit and should be regularized. But value-equivalence isn't free: too small/mis-shaped a latent loses structure a *planner* needs (MuZero/Sampled-MuZero/EfficientZero show strong search demands a rich-enough latent). The implication for a regularized **spectral** latent: size/regularize it toward reward-/value-equivalence (preserve the spectrum driving reward and planning, discard task-irrelevant modes) and pair it with a sampling planner (CEM/MPPI as in TD-MPC, or sampled-action MCTS) whose horizon/strength is itself regularized.

---

## Cross-cutting synthesis & implications for research

The apparatus occupies an under-explored corner: it fuses the **Dreamer latent-imagination lineage** (§1) with the **spectral/Koopman + spectral-RL lineage** (§2). Most world models commit to *nonlinear* latent dynamics (RSSM, Transformer, diffusion); the apparatus instead commits to **operator-style (linear/spectral) latent dynamics with a closed-form reward fit**. Its nearest neighbors each cover only part of this: deep-Koopman dynamics (Lusch; Mondal et al.) give the linear latent operator but no imagination-trained policy; successor features / SPEDER give the linear reward recovery but not a world model for imagination; value-equivalent latent models (MuZero, TD-MPC, DeepMDP) give planning-shaped latents but with learned nonlinear dynamics and no closed-form fit. The combination — *spectral observation + operator latent + closed-form latent reward + latent imagination* — appears to be the apparatus's distinctive bet.

Concrete implications and research directions:

1. **Linear-latent expressivity is the central empirical question.** Characterize when a linear/spectral latent operator is expressive enough for imagination-based policy learning versus when nonlinearity is required. The apparatus's **dual-latent** variant is naturally read as the hedge (a stochastic/nonlinear path alongside the operator path); §1's DreamerV2/STORM and §2's diagonalized-Koopman (Mondal et al.) bracket the design space to test against.

2. **The cf4 stabilizers can likely be simplified by adopting upstream principled forms.** Two literature-backed swaps worth ablating: PopArt-style *adaptive* return normalization in place of (or before) hard **return clipping**; and **layer-norm critics** (Nauman et al. 2024 found layernorm beats purpose-built overestimation fixes), which could reduce reliance on clipped-double-value. The numerical guard (non-finite-loss skipping) has no clean upstream equivalent and should stay.

3. **For a spectral latent, the natural capacity knob is the retained spectral mode count.** Rather than generic latent-dim regularization, truncating the operator's spectrum is a cleaner, interpretable capacity control — and it operationalizes the value-equivalence principle (§4): keep the modes that drive reward/value, drop the rest. Bisimulation (DBC), VIB, and IterVAML supply the objective for *which* modes matter.

4. **Planning should be regularized jointly with the latent.** Sampled MuZero's sampled-action search is the right primitive for the combinatorial action sets of proof reconstruction; Arumugam et al.'s planner-overfitting result says the planner's horizon/strength must be co-regularized with the (residual-error-bearing) latent model.

5. **The frontier is the symbolic port.** Almost all of this work is vision/continuous-control. Proof reconstruction is discrete, sparse-reward, and combinatorial. AlphaProof (Hubert et al. 2025) shows RL with search over a formal Lean proof environment already reaches silver-medal IMO level — but it treats the verifier as a ground-truth environment and learns no world model, exactly the gap a learned spectral world model would fill by enabling imagination and planning over proof states without exhaustive verifier calls. The apparatus's "spectral = observation" choice connects directly to **graph spectral theory**: the Laplacian/transition spectrum of proof-state graphs is the discrete analogue of the Koopman spectrum, and successor-feature / SPEDER-style linear reward recovery is the closest precedent for a closed-form reward fit over discrete states. Defining the spectral observation and operator over Metamath proof-state graphs — and testing whether linear reward recovery holds there — is the highest-leverage open problem for this program.

---

*Maintenance: append new work to the relevant section with the same entry format (**Title** (Authors, Year, Venue) — URL — summary — relevance) and update the per-section synthesis + the cross-cutting implications when a finding shifts them. Keep citations verified.*
