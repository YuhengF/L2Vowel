# L2 Vowel Transfer Modeling: From Concept to Implementation

**Original prompt:**

> I'm framing up the core skeleton of my question and think about some proper simple ML/NN models to solve it. So as you can learn from my proposal etc, my ultimate goal is to predict the main features (if not directly output the whole spectrogram) of L2 ENG vowels given the L1 vowels. I'm trying to divide the problem into steps. So I first pool all the major acoustic features of vowels (F1, 2, 3, duration, etc etc) to form a higher dimensional space (higher than F1-F2 alone), then I do clustering and some proper dimension reduction to locate the most expressive dimension combinations. I do this for both L1 languages, L2 ENGs, and also L1 ENG. So I end up having in a N dimensional space some clusters of points each corresponding to one vowel in either L1, or L2. I can then represent each cluster using its centroid coordinate and ellipsoid radius etc (we can even start from sphere). Now here is the key part, I want to conceptualize my core modeling question as (it might be not so good so feel free to improve it): I want to learn the force coming from L1 (non-english) clusters that push or pull or whatever transformation that move/creates a new coordinate/shape for each english vowel from its original space. That's basically a mapping from native english to L2 english.

## Assessment of the "Force Field" / Conditional Mapping Concept

Your intuition is linguistically well-grounded -- it maps directly onto established SLA phonetics theory:

- **Flege's Speech Learning Model (SLM):** L2 categories are "pulled" toward similar L1 categories; new L2 sounds without L1 equivalents may resist or attract.
- **Your formulation:** Learn a displacement function `delta(v_eng, L1_system) = v_L2 - v_eng_native`, where L1 clusters exert "forces" on native English vowel positions.

This is a strong framing. But I'd suggest a refinement: rather than a physics "force" metaphor (which implies distance-based interactions requiring many free parameters), model it as a **conditional mapping** -- "given that L1 has these vowel cluster positions/shapes, what displacement does each English vowel undergo?"

This reframing keeps the same spirit but opens cleaner model families.

## The Core Modeling Problem (refined)

```
Input:  L1 vowel system summary (centroids + dispersions in N-dim acoustic space)
Output: Per-English-vowel displacement vectors (or directly, L2 vowel cluster centroids/shapes)
```

Concretely, for each speaker `s`:

- **X_s**: Feature vector summarizing their L1 vowel system (extracted from L1 task recordings)
- **Y_s**: Feature vector summarizing their L2 English vowel system (extracted from ENG task recordings)
- **Baseline**: `Y_native` = per-vowel centroids from 26 ENG-ENG native speakers (now available in corpus)
- **Target**: Learn `f(X_s) -> Y_s - Y_native` (the displacement from native English)

## Proposed Pipeline (3 Phases)

### Phase 1: Feature Engineering and Clustering (unsupervised)

**Expand acoustic features** (currently only F1, F2, duration):

- **F3** -- already partially implemented in the outlier inspector; extract it as a column
- **F0 (pitch)** -- available via Parselmouth; captures prosodic/tonal L1 effects
- **Spectral moments** (center of gravity, skewness) -- captures spectral tilt differences
- **Duration** -- already extracted

So the token-level feature vector becomes ~6-7D: `[F1, F2, F3, F0, duration, spectral_COG, spectral_skewness]`.

**Per-speaker vowel system summarization:**

For each speaker, for each vowel category `v`:

- Centroid: `mu_v = [mean_F1, mean_F2, mean_F3, mean_F0, mean_dur, ...]` (N-dim)
- Spread: `sigma_v = [std_F1, std_F2, ...]` (N-dim, or use covariance if samples allow)
- Token count: `n_v`

**Normalization:** Lobanov (already implemented) or use speaker-intrinsic z-scores per feature.

**Clustering / dimensionality reduction:**

- Fit GMMs per language group to validate that clusters match known phoneme categories
- Apply PCA or UMAP on the pooled multi-language vowel space to find the most expressive 2-3D projections
- This is exploratory -- it informs which features matter most, but you don't need to reduce dimensions for the models (the feature space is small enough)

### Phase 2: Data Representation -- The Unit-of-Analysis Problem

This is your biggest practical challenge. You have ~65 speakers, which is too few if each speaker = one sample.

**Strategy: Vowel-level pairing (recommended)**

Instead of one sample per speaker, create one sample **per (speaker x English vowel)** pair:

| Row | X (input) | Y (target) |
| --- | --------- | ---------- |

For each speaker `s` and English vowel `v_eng`:

- **Y**: L2 production of `v_eng` by speaker `s` -- centroid in N-dim space (from their ENG task recordings)
- **X**: The speaker's full L1 vowel system summary -- concatenated centroids + dispersions of all L1 vowels

This gives you: **~65 speakers x 8 English monophthongs = ~520 samples.** Much more workable.

**Feature vector structure for X:**

Option A -- Fixed-size L1 summary per speaker:

- For each of the K L1 vowels: `[mu_F1, mu_F2, ..., sigma_F1, sigma_F2, ...]` concatenated
- Problem: K varies across L1s (Mandarin has different vowels than French)

Option B (recommended) -- Language-agnostic L1 summary:

- Compute summary statistics of the L1 vowel space that are **vowel-inventory-agnostic**:
  - Global centroid of all L1 vowels
  - Overall dispersion (total variance, convex hull area in F1-F2)
  - Number of vowel categories
  - Min/max along each acoustic dimension
  - Pairwise distances between L1 vowel centroids (mean, min, max)
  - The closest L1 vowel centroid to the target English vowel (distance + its features)
- Plus speaker metadata: gender (binary), L1 (one-hot or learned embedding)
- This gives a fixed-length feature vector (~20-40D) regardless of L1 inventory size

Option C -- Include both: Use Option B features + pad/mask Option A to max inventory size.

### Phase 3: Model Selection (progressive complexity)

Given ~520 samples with ~20-40 input features and ~6-7 output features:

**Tier 1 -- Baselines (start here):**

- **Ridge Regression**: `Y = WX + b` with L2 regularization. Fast, interpretable, establishes baseline. Multi-output ridge predicts all formant dimensions at once.
- **Per-vowel Ridge**: Train 8 separate models, one per English vowel. May capture vowel-specific transfer patterns better.

**Tier 2 -- Nonlinear but data-efficient:**

- **Gaussian Process Regression (GPR)**: Excellent for small datasets, gives uncertainty estimates (very useful for your "how confident is the prediction" question). Use ARD kernel to learn feature importance automatically.
- **Random Forest / Gradient Boosting**: Good with tabular data, handles feature interactions, built-in feature importance.

**Tier 3 -- Neural (only if Tiers 1-2 leave room):**

- **Small MLP** (2-3 hidden layers, 32-64 units, dropout): Input = L1 features + English vowel identity (one-hot), Output = predicted L2 formants. Shared architecture across vowels lets the model learn cross-vowel transfer patterns.
- **L1 Vowel System Encoder**: A small network that takes the variable-size L1 vowel inventory (set of centroids) and produces a fixed-size "L1 system embedding" via attention/pooling, then predicts L2 shifts. Elegant but needs more data to train well.

**Recommended starting point:** Ridge regression (baseline) then GPR (best effort for this data size).

### Evaluation: Leave-One-Language-Out CV

- Hold out all speakers of one L1 at a time (10 folds, one per non-English L1)
- Train on remaining 9 L1s
- Test: Can the model predict L2 English vowels for a completely unseen L1?
- Also do **leave-one-speaker-out** within each L1 for a less aggressive test

**Metrics:**

- RMSE / MAE on predicted vs. actual L2 formants (Hz or Lobanov-normalized)
- Correlation between predicted and actual displacement vectors
- Qualitative: plot predicted vs. actual L2 vowel quadrilaterals

## Data Inventory

**Corpus totals:** 755 WAV files, 728 TextGrids, 97 sub-folders.

- **Native English (ENG-ENG):** 26 speakers, 129 files (5 tasks), all with WAV + TextGrid. This is the ground-truth baseline -- no need for external norms.
- **L2 speakers:** ~65 speakers across 10 L1 backgrounds, 626 files (L1 + L2 tasks).
- **L1 diversity:** 10 languages (CMN, FRA, GER, GRE, JPN, KOR, RUS, SPA, TUR, VIE).

**Data sufficiency:**

- Native English reference: **Solved** -- 26 ENG-ENG speakers provide in-corpus native vowel targets matched on tasks and recording conditions.
- L1 vowel tokens: Available in `*_{L1}_*` task folders -- need extraction.
- L2 English tokens: Available in `*_ENG_*` task folders -- already partially extracted.
- Speakers for modeling: ~65 L2 speakers x 8 vowels = ~520 samples (vowel-level pairing).
- L1 diversity: 10 languages is marginal for leave-one-L1-out but workable for leave-one-speaker-out.

## Implementation Steps

1. **Expand feature extraction** -- add F3, F0, spectral moments to `extract_formants()` in the notebook
2. **Extract native English baseline** -- process all 26 ENG-ENG speakers across 5 tasks; compute per-vowel centroids as ground truth
3. **Extract L1 vowel features** -- run the pipeline on L1 task folders (currently only L2 ENG tasks are fully processed)
4. **Build speaker-level summaries** -- per-speaker, per-vowel centroids + dispersions for L1, L2, and native English
5. **Unsupervised exploration** -- GMM clustering, PCA/UMAP visualization across languages
6. **Construct paired dataset** -- (speaker x English vowel) rows; Y = L2 centroid - native centroid (displacement); X = L1 system features
7. **Train baseline models** -- Ridge, then GPR, then MLP
8. **Evaluate** -- Leave-one-language-out and leave-one-speaker-out CV

## TODO Tracker

| ID | Task | Status |
|----|------|--------|
| expand-features | Add F3, F0, spectral moments extraction to the formant pipeline in the notebook | pending |
| extract-native | Run vowel extraction on ENG-ENG folders (26 native speakers) to build the ground-truth baseline | pending |
| extract-l1 | Run vowel extraction on L1 task folders (currently only L2 ENG tasks are processed) | pending |
| speaker-summaries | Build per-speaker, per-vowel centroid + dispersion summaries for both L1 and L2 systems | pending |
| unsupervised-explore | GMM clustering and PCA/UMAP dimensionality reduction exploration across languages | pending |
| paired-dataset | Construct the (speaker x English vowel) paired dataset: Y = L2 centroid - native ENG centroid (displacement), X = L1 system features | pending |
| baseline-models | Implement Ridge regression and GPR baselines with leave-one-language-out CV | pending |
| nn-models | Implement MLP and potentially L1 system encoder if baselines show room for improvement | pending |
