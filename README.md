# L2 Vowel Formant Analysis

Predicting L2 English vowel production from L1 acoustic features using machine learning.

## Project Overview

Second-language (L2) speakers systematically deviate from native vowel targets in ways shaped by their first language (L1). This project asks: *Can a machine-learning model trained on paired L1–L2 vowel acoustics predict the "accent signature" of L2 English vowels from L1 features, and does this mapping generalize across different L1 languages?*

Foreign accent in L2 vowel production is conceptualized here as a systematic **displacement** of each L2 vowel away from the native English target, conditioned on properties of the speaker's L1. This displacement is operationalized as a regression target, and a structured input encoding is used in which each feature group corresponds to a candidate scientific driver of accent.

The project produced three substantive findings:

1. **Cross-validated regression generalizes across L1s.** Three regression models (ridge, Gaussian process, gradient-boosted) achieve moderate-to-strong cross-validated correlations on F1, F2, F3, and duration displacement targets, and gradient-boosted regression's leave-one-language-out (LOLO) and leave-one-speaker-out (LOSO) confidence intervals overlap on every target — the model generalizes to held-out L1s and held-out speakers indistinguishably.
2. **A per-speaker centering diagnostic separates anatomical baseline from vowel-specific phonological transfer.** Roughly 63% of disp_F3 variance turns out to be speaker-level baseline shift; F1 and F2 retain real vowel-specific structure after centering; duration is uncontaminated and is the cleanest single piece of phonological-transfer evidence.
3. **PAM-style perceptual-assimilation features emerge after centering.** The "closest L1 vowel to each English target" feature group is negligible on the original target but becomes the second-most-important predictor on the centered target — masked by the speaker-baseline confound on the original.

A complementary analysis replicates Kartushina & Frauenfelder (2014)'s L1-compactness × L2-accuracy result on this corpus, and a partial-correlation analysis shows that ~76% of the K&F effect survives controlling for a general L2 articulatory precision trait.

## Modeling Approach

The core question: *given a speaker's L1 vowel system (cluster centroids + dispersions in N-dimensional acoustic space), can we predict how each English vowel will shift from its native target?*

This is grounded in Flege's Speech Learning Model (SLM/SLM-r) and the Perceptual Assimilation Model for L2 (PAM-L2): L2 vowel categories are attracted toward similar L1 categories, so each L1 background produces a characteristic "accent signature" — a systematic displacement of English vowels in acoustic space.

**Formal setup.** For each L2 speaker *s* and English vowel *v*:

- **Y** = L2 production centroid of *v* by speaker *s* (from `*_ENG_*` task recordings)
- **Y_native** = gender-matched native English centroid of *v* (from 26 ENG-ENG control speakers)
- **X** = structured summary of speaker *s*'s L1 vowel system (from `*_{L1}_*` task recordings) plus the L1 vowel closest to *v* in F1–F2 space, plus speaker metadata
- **Target**: learn `f(X) → Y − Y_native` (the 4-D displacement vector ΔF1, ΔF2, ΔF3, Δduration)

**Unit of analysis.** One sample per (speaker × English vowel) pair: ~65 L2 speakers × 8 monophthongs ≈ 520 samples with 40 input features and 4 output dimensions.

**Models actually fit and reported:**

| Model                       | Rationale                                                   |
| --------------------------- | ----------------------------------------------------------- |
| Ridge regression            | Linear baseline; closed-form, lowest variance               |
| Gaussian process regression | Smooth nonlinear similarity (RBF + white-noise kernel)      |
| Gradient-boosted regression | Discrete feature interactions (200 trees, depth 4, lr 0.05) |

The three models span the bias-variance axis. If all three converge on the same per-target ordering, that ordering reflects data structure rather than any one model's inductive bias.

**Feature groups (each a testable hypothesis about what drives accent):**

| Group                 | # features | Hypothesis                                                   |
| --------------------- | ---------- | ------------------------------------------------------------ |
| L1 system geometry    | 17         | Gestalt properties of the L1 vowel inventory condition L2 displacement |
| Closest L1 vowel      | 5          | PAM-L2 perceptual assimilation: the specific L1 category an English target maps to drives the displacement |
| Target vowel identity | 9          | Different English vowels have different residual patterns regardless of L1 |
| L1 identity           | 8          | Catches any unmodeled L1-specific structure                  |
| Gender                | 1          | Residual gender effects after Lobanov normalization          |

**Evaluation:**

- **Leave-one-language-out CV** (8 folds): can the model predict L2 vowels for an entirely unseen L1?
- **Leave-one-speaker-out CV** (61 folds): per-speaker generalization within seen L1s
- **Per-speaker centering diagnostic**: refit on `(L2 − speaker_grand_mean) − (native − native_grand_mean)` to separate vowel-specific transfer from each speaker's overall acoustic baseline
- **Feature-group ablation**: drop each feature group and measure Δr, on both original and centered targets
- **Metrics**: Pearson r per target, with 95% bootstrap CIs from resampling speakers (1000 reps)

## Repository Structure

```
L2Vowel_ML/                          # Git-tracked code (this repo)
├── README.md
├── L2_vowel_formant_analysis.ipynb  # EDA notebook: formant extraction, plotting, outlier inspection,
│                                    # ground-truth validation against Hagiwara (1997)
├── l2_vowel_transfer_modeling.ipynb # ML pipeline: feature engineering → clustering → supervised models
│                                    # → centering diagnostic → feature-group ablation → K&F replication
├── modeling_plan.md                 # Detailed modeling plan and design rationale
├── batch_optimize_ceilings.py       # Overnight script to find optimal formant ceilings per speaker
├── figures/                         # Generated figures (referenced from the report)
└── CS289_26Spring_Graduate_project_proposal.docx

Data/                                # NOT in repo — see "Data Setup" below
├── ALLSSTAR/                        # Extracted ALLSSTAR corpus
│   ├── ALL_ENG_ENG_LPP/             # Native English controls (26 speakers)
│   ├── ALL_CMN_ENG_LPP/             # L2 English by Mandarin speakers
│   ├── ALL_CMN_CMN_DHR/             # L1 Mandarin recordings
│   ├── ...                          # 97 sub-folders total
│   └── ALL_VIE_VIE_NWS/
├── speaker_ceilings.json            # Per-speaker optimized formant ceilings (generated)
└── *.parquet                        # Cached extracted features (generated, see notes below)
```

## Data Setup

The audio data is too large to include in the repository. You need to download and extract it manually.

### ALLSSTAR Corpus

The ALLSSTAR corpus (Archive of L1 and L2 Scripted and Spontaneous Transcripts and Recordings) is hosted by Northwestern.

1. Go to the [ALLSSTAR Page](https://speechbox.linguistics.northwestern.edu/ALLSSTARcentral/#!/recordings).
2. Select data for the following 11 L1 backgrounds. For each non-English L1, download both L1 and L2 (English) task data. For native English, download the ENG-ENG data:
   - **ENG** (native English controls)
   - **CMN** (Mandarin), **FRA** (French), **GER** (German), **GRE** (Greek), **JPN** (Japanese), **KOR** (Korean), **RUS** (Russian), **SPA** (Spanish), **TUR** (Turkish), **VIE** (Vietnamese)
3. After selecting datasets, a download link will appear. Click it and input your email. In a minute or two you will receive an email with a link to download the dataset.
4. Extract them into `Data/ALLSSTAR/` so that the folder structure matches the layout described below.

### (extra dataset that hasn't been explored) L2-Arctic Corpus

A natural extension of this work is cross-corpus validation on L2-ARCTIC; the data layout for that corpus is included here for completeness:

1. Go to the [L2-Arctic page](https://psi.engr.tamu.edu/l2-arctic-corpus/).
2. Download `l2arctic_release_v5.0.zip`.
3. Extract into `Data/l2arctic_release_v5.0/`.

### Expected Data Layout After Extraction

```
Data/ALLSSTAR/
├── ALL_{L1}_{TaskLang}_{Task}/
│   ├── ALL_{ID}_{Gender}_{L1}_{TaskLang}_{Task}.wav
│   └── ALL_{ID}_{Gender}_{L1}_{TaskLang}_{Task}.TextGrid
└── ...
```

| Field      | Description                           | Examples                                                   |
| ---------- | ------------------------------------- | ---------------------------------------------------------- |
| `L1`       | Speaker's native language (ISO 639-3) | ENG, CMN, FRA, GER, GRE, JPN, KOR, RUS, SPA, TUR, VIE      |
| `TaskLang` | Language of the recording task        | ENG (L2 English, or native for ENG speakers) or same as L1 |
| `Task`     | Elicitation task                      | DHR, HT1, HT2, LPP, NWS                                    |
| `ID`       | Numeric participant ID (zero-padded)  | 005, 011, 012, ...                                         |
| `Gender`   | Speaker gender                        | M, F                                                       |

**Task codes:**

- **DHR** — Diapix (Hierarchical Referencing)
- **HT1** — HINT sentences, set 1
- **HT2** — HINT sentences, set 2
- **LPP** — Le Petit Prince paragraph reading
- **NWS** — News passage reading

The full corpus contains **755 files** (728 with TextGrids) across **97 sub-folders**:

- **Native English (ENG-ENG):** 26 speakers, 129 files across all 5 tasks — serves as the ground-truth vowel baseline
- **L2 speakers:** ~65 participants across 10 L1 backgrounds, 626 files (L1 + L2 English tasks)

### TextGrid Tier Conventions

- **`*_ENG_*` files** (L2 English and native English tasks): typically 3 tiers — `"utt"` / `"Speaker - word"` / `"Speaker - phone"` (ARPAbet labels). Exception: some ENG-ENG NWS files use 2 tiers as `"utt - words"` / `"utt - phones"`.
- **`*_{L1}_*` files** (L1 tasks): 2–3 tiers — `"sentence"/"utt"` / `["sentence - words"]` / `"sentence - phones"` (language-specific phone labels)

The phone tier is identified as the first tier whose name contains `"phone"`, falling back to the last tier.

## Reproducing the Results

After cloning the repo and setting up the environment (see "Environment Setup" below), the full pipeline runs in three stages.

### Stage 1: Per-speaker formant-ceiling optimization (~30–60 min, run once)

This sweeps max-formant ceilings (4500–6000 Hz in 100 Hz steps) per speaker and caches the optimal ceiling for each in `Data/speaker_ceilings.json`. Results are cached so re-runs skip already-processed speakers. Run from the repo root:

```bash
python batch_optimize_ceilings.py
```

You can interrupt and resume; speakers without a cache entry will be processed.

### Stage 2: EDA notebook — `L2_vowel_formant_analysis.ipynb`

This notebook produces the pipeline-validation figures (Figures 1 and 2 in the report) by extracting native-English citation-form vowels from ALLSSTAR and comparing the resulting vowel space against published Hagiwara (1997) norms.

```bash
jupyter notebook L2_vowel_formant_analysis.ipynb
```

Run cells in order. Cells 13–16 produce the corpus-wide vowel quadrilaterals, including the gender-stratified citation-form comparison against Hagiwara that validates the feature-extraction pipeline. Expected output:

- `figures/step6A_native_eng_vs_hagiwara.png` — native ENG citation centroids vs Hagiwara, gender-stratified
- `figures/step6B_l2_vs_corpus_eng_citation.png` — L2 citation centroids by L1 × gender

The pipeline output should match Hagiwara within median |ΔF1| ≈ 35–40 Hz, |ΔF2| ≈ 70 Hz on the citation-like subset (stressed vowels in content words with token duration ≥ 100 ms).

### Stage 3: Modeling notebook — `l2_vowel_transfer_modeling.ipynb`

This notebook runs the supervised pipeline: feature engineering, clusterability ablation, regression sweep, per-speaker centering diagnostic, feature-group ablation, and the K&F mediation analysis.

```bash
jupyter notebook l2_vowel_transfer_modeling.ipynb
```

The first run extracts and caches per-speaker formant data (~10–20 min); subsequent runs reuse the parquet caches. Cells are organized by step, listed below in the order they should be run.

| Cells  | Step | Stage / output                                               |
| ------ | ---- | ------------------------------------------------------------ |
| 1–2    | 1    | Imports, corpus scan, helper functions (TextGrid parser, vowel inventories, Lobanov) |
| 3      | 2    | Expanded feature extraction: F1–F3, F0, duration, spectral moments via Parselmouth |
| 4–6    | 3    | Extract three vowel pools (native English, L2 English, L1) with disk caching |
| 7      | 4    | Build per-speaker vowel-system summaries (centroids, dispersions, hull area, pairwise dists) |
| 8      | 5    | Within-language clusterability ladder (Table 2 in the report) |
| 8c, 8d | 5    | 3D vowel-space scatters per language (sanity check; not in report) |
| 9      | 6    | Construct paired (speaker × vowel) dataset for supervised modeling |
| 10     | 7    | Shared evaluation framework + ridge baseline (LOSO/LOLO with bootstrap CIs) |
| 11     | 7    | Gaussian process + gradient-boosted regression               |
| 11a    | 7    | Per-L1 LOLO + LOSO breakdown (sanity check on imbalanced L1s) |
| 11b    | 7    | Bootstrap CIs on r (1000 resamples)                          |
| 14a    | 8    | Permutation importance per (target, CV scheme)               |
| 14b    | 8    | Feature-group ablation: drop each group, measure Δr          |
| 14c    | 8    | **Centering diagnostic**: refit on per-speaker-centered targets (Table 4 in the report) |
| 14d    | 8    | Figure 4 — feature-group Δr, original vs centered displacement |
| 17     | 9    | K&F replication (within-L1, all in Lobanov units)            |
| 19     | 9    | K&F mediation test — partial correlation controlling for L2 compactness |
| 20     | 9    | Figure 3 — paper-quality K&F mediation diagnostic (4-panel)  |

Cells 12 (MLP) and 13–16 (older comparison/visualization cells) are present in the notebook for completeness but are not used to generate any reported result; they can be skipped.

Cells produce the data behind the report's Tables 3 and 4 and Figures 3 and 4 directly. To regenerate the figures used in the report, run cells 14d and 20 after the upstream cells they depend on.

### Expected Results (sanity check)

If the pipeline runs correctly, you should reproduce within a few percent:

- **Sanity check (§4.1 / §6.1):** native-English citation centroids match Hagiwara (1997) within median |ΔF1| ≈ 35–40 Hz, |ΔF2| ≈ 70 Hz.
- **Clusterability (§4.2 / §6.2, Table 2):** native pool peaks at ARI ≈ 0.40 with F1+F2 Lobanov, dropping to ≈ 0.17 with the full feature set; L2 pool similar pattern with weaker magnitude.
- **Regression (§4.4 / §6.4, Table 3):** GBR LOLO reaches r ≈ 0.83 (F3), 0.77 (F2), 0.68 (F1), 0.38 (duration); LOSO and LOLO bootstrap CIs overlap on every target under GBR.
- **Centering (§4.5 / §6.5, Table 4):** disp_F3 r drops from 0.83 → 0.17 under per-speaker centering; F1/F2 partially survive (r ≈ 0.50–0.55); duration r preserved (~+0.03).
- **Feature ablation (§4.6 / §6.6, Figure 4):** Closest_L1_vowel emerges as the second-most-important group on centered targets (Δr_F3 LOLO ≈ +0.12, Δr_F1 LOLO ≈ +0.07).
- **K&F replication (§4.3 / §6.3, Figure 3):** within-L1 r = +0.34 (p = 0.008); partial r controlling for L2 compactness = +0.25 (p = 0.047).

## Notebook Walkthrough — `L2_vowel_formant_analysis.ipynb`

This notebook is for EDA and pipeline validation; it does not produce modeling results.

| Cells | Stage                       | What it does                                                 |
| ----- | --------------------------- | ------------------------------------------------------------ |
| 0–1   | Data scan                   | Walks `Data/ALLSSTAR/`, builds a `file_metadata` DataFrame   |
| 2     | File selection              | Cascading dropdown widgets: L1 → task language → task → file |
| 3     | TextGrid parsing            | Parses the selected TextGrid into tier/interval structures   |
| 4     | Vowel inventory             | Defines vowel sets for all 10 L1 phone systems               |
| 5     | Formant extraction (1 file) | Extracts F1/F2 at the middle third of each vowel; Lobanov normalization; ARPAbet→IPA mapping |
| 6     | Vowel plot (1 participant)  | Scatter + mean ± SD ellipses on the F1×F2 plane              |
| 7     | Outlier inspector           | Interactive GUI: spectrogram, formant track, vowel-space highlight, audio playback |
| 8     | All tasks (1 participant)   | Pools formants across every task for the selected speaker    |
| 9     | Folder-level extraction     | Processes all files in a selected `ALL_{L1}_{TaskLang}_{Task}` folder |
| 10–12 | Folder-level plots          | Group vowel quadrilaterals: all speakers / F / M             |
| 13    | Corpus-wide extraction      | All subjects × all tasks combined                            |
| 14–16 | Corpus-wide plots           | Group-level quadrilateral plots: all / F / M, with Hagiwara reference overlay (Figures 1, 2) |

### Key Functions

- `parse_textgrid(path)` — regex-based TextGrid parser returning a list of tier dicts
- `extract_formants(audio_segment, sr, max_formant)` — Praat-based (via Parselmouth) F1/F2 extraction averaged over the middle third of a segment
- `extract_vowel_formants_from_file(meta_row)` — end-to-end: read WAV + TextGrid, filter vowels, extract formants
- `lobanov_normalize(df)` — per-speaker z-score normalization of F1 and F2
- `plot_vowel_space(df)` — vowel quadrilateral with scatter + mean ± SD ellipses
- `find_outliers(df)` / `launch_outlier_inspector()` — IQR-based outlier detection with interactive review

## Formant Ceiling Optimization

`batch_optimize_ceilings.py` sweeps max-formant ceilings (4500–6000 Hz in 100 Hz steps) per speaker, pooling all their files. The optimal ceiling is the one that maximizes inter-vowel centroid separation among the 8 English monophthongs (IY, IH, EH, AE, AA, AH, UH, UW). Results are cached in `Data/speaker_ceilings.json` so already-processed speakers are skipped on re-runs.

Run with:

```bash
python batch_optimize_ceilings.py
```

## Environment Setup

**Python 3.10** via Conda.

```bash
conda create -n comp_ling_project python=3.10
conda activate comp_ling_project
pip install numpy pandas matplotlib scipy parselmouth soundfile ipywidgets scikit-learn umap-learn pyarrow
```

Key dependencies:

| Package      | Version  | Purpose                                                     |
| ------------ | -------- | ----------------------------------------------------------- |
| numpy        | 2.2.6    | Numerical operations                                        |
| pandas       | 2.3.3    | DataFrames, parquet caching                                 |
| pyarrow      | (latest) | Parquet I/O backend for pandas (caching extracted features) |
| matplotlib   | 3.10.8   | Plotting                                                    |
| scipy        | 1.15.3   | Statistical functions, Mahalanobis outlier removal          |
| parselmouth  | 0.4.7    | Praat formant analysis in Python                            |
| soundfile    | 0.13.1   | WAV I/O                                                     |
| ipywidgets   | 8.1.8    | Interactive notebook widgets                                |
| scikit-learn | (latest) | Ridge, GPR, gradient boosting, GMM, ARI, CV utilities       |
| umap-learn   | (latest) | UMAP dimensionality reduction (used in EDA only)            |

## References

- Bradlow, A. R., et al. (2021). ALLSSTAR corpus. [OSF](https://osf.io/q9e2y/)
- Hagiwara, R. (1997). Dialect variation and formant frequency: The American English vowels revisited. *JASA* 102(1), 655–658.
- Kartushina, N. & Frauenfelder, U. H. (2014). On the effects of L2 perception and of individual differences in L1 production on L2 pronunciation. *Frontiers in Psychology* 5, 1246.
- Flege, J. E. (1995). Second language speech learning: Theory, findings, and problems. *Speech perception and linguistic experience*.
- Best, C. T. & Tyler, M. D. (2007). Nonnative and second-language speech perception: Commonalities and complementarities (PAM-L2).
- Lobanov, B. M. (1971). Classification of Russian vowels spoken by different speakers. *JASA* 49(2B), 606–608.
- Zhao, G., et al. (2018). L2-Arctic: A non-native English speech corpus. [TAMU](https://psi.engr.tamu.edu/l2-arctic-corpus/)
