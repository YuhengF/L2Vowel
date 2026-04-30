# L2 Vowel Formant Analysis

Predicting L2 English vowel production from L1 acoustic features using machine learning.

**Course:** CS 289A, Spring 2026

## Project Overview

Second-language (L2) speakers systematically deviate from native vowel targets in ways shaped by their first language (L1). This project asks: *Can a machine-learning model trained on paired L1–L2 vowel acoustics predict the "accent signature" of L2 English vowels from L1 features, and does this mapping generalize across different L1 languages?*

The analysis proceeds in three phases:

1. **(Finished) Formant extraction and visualization.** Extract vowel formant measurements (F1, F2) from the ALLSSTAR corpus for both L2 English speakers and native English controls, apply per-speaker Lobanov normalization with optimized formant ceilings, and produce F1-F2 vowel quadrilateral plots. The goal is to replicate known vowel-space patterns and validate against ground truth from previous studies.
2. **(In progress) Expanded feature space and unsupervised exploration.** Extract additional acoustic features (F3, F0, duration, spectral moments) to form a higher-dimensional vowel space. Apply GMM clustering and dimensionality reduction (PCA/UMAP) to reveal cross-language vowel structure. Represent each vowel category per speaker as a centroid + dispersion ellipsoid. These unsupervised findings serve as a sanity check for the supervised models.
3. **(Planned) Supervised L1-to-L2 transfer modeling.** Learn a mapping from L1 vowel-system features to L2 English vowel displacements (relative to native English baseline). Train progressively complex models -- ridge regression, Gaussian-process regression, and feedforward networks -- evaluated via **leave-one-language-out** and **leave-one-speaker-out** cross-validation.

See `CS289_26Spring_Graduate_project_proposal.docx` for the full proposal.

## Modeling Approach

The core question: *given a speaker's L1 vowel system (cluster centroids + dispersions in N-dimensional acoustic space), can we predict how each English vowel will shift from its native target?*

This is grounded in Flege's Speech Learning Model: L2 vowel categories are attracted toward similar L1 categories, so each L1 background produces a characteristic "accent signature" -- a systematic displacement of English vowels in acoustic space.

**Formal setup.** For each L2 speaker *s* and English vowel *v*:
- **Y** = L2 production centroid of *v* by speaker *s* (from `*_ENG_*` task recordings)
- **Y_native** = native English centroid of *v* (from 26 ENG-ENG control speakers)
- **X** = language-agnostic summary of speaker *s*'s L1 vowel system (from `*_{L1}_*` task recordings): global centroid, dispersion, inventory size, pairwise distances, distance from each L1 vowel to the target English vowel, plus speaker metadata (gender, L1 one-hot)
- **Target**: learn `f(X) -> Y - Y_native` (the displacement vector)

**Unit of analysis.** One sample per (speaker x English vowel) pair: ~65 L2 speakers x 8 monophthongs = ~520 samples with ~20-40 input features and ~6-7 output dimensions.

**Models (progressive complexity):**

| Tier | Model | Rationale |
|------|-------|-----------|
| 1 | Ridge regression | Interpretable baseline; multi-output linear mapping |
| 1 | Per-vowel ridge | One model per English vowel; captures vowel-specific transfer |
| 2 | Gaussian process regression (GPR) | Nonlinear, data-efficient, provides uncertainty estimates; ARD kernel for feature selection |
| 2 | Gradient boosting | Handles feature interactions; built-in feature importance |
| 3 | Small MLP (2-3 layers, 32-64 units) | Input = L1 features + vowel identity; shared architecture learns cross-vowel patterns |
| 3 | L1 vowel-system encoder | Attention/pooling over variable-size L1 inventory; produces fixed-size L1 embedding |

**Evaluation:**
- **Leave-one-language-out CV** (10 folds): can the model predict L2 vowels for an entirely unseen L1?
- **Leave-one-speaker-out CV** (within each L1): less aggressive but validates per-speaker generalization
- **Metrics:** RMSE / MAE on predicted vs. actual displacements (Hz and Lobanov-normalized); predicted vs. actual vowel quadrilateral overlays

## Repository Structure

```
L2Vowel_ML/                         # Git-tracked code (this repo)
├── README.md
├── L2_vowel_formant_analysis.ipynb  # EDA notebook: formant extraction, plotting, outlier inspection
├── l2_vowel_transfer_modeling.ipynb # ML pipeline: feature engineering → clustering → supervised models
├── modeling_plan.md                 # Detailed modeling plan and design rationale
├── batch_optimize_ceilings.py       # Overnight script to find optimal formant ceilings per speaker
└── CS289_26Spring_Graduate_project_proposal.docx

Data/                                # NOT in repo — see "Data Setup" below
├── ALLSSTAR/                        # Extracted ALLSSTAR corpus
│   ├── ALL_ENG_ENG_LPP/            # Native English controls (26 speakers)
│   ├── ALL_CMN_ENG_LPP/            # L2 English by Mandarin speakers
│   ├── ALL_CMN_CMN_DHR/            # L1 Mandarin recordings
│   ├── ...                          # 97 sub-folders total
│   └── ALL_VIE_VIE_NWS/
├── speaker_ceilings.json            # Per-speaker optimized formant ceilings (generated)
```

## Data Setup

The audio data is too large to include in the repository. You need to download and extract it manually.

### ALLSSTAR Corpus

The ALLSSTAR corpus (Archive of L1 and L2 Scripted and Spontaneous Transcripts and Recordings) is hosted by the Open Science Framework.

1. Go to the [ALLSSTAR Page](https://speechbox.linguistics.northwestern.edu/ALLSSTARcentral/#!/recordings).
2. Select data for the following 11 L1 backgrounds. For each non-English L1, download both L1 and L2 (English) task data. For native English, download the ENG-ENG data:
   - **ENG** (native English controls)
   - **CMN** (Mandarin), **FRA** (French), **GER** (German), **GRE** (Greek), **JPN** (Japanese), **KOR** (Korean), **RUS** (Russian), **SPA** (Spanish), **TUR** (Turkish), **VIE** (Vietnamese)
3. After selecting datasets, a download link will appear. Click it and input your email. In a minute or two you will receive an email with a link to download the dataset.
4. Extract them into `Data/ALLSSTAR/` so that the folder structure matches the layout described below.

### (extra dataset that hasn't been explored) L2-Arctic Corpus

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

| Field       | Description                            | Examples                             |
|-------------|----------------------------------------|--------------------------------------|
| `L1`        | Speaker's native language (ISO 639-3)  | ENG, CMN, FRA, GER, GRE, JPN, KOR, RUS, SPA, TUR, VIE |
| `TaskLang`  | Language of the recording task         | ENG (L2 English, or native for ENG speakers) or same as L1 |
| `Task`      | Elicitation task                       | DHR, HT1, HT2, LPP, NWS             |
| `ID`        | Numeric participant ID (zero-padded)   | 005, 011, 012, ...                   |
| `Gender`    | Speaker gender                         | M, F                                 |

**Task codes:**
- **DHR** — Diapix (Hierarchical Referencing)
- **HT1** — HINT sentences, set 1
- **HT2** — HINT sentences, set 2
- **LPP** — Le Petit Prince paragraph reading
- **NWS** — News passage reading

The full corpus contains **755 files** (728 with TextGrids) across **97 sub-folders**:
- **Native English (ENG-ENG):** 26 speakers, 129 files across all 5 tasks -- serves as the ground-truth vowel baseline
- **L2 speakers:** ~65 participants across 10 L1 backgrounds, 626 files (L1 + L2 English tasks)

### TextGrid Tier Conventions

- **`*_ENG_*` files** (L2 English and native English tasks): typically 3 tiers — `"utt"` / `"Speaker - word"` / `"Speaker - phone"` (ARPAbet labels). Exception: some ENG-ENG NWS files use 2 tiers as `"utt - words"` / `"utt - phones"`.
- **`*_{L1}_*` files** (L1 tasks): 2–3 tiers — `"sentence"/"utt"` / `["sentence - words"]` / `"sentence - phones"` (language-specific phone labels)

The phone tier is identified as the first tier whose name contains `"phone"`, falling back to the last tier.

## Notebook Walkthrough

`L2_vowel_formant_analysis.ipynb` is organized as follows:

| Cells  | Stage                          | What it does                                                        |
|--------|--------------------------------|---------------------------------------------------------------------|
| 0–1    | Data scan                      | Walks `Data/ALLSSTAR/`, builds a `file_metadata` DataFrame          |
| 2      | File selection                 | Cascading dropdown widgets: L1 → task language → task → file        |
| 3      | TextGrid parsing               | Parses the selected TextGrid into tier/interval structures          |
| 4      | Vowel inventory                | Defines vowel sets for all 10 L1 phone systems; filters phone segments to vowels |
| 5      | Formant extraction (1 file)    | Extracts F1/F2 at the mid-third of each vowel; Lobanov normalization; ARPAbet→IPA mapping |
| 6      | Vowel plot (1 participant)     | Scatter + mean ± SD ellipses on the F1×F2 plane                    |
| 7      | Outlier inspector              | Interactive GUI: spectrogram, formant track, vowel-space highlight, audio playback |
| 8      | All tasks (1 participant)      | Pools formants across every task for the selected speaker           |
| 9      | Folder-level extraction        | Processes all files in a selected `ALL_{L1}_{TaskLang}_{Task}` folder |
| 10–12  | Folder-level plots             | Group vowel quadrilaterals: all speakers / F / M                   |
| 13     | Corpus-wide extraction         | All subjects × all tasks combined                                  |
| 14–16  | Corpus-wide plots              | Group-level quadrilateral plots: all / F / M                       |

### Key Functions

- `parse_textgrid(path)` — regex-based TextGrid parser returning a list of tier dicts
- `extract_formants(audio_segment, sr, max_formant)` — Praat-based (via Parselmouth) F1/F2 extraction at the middle third of a segment
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
pip install numpy pandas matplotlib scipy parselmouth soundfile ipywidgets scikit-learn umap-learn
```

Key dependencies:

| Package       | Version | Purpose                          |
|---------------|---------|----------------------------------|
| numpy         | 2.2.6   | Numerical operations             |
| pandas        | 2.3.3   | DataFrames                       |
| matplotlib    | 3.10.8  | Plotting                         |
| scipy         | 1.15.3  | Statistical functions            |
| parselmouth   | 0.4.7   | Praat formant analysis in Python |
| soundfile     | 0.13.1  | WAV I/O                          |
| ipywidgets    | 8.1.8   | Interactive notebook widgets     |
| scikit-learn  | (latest) | Ridge, GPR, gradient boosting, CV utilities |
| umap-learn    | (latest) | UMAP dimensionality reduction    |

## References

- Bradlow, A. R., et al. (2021). ALLSSTAR corpus. [OSF](https://osf.io/q9e2y/)
- Zhao, G., et al. (2018). L2-Arctic: A non-native English speech corpus. [TAMU](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- Kartushina, N. & Frauenfelder, U. H. (2014). On the effects of L2 perception and of individual differences in L1 production on L2 pronunciation.
- Flege, J. E. (1995). Second language speech learning: Theory, findings, and problems. *Speech perception and linguistic experience*.
