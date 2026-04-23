r"""Batch-optimize max_formant ceilings for ALL speakers in ALLSSTAR.

Run overnight:
    & "C:\Users\BatLab\anaconda3\envs\comp_ling_project\python.exe" L2Vowel_ML\batch_optimize_ceilings.py

Reads every WAV + TextGrid, sweeps 4500–6000 Hz per speaker (all their
files pooled), picks the ceiling that maximizes inter-vowel centroid
separation, and saves results to Data/speaker_ceilings.json.

Already-cached speakers are skipped automatically.
"""
import os
import re
import json
import time
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from itertools import combinations
from collections import defaultdict

# ── Paths ──
DATA_ROOT = Path(r"C:\Users\Yuheng\l2Vowel_ML\Data\ALLSSTAR")
CACHE_PATH = Path(r"C:\Users\Yuheng\l2Vowel_ML\Data\speaker_ceilings.json")

CEILING_MIN  = 4500
CEILING_MAX  = 6000
CEILING_STEP = 100

# Only optimize for the 8 plotted monophthongs (normalized ARPAbet labels)
_CEILING_TARGET_VOWELS = {"IY", "IH", "EH", "AE", "AA", "UH", "UW", "AH1", "AH2"}

# Target language for the current batch run
TARGET_LANGUAGE = "ENG"

def _ceiling_context_key(target_language, target_vowels):
    """Build a deterministic cache key from language + vowel set."""
    return f"{target_language}:{','.join(sorted(target_vowels))}"

# ═══════════════════════════════════════════════════════════════════════
# Vowel inventories (copied from notebook Cell 4)
# ═══════════════════════════════════════════════════════════════════════

ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AX", "AY",
    "EH", "ER", "EY",
    "IH", "IY", "IX",
    "OW", "OY",
    "UH", "UW", "UX",
}

CMN_VOWEL_BASES = {
    "a", "e", "i", "o", "u", "v", "ii",
    "ai", "ao", "ei", "ou",
    "ia", "iao", "ie", "iu", "iou",
    "ua", "uai", "ue", "uei", "uo",
    "va",
}

FRA_VOWELS = {
    "a", "e", "i", "o", "u", "y",
    "AE", "E", "EU", "O", "OE", "AX",
    "A~", "E~", "OE~", "o~",
}

GER_VOWELS = {
    "a", "e", "i", "o", "u", "ae", "oe", "ue",
    "al", "el", "il", "ol", "ul", "ael", "oel", "uel",
    "atu", "etu",
    "aI", "aU",
}

KOR_VOWELS = {
    "A", "AE", "E", "EO", "EU", "I", "O", "OE", "U", "UE",
    "iE", "iEO", "iO", "iU", "euI",
    "oA", "uEO",
}

RUS_VOWELS = {"a", "i", "i2", "o", "u", "jA", "jE", "jU"}
SPA_VOWELS = {"a", "e", "i", "o", "u", "a+", "i+", "o+", "u+", "eU"}
TUR_VOWELS = {"ab", "e", "i", "i2", "o", "oe", "u", "ue"}

VIE_VOWEL_BASES = {
    "a1", "a2", "a3", "e1", "e2", "i",
    "o1", "o2", "o3", "u1", "u2",
    "ai", "ao", "au", "ay", "ay3",
    "eo", "eu", "ie2", "ieu",
    "oa", "oi", "oi2", "oi3",
    "ua", "ua2", "uu2", "uy", "uoi3",
}

_re_vowel = re

def _detect_phone_system(task_language):
    return "ARPABET" if task_language == "ENG" else task_language

def is_vowel(phone_label, task_language="ENG"):
    system = _detect_phone_system(task_language)
    if system == "ARPABET":
        return phone_label.rstrip("0123456789") in ARPABET_VOWELS
    if system == "CMN":
        return phone_label.rstrip("12345") in CMN_VOWEL_BASES
    if system == "FRA":
        return phone_label in FRA_VOWELS
    if system == "GER":
        return phone_label in GER_VOWELS
    if system == "KOR":
        return phone_label in KOR_VOWELS
    if system == "RUS":
        return phone_label in RUS_VOWELS
    if system == "SPA":
        return phone_label in SPA_VOWELS
    if system == "TUR":
        return phone_label in TUR_VOWELS
    if system == "VIE":
        base = _re_vowel.sub(r"_T\d$", "", phone_label)
        return base in VIE_VOWEL_BASES
    return phone_label.rstrip("0123456789") in ARPABET_VOWELS

def normalize_vowel_label(phone_label, task_language="ENG"):
    system = _detect_phone_system(task_language)
    if system == "ARPABET":
        base = phone_label.rstrip("0123456789")
        if base == "AH" and len(phone_label) > len(base):
            return phone_label
        return base
    if system == "CMN":
        return phone_label.rstrip("12345")
    if system == "VIE":
        return _re_vowel.sub(r"_T\d$", "", phone_label)
    return phone_label

# ═══════════════════════════════════════════════════════════════════════
# TextGrid parser (copied from notebook Cell 3)
# ═══════════════════════════════════════════════════════════════════════

def parse_textgrid(path):
    for enc in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    tiers = []
    tier_blocks = re.split(r'item\s*\[\d+\]\s*:', raw)
    for block in tier_blocks[1:]:
        tier = {}
        tier["class"] = re.search(r'class\s*=\s*"([^"]+)"', block).group(1)
        tier["name"] = re.search(r'name\s*=\s*"([^"]*)"', block).group(1)
        tier["xmin"] = float(re.search(r'(?<!\[)xmin\s*=\s*(-?[\d.]+)', block).group(1))
        tier["xmax"] = float(re.search(r'(?<!\[)xmax\s*=\s*(-?[\d.]+)', block).group(1))
        intervals = []
        for m in re.finditer(
            r'intervals\s*\[\d+\]\s*:\s*'
            r'xmin\s*=\s*(-?[\d.]+)\s*'
            r'xmax\s*=\s*(-?[\d.]+)\s*'
            r'text\s*=\s*"([^"]*)"',
            block,
        ):
            intervals.append({
                "xmin": float(m.group(1)),
                "xmax": float(m.group(2)),
                "text": m.group(3).strip(),
            })
        tier["intervals"] = intervals
        tiers.append(tier)
    return tiers

# ═══════════════════════════════════════════════════════════════════════
# Formant extraction + ceiling optimization
# ═══════════════════════════════════════════════════════════════════════

def extract_formants(audio_segment, sr, max_formant=5500, n_formants=5,
                     n_samples=5):
    snd = parselmouth.Sound(audio_segment, sampling_frequency=sr)
    formant_obj = call(snd, "To Formant (burg)", 0.0, n_formants, max_formant, 0.025, 50.0)
    t_start = snd.duration / 3.0
    t_end   = snd.duration * 2.0 / 3.0
    times = np.linspace(t_start, t_end, n_samples)
    f1_vals, f2_vals = [], []
    for t in times:
        f1 = call(formant_obj, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant_obj, "Get value at time", 2, t, "Hertz", "Linear")
        if not np.isnan(f1) and f1 > 0:
            f1_vals.append(f1)
        if not np.isnan(f2) and f2 > 0:
            f2_vals.append(f2)
    if not f1_vals or not f2_vals:
        return float("nan"), float("nan")
    return np.mean(f1_vals), np.mean(f2_vals)

def _sweep_formants(aud, sr, intervals, task_lang, ceiling,
                    target_vowels=None):
    results = []
    for iv in intervals:
        lbl = iv["text"].strip()
        if not lbl or lbl in ("sil", "sp", "SIL", "SP"):
            continue
        if not is_vowel(lbl, task_lang):
            continue
        dur = iv["xmax"] - iv["xmin"]
        if dur < 0.05:
            continue
        base = normalize_vowel_label(lbl, task_lang)
        if target_vowels and base not in target_vowels:
            continue
        s0 = int(iv["xmin"] * sr)
        s1 = int(iv["xmax"] * sr)
        seg = aud[s0:s1]
        try:
            f1, f2 = extract_formants(seg, sr, max_formant=ceiling)
        except Exception:
            continue
        if np.isnan(f1) or np.isnan(f2) or f1 <= 0 or f2 <= 0:
            continue
        results.append((base, f1, f2))
    return results

def find_optimal_ceiling(file_rows):
    """file_rows: list of dicts with wav_path, textgrid_path, task_language, gender."""
    file_data = []
    for row in file_rows:
        tg_p = row["textgrid_path"]
        wav_p = row["wav_path"]
        if tg_p is None or wav_p is None:
            continue
        if not os.path.exists(wav_p):
            continue
        tg = parse_textgrid(tg_p)
        pt = None
        for t in tg:
            if "phone" in t["name"].lower():
                pt = t
                break
        if pt is None:
            pt = tg[-1]
        aud, sr = sf.read(wav_p)
        if aud.ndim > 1:
            aud = aud[:, 0]
        file_data.append((aud, sr, pt["intervals"], row["task_language"]))

    if not file_data:
        return 5500 if file_rows[0]["gender"] == "F" else 5000

    best_ceiling = 5500 if file_rows[0]["gender"] == "F" else 5000
    best_score = -1

    for ceiling in range(CEILING_MIN, CEILING_MAX + 1, CEILING_STEP):
        all_tokens = []
        for aud, sr, intervals, tl in file_data:
            all_tokens.extend(_sweep_formants(aud, sr, intervals, tl, ceiling,
                                              target_vowels=_CEILING_TARGET_VOWELS))
        if len(all_tokens) < 4:
            continue
        groups = defaultdict(list)
        for v, f1, f2 in all_tokens:
            groups[v].append((f1, f2))
        centroids = {}
        for v, pts in groups.items():
            if len(pts) >= 2:
                arr = np.array(pts)
                centroids[v] = arr.mean(axis=0)
        if len(centroids) < 2:
            continue
        score = sum(
            np.linalg.norm(centroids[a] - centroids[b])
            for a, b in combinations(centroids.keys(), 2)
        )
        if score > best_score:
            best_score = score
            best_ceiling = ceiling

    return best_ceiling

# ═══════════════════════════════════════════════════════════════════════
# Build metadata (same logic as notebook Cell 1)
# ═══════════════════════════════════════════════════════════════════════

def build_metadata():
    pattern = re.compile(
        r"^ALL_(\d+)_([MF])_([A-Z]{3})_([A-Z]{3})_([A-Z0-9]{2,3})$"
    )
    records = []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        wav_lookup = {f.stem: str(f.resolve()) for f in folder.glob("*.wav")}
        tg_lookup = {f.stem: str(f.resolve()) for f in folder.glob("*.TextGrid")}
        all_stems = set(wav_lookup) | set(tg_lookup)
        for stem in sorted(all_stems):
            m = pattern.match(stem)
            if not m:
                continue
            pid, gender, native_lang, task_lang, task = m.groups()
            records.append({
                "filename_stem": stem,
                "participant_id": int(pid),
                "gender": gender,
                "native_language": native_lang,
                "task_language": task_lang,
                "task": task,
                "wav_path": wav_lookup.get(stem),
                "textgrid_path": tg_lookup.get(stem),
            })
    return records

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    from datetime import datetime, timezone

    ctx_key = _ceiling_context_key(TARGET_LANGUAGE, _CEILING_TARGET_VOWELS)
    print(f"Target: {TARGET_LANGUAGE}, vowels: {sorted(_CEILING_TARGET_VOWELS)}")
    print(f"Context key: {ctx_key}")
    print()

    print("Building file metadata ...")
    all_records = build_metadata()
    print(f"  {len(all_records)} files, ", end="")

    by_speaker = defaultdict(list)
    for r in all_records:
        by_speaker[r["participant_id"]].append(r)
    print(f"{len(by_speaker)} speakers")

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)

    already = 0
    for pid in by_speaker:
        key = str(pid)
        entry = cache.get(key)
        if entry and ctx_key in entry.get("ceilings", {}):
            already += 1
    remaining = len(by_speaker) - already
    print(f"  Cache: {already} done, {remaining} remaining")
    print("=" * 60)

    if remaining == 0:
        print("All speakers already cached for this context. Nothing to do.")
        return

    t_start_all = time.time()
    done = 0

    for pid in sorted(by_speaker.keys()):
        key = str(pid)
        entry = cache.get(key)
        if entry and ctx_key in entry.get("ceilings", {}):
            continue

        rows = by_speaker[pid]
        gender = rows[0]["gender"]
        native_language = rows[0]["native_language"]
        n_files = len(rows)
        print(f"[{done+1}/{remaining}] Participant {pid} "
              f"({gender}, L1={native_language}, {n_files} files) ...",
              end=" ", flush=True)

        t0 = time.time()
        ceiling = find_optimal_ceiling(rows)
        elapsed = time.time() - t0

        print(f"{ceiling} Hz  ({elapsed:.1f}s)")

        if key not in cache:
            cache[key] = {
                "gender": gender,
                "native_language": native_language,
                "ceilings": {},
            }
        entry = cache[key]
        if "ceilings" not in entry:
            entry["ceilings"] = {}
        entry["ceilings"][ctx_key] = {
            "ceiling": ceiling,
            "target_language": TARGET_LANGUAGE,
            "target_vowels": sorted(_CEILING_TARGET_VOWELS),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        done += 1

        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)

    total_time = time.time() - t_start_all
    print("=" * 60)
    print(f"Done. {done} speakers optimized in {total_time:.0f}s")
    print(f"Cache saved to {CACHE_PATH}")

if __name__ == "__main__":
    main()
