import os
import re
import json
import argparse
import unicodedata
import numpy as np
import librosa
import torchaudio
import scipy.signal as signal
from torchaudio import transforms as T
from typing import Dict, List, Tuple

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

def load_audio_mono(path: str, sr: int) -> np.ndarray:
    """Load mono float32 @ sr (torchaudio first, fallback librosa)."""
    try:
        wav, in_sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  
        if int(in_sr) != int(sr):
            wav = T.Resample(orig_freq=int(in_sr), new_freq=int(sr))(wav)
        return wav.numpy().astype(np.float32)
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)

def extract_melody_one_hot(
    audio_path: str, sr: int = 44100, cutoff: float = 261.2, win_length: int = 2048, hop_length: int = 256
) -> np.ndarray:
    y = load_audio_mono(audio_path, sr)
    nyq = 0.5 * sr
    b, a = signal.butter(N=2, Wn=float(cutoff)/float(nyq), btype="high", analog=False)
    try:
        y = signal.filtfilt(b, a, y)
    except Exception:
        y = signal.lfilter(b, a, y)

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=win_length, win_length=win_length, hop_length=hop_length
    )
    idx = np.argmax(chroma, axis=0)               # (T,)
    one_hot = np.zeros_like(chroma, dtype=np.float32)  # (12, T)
    one_hot[idx, np.arange(chroma.shape[1])] = 1.0
    return one_hot

def circular_best_accuracy(A: np.ndarray, B: np.ndarray) -> float:
    Tlen = min(A.shape[1], B.shape[1])
    if Tlen == 0:
        return 0.0
    A = A[:, :Tlen]
    B = B[:, :Tlen]
    best = 0.0
    for shift in range(12):
        B_roll = np.roll(B, shift=shift, axis=0)
        matches = ((A == B_roll) & (A == 1)).sum()
        acc = matches / Tlen
        if acc > best:
            best = acc
    return float(best)

def dtw_align(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sa = np.argmax(A, axis=0).astype(np.int32)
    sb = np.argmax(B, axis=0).astype(np.int32)
    a_feat = sa[None, :]
    b_feat = sb[None, :]
    D, wp = librosa.sequence.dtw(
        X=a_feat, Y=b_feat, metric=lambda x, y: 0.0 if x == y else 1.0
    )
    wp = wp[::-1]  
    Tnew = len(wp)
    A_new = np.zeros((12, Tnew), dtype=np.float32)
    B_new = np.zeros((12, Tnew), dtype=np.float32)
    for t, (ia, ib) in enumerate(wp):
        A_new[sa[ia], t] = 1.0
        B_new[sb[ib], t] = 1.0
    return A_new, B_new

def accuracy_between_paths(path_a: str, path_b: str, args) -> float:
    A = extract_melody_one_hot(path_a, sr=args.sr, win_length=args.win_length, hop_length=args.hop_length)
    B = extract_melody_one_hot(path_b, sr=args.sr, win_length=args.win_length, hop_length=args.hop_length)
    if args.align_time == "dtw":
        A, B = dtw_align(A, B)
    if args.no_key_invariant:
        Tlen = min(A.shape[1], B.shape[1])
        if Tlen == 0:
            return 0.0
        matches = ((A[:, :Tlen] == B[:, :Tlen]) & (A[:, :Tlen] == 1)).sum()
        return float(matches / Tlen)
    else:
        return circular_best_accuracy(A, B)

def is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS

def slug(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = s.replace("’", "'").replace("｜", "|").replace("⧸", "/")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def list_audio_basenames(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if is_audio(name):
            files.append(name)
    return sorted(files)

def build_pairs_from_dirs(target_root: str, gen_dir: str, gen_suffix: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    gen_paths = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if is_audio(f)]
    gen_by_name = {os.path.basename(p).lower(): p for p in gen_paths}

    def stems_index(paths):
        idx = {}
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            idx.setdefault(stem, []).append(p)
        return idx

    gen_by_stem = stems_index(gen_paths)

    gen_by_slug = {}
    for p in gen_paths:
        st = os.path.splitext(os.path.basename(p))[0]
        gen_by_slug.setdefault(slug(st), []).append(p)

    target_files = [f for f in os.listdir(target_root) if is_audio(f)]

    for fname in sorted(target_files):
        stem, ext = os.path.splitext(fname)
        stem_l = stem.lower()
        candidates: List[str] = []


        for name in [stem, stem + gen_suffix, stem_l, stem_l + gen_suffix]:
            cand = f"{name}.wav"
            p = gen_by_name.get(cand.lower())
            if p:
                candidates.append(p)

      
        if not candidates:
            for name in [stem, stem + gen_suffix, stem_l, stem_l + gen_suffix]:
                lst = gen_by_stem.get(name.lower(), [])
                candidates.extend(lst)

        if not candidates:
            for name in [stem, stem + gen_suffix, stem_l, stem_l + gen_suffix]:
                lst = gen_by_slug.get(slug(name), [])
                candidates.extend(lst)

        uniq, seen = [], set()
        for p in candidates:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        uniq.sort(key=lambda p: 0 if p.lower().endswith(".wav") else 1)

        if uniq:
            pairs[fname] = os.path.abspath(uniq[0])
        else:
            print(f"[WARN] no generated match for target: {fname}")

    print(f"[INFO] Auto-match pairs: {len(pairs)} / {len(target_files)}")
    return pairs

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs-json", type=str, default=None,
                    help="JSON: {target_filename.ext: /abs/generated.ext, ...}")
    ap.add_argument("--gen-dir", type=str, default=None,
                    help="Directory of generated audios (auto-match by filename)")
    ap.add_argument("--gen-suffix", type=str, default="_gen",
                    help="Suffix to append before extension when matching in --gen-dir ('' to disable)")
    ap.add_argument("--target-root", type=str, required=True,
                    help="Directory containing TARGET audio files (any of supported audio extensions)")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--win-length", type=int, default=2048)
    ap.add_argument("--hop-length", type=int, default=256)
    ap.add_argument("--align-time", type=str, default="none", choices=["none", "dtw"])
    ap.add_argument("--no-key-invariant", action="store_true")
    ap.add_argument("--out", type=str, default="results_task2/melody_acc.tsv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.pairs_json:
        with open(args.pairs_json, "r", encoding="utf-8") as f:
            pairs = json.load(f)   
    elif args.gen_dir:
        if not os.path.isdir(args.gen_dir):
            raise ValueError(f"--gen-dir not found: {args.gen_dir}")
        pairs = build_pairs_from_dirs(args.target_root, args.gen_dir, args.gen_suffix)
    else:
        raise ValueError("Please provide either --pairs-json or --gen-dir for generated audios.")

    results, total, s = [], 0, 0.0

    for target_name, gen_path in pairs.items():

        tgt_path = os.path.normpath(os.path.join(args.target_root, target_name))
        if not os.path.isfile(tgt_path):

            base, _ = os.path.splitext(target_name)
            found = None
            for ext in AUDIO_EXTS:
                p = os.path.join(args.target_root, base + ext)
                if os.path.isfile(p):
                    found = p
                    break
            tgt_path = found if found else tgt_path

        if not os.path.isfile(tgt_path):
            print(f"[WARN] missing target: {tgt_path}")
            continue
        if not os.path.isabs(gen_path):
            gen_path = os.path.abspath(gen_path)
        if not os.path.isfile(gen_path):
            print(f"[WARN] missing generated: {gen_path}")
            continue

        try:
            acc = accuracy_between_paths(tgt_path, gen_path, args)
        except Exception as e:
            print(f"[WARN] skip pair (feature fail): {tgt_path} | {gen_path} -> {e}")
            continue

        results.append((tgt_path, gen_path, 1, acc))
        total += 1
        s += acc

    avg = (s / total) if total else 0.0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("target\tgenerated\trank\tmelody_acc\n")
        for tgt, gen, rank, acc in results:
            f.write(f"{tgt}\t{gen}\t{rank}\t{acc:.6f}\n")
        f.write(f"# AVG\t-\t-\t{avg:.6f}\n")
    print(f"[OK] saved: {args.out} | AVG melody_acc={avg:.6f}")

if __name__ == "__main__":
    main()
