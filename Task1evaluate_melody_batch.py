import os
import json
import argparse
import numpy as np
import librosa
import torchaudio
import scipy.signal as signal
from torchaudio import transforms as T

def load_audio_mono(path, sr):
    try:
        wav, in_sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  # mono
        if in_sr != sr:
            wav = T.Resample(orig_freq=int(in_sr), new_freq=int(sr))(wav)
        y = wav.numpy().astype(np.float32)
        return y
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)

def extract_melody_one_hot(
    audio_path,
    sr=44100,
    cutoff=261.2,
    win_length=2048,
    hop_length=256
):
    y = load_audio_mono(audio_path, sr)

    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    try:
        y = signal.filtfilt(b, a, y)
    except Exception:
 
        y = signal.lfilter(b, a, y)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length
    )

    pitch_idx = np.argmax(chroma, axis=0)
    one_hot = np.zeros_like(chroma, dtype=np.float32)
    one_hot[pitch_idx, np.arange(chroma.shape[1])] = 1.0
    return one_hot  # (12, T)

def circular_best_accuracy(A, B):
    """
    Key-invariant accuracy by circularly shifting B along pitch classes.
    A,B: (12, T) one-hot
    """
    Tlen = min(A.shape[1], B.shape[1])
    A = A[:, :Tlen]
    B = B[:, :Tlen]
    best = 0.0
    for shift in range(12):
        B_roll = np.roll(B, shift=shift, axis=0)
        matches = ((A == B_roll) & (A == 1)).sum()
        acc = matches / Tlen
        if acc > best:
            best = acc
    return best

def dtw_align(A, B):
    """
    DTW on argmax pitch sequences; return aligned one-hot (12, T').
    """
    sa = np.argmax(A, axis=0).astype(np.int32)
    sb = np.argmax(B, axis=0).astype(np.int32)
    # cost: 0 if equal else 1
    a_feat = sa[None, :]
    b_feat = sb[None, :]
    D, wp = librosa.sequence.dtw(X=a_feat, Y=b_feat, metric=lambda x, y: 0.0 if x == y else 1.0)
    wp = wp[::-1]
    Tnew = len(wp)
    A_new = np.zeros((12, Tnew), dtype=np.float32)
    B_new = np.zeros((12, Tnew), dtype=np.float32)
    for t, (ia, ib) in enumerate(wp):
        A_new[sa[ia], t] = 1.0
        B_new[sb[ib], t] = 1.0
    return A_new, B_new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-json", type=str, default="retrieval_topk.json")
    ap.add_argument("--target-root", type=str, required=True,
                    help="directory containing target audio files named by the JSON keys")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--win-length", type=int, default=2048)
    ap.add_argument("--hop-length", type=int, default=256)
    ap.add_argument("--align-time", type=str, default="none", choices=["none", "dtw"])
    ap.add_argument("--no-key-invariant", action="store_true")
    ap.add_argument("--out", type=str, default="results_task1/melody_acc.tsv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.topk_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)  

    results = []
    total_pairs = 0
    sum_acc = 0.0

    for target_name, cand_list in mapping.items():
        tgt_path = os.path.normpath(os.path.join(args.target_root, target_name))
        if not os.path.isfile(tgt_path):
            print(f"[WARN] missing target: {tgt_path}")
            continue

        try:
            tgt_mel = extract_melody_one_hot(
                tgt_path, sr=args.sr, win_length=args.win_length, hop_length=args.hop_length
            )
        except Exception as e:
            print(f"[WARN] skip target (fail chroma): {tgt_path} -> {e}")
            continue


        cand_list = sorted(cand_list, key=lambda x: x.get("rank", 1e9))
        for item in cand_list[: args.topk]:
            cand_path = item.get("ref_path")
            if not cand_path:
                continue
            if not os.path.isabs(cand_path):
                cand_path = os.path.abspath(cand_path)
            if not os.path.isfile(cand_path):
                print(f"[WARN] missing candidate: {cand_path}")
                continue

            try:
                ret_mel = extract_melody_one_hot(
                    cand_path, sr=args.sr, win_length=args.win_length, hop_length=args.hop_length
                )
            except Exception as e:
                print(f"[WARN] skip cand (fail chroma): {cand_path} -> {e}")
                continue

            A, B = tgt_mel, ret_mel
            if args.align_time == "dtw":
                A, B = dtw_align(A, B)

            if args.no_key_invariant:
                Tlen = min(A.shape[1], B.shape[1])
                matches = ((A[:, :Tlen] == B[:, :Tlen]) & (A[:, :Tlen] == 1)).sum()
                acc = matches / Tlen
            else:
                acc = circular_best_accuracy(A, B)

            results.append((tgt_path, cand_path, item.get("rank", -1), acc))
            total_pairs += 1
            sum_acc += acc

    avg_acc = (sum_acc / total_pairs) if total_pairs > 0 else 0.0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("target\tcandidate\trank\tmelody_acc\n")
        for tgt, cand, rank, acc in results:
            f.write(f"{tgt}\t{cand}\t{rank}\t{acc:.6f}\n")
        f.write(f"# AVG\t-\t-\t{avg_acc:.6f}\n")

    print(f"[OK] saved: {args.out} | AVG melody_acc={avg_acc:.6f}")

if __name__ == "__main__":
    main()
