
from __future__ import annotations
import os, sys, json, glob, argparse, warnings, tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import librosa
import soundfile as sf
import torch
from tqdm import tqdm

AUDIO_GLOBS = ["*.wav","*.WAV","*.mp3","*.MP3","*.flac","*.FLAC","*.m4a","*.M4A","*.ogg","*.OGG"]

def gather_files(folder: Path) -> List[str]:
    files: List[str] = []
    for pat in AUDIO_GLOBS:
        files += glob.glob(str(folder / pat))
    return sorted(files)

def human(n: int) -> str:
    return f"{n:,}"

def load_clap(device: str = "cpu"):
    from laion_clap import CLAP_Module
    model = CLAP_Module(enable_fusion=False)
    # 會自動下載官方 checkpoint（若未快取）
    model.load_ckpt()
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def embed_paths_filelist(model, paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    ok_vecs, ok_paths = [], []
    for p in tqdm(paths, desc="CLAP embeddings (filelist)"):
        try:
            emb = model.get_audio_embedding_from_filelist([p], use_tensor=True)  # (1, D)
            v = emb.detach().cpu().numpy()[0]
            if not np.isfinite(v).all() or np.linalg.norm(v) == 0:
                raise RuntimeError("invalid/zero embedding")
            ok_vecs.append(v); ok_paths.append(p)
        except Exception as e:
            warnings.warn(f"[skip] {p} -> {e}")
    if len(ok_vecs) == 0:
        raise RuntimeError("no valid embeddings")
    return np.vstack(ok_vecs), ok_paths

def slice_segments(y: np.ndarray, sr: int, segment_sec: float, mode: str, multi_crops: int) -> List[np.ndarray]:
    seg_len = int(segment_sec * sr)
    if seg_len <= 0:
        return [y.astype(np.float32)]

    ylen = len(y)
    segs: List[np.ndarray] = []

    def safe_slice(start: int):
        s = max(0, min(start, max(0, ylen - seg_len)))
        return y[s:s+seg_len]

    if mode == "front":
        segs = [safe_slice(0)]
    elif mode == "middle":
        segs = [safe_slice((ylen - seg_len)//2)]
    elif mode == "back":
        segs = [safe_slice(ylen - seg_len)]
    elif mode == "uniform":
        if multi_crops <= 1:
            segs = [safe_slice(0)]
        else:
            # 均勻抽 multi_crops 段
            gap = max(1, (ylen - seg_len) // max(1, (multi_crops - 1)))
            starts = [i*gap for i in range(multi_crops)]
            segs = [safe_slice(s) for s in starts]
    elif mode == "front-middle-back":
        starts = [0, (ylen - seg_len)//2, max(0, ylen - seg_len)]
        segs = [safe_slice(s) for s in starts]
    else:
        # 預設前段
        segs = [safe_slice(0)]

    # RMS normalize，避免超小振幅
    outs = []
    for seg in segs:
        if len(seg) == 0:
            continue
        seg = seg.astype(np.float32)
        rms = np.sqrt(np.mean(seg**2) + 1e-8)
        seg = seg / rms
        outs.append(seg)
    return outs if outs else [y.astype(np.float32)]

def write_temp_wav_and_embed(model, y_list: List[np.ndarray], sr: int) -> np.ndarray:
    tmp_paths = []
    try:
        for y in y_list:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp_path, y, sr, subtype="PCM_16")
            tmp_paths.append(tmp_path)
        # 逐段取 embedding，再平均
        vecs = []
        for p in tmp_paths:
            emb = model.get_audio_embedding_from_filelist([p], use_tensor=True)
            v = emb.detach().cpu().numpy()[0]
            vecs.append(v)
        vecs = np.stack(vecs, axis=0)  # (N, D)
        v_mean = vecs.mean(axis=0)
        # 防呆
        if not np.isfinite(v_mean).all() or np.linalg.norm(v_mean) == 0:
            raise RuntimeError("invalid/zero averaged embedding")
        return v_mean
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

@torch.no_grad()
def embed_with_optional_cropping(model, paths: List[str], segment_sec: float,
                                 segment_mode: str, multi_crops: int) -> Tuple[np.ndarray, List[str]]:
    if segment_sec <= 0:
        # 直接整檔
        return embed_paths_filelist(model, paths)

    ok_vecs, ok_paths = [], []
    for p in tqdm(paths, desc=f"CLAP embeddings (segment={segment_sec}s, mode={segment_mode}, crops={multi_crops})"):
        try:
            y, sr = librosa.load(p, sr=48_000, mono=True)  # 對齊 CLAP 預設 48k
            if y is None or y.size == 0:
                raise RuntimeError("empty audio")
            segs = slice_segments(y, sr, segment_sec, segment_mode, multi_crops)
            v = write_temp_wav_and_embed(model, segs, sr)
            ok_vecs.append(v); ok_paths.append(p)
        except Exception as e:
            warnings.warn(f"[skip] {p} -> {e}")
    if len(ok_vecs) == 0:
        raise RuntimeError("no valid embeddings after cropping")
    return np.vstack(ok_vecs), ok_paths

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

def main():
    ap = argparse.ArgumentParser(description="Task 1 Retrieval with CLAP (clean baseline)")
    ap.add_argument("--target", type=str, required=True, help="target_music_list_60s directory")
    ap.add_argument("--ref",    type=str, required=True, help="reference_music_list_60s directory")
    ap.add_argument("--topk",   type=int, default=5, help="Top-K to report per target")
    ap.add_argument("--out",    type=str, default="retrieval_topk.json", help="Output JSON file")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cuda","cpu"])
    # 裁切與多裁切參數（可選）
    ap.add_argument("--segment-sec", type=float, default=0.0,
                    help="If >0, cut each audio into fixed-length segments before embedding (recommended: 10~30)")
    ap.add_argument("--segment-mode", type=str, default="front",
                    choices=["front","middle","back","uniform","front-middle-back"],
                    help="How to pick segments when --segment-sec > 0")
    ap.add_argument("--multi-crops", type=int, default=1,
                    help="If >1 and segment-mode is 'uniform', take N evenly spaced crops and average embeddings")

    args = ap.parse_args()

    target_dir = Path(args.target).expanduser().resolve()
    ref_dir    = Path(args.ref).expanduser().resolve()
    out_path   = Path(args.out).expanduser().resolve()

    print(f"[info] TARGET_DIR = {target_dir}")
    print(f"[info] REF_DIR    = {ref_dir}")
    print(f"[info] OUT_JSON   = {out_path}")
    print(f"[info] device     = {args.device}")
    print(f"[info] segment    = {args.segment_sec}s, mode={args.segment_mode}, crops={args.multi_crops}")

    if not target_dir.exists():
        sys.exit(f"[error] target dir not found: {target_dir}")
    if not ref_dir.exists():
        sys.exit(f"[error] ref dir not found: {ref_dir}")

    tgt_files_all = gather_files(target_dir)
    ref_files_all = gather_files(ref_dir)
    print(f"[info] #targets = {human(len(tgt_files_all))}")
    print(f"[info] #refs    = {human(len(ref_files_all))}")
    if len(tgt_files_all) == 0 or len(ref_files_all) == 0:
        sys.exit("[error] No audio found. Check folder names and extensions.")

    # Load CLAP
    model = load_clap(args.device)

    # Embeddings（依設定選「整檔」或「裁切/多裁切」）
    tgt_emb, tgt_files = embed_with_optional_cropping(model, tgt_files_all,
                                                      args.segment_sec, args.segment_mode, args.multi_crops)
    ref_emb, ref_files = embed_with_optional_cropping(model, ref_files_all,
                                                      args.segment_sec, args.segment_mode, args.multi_crops)

    # Similarity (targets x refs)
    S = cosine_matrix(tgt_emb, ref_emb)
    topk = max(1, min(args.topk, S.shape[1]))
    top_idx = np.argsort(-S, axis=1)[:, :topk]
    top_val = np.take_along_axis(S, top_idx, axis=1)

    # Build report
    report: Dict[str, List[Dict]] = {}
    for i, tpath in enumerate(tgt_files):
        items = []
        for k in range(topk):
            ridx = int(top_idx[i, k])
            items.append({
                "rank": k + 1,
                "ref_path": str(ref_files[ridx]),
                "clap_cosine": float(top_val[i, k])
            })
        report[Path(tpath).name] = items

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved → {out_path}")
    any_key = next(iter(report))
    print(f"🔎 example: {any_key} → {report[any_key][0]}")

if __name__ == "__main__":
    main()



