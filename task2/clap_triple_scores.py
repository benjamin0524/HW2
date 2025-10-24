import os
import re
import glob
import json
import argparse
from typing import List, Dict
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
def load_clap(device: str = "cpu"):
    from laion_clap import CLAP_Module
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()  
    model.to(device)
    model.eval()
    return model

def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(l2n(vec1), l2n(vec2)))


def peak_normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(y)) if y.size else 0.0
    return y * (peak / m) if m > 0 else y

def slice_segments(y: np.ndarray, sr: int, seg_len: int, mode: str, multi_crops: int) -> List[np.ndarray]:
    ylen = len(y)
    outs = []

    def safe_slice(start: int):
        s = max(0, min(start, max(0, ylen - seg_len)))
        seg = y[s:s + seg_len].astype(np.float32)
        # RMS normalize per-segment for robustness
        rms = np.sqrt(np.mean(seg**2) + 1e-8)
        seg = seg / (rms if rms > 0 else 1.0)
        return seg

    if ylen <= seg_len or multi_crops <= 1:
        return [safe_slice(0)]

    if mode == "uniform":
        crops = max(1, int(multi_crops))
        gap = max(1, (ylen - seg_len) // (crops - 1)) if crops > 1 else 1
        starts = [i * gap for i in range(crops)]
        outs = [safe_slice(s) for s in starts]
    elif mode == "head":
        outs = [safe_slice(0)]
    elif mode == "tail":
        outs = [safe_slice(ylen - seg_len)]
    else:
        crops = max(1, int(multi_crops))
        gap = max(1, (ylen - seg_len) // (crops - 1)) if crops > 1 else 1
        starts = [i * gap for i in range(crops)]
        outs = [safe_slice(s) for s in starts]

    return outs


def write_temp_wav_and_embed(model, y_list: List[np.ndarray], sr: int) -> np.ndarray:
    import tempfile
    tmp_paths = []
    try:
        for y in y_list:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp_path, y, sr, subtype="PCM_16")
            tmp_paths.append(tmp_path)

        vecs = []
        for p in tmp_paths:
            emb = model.get_audio_embedding_from_filelist([p], use_tensor=True)  # (1, D)
            v = emb.detach().cpu().numpy()[0]
            vecs.append(v)
        v_mean = np.mean(np.stack(vecs, axis=0), axis=0)
        if not np.isfinite(v_mean).all() or np.linalg.norm(v_mean) == 0:
            raise RuntimeError("invalid/zero averaged embedding")
        return v_mean
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass


def audio_embed_clap(model, path: str, segment_sec: float = 30, segment_mode: str = "uniform", multi_crops: int = 3) -> np.ndarray:
    y, sr = librosa.load(path, sr=48000, mono=True)
    if y is None or y.size == 0:
        raise RuntimeError(f"empty audio: {path}")
    y = peak_normalize(y, 0.99)
    seg_len = int(segment_sec * sr)
    segs = slice_segments(y, sr, seg_len, segment_mode, multi_crops)
    return write_temp_wav_and_embed(model, segs, sr)


def text_embed_clap(model, text: str) -> np.ndarray:
    v = model.get_text_embedding([text], use_tensor=True)  # (1, D)
    return v.detach().cpu().numpy()[0]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def list_audio_files(root: str) -> List[str]:
    paths = []
    for ext in AUDIO_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
    return sorted(paths)


def base_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_target_index(target_dir: str) -> Dict[str, str]:
    idx = {}
    for p in list_audio_files(target_dir):
        idx[base_of(p)] = p
    return idx


def build_caption_index(caps_json: str) -> Dict[str, str]:
    with open(caps_json, "r", encoding="utf-8") as f:
        caps = json.load(f)
    idx = {}
    for k, v in caps.items():
        idx[base_of(k)] = v
    return idx


def guess_base_from_gen(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"_gen(_\d+)?$", "", base)  
    base = re.sub(r"^gen_", "", base)       
    return base

def parse_args():
    ap = argparse.ArgumentParser(description="Compute CLAP cosine among (Target, Text, Generated)")
    ap.add_argument("--caps", type=str, default="captions_qwen_audio_best.json", help="caption JSON 檔 (name->caption)")
    ap.add_argument("--target-dir", type=str, default="target_music_list_60s", help="目標音檔資料夾（*.wav/*.mp3/...）")
    ap.add_argument("--gen-dir", type=str, default="gen_audio_qwen", help="生成音檔資料夾（*.wav）")
    ap.add_argument("--out-json", type=str, default="clap_triple_results.json", help="輸出 JSON 路徑")
    ap.add_argument("--out-csv",  type=str, default="clap_triple_results.csv",  help="輸出 CSV 路徑")
    ap.add_argument("--segment-sec", type=float, default=30.0, help="每段切片秒數（影響 CLAP 音訊嵌入）")
    ap.add_argument("--multi-crops", type=int, default=3, help="每首做幾個切片平均（3~5 通常夠）")
    ap.add_argument("--segment-mode", type=str, default="uniform", choices=["uniform", "head", "tail"], help="切片策略")
    return ap.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clap(device)

    # 建立索引
    target_index  = build_target_index(args.target_dir)
    caption_index = build_caption_index(args.caps)

    # 只掃 .wav 的生成輸出（依你習慣）
    gen_paths = [p for p in list_audio_files(args.gen_dir) if p.lower().endswith(".wav")]
    if not gen_paths:
        raise FileNotFoundError(f"No wav files found in {args.gen_dir}")

    results = {}
    miss_target = []
    miss_caption = []

    for p in tqdm(gen_paths, desc="CLAP(Text↔Target, Text↔Gen, Gen↔Target)"):
        gen_name = os.path.basename(p)
        base = guess_base_from_gen(gen_name)

        tgt_path = target_index.get(base, None)
        if tgt_path is None:
            miss_target.append(gen_name)
            continue

        cap = caption_index.get(base, None)
        if cap is None:
            miss_caption.append(gen_name)
            continue

        rec = {
            "base_name": base,
            "target_audio": os.path.basename(tgt_path),
            "generated_audio": gen_name,
            "caption": cap
        }

        try:
            # 音訊嵌入
            emb_gen = audio_embed_clap(
                model, p,
                segment_sec=float(args.segment_sec),
                segment_mode=args.segment_mode,
                multi_crops=int(args.multi_crops),
            )
            emb_tgt = audio_embed_clap(
                model, tgt_path,
                segment_sec=float(args.segment_sec),
                segment_mode=args.segment_mode,
                multi_crops=int(args.multi_crops),
            )
            # 文字嵌入
            emb_txt = text_embed_clap(model, cap)

            # 三組相似度
            rec["clap_text_target"] = compute_cosine_similarity(emb_txt, emb_tgt)  # i. Target ↔ Text
            rec["clap_text_gen"]    = compute_cosine_similarity(emb_txt, emb_gen)  # ii. Text ↔ Gen
            rec["clap_gen_target"]  = compute_cosine_similarity(emb_gen, emb_tgt)  # iii. Gen ↔ Target

        except Exception as e:
            rec["error"] = str(e)

        results[gen_name] = rec

    out_obj = {
        "config": {
            "gen_dir": args.gen_dir,
            "target_dir": args.target_dir,
            "caps": args.caps,
            "segment_sec": args.segment_sec,
            "multi_crops": args.multi_crops,
            "segment_mode": args.segment_mode,
            "device": device,
        },
        "missing_target_matches": miss_target,
        "missing_caption_matches": miss_caption,
        "results": results,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"✅ Done. Saved -> {args.out_json}")

    # 也輸出 CSV
    try:
        import pandas as pd
        rows = []
        for k, v in results.items():
            rows.append({
                "generated": k,
                "target": v.get("target_audio", ""),
                "base": v.get("base_name", ""),
                "clap_text_target": v.get("clap_text_target", np.nan),
                "clap_text_gen": v.get("clap_text_gen", np.nan),
                "clap_gen_target": v.get("clap_gen_target", np.nan),
            })
        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ CSV saved -> {args.out_csv}")
        if len(df):
            print("📊 Means:", df[["clap_text_target","clap_text_gen","clap_gen_target"]].mean(numeric_only=True).to_dict())
    except Exception as e:
        print(f"ℹ️ CSV 存檔失敗：{e}")


if __name__ == "__main__":
    main()



