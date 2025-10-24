import os
import csv
import argparse
from pathlib import Path
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
def list_audio_files(folder: Path):
    files = {}
    for p in folder.glob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            files[p.stem] = str(p)
    return files

def find_generated_match(gen_dir: Path, stem: str):
    patterns = [
        f"{stem}_gen",
        f"generated_{stem}",
        f"{stem}"
    ]
    for pat in patterns:
        for ext in AUDIO_EXTS:
            cand = gen_dir / f"{pat}{ext}"
            if cand.exists():
                return str(cand)
    return None

def safe_forward(predictor, path):
    try:
        res = predictor.forward([{"path": path}])
        return res[0] if res and len(res) > 0 else {}
    except Exception as e:
        print(f"⚠️ {path} 評分失敗：{e}")
        return {}

def main():
    ap = argparse.ArgumentParser(description="Audiobox-Aesthetics 比較：Target vs Generated")
    ap.add_argument("--targets-dir", required=True, help="目標音檔資料夾")
    ap.add_argument("--gen-dir", required=True, help="生成音檔資料夾")
    ap.add_argument("--out-csv", default="aesthetics_targets_gen.csv", help="輸出 CSV 路徑")
    ap.add_argument("--ckpt", default=None, help="可選：指定 checkpoint.pt")
    args = ap.parse_args()

    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except Exception as e:
        raise RuntimeError("請先安裝 audiobox_aesthetics：`pip install audiobox_aesthetics`") from e

    predictor = initialize_predictor(ckpt=args.ckpt)
    print("✅ Predictor ready!")
    targets = list_audio_files(Path(args.targets_dir))
    gen_dir = Path(args.gen_dir)
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stem",
            "target_path", "gen_path",
            "T_CE", "T_CU", "T_PC", "T_PQ",
            "G_CE", "G_CU", "G_PC", "G_PQ"
        ])

        for stem, tgt_path in targets.items():
            gen_path = find_generated_match(gen_dir, stem)
            if not gen_path:
                print(f"⚠️ 找不到 {stem} 對應的生成音檔，略過。")
                continue

            t_score = safe_forward(predictor, tgt_path)
            g_score = safe_forward(predictor, gen_path)

            writer.writerow([
                stem,
                tgt_path, gen_path,
                t_score.get("CE"), t_score.get("CU"), t_score.get("PC"), t_score.get("PQ"),
                g_score.get("CE"), g_score.get("CU"), g_score.get("PC"), g_score.get("PQ"),
            ])

    print(f"✅ 結果已輸出至: {args.out_csv}")

if __name__ == "__main__":
    main()



