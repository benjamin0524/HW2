import json
import argparse
import csv
import os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-json", type=str, default="retrieval_topk.json",
                    help="包含 target → 候選列表（含 ref_path）的 JSON 檔")
    ap.add_argument("--out-csv", type=str, default="results_task1/aesthetics_top1.csv",
                    help="輸出 CSV 路徑")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="可選：自訂 checkpoint.pt 路徑；預設自動下載官方權重")
    args = ap.parse_args()

    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except Exception as e:
        raise RuntimeError(
            "audiobox-aesthetics 未正確安裝；請先 `pip install audiobox_aesthetics`"
        ) from e


    with open(args.topk_json, "r") as f:
        data = json.load(f)


    os.makedirs(Path(args.out_csv).parent, exist_ok=True)


    print("🔄 Initializing predictor ...")
    predictor = initialize_predictor(ckpt=args.ckpt)
    print("✅ Predictor ready!")

    # 輸出 CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "ref_path", "CE", "CU", "PC", "PQ"])

        for target, items in data.items():

            ref_path = items[0]["ref_path"]
            try:

                scores_list = predictor.forward([{"path": ref_path}])
                scores = scores_list[0] if scores_list else {}
                CE = scores.get("CE")
                CU = scores.get("CU")
                PC = scores.get("PC")
                PQ = scores.get("PQ")
            except Exception as e:
                print(f"⚠️  評分失敗：{ref_path} -> {e}")
                CE = CU = PC = PQ = None

            writer.writerow([target, ref_path, CE, CU, PC, PQ])

    print(f"\n✅ Aesthetic evaluation finished. Results saved to: {args.out_csv}")

if __name__ == "__main__":
    main()








