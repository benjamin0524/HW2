# 🎵 HW2 – Music Generation & Evaluation

This repository contains the code, environment, and evaluation pipeline for the **Music Generation Assignment (HW2)**, including:
- **Task 1:** Retrieval-based evaluation using CLAP, Melody accuracy, and Aesthetic scoring  
- **Task 2:** Music generation and full evaluation pipeline  
- **Environment:** Reproducible setup with `hw2_environment.yaml`

---

## 🧩 Environment Setup

Clone this repo and create the environment:
git clone https://github.com/benjamin0524/HW2.git
cd HW2
conda env create -f hw2_environment.yaml
conda activate hw2


🎼 Task 1 – Retrieval & Evaluation

1️⃣ CLAP Retrieval (Task1/retrieval_clap.py)

Compute CLAP-based similarity between target and reference music.

python Task1/retrieval_clap.py \
  --target dataset/.../target_music_list_60s \
  --ref    dataset/.../reference_music_list_60s \
  --topk 5 \
  --out retrieval_topk.json
  
2️⃣ Melody Accuracy (Task1/evaluate_melody_batch.py)

Compare melodic similarity between target and retrieved audio (key-invariant, DTW optional).

python Task1/evaluate_melody_batch.py \
  --topk-json retrieval_topk.json \
  --target-root dataset/.../target_music_list_60s \
  --topk 1 \
  --align-time dtw \
  --out results_task1/melody_acc.tsv
  
3️⃣ Aesthetic Evaluation (Task1/evaluate_aesthetics.py)

python Task1/evaluate_aesthetics.py \
  --topk-json retrieval_topk.json \
  --out-csv results_task1/aesthetics_top1.csv

  
🎧 Task 2 – Music Generation & Full Evaluation

1️⃣ Generate Captions (task2/caption_qwen_audio_best.py)

Generate concise English captions using Qwen2-Audio and CLAP selection.

python task2/caption_qwen_audio_best.py

2️⃣ Music Generation (MuseControlLite)

python task2/MuseControlLite.py

3️⃣ CLAP Triple Similarity (task2/clap_triple_scores.py)
Compute CLAP similarity between:
Text ↔ Target
Text ↔ Generated
Generated ↔ Target

python task2/clap_triple_scores.py \
  --caps captions_qwen_audio_best.json \
  --target-dir dataset/.../target_music_list_60s \
  --gen-dir music_control_results \
  --out-json clap_triple_results.json \
  --out-csv clap_triple_results.csv
  
4️⃣ Aesthetic Comparison (task2/evaluate_aesthetics.py)

Compare Aesthetic scores between target and generated audios.

python task2/evaluate_aesthetics.py \
  --targets-dir dataset/.../target_music_list_60s \
  --gen-dir music_control_results \
  --out-csv aesthetics_targets_gen.csv

5️⃣ Melody Accuracy (task2/evaluate_melody_batch.py)

Compare melodic structure between generated and target music (with DTW alignment).

python task2/evaluate_melody_batch.py \
  --target-root dataset/.../target_music_list_60s \
  --gen-dir music_control_results \
  --gen-suffix _gen \
  --align-time dtw \
  --out melody_acc.tsv


