import os
import json
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from laion_clap import CLAP_Module
from tqdm import tqdm

AUDIO_DIR = "/home/ubuntu/music/HW2/dataset/fundwotsai/Deep_MIR_hw2/target_music_list_60s"
OUT_JSON = "captions_qwen_audio_best.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2-Audio-7B-Instruct" 

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map=None 
).to(device)


def load_clap(device: str = "cpu"):
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt() 
    model.to(device)
    model.eval()
    return model


PROMPT_ZH = (
    "Output concise, comma-separated English tags describing the music. "
    "Be accurate about genre, instruments, tempo, and mood. "
    "If the music has Chinese instruments (like bamboo flute or guzheng), include 'Chinese traditional' or 'world'. "
    "If it is film score, include 'soundtrack' or 'cinematic'. "
    "Follow the structure: [genre], [main instruments], [BPM number + 'bpm'], [time signature], [mood/emotion], [typical usage]. "
    "Use lowercase, no full sentences, no explanations.\n"
    "Examples:\n"
    "- world, bamboo flute, piano, 100 bpm, 4/4, emotional, cinematic soundtrack\n"
    "- jazz fusion, bass, drums, 120 bpm, 4/4, smooth, live performance\n"
    "- classical, solo piano, 90 bpm, 3/4, nostalgic, movie soundtrack"
)

def load_audio(path):
    audio, sr = sf.read(path, always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / (scale if scale > 0 else 1.0)
    else:
        audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


def generate_caption(audio_path: str, temperature: float = 0.6) -> str:
    audio, sr = load_audio(audio_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio, "sampling_rate": sr},
            {"type": "text", "text": PROMPT_ZH},
        ],
    }]
    text_in = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_in], audio=audio, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=temperature, 
            top_p=0.9,
            repetition_penalty=1.1
        )
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
 
    text = text.split("assistant:")[-1].strip().lower()
    text = text.replace("\n", " ").replace(" ,", ",")
    text = text.strip(" .")
    return text


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def generate_multiple_captions(audio_path: str, num_candidates: int = 5) -> List[str]:
    captions = []
    temperatures = [0.5, 0.6, 0.7, 0.8, 0.9]  
    for temperature in temperatures[:num_candidates]:
        caption = generate_caption(audio_path, temperature)
        captions.append(caption)
    return captions


def select_best_caption(captions: List[str], audio_path: str, clap_model) -> str:
    best_caption = ""
    best_score = -1
    for caption in captions:
      
        text_embedding = text_embed_clap(clap_model, caption)
        audio_embedding = audio_embed_clap(clap_model, audio_path, segment_sec=30, segment_mode="uniform", multi_crops=3)
        score = float(l2n(audio_embedding) @ l2n(text_embedding))
        if score > best_score:
            best_score = score
            best_caption = caption
    return best_caption

def main():
    results = {}
    names = [n for n in sorted(os.listdir(AUDIO_DIR))
             if n.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))]


    clap_model = load_clap(device)

    for name in names:
        path = os.path.join(AUDIO_DIR, name)
        try:
 
            captions = generate_multiple_captions(path, num_candidates=5)

            best_caption = select_best_caption(captions, path, clap_model)
            print(f"{name} -> {best_caption}")
            results[name] = best_caption
        except Exception as e:
            print("❌ FAIL:", name, e)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Saved best captions to", OUT_JSON)

if __name__ == "__main__":
    main()

