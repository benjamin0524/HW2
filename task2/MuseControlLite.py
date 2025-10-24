import os
import json
from pathlib import Path
import torch
import soundfile as sf
from tqdm import tqdm
import numpy as np
import librosa

import sys
sys.path.append('./MuseControlLite')
from MuseControlLite.MuseControlLite_setup import (
    setup_MuseControlLite,
    initialize_condition_extractors,
)


class MuseControlLiteGenerator:
   
    def __init__(
        self,
        target_dir="target_music_list_60s",
        captions_json="captions_qwen_audio_best.json",
        output_dir="gen_musecontrollite",
        condition_types=["melody_mono"],
        guidance_scale_text=7.0,
        guidance_scale_con=1.5,
        denoise_steps=50,
        device="cuda",
    ):

        self.target_dir = Path(target_dir)
        self.captions_json = Path(captions_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.condition_dir = self.output_dir / "conditions"
        self.condition_dir.mkdir(exist_ok=True)


        self.condition_types = [c for c in condition_types if c != "audio"]
        self.guidance_scale_text = guidance_scale_text
        self.guidance_scale_con = guidance_scale_con
        self.denoise_steps = denoise_steps
        self.device = device

        self.config = self._create_config()

 
        ckpt_dir = Path("./MuseControlLite/checkpoints")
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"找不到 MuseControlLite checkpoints 目錄：{ckpt_dir}\n"
                "請依 repo 指示先下載到 ./MuseControlLite/checkpoints/"
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        weight_dtype = torch.float16

        self.condition_extractors, transformer_ckpt = initialize_condition_extractors(self.config)
        self.model = setup_MuseControlLite(self.config, weight_dtype, transformer_ckpt).to(device)

        print("✓ MuseControlLite 初始化完成（禁止 audio condition）")


        if not self.captions_json.exists():
            raise FileNotFoundError(f"找不到 captions 檔案：{self.captions_json}")
        with open(self.captions_json, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
        print(f"✓ 已載入 captions：{len(self.captions)} 筆")


    def _create_config(self):
        """MuseControlLite 的必要設定（禁用 audio condition）"""
        return {
            "condition_type": self.condition_types,
            "GPU_id": "0",
            "apadapter": True,
            "ap_scale": 1.0,
            "guidance_scale_text": self.guidance_scale_text,
            "guidance_scale_con": self.guidance_scale_con,
            "guidance_scale_audio": 1.0,
            "denoise_step": self.denoise_steps,
            "sigma_min": 0.3,
            "sigma_max": 500,
            "weight_dtype": "fp16",
            "negative_text_prompt": "",
            "audio_mask_start_seconds": 14,
            "audio_mask_end_seconds": 47,
            "musical_attribute_mask_start_seconds": 0,
            "musical_attribute_mask_end_seconds": 0,
            "no_text": False,
            "show_result_and_plt": False,
            "transformer_ckpt_musical": "./MuseControlLite/checkpoints/woSDD-all/model_3.safetensors",
            "extractor_ckpt_musical": {
                "dynamics": "./MuseControlLite/checkpoints/woSDD-all/model_1.safetensors",
                "melody": "./MuseControlLite/checkpoints/woSDD-all/model.safetensors",
                "rhythm": "./MuseControlLite/checkpoints/woSDD-all/model_2.safetensors",
            },
            "transformer_ckpt_melody_mono": "./MuseControlLite/checkpoints/40000_Melody_mono/model_1.safetensors",
            "extractor_ckpt_melody_mono": {
                "melody": "./MuseControlLite/checkpoints/40000_Melody_mono/model.safetensors",
            },
        }

    @staticmethod
    def _prepare_condition_tensor(raw_condition_np, extractor, target_len=1024):
        import torch.nn.functional as F
        import numpy as np

        if raw_condition_np is None:
            return None

        if raw_condition_np.ndim == 1:
            cond = torch.from_numpy(raw_condition_np).float().unsqueeze(0).cuda()
        else:
            if raw_condition_np.shape[0] < raw_condition_np.shape[-1]:
                cond = torch.from_numpy(raw_condition_np).float().cuda()
            else:
                cond = torch.from_numpy(raw_condition_np.T).float().cuda()

        extracted = extractor(cond)
        if extracted.ndim == 2:
            extracted = extracted.unsqueeze(0)

        if extracted.shape[-1] != target_len:
            extracted = F.interpolate(extracted, size=target_len, mode="linear", align_corners=False)

        masked = torch.zeros_like(extracted)
        final_condition = torch.cat([masked, masked, extracted], dim=0)
        final_condition = final_condition.transpose(1, 2)
        return final_condition


    def _extract_melody(self, audio_path):
        import numpy as np, librosa
        try:
            from MuseControlLite.utils.extract_conditions import compute_melody
            melody = compute_melody(str(audio_path))
            return melody
        except Exception:
            y, sr = librosa.load(str(audio_path), sr=44100, mono=True)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            idx = np.argmax(chroma, axis=0)
            one_hot = np.zeros_like(chroma, dtype=np.float32)
            one_hot[idx, np.arange(chroma.shape[1])] = 1.0
            return one_hot

    def generate_all(self, seed=42):
        """批次生成所有音樂"""
        audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        targets = sorted([p for p in self.target_dir.glob("*") if p.suffix.lower() in audio_exts])
        print(f"🎵 找到 {len(targets)} 首目標音樂")

        for p in tqdm(targets, desc="MuseControlLite 生成中"):
            base = p.stem
            prompt = self.captions.get(p.name, self.captions.get(base, ""))
            if not prompt:
                print(f"[WARN] 找不到 {p.name} 的 caption，跳過。")
                continue

            try:
                melody_np = self._extract_melody(p)
                np.save(self.condition_dir / f"{base}_melody.npy", melody_np)
            except Exception as e:
                print(f"[WARN] 無法提取旋律條件 {p.name}: {e}")
                continue

            extractor = self.condition_extractors.get("melody")
            final_condition = self._prepare_condition_tensor(melody_np, extractor)

            with torch.no_grad():
                out = self.model(
                    extracted_condition=final_condition,
                    extracted_condition_audio=None,
                    prompt=prompt,
                    negative_prompt=self.config["negative_text_prompt"],
                    num_inference_steps=self.denoise_steps,
                    guidance_scale_text=self.guidance_scale_text,
                    guidance_scale_con=self.guidance_scale_con,
                    guidance_scale_audio=self.config["guidance_scale_audio"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=2097152 / 44100,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                )
                audios = out.audios

            out_path = self.output_dir / f"{base}_gen.wav"
            wav = audios[0].T.float().cpu().numpy()
            sr = getattr(self.model.vae, "sampling_rate", 44100)
            sf.write(str(out_path), wav, sr)
            print(f"✅ {p.name} → {out_path}")

        print("\n🎉 全部完成！")


if __name__ == "__main__":
    gen = MuseControlLiteGenerator(
        target_dir="/home/ubuntu/music/HW2/dataset/fundwotsai/Deep_MIR_hw2/target_music_list_60s",
        captions_json="captions_qwen_audio_best.json",
        output_dir="music_control_results",
        condition_types=["melody_mono"],
        guidance_scale_text=7.0,
        guidance_scale_con=1.5,
        denoise_steps=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    gen.generate_all(seed=42)



