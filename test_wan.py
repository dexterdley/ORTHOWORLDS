import os
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Using device: {device} with dtype: {dtype}")

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    output_path = "wan_test_output.mp4"

    print(f"Loading Wan2.1-T2V-1.3B pipeline from '{model_id}'...")

    # Load 3D VAE in float32 for numerical stability
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )

    # Load full Wan pipeline
    pipe = WanPipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=dtype
    )

    # Memory optimizations for consumer GPUs (~8-12GB VRAM)
    if device == "cuda":
        # Offload components to CPU when not active to save VRAM
        pipe.enable_model_cpu_offload()
        # Enable VAE tiling to decode long video sequences without OOM
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    else:
        pipe.to(device)

    # Example multi-agent racing prompt
    prompt = (
        "Multiplayer arcade racing game: Player 1 in a red sports car drifts sharply around a curve, "
        "while Player 2 in a blue supercar accelerates closely behind, high quality, 4k resolution, cinematic lighting."
    )
    negative_prompt = "blurry, low quality, distorted, jitter, static frame, glitch"

    print("\nPrompt:", prompt)
    print("Generating video frames (this may take a few minutes depending on your GPU)...")

    # Generate 49 frames (~3 seconds at 16 fps) at 480x832 resolution
    # You can reduce num_frames (e.g. 17 or 33) or height/width for faster testing
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=49,             # 16n + 1 (e.g., 17, 33, 49, 81)
            height=480,
            width=832,
            num_inference_steps=30,    # standard flow matching steps
            guidance_scale=6.0,
        )

    frames = output.frames[0]
    print(f"Generation finished! Exporting {len(frames)} frames to '{output_path}'...")
    export_to_video(frames, output_path, fps=16)
    print(f"Done! Video saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
