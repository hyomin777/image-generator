import torch
from PIL import Image
from model import TextToImageModel
import argparse
import os
from diffusers import DPMSolverMultistepScheduler
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate images from text')
    parser.add_argument("--prompt", type=str, required=True, help="Description of the image to generate")
    parser.add_argument("--negative_prompt", type=str, default="", help="Elements to exclude from the image")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Path to save the generated image")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Strength of text guidance")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--scheduler", type=str, default="dpm", choices=["dpm", "ddim"], help="Noise scheduler to use")
    args = parser.parse_args()

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("Loading model...")
    model = TextToImageModel()

    # Load trained model if specified
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print(f"Loaded trained model from {args.model_path}")

    model = model.to(device)

    # Configure scheduler
    if args.scheduler == "dpm":
        model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
        print("Using DPM scheduler")
    else:
        print("Using DDIM scheduler")

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to {args.seed}")

    # Generate image
    print(f"Generating image for prompt: '{args.prompt}'...")
    with torch.no_grad():
        image = model(
            text=args.prompt
        )

    # Save image
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype("uint8")
    image = Image.fromarray(image)
    image.save(args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
