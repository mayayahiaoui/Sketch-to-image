from controlnet_aux import LineartDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
import os
import torch
from PIL import Image
import io, base64, random, threading, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps


device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if device == "cuda" else torch.float32

lineart_detector = LineartDetector.from_pretrained(
    "lllyasviel/Annotators"
)
# Load pretrained ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart",
    torch_dtype=dtype
)

# Load Stable Diffusion + ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
)

pipe.to(device)

# def preprocess_sketch(image, size=512):

#     # Resize while keeping the image square
#     image = image.convert("RGB")
#     image = image.resize((size, size))

#     # Extract lineart
#     lineart = lineart_detector(
#         image,
#         coarse=False
#     )

#     return lineart

def generate_image(
    control_image,
    prompt,
    negative_prompt="",
    control_strength=1.0,
    num_steps=30,
    guidance_scale=7.5,
    seed=42
):

    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=control_strength,
        generator=generator
    )

    return result.images[0]



app = Flask(__name__)
# Allow the HTML page (opened from your computer) to call this API.
# The ngrok-skip-browser-warning header lets the page bypass ngrok's free-tier warning page.
CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "ngrok-skip-browser-warning"])

gpu_lock = threading.Lock()  # one generation at a time on the GPU


def decode_image(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        d = request.get_json()
        sketch = decode_image(d["image"]).convert("RGB").resize((512, 512))

        if d.get("use_detector"):
            # photo / detailed drawing -> extract lineart (white lines on black)
            control = lineart_detector(sketch, coarse=False)
        else:
            # hand-drawn strokes are already lines: black on white -> invert to white on black
            control = ImageOps.invert(sketch.convert("L"))
        control = control.convert("RGB").resize((512, 512))

        seed = int(d.get("seed", 42))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        with gpu_lock:
            result = generate_image(
                control,
                prompt=d.get("prompt", ""),
                negative_prompt=d.get("negative_prompt", ""),
                control_strength=float(d.get("control_strength", 1.0)),
                num_steps=int(d.get("num_steps", 30)),
                guidance_scale=float(d.get("guidance_scale", 7.5)),
                seed=seed,
            )

        return jsonify(result=encode_image(result), control=encode_image(control), seed=seed)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

PORT = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT)