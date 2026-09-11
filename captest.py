import dxcam
import numpy as np
from PIL import Image

camera = dxcam.create(output_color="RGB")
frame = camera.grab(new_frame_only=False)

base = Image.fromarray(frame).convert("RGB")
mask = Image.open("guidance.png").convert("RGBA")
if mask.size != base.size:
    mask = mask.resize(base.size, Image.Resampling.NEAREST)

frame_np = np.asarray(base, dtype=np.float32)
alpha = np.asarray(mask.split()[-1], dtype=np.float32) / 255.0  # 0..1
strength = 1.0  # 1.0 = fully black where alpha is 255

out_np = (frame_np * (1.0 - alpha[..., None] * strength)).clip(0, 255).astype(np.uint8)
Image.fromarray(out_np).save("frame.png")

#?? todo: why is this throwing an error on release?
#camera.release()

