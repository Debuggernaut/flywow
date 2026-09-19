import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as tf

GLB_PATH = Path(__file__).with_name("dragon.glb")
glb_bytes = GLB_PATH.read_bytes()

server = viser.ViserServer()
server.scene.add_grid("/grid", width=20.0, height=20.0)

# Hero, full size, parked at the origin
hero = server.scene.add_glb(
    "/hero",
    glb_data=glb_bytes,
    scale=1.0,
    position=(0.0, 0.0, 0.0),
    wxyz=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
)

# Smaller copies — same bytes, different names / poses
minis = []
for i, x in enumerate((-3.0, 3.0, 0.0)):
    h = server.scene.add_glb(
        f"/mini/{i}",
        glb_data=glb_bytes,
        scale=0.35,
        position=(x, 0.0, 2.0),
    )
    minis.append(h)

print("open http://localhost:8080")

t0 = time.time()
while True:
    t = time.time() - t0

    # park the hero, spin it slowly about Y
    hero.position = (0.0, 0.0, 0.0)
    hero.wxyz = tf.SO3.from_y_radians(0.3 * t).wxyz

    # orbit / spin the minis
    for i, h in enumerate(minis):
        angle = t + i * (2 * np.pi / 3)
        h.position = (3.0 * np.cos(angle), 0.0, 3.0 * np.sin(angle))
        h.wxyz = tf.SO3.from_y_radians(angle).wxyz
        # h.scale = 0.25 + 0.1 * np.sin(t)   # optional live scale

    time.sleep(1 / 30)