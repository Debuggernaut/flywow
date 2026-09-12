import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf

OBJ = Path(__file__).with_name("dragon.obj")

mesh = trimesh.load(OBJ, force="mesh")
mesh.vertices -= mesh.centroid          # sit on the origin
# mesh.apply_scale(0.05)                # uncomment if it is huge / tiny

print(f"loaded {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

server = viser.ViserServer()
server.scene.add_grid("/grid", width=20.0, height=20.0, position=(0.0, 0.0, -1.0))

handle = server.scene.add_mesh_simple(
    "/dragon",
    vertices=np.asarray(mesh.vertices),
    faces=np.asarray(mesh.faces),
    color=(200, 80, 40),
    wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,  # OBJ is often Z-up; viser is Y-up
)

print("open http://localhost:8080")

# live spin — swap this loop for your Tk / sim updates later
t0 = time.time()
while True:
    yaw = 0.4 * (time.time() - t0)
    handle.wxyz = (np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)  # rotate about Y
    time.sleep(1 / 30)