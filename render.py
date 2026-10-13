"""Render 3D files into 2x2 contact sheets (front, side, top, three-quarter) with Blender.

The first half runs in cat3d and starts Blender processes. The second half runs inside
Blender, which is started as:

    blender -b --factory-startup --python render.py -- jobs.json results.jsonl
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:  # only importable when Blender runs this file
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    pass


# --- cat3d side ---------------------------------------------------------------------------

def render_chunk(blender: str, jobs: list[dict], size: int, file_timeout: float) -> list[dict]:
    """Render jobs ({"src", "sheet"}) in one Blender process. Returns one result per job.

    If Blender dies, the first unfinished file is retried alone (it's the likely culprit)
    and the rest go round again, so one bad file can't take its neighbours down with it.
    """
    with tempfile.TemporaryDirectory(prefix="cat3d-") as tmp:
        jobs_file, results_file = Path(tmp, "jobs.json"), Path(tmp, "results.jsonl")
        jobs_file.write_text(json.dumps({"size": size, "jobs": jobs}))
        command = [blender, "-b", "--factory-startup", "--python", __file__, "--", str(jobs_file), str(results_file)]
        try:
            done = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                  timeout=file_timeout * len(jobs))
            note = f"Blender exited with {done.returncode}: {done.stdout[-600:]}" if done.returncode else ""
        except subprocess.TimeoutExpired:
            note = f"timed out after {file_timeout * len(jobs):.0f}s"
        results = {}
        for line in results_file.read_text().splitlines() if results_file.exists() else []:
            try:
                result = json.loads(line)
                results[result["src"]] = result
            except json.JSONDecodeError:
                pass  # half-written line from a crash

    missing = [j for j in jobs if j["src"] not in results]
    if not missing:
        return list(results.values())
    if len(jobs) == 1:
        return [{"src": jobs[0]["src"], "ok": False, "error": note or "Blender produced no result"}]
    retried = render_chunk(blender, missing[:1], size, file_timeout)
    rest = render_chunk(blender, missing[1:], size, file_timeout) if missing[1:] else []
    return list(results.values()) + retried + rest


# --- Blender side -------------------------------------------------------------------------

BACKGROUND = (0.0, 0.0, 0.0)
MAX_NAMES = 40  # raw names per kind; llm.py drops the junk and keeps the best few
IMPORTERS = {
    ".obj": lambda path: bpy.ops.wm.obj_import(filepath=path),
    ".fbx": lambda path: bpy.ops.import_scene.fbx(filepath=path),
    ".glb": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".gltf": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".stl": lambda path: bpy.ops.wm.stl_import(filepath=path),
    ".ply": lambda path: bpy.ops.wm.ply_import(filepath=path),
}
# Camera direction, fixed rotation (None = aim at the centre), and the two axes that must fit the frame.
VIEWS = [
    ((0, -1, 0), (math.pi / 2, 0, 0), (0, 2)),           # front
    ((1, 0, 0), (math.pi / 2, 0, math.pi / 2), (1, 2)),  # side
    ((0, 0, 1), (0, 0, 0), (0, 1)),                      # top
    ((1, -1, 0.8), None, None),                          # three-quarter
]


def render_sheet(src: str, sheet: str, size: int, scratch: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(src)[1].lower()
    if ext not in IMPORTERS or "FINISHED" not in IMPORTERS[ext](src):
        raise RuntimeError(f"can't import {ext} file")
    objects = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.visible_get()]
    if not objects:
        raise ValueError("no visible mesh objects")

    corners = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    extent, centre = hi - lo, (lo + hi) / 2
    radius = max(extent.length / 2, 1e-6)

    # Names inside the file ("Chair_LOD0", "wood_albedo.png") are strong hints for the model.
    info = {
        "stats": {"bones": sum(len(a.bones) for a in bpy.data.armatures)},
        "names": {
            # Every object but cameras and lights: empties and rigs often carry the telling names.
            "objects": sorted({o.name for o in bpy.context.scene.objects if o.type not in {"CAMERA", "LIGHT"}})[:MAX_NAMES],
            "materials": sorted({s.material.name for o in objects for s in o.material_slots if s.material})[:MAX_NAMES],
            "textures": sorted({os.path.basename(i.filepath) or i.name
                                for i in bpy.data.images if i.source == "FILE"})[:MAX_NAMES],
            "animations": sorted({a.name for a in bpy.data.actions})[:MAX_NAMES],
        },
    }
    # Workbench draws a missing texture file bright magenta, which the model reads as paint.
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file and not os.path.exists(bpy.path.abspath(image.filepath)):
            image.source, image.generated_color = "GENERATED", (*BACKGROUND, 1.0)

    scene, tile = bpy.context.scene, size // 2
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = tile
    scene.view_settings.view_transform = "Standard"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_cavity = True
    scene.world = bpy.data.worlds.new("background")
    scene.world.color = BACKGROUND
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type, camera.data.clip_start, camera.data.clip_end = "ORTHO", radius * 0.01, radius * 10
    scene.collection.objects.link(camera)
    scene.camera = camera

    tiles = []
    for direction, rotation, frame_axes in VIEWS:
        direction = Vector(direction).normalized()
        camera.location = centre + direction * radius * 4
        if rotation is None:
            camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
            camera.data.ortho_scale = radius * 2.1
        else:
            camera.rotation_euler = rotation
            camera.data.ortho_scale = max(max(extent[a] for a in frame_axes), 1e-6) * 1.1
        scene.render.filepath = os.path.join(scratch, "view.png")
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(scene.render.filepath)
        pixels = np.empty(tile * tile * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        bpy.data.images.remove(image)
        tiles.append(pixels.reshape(tile, tile, 4))

    # Blender stores images bottom row first, so the sheet's top row is the upper half of the array.
    grid = np.ones((size, size, 4), dtype=np.float32)
    for (row, col), pixels in zip([(tile, 0), (tile, tile), (0, 0), (0, tile)], tiles):
        grid[row:row + tile, col:col + tile] = pixels
    grid[tile - 1:tile + 1, :, :3] = grid[:, tile - 1:tile + 1, :3] = 0.35  # separator lines
    grid[:, :, 3] = 1.0
    out = bpy.data.images.new("sheet", size, size)
    out.pixels.foreach_set(grid.ravel())
    out.filepath_raw, out.file_format = sheet, "PNG"
    out.save()
    return info


def blender_main() -> None:
    jobs_path, results_path = sys.argv[sys.argv.index("--") + 1:][:2]
    spec = json.loads(Path(jobs_path).read_text())
    with tempfile.TemporaryDirectory() as scratch, open(results_path, "a") as out:
        for job in spec["jobs"]:
            started = time.perf_counter()
            try:
                result = {"src": job["src"], "sheet": job["sheet"], "ok": True,
                          **render_sheet(job["src"], job["sheet"], spec["size"], scratch)}
            except Exception as e:  # one bad file must not stop the rest
                result = {"src": job["src"], "ok": False, "error": f"{type(e).__name__}: {e}"}
            result["render_ms"] = round((time.perf_counter() - started) * 1000)
            out.write(json.dumps(result) + "\n")
            out.flush()


if __name__ == "__main__":
    blender_main()
