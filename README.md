# cat3d

Rename 3D model files by what they look like.

cat3d opens each file in Blender, renders it from four angles into one image, asks a
local vision model (Qwen3.5-2B via llama.cpp) what the object is, and renames the
file to the answer: `mesh_0042.fbx` becomes `mirror_hand_001.fbx`. Nothing leaves your machine.

Formats: `.obj` `.fbx` `.glb` `.gltf` `.stl` `.ply`

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- [Blender](https://www.blender.org/download/) 4.2+ (tested with 5.2.1)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, built for your GPU or a prebuilt release

## Use

```bash
uv run cat3d rename ~/meshes   # names every file after what the model sees
uv run cat3d undo              # puts back the names from the last rename

uv run cat3d plan ~/meshes     # only suggests names: "planned" in results.json
uv run cat3d apply             # renames every file that has a planned name
```

`rename` and `plan` print each file's answer as it comes in. Use `plan` to review the
suggestions first, and edit any `planned` name in `results.json` before `apply`.

If no llama-server is running at `server_url`, cat3d starts one for the run and stops
it afterwards. Without a local `model` set, the server downloads
`unsloth/Qwen3.5-2B-GGUF:Q4_K_M` once (about 2 GB).

## results.json

One record per asset, keyed by its current file name. cat3d assumes names are unique across
your library, so it numbers them library-wide.

```json
"bottle_001.glb": {
 "original": "00fd0618c93b4979b258f25df0109a6a.glb",
 "folder": "/assets/props",
 "guess": "bottle",
 "description": "A vintage brown glass bottle with a cork stopper, labeled \"Le Coq Extrakt\".",
 "confidence": 0.61,
 "colors": ["brown"],
 "geometry": "8920v_8678f_0.0812x0.0812x0.2410",
 "image": "logs/2026-10-12_124025/bottle_001.png",
 "time_ms": 716,
 "history": [{"run": "2026-10-12_124100", "from": "00fd0618c93b4979b258f25df0109a6a.glb", "to": "bottle_001.glb"}]
}
```

- `guess` is the model's generic, searchable name, with no brands or proper names. The specifics go in
  `description`, for deeper searches.
- `confidence` (0 to 1) is how sure the model was of the least certain word in its guess. Every file
  gets the model's best guess; use `confidence` to decide which ones to trust. As a rough guide, in an
  earlier test on 295 labelled objects, names above 0.6 were right or close about 9 times in 10.
- `history` lists every rename, so the original name is never lost and `undo` can reverse the last run.
- `time_ms` is how long the file took to open, render and name.
- `colors` lists the colours from the guess and description. `geometry` (vertex and face counts and size)
  is how cat3d spots colour variants of the same mesh.

## Naming

- The model answers in plain English ("a red soda can"); cat3d turns that into the file name itself:
  lowercase, noun first and then the most specific words, at most four words, without filler like
  `3d`, `model` or `low poly`: `can_soda_red_001.fbx`. Details after "with", "of" and similar words follow
  the noun: "black robe with red patterns" becomes `robe_black_patterns_red`.
- Phrases listed under `keep_together` in `cat3d.toml` stay whole instead of being flipped:
  "brown teddy bear" becomes `teddy_bear_brown`, not `bear_teddy_brown`. List the ones that aren't a kind
  of their last word (a teddy bear is a toy, flip flops are sandals).
- Colours are metadata, not part of the name: "red and black flip-flops" becomes `flip_flops`, with
  `"colors": ["red", "black"]` in `results.json`. Metal colours describe the material, so they stay:
  "gold wedding ring" becomes `ring_wedding_gold`. "dark" always goes ("dark wood cabinet" becomes
  `cabinet_wood`); "light" only goes next to a colour, so `fixture_light` keeps it. A colour joins the name only to tell apart files that
  would share a name and have identical geometry: `chair_wooden_red_001` and `chair_wooden_blue_001`.
- No word appears twice: "table tennis table" becomes `table_tennis`.
- Names are numbered across the whole library: `can_soda_red_001.fbx`, `can_soda_red_002.fbx`, …
- A file only becomes `unknown_001.fbx` when the model gives no usable name at all ("object", "3d model").
- Files Blender can't open keep their names, and `results.json` says why.
- Running `rename` again on renamed files leaves them alone. Files are never overwritten.

## Settings

Put settings in `cat3d.toml` in this folder, or pass them as flags (`--workers 8`).
Every setting with its default is at the top of `cat3d.py`. Example:

```toml
blender = "/opt/blender-5.2.1-linux-x64/blender"
llama_server = "/path/to/llama.cpp/build/bin/llama-server"
model = "/path/to/Qwen3.5-2B-Q4_K_M.gguf"
mmproj = "/path/to/mmproj-F16.gguf"
```

Each run also gets a dated folder in `logs/`, such as `logs/2026-10-12_101500/`, holding the
llama-server log, a `renames.csv` of every rename it made, and the rendered image of each file,
named like the file (`ring_001.fbx` → `ring_001.png`) so you can flip through them to check the model.

## Code

| File | Does |
|---|---|
| `cat3d.py` | Command line, settings, the run, results.json, renaming and undo |
| `render.py` | Starts Blender processes, and is the script Blender runs to render each sheet |
| `llm.py` | Starts llama-server and asks the model |
| `naming.py` | Turns answers into unique file names |

Measured on a Ryzen 9 9900X with a Radeon RX 7800 XT: about 0.18 s per file, so 10k files in roughly 30 minutes.
