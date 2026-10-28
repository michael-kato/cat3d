"""Ask a local vision model, served by llama-server, what a contact sheet shows."""

import base64
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

QUESTION = (
    "The image shows a 3D model from four angles: front, side, top, and three-quarter.\n"
    "What is this object? Reply with a short plain-English noun phrase of 4 words or fewer, the way you "
    "would say it in a sentence. Name the object itself, not the picture: never mention views, angles or "
    "that it is a 3D model. If it is a character or figure, name the figure, not what it holds or wears. "
    "Avoid brand names, model numbers, proper nouns and words read from text in the image; those belong "
    "in the description. If unsure, give your best guess. Reply with just the phrase."
)
FOLLOW_UP = ("Now describe it in one sentence for a search index: what exactly it is, its style, color, era, or origin, "
             "and any brand, title or text you can read.")  


# Names that exporters and tools generate on their own, after stripping numbers and suffixes.
# Every word here was seen in at least one of 313 sample files (295 Sketchfab GLBs, 18 mixed formats).
JUNK = set("sketchfabmodel image object rootnode material root gltfscenerootnode lambert cube sceneroot "
           "colladavisualscenegroup defaultmaterial cylinder pcylinder plane icosphere sphere mesh "
           "pcylinderlambert group default model blinn initialshadinggroup wire camera mat polysurface "
           "pcube phong take box node aistandardsurface gltfcreated gltfcreatedrootjoint pcubelambert "
           "defaultmat modelmaterial matrl armature empty scene standardsurface "
           # Also seen: Spanish and Italian Blender defaults, SketchUp, USD and OBJ export names.
           "cubo cilindro materiale cerchiobezier edgecolor color edge geom usdroot obj".split())
FILE_SUFFIX = re.compile(r"\.(obj|fbx|stl|ply|glb|gltf|blend|dae|3ds|max|ma|mb|c4d|png|jpe?g|tiff?|tga|bmp|exr|dds|webp)\b.*$", re.I)


def clean_names(names: list[str], keep: int = 8) -> list[str]:
    """Drop generated junk ('Object_12', 'aiStandardSurface10SG', hex ids) and export suffixes:
    'stool.obj.cleaner.materialmerger.gles' -> 'stool'."""
    kept, seen = [], set()
    for name in names:
        name = FILE_SUFFIX.sub("", name)
        # One trailing number or 'SG' at a time: a single regex for the whole run backtracks
        # exponentially on names like '118280388_302057217738075_7437610194166637494_n'.
        while (stripped := re.sub(r"(?:[._\-\s]*\d+|sg)$", "", name, flags=re.I)) != name:
            name = stripped
        name = name.strip(" ._-")
        key = re.sub(r"[^a-z]", "", name.lower())
        words = [w for w in re.split(r"[^a-z]+", name.lower()) if w]
        if len(key) < 2 or key in JUNK or all(w in JUNK for w in words) or key in seen \
                or "\ufffd" in name or re.fullmatch(r"[0-9a-f]{16,}", name.lower()):
            continue
        seen.add(key)
        kept.append(name)
    return kept[:keep]


def looks_like_id(text: str) -> bool:
    """'c3de8f6d87bd4cd19ccd1f6ead3d4d2b' or '73781': a database id, not a name."""
    return re.fullmatch(r"[0-9a-f]{8,}|\d+", text.lower()) is not None


def build_prompt(file_name: str, stats: dict, names: dict) -> str:
    hints = [] if looks_like_id(Path(file_name).stem) else [f'original file name: "{file_name}"']
    for kind, found in names.items():
        if cleaned := clean_names(found):
            hints.append(f"{kind[:-1]} names: " + ", ".join(cleaned))
    if stats["bones"]:
        hints.append(f"rigged with {stats['bones']} bones")
    if not hints:
        return QUESTION
    return (QUESTION + "\nHints from the 3D file itself, for when the image is hard to recognize. They may be "
            "auto-generated or in another language; trust the image over them:\n- " + "\n- ".join(hints))


# Changes whenever the questions or the hints do, so answers to an older prompt are asked again.
PROMPT_ID = hashlib.sha1("".join([QUESTION, FOLLOW_UP, " ".join(sorted(JUNK)), inspect.getsource(clean_names),
                                  inspect.getsource(looks_like_id), inspect.getsource(build_prompt)]).encode()).hexdigest()[:8]


def _chat(server_url: str, messages: list[dict], max_tokens: int) -> dict:
    body = {
        "messages": messages,
        # Plain text rather than a JSON schema: grammar-forced tokens make the logprobs meaningless.
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stop": ["\n"],
        "logprobs": True,
    }
    request = urllib.request.Request(server_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]


def ask(server_url: str, png: bytes, prompt: str) -> dict:
    """The model's generic name for one contact sheet, how sure it was, and a one-sentence description.

    Two turns of one conversation: asked together, the 2B model writes a sentence as the name.
    confidence is its probability (0-1) for the least certain token of the name; on 295 labelled
    objects that separated right from wrong names best of the measures tried.
    """
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
    messages = [{"role": "user", "content": [image, {"type": "text", "text": prompt}]}]
    named = _chat(server_url, messages, max_tokens=16)
    guess = (named["message"].get("content") or "").strip()
    logprobs = [t["logprob"] for t in (named.get("logprobs") or {}).get("content") or [] if t["token"].strip()]
    messages += [{"role": "assistant", "content": guess}, {"role": "user", "content": FOLLOW_UP}]
    described = _chat(server_url, messages, max_tokens=60)
    return {"guess": guess, "description": (described["message"].get("content") or "").strip(),
            "confidence": round(math.exp(min(logprobs)), 3) if logprobs else None}


def _healthy(server_url: str) -> bool:
    try:
        with urllib.request.urlopen(server_url.rstrip("/") + "/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


@contextmanager
def running_server(settings, log_path: Path):
    """Use the llama-server at settings.server_url, or start one for the run and stop it afterwards."""
    if _healthy(settings.server_url):
        yield settings.server_url
        return
    url = urlsplit(settings.server_url)
    if settings.model:
        model = ["-m", settings.model] + (["--mmproj", settings.mmproj] if settings.mmproj else [])
    else:
        model = ["-hf", settings.hf_repo]  # downloads the model and its mmproj on first use
    command = [settings.llama_server, *model, "--host", url.hostname or "127.0.0.1", "--port", str(url.port or 8080),
               "-ngl", "99", "-fa", "on", "-np", str(settings.parallel), "-c", str(settings.parallel * 4096)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900  # a first -hf run has to download the model
        while not _healthy(settings.server_url):
            if process.poll() is not None:
                hint = ""
                if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
                    hint = ("\nHSA_OVERRIDE_GFX_VERSION is set, which crashes llama.cpp builds made for your exact GPU;"
                            " try: env -u HSA_OVERRIDE_GFX_VERSION cat3d ...")
                raise RuntimeError(f"llama-server exited with code {process.returncode}; see {log_path}{hint}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server did not start; see {log_path}")
            time.sleep(0.5)
        yield settings.server_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
