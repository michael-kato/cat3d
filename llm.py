"""Ask a local vision model, served by llama-server, what a contact sheet shows."""

import base64
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
    "Name this object with a professional game studio style naming convention. If unsure, give your best guess."
)
FOLLOW_UP = ("Now describe it in one sentence for a search index: what exactly it is, its style, color, era, or origin, "
             "and any brand, title or text you can read.")


# Names exporters and tools generate on their own; they say nothing about the object.
JUNK = set("object mesh node root rootnode scene gltfscenerootnode sketchfabmodel sketchfabscene default "
           "defaultmaterial material mat lambert blinn phong aistandardsurface standardsurface principledbsdf "
           "cube plane sphere cylinder empty armature camera light geometry group polysurface image texture".split())
FILE_SUFFIX = re.compile(r"\.(obj|fbx|stl|ply|glb|gltf|blend|dae|3ds|max|ma|mb|c4d|png|jpe?g|tiff?|tga|bmp|exr|dds|webp)\b.*$", re.I)


def clean_names(names: list[str], keep: int = 8) -> list[str]:
    """Drop generated junk ('Object_12', 'aiStandardSurface10SG', hex ids) and export suffixes:
    'stool.obj.cleaner.materialmerger.gles' -> 'stool'."""
    kept, seen = [], set()
    for name in names:
        name = FILE_SUFFIX.sub("", name)
        name = re.sub(r"([._\-\s]*\d+|sg)+$", "", name, flags=re.I).strip(" ._-")
        key = re.sub(r"[^a-z]", "", name.lower())
        if len(key) < 2 or key in JUNK or key in seen or "\ufffd" in name or re.fullmatch(r"[0-9a-f]{16,}", name.lower()):
            continue
        seen.add(key)
        kept.append(name)
    return kept[:keep]


def build_prompt(file_name: str, stats: dict, names: dict) -> str:
    hints = [f'original file name: "{file_name}"']
    for kind, found in names.items():
        if cleaned := clean_names(found):
            hints.append(f"{kind[:-1]} names: " + ", ".join(cleaned))
    hints.append(f"face count: {stats['faces']}")
    return QUESTION + "\nHints from inside the file (they may be meaningless):\n- " + "\n- ".join(hints)


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
