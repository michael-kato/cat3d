"""cat3d: rename 3D model files by what they look like.

    cat3d plan PATH...     render each file, ask the model what it is, record planned names in results.json
    cat3d rename PATH...   plan, then rename the files
    cat3d apply            rename every file that has a planned name in results.json (edit them first if you like)
    cat3d undo             put back the names changed by the last rename

results.json, next to this script, is the record of every asset: keyed by its current file name,
with its original name, the model's guess, description and confidence, and every rename so far.
It assumes names are unique across your library. Each run's images and llama-server log go to
logs/<date>_<time>/. Settings come from flags, then cat3d.toml next to this script, then the defaults below.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import PROMPT_ID, ask, build_prompt, running_server
from naming import colours_in, plan_names, split_name
from render import render_chunk

EXTENSIONS = {".obj", ".fbx", ".glb", ".gltf", ".stl", ".ply"}
_checkout = Path(__file__).resolve().parent
HOME = _checkout if (_checkout / "pyproject.toml").is_file() else Path.cwd()
RESULTS = HOME / "results.json"
RUN = HOME / "logs" / time.strftime("%Y-%m-%d_%H%M%S")  # this run's images, server log and renames log

DEFAULTS = {
    "blender": "blender",
    "llama_server": "llama-server",
    "server_url": "http://127.0.0.1:8080",
    "model": None,  # local .gguf; when unset llama-server downloads hf_repo
    "mmproj": None,  # vision projector .gguf that goes with `model`
    "hf_repo": "unsloth/Qwen3.5-2B-GGUF:Q4_K_M",
    "sheet_size": 512,
    "workers": max(1, (os.cpu_count() or 4) // 4),  # Blender processes at once
    "chunk_size": 25,  # files per Blender process
    "file_timeout": 120.0,  # seconds per file before Blender is killed
    "parallel": 4,  # llama-server slots
    "keep_together": [],  # phrases that stay whole in names ("teddy bear"); set in cat3d.toml only
}


def find_files(paths: list[Path]) -> list[Path]:
    found = set()
    for root in (p.absolute() for p in paths):
        for path in root.rglob("*") if root.is_dir() else [root]:
            hidden = path != root and any(part.startswith(".") for part in path.relative_to(root).parts)
            if path.suffix.lower() in EXTENSIONS and path.is_file() and not hidden:
                found.add(path)
    return sorted(found)


def load_results() -> dict[str, dict]:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save_results(results: dict[str, dict]) -> None:
    temp = RESULTS.with_name(f".{RESULTS.name}.{uuid.uuid4().hex[:8]}")
    temp.write_text(json.dumps(dict(sorted(results.items())), indent=1))
    os.replace(temp, RESULTS)  # never leaves half a file behind


def records_for(files: list[Path], results: dict[str, dict]) -> dict[Path, dict]:
    """The results.json record of each file, created if new. Fails if two files share a name."""
    clashes = [name for name, n in Counter(f.name for f in files).items() if n > 1]
    for f in files:
        other = results.get(f.name)
        if other and other["folder"] != str(f.parent) and Path(other["folder"], f.name).exists():
            clashes.append(f.name)
    if clashes:
        raise ValueError(f"names must be unique across the library, but these appear twice: {', '.join(sorted(set(clashes))[:10])}")
    records = {}
    for f in files:
        record = results.setdefault(f.name, {"original": f.name, "history": []})
        record["folder"] = str(f.parent)
        records[f] = record
    return records


def free_image_name(folder: Path, stem: str) -> Path:
    """folder/stem.png, or stem-2.png, ... in the rare case another image there already has the name."""
    path, n = folder / f"{stem}.png", 1
    while path.exists():
        n += 1
        path = folder / f"{stem}-{n}.png"
    return path


def classify(records: dict[Path, dict], results: dict[str, dict], settings) -> None:
    """Fill in each record's guess, description and confidence. Files answered by an earlier run are skipped."""
    model_id = Path(settings.model).name if settings.model else settings.hf_repo
    todo = [f for f, r in records.items() if not (r.get("model") == model_id and r.get("prompt_id") == PROMPT_ID and "guess" in r)]
    print(f"{len(records)} files: {len(records) - len(todo)} answered by an earlier run, {len(todo)} to do")
    if not todo:
        return
    RUN.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = 0

    def work(chunk: list[Path]) -> None:
        nonlocal done
        with tempfile.TemporaryDirectory(prefix="cat3d-") as tmp:
            jobs = [{"src": str(f), "sheet": os.path.join(tmp, f"{i}.png")} for i, f in enumerate(chunk)]
            for result in render_chunk(settings.blender, jobs, settings.sheet_size, settings.file_timeout):
                f = Path(result["src"])
                answer, started = {"model": model_id, "prompt_id": PROMPT_ID, "error": None}, time.perf_counter()
                try:
                    if not result["ok"]:
                        raise RuntimeError(f"render: {result['error']}")
                    # The original name, not the current one: a name cat3d gave earlier must not steer it again.
                    prompt = build_prompt(records[f]["original"], result["stats"], result["names"])
                    answer.update(ask(url, Path(result["sheet"]).read_bytes(), prompt))
                    image = free_image_name(RUN, f.stem)  # renamed to match the planned name later
                    shutil.move(result["sheet"], image)
                    answer["image"] = str(image.relative_to(HOME))
                    answer["geometry"] = result["stats"]["geometry"]  # tells colour variants of one mesh apart
                except Exception as e:  # one failure leaves one file untouched; the next run retries it
                    answer["error"] = str(e)
                # Time to open, render and name this one file (Blender measures its part).
                answer["time_ms"] = result.get("render_ms", 0) + round((time.perf_counter() - started) * 1000)
                with lock:
                    records[f].update(answer)
                    done += 1
                    outcome = answer["error"] or f"{answer['guess']!r} (confidence {_confidence(answer)}, {answer['time_ms']} ms)"
                    print(f"[{done}/{len(todo)}] {f}: {outcome}", flush=True)
        with lock:
            save_results(results)  # after every chunk, so an interrupted run loses at most one chunk

    chunks = [todo[i:i + settings.chunk_size] for i in range(0, len(todo), settings.chunk_size)]
    with running_server(settings, RUN / "llama-server.log") as url, ThreadPoolExecutor(settings.workers) as pool:
        list(pool.map(work, chunks))  # list() re-raises anything a worker didn't catch


def _confidence(record: dict) -> str:
    return "?" if record.get("confidence") is None else f"{record['confidence']:.2f}"


def label_images(records: dict[Path, dict]) -> None:
    """Name each file's image after its planned (or current) name: ring_001.fbx -> ring_001.png.

    Images whose names change go to temporary names first, so they can't clash with each other.
    """
    relabel = []
    for f, record in records.items():
        image = HOME / record["image"] if record.get("image") else None
        if image and image.exists() and image.stem != Path(record.get("planned", f.name)).stem:
            temp = image.with_name(f".relabel-{uuid.uuid4().hex[:8]}.png")
            image.rename(temp)
            relabel.append((record, temp, Path(record.get("planned", f.name)).stem))
    for record, temp, stem in relabel:
        image = free_image_name(temp.parent, stem)
        temp.rename(image)
        record["image"] = str(image.relative_to(HOME))


def add_variant_colours(records: dict[Path, dict], slugs: dict[Path, str | None]) -> None:
    """Colours stay out of names, except to tell apart files that would share a name and have identical
    geometry but different colours: chair_wooden_red_001 and chair_wooden_blue_001."""
    same = defaultdict(list)
    for f, record in records.items():
        if slugs[f] and record.get("geometry"):
            same[slugs[f], record["geometry"]].append(f)
    for files in same.values():
        if len({records[f]["colors"][0] for f in files if records[f]["colors"]}) > 1:
            for f in files:
                if records[f]["colors"]:
                    slugs[f] += "_" + records[f]["colors"][0]


def plan(settings) -> list[str]:
    """Work out a new name for every file and record it in results.json as "planned". Renames nothing."""
    results = load_results()
    records = records_for(find_files(settings.paths), results)
    classify(records, results, settings)
    named = {f: r for f, r in records.items() if not r.get("error")}
    taken = set(results) - {f.name for f in named}
    for folder in {f.parent for f in named}:
        taken |= set(os.listdir(folder)) - {f.name for f in named}
    slugs = {}
    for f, record in named.items():
        slugs[f], colours = split_name(record["guess"], settings.keep_together)
        record["colors"] = list(dict.fromkeys(colours + colours_in(record.get("description", ""))))
        slugs[f] = slugs[f] or None
    add_variant_colours(named, slugs)
    destinations = plan_names(slugs, taken)
    for f, record in named.items():
        if destinations[f] == f:
            record.pop("planned", None)
        else:
            record["planned"] = destinations[f].name
    label_images(named)
    save_results(results)

    for f, record in records.items():
        if record.get("error"):
            print(f"{f}  kept (failed: {record['error']})")
        elif "planned" in record:
            print(f"{f} -> {record['planned']}  (confidence {_confidence(record)})")
    planned = [f.name for f, r in named.items() if "planned" in r]
    failed = sum(bool(r.get("error")) for r in records.values())
    print(f"{len(planned)} to rename, {len(named) - len(planned)} already named, {failed} failed; see {RESULTS}")
    return planned


def rename_all(pairs: list[tuple[Path, Path]], log_path: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Rename every (src, dst), never overwriting, and log each rename as it happens.

    Two passes via temporary names, so swaps and chains (a->b, b->c) work in any order.
    """
    done, problems, staged = [], [], []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        log = csv.writer(f)

        def move(a: Path, b: Path) -> None:
            if b.exists():  # os.rename would silently overwrite on Linux and macOS
                raise FileExistsError(f"{b.name} already exists")
            os.rename(a, b)
            log.writerow([a, b])
            f.flush()

        for src, dst in pairs:
            temp = src.with_name(f".cat3d-{uuid.uuid4().hex[:8]}{src.suffix}")
            try:
                move(src, temp)
                staged.append((temp, src, dst))
            except OSError as e:
                problems.append(f"{src}: {e}")
        for temp, src, dst in staged:
            try:
                move(temp, dst)
                done.append((src, dst))
            except OSError as e:
                move(temp, src)
                problems.append(f"{src} kept its name: {e}")
    return done, problems


def apply(only: list[str] | None = None) -> int:
    """Rename files to their planned names and record each rename in its results.json history."""
    results = load_results()
    keys = [k for k, r in results.items() if "planned" in r and (only is None or k in only)]
    pairs = [(Path(results[k]["folder"], k), Path(results[k]["folder"], results[k]["planned"])) for k in keys]
    done, problems = rename_all(pairs, RUN / "renames.csv")
    # Take every renamed record out before re-adding any: a new name can be another file's old one.
    moved = [(src, dst, results.pop(src.name)) for src, dst in done]
    for src, dst, record in moved:
        record["history"].append({"run": RUN.name, "from": src.name, "to": dst.name})
        record.pop("planned", None)
        results[dst.name] = record
    save_results(results)
    print(f"renamed {len(done)} files; to put them back: cat3d undo")
    for p in problems:
        print(f"  skipped {p}")
    return 1 if problems else 0


def undo() -> int:
    """Put back the names changed by the most recent run that renamed anything."""
    results = load_results()
    runs = [r["history"][-1]["run"] for r in results.values() if r["history"]]
    if not runs:
        print("nothing to undo")
        return 0
    last = max(runs)
    keys = [k for k, r in results.items() if r["history"] and r["history"][-1]["run"] == last]
    pairs = [(Path(results[k]["folder"], k), Path(results[k]["folder"], results[k]["history"][-1]["from"])) for k in keys]
    done, problems = rename_all(pairs, RUN / "renames.csv")
    moved = [(dst, results.pop(src.name)) for src, dst in done]  # all out first, as in apply()
    for dst, record in moved:
        record["history"].pop()
        results[dst.name] = record
    save_results(results)
    print(f"put back {len(done)} names from run {last}")
    for p in problems:
        print(f"  skipped {p}")
    return 1 if problems else 0


def load_settings() -> dict:
    path = HOME / "cat3d.toml"
    from_file = tomllib.loads(path.read_text()) if path.exists() else {}
    # Paths may use $HOME, $USER, %USERNAME% (Windows) or ~ so the file isn't tied to one account.
    from_file = {k: os.path.expandvars(os.path.expanduser(v)) if isinstance(v, str) else v
                 for k, v in from_file.items()}
    unknown = set(from_file) - set(DEFAULTS)
    if unknown:
        sys.exit(f"cat3d: unknown setting(s) in {path}: {', '.join(sorted(unknown))}")
    return {**DEFAULTS, **from_file}


def main(argv: list[str] | None = None) -> int:
    settings = argparse.ArgumentParser(add_help=False)
    for key, default in DEFAULTS.items():
        if isinstance(default, list):
            continue  # lists live in cat3d.toml, not on the command line
        settings.add_argument("--" + key.replace("_", "-"), dest=key, type=str if default is None else type(default))
    settings.set_defaults(**load_settings())

    parser = argparse.ArgumentParser(prog="cat3d", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("plan", "suggest names into results.json, renaming nothing"), ("rename", "plan, then rename")):
        command = commands.add_parser(name, parents=[settings], help=help_text)
        command.add_argument("paths", nargs="+", type=Path, help="files or folders (searched recursively)")
    commands.add_parser("apply", help="rename every file with a planned name in results.json")
    commands.add_parser("undo", help="put back the names changed by the last rename")

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan(args)
            print("nothing renamed yet; to rename as planned: cat3d apply")
            return 0
        if args.command == "rename":
            return apply(plan(args))
        return apply() if args.command == "apply" else undo()
    except (OSError, RuntimeError, ValueError) as e:
        print(f"cat3d: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
