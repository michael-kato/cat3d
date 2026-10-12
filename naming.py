"""Turn the model's answers into file names. No file access: names in, names out."""

import re
import unicodedata
from itertools import count
from pathlib import Path

FILLER = set("a an the of 3d model mesh render object low high poly lowpoly highpoly".split())
GENERIC = set("unknown thing item shape part piece geometry abstract form".split())


def noun_first(words: list[str]) -> list[str]:
    """'red soda can' -> can soda red, so names sort by what the thing is, most specific word next.

    English puts the noun last, except before "of": 'bouquet of roses' -> bouquet roses.
    """
    if "of" in words:
        i = words.index("of")
        return words[i - 1::-1] + words[i + 1:] if i else words[1:]
    return words[::-1]


def slugify(text: str, max_words: int = 5, max_chars: int = 48) -> str:
    """'A Red 3D Soda Can!' -> 'can_soda_red'. Empty when nothing specific is left."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    words = noun_first([w for w in re.split(r"[^a-z0-9]+", text) if w])
    words = [w for w in words if w not in FILLER]
    # Generic words or bare numbers (the model echoing "73781.obj") say nothing about the object.
    if not any(w not in GENERIC and re.search("[a-z]", w) for w in words):
        return ""
    slug = ""
    for word in words[:max_words]:
        candidate = f"{slug}_{word}" if slug else word
        if len(candidate) > max_chars:
            break
        slug = candidate
    return slug


def plan_names(slugs: dict[Path, str | None], taken: set[str]) -> dict[Path, Path]:
    """A new path for every file, unique across the whole library: chair_001, chair_002, ...

    A slug of None (the model gave no usable name) becomes unknown_001, unknown_002, ...
    `taken` holds the names of all files that keep theirs. Numbers count names regardless of
    extension, and names compare case-insensitively (macOS, Windows). A file whose current
    name already fits keeps it, so planning a renamed library again changes nothing.
    """
    taken = {Path(n).stem.casefold() for n in taken}
    plan, pending = {}, []
    for src in sorted(slugs):
        label = slugs[src] or "unknown"
        if re.fullmatch(re.escape(label) + r"_\d{3,}", src.stem) and src.stem.casefold() not in taken:
            plan[src] = src
            taken.add(src.stem.casefold())
        else:
            pending.append(src)
    for src in pending:
        label = slugs[src] or "unknown"
        stem = next(stem for n in count(1) if (stem := f"{label}_{n:03d}").casefold() not in taken)
        taken.add(stem.casefold())
        plan[src] = src.with_name(stem + src.suffix)
    return plan
