"""Turn the model's answers into file names. No file access: names in, names out."""

import re
import unicodedata
from itertools import count
from pathlib import Path

# Words the model adds that never belong in an asset's name (counted in 5,509 of its answers:
# "3d" 118 times, "model" 95, "low poly" about 96, "mesh" 7, "asset" 4, "render" 1).
NOT_NAMES = set("3d model mesh render lowpoly highpoly asset".split())
# A name made only of these words says nothing about the object.
FILLER = set("a an the of 3d model mesh render object low high poly lowpoly highpoly".split())
GENERIC = set("unknown thing item shape part piece geometry abstract form".split())


def drop_not_names(words: list[str]) -> list[str]:
    """'low_poly_wolf_3d_model' -> wolf. "low poly" goes as a pair, so 'low_table' keeps its 'low'."""
    kept, i = [], 0
    while i < len(words):
        if words[i].lower() in ("low", "high") and i + 1 < len(words) and words[i + 1].lower() == "poly":
            i += 2
            continue
        if words[i].lower() not in NOT_NAMES:
            kept.append(words[i])
        i += 1
    return kept


def slugify(text: str, max_words: int = 6, max_chars: int = 48) -> str:
    """Make the model's name safe as a file name without restyling it: 'SM_Bottle_Glass_01' -> 'SM_Bottle_Glass'.

    Keeps its words, order and case. Drops characters files can't hold, an extension the model
    added anyway, and its own trailing number, since cat3d numbers files itself.
    Empty when nothing specific is left ('3D model', 'unknown', a bare number).
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().strip()
    text = re.sub(r"\.[A-Za-z][A-Za-z0-9]{1,3}$", "", text)
    # Hex ids the model copied from a file name ('c3de8f6d87bd') are not words.
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text) if w and not re.fullmatch(r"[0-9a-f]{8,}", w.lower())]
    while words and words[-1].isdigit():
        words.pop()
    words = drop_not_names(words)
    if not any(w.lower() not in GENERIC | FILLER and re.search("[a-z]", w.lower()) for w in words):
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
        if re.fullmatch(re.escape(label) + r"_\d{3,}", src.stem, re.I) and src.stem.casefold() not in taken:
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
