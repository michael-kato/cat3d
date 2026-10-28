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


# Colours are metadata, not part of names. Every colour word seen in the model's answers and descriptions,
# except metal colours (gold, golden, silver, copper, bronze): those describe the material, so they stay.
COLORS = set("red black blue brown green gray grey white yellow purple orange pink teal beige "
             "cream maroon tan navy cyan magenta turquoise colorful".split())
# Shades count as colour only next to one: "light blue vase" -> vase, but "light fixture" keeps "light".
SHADES = set("light pale bright deep".split())
# Always a tone, never an identity in the model's answers ("dark wood cabinet", "dark glass bottle"), so it
# leaves names everywhere. Not a colour either, so it never becomes a variant's colour.
TONES = {"dark"}

# Words that start the extra-detail part of a phrase; the noun comes right before them.
DETAIL_STARTS = set("of with in on for from at by".split())
# Actions that also start a detail ("cow standing on a platform"). Only the ones seen in the model's
# answers: most words ending in -ing there are nouns ("ring" 17 times, "building", "casing").
ACTIONS = set("standing hanging".split())


def drop_not_names(words: list[str]) -> list[str]:
    """low poly wolf 3d model -> wolf. "low poly" goes as a pair, so 'low table' keeps its 'low'."""
    kept, i = [], 0
    while i < len(words):
        if words[i] in ("low", "high") and i + 1 < len(words) and words[i + 1] == "poly":
            i += 2
            continue
        if words[i] not in NOT_NAMES:
            kept.append(words[i])
        i += 1
    return kept


def join_kept(words: list[str], phrases: list[str]) -> list[str]:
    """Make each keep-together phrase one word, so noun_first can't split it: brown teddy bear -> brown teddy_bear.

    Spacing, hyphens and a plural 's' don't matter: "flip-flops" matches "flip flop".
    """
    def key(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w
    targets = sorted(([key(w) for w in re.split(r"[^a-z0-9]+", p.lower()) if w] for p in phrases), key=len, reverse=True)
    joined, i = [], 0
    while i < len(words):
        match = next((t for t in targets if [key(w) for w in words[i:i + len(t)]] == t), None)
        joined.append("_".join(words[i:i + len(match)]) if match else words[i])
        i += len(match) if match else 1
    return joined


def noun_first(words: list[str]) -> list[str]:
    """red bell pepper -> pepper bell red: the noun, then the closest (most specific) words first.

    English puts the noun last, except before a detail like "with red patterns" or "standing on a
    platform": black robe with red patterns -> robe black patterns red. Tested against NLTK's tagger
    on 4,373 of the model's answers: this rule picked the right noun in nearly every disagreement.
    """
    words = [w for w in words if w not in ("a", "an", "the", "and")]
    for i, word in enumerate(words):
        if i and (word in DETAIL_STARTS or word in ACTIONS):
            detail = words[i + 1:]
            while detail and detail[0] in DETAIL_STARTS:  # "standing on ..." -> "..."
                detail = detail[1:]
            return words[:i][::-1] + noun_first(detail)  # the detail is a phrase of its own
    return words[::-1]


def colour_words(words: list[str]) -> set[str]:
    """The colours and tones in a phrase, plus any shade standing next to a colour."""
    found = set()
    for i, word in enumerate(words):
        neighbours = {words[j] for j in (i - 1, i + 1) if 0 <= j < len(words)}
        if word in COLORS or word in TONES or (word in SHADES and neighbours & COLORS):
            found.add(word)
    return found


def unique_words(tokens: list[str]) -> list[str]:
    """No word twice, even inside kept phrases: table table_tennis -> table tennis."""
    used, kept = set(), []
    for token in tokens:
        parts = [p for p in token.split("_") if p not in used]
        used.update(parts)
        if parts:
            kept.append("_".join(parts))
    return kept


def split_name(text: str, keep_together: list[str] = (), max_words: int = 4, max_chars: int = 48) -> tuple[str, list[str]]:
    """The model's plain-English answer as (file name, colours):
    'red and black flip-flops' -> ('flip_flops', ['red', 'black']).

    Lowercase, noun first, no filler ('3d', 'model', 'low poly'), no word twice, at most `max_words`
    words. Phrases in `keep_together` stay whole: 'brown teddy bear' -> 'teddy_bear'. Colours go to the
    second value instead of the name, unless the colour is the noun itself ('an orange'). Drops an
    extension or trailing number the model added, since cat3d numbers files itself. The name is empty
    when nothing specific is left ('3D model', 'unknown', a bare number).
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().strip().lower()
    text = re.sub(r"\.[a-z][a-z0-9]{1,3}$", "", text)
    # Hex ids the model copied from a file name ('c3de8f6d87bd') are not words.
    words = [w for w in re.split(r"[^a-z0-9]+", text) if w and not re.fullmatch(r"[0-9a-f]{8,}", w)]
    while words and words[-1].isdigit():
        words.pop()
    tokens = join_kept(drop_not_names(words), keep_together)
    colours = colour_words(tokens)
    ordered = unique_words(noun_first(tokens))
    if not ordered:
        return "", []
    head = ordered[0]
    name = [head] + [w for w in ordered[1:] if w not in colours]
    found = list(dict.fromkeys(w for w in tokens if w in colours and w in COLORS and w != head))
    if not any(w not in GENERIC | FILLER and re.search("[a-z]", w) for w in name):
        return "", found
    slug = ""
    for word in name[:max_words]:
        candidate = f"{slug}_{word}" if slug else word
        if len(candidate) > max_chars:
            break
        slug = candidate
    return slug, found


def colours_in(text: str) -> list[str]:
    """Colour words in free text, in order: for descriptions."""
    return list(dict.fromkeys(w for w in re.split(r"[^a-z]+", text.lower()) if w in COLORS))


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
