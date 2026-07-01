"""One-shot migration for legacy storage artifacts damaged before the refusal /
embedding / meta-skill fixes landed.

Cleans two failure modes found across existing outputs/ run dirs:

  1. Poisoned SKILL.md — compress() once did a monolithic LLM rewrite with no
     refusal guard, so a safety refusal ("I cannot fulfill your request to ...")
     overwrote the whole playbook (ACE "context collapse"). This script strips
     refusal paragraphs and re-applies Agent-Skills YAML frontmatter; if nothing
     survives, it resets to a clean stub.

  2. Garbage memory_l4/meta_skills.json — meta-skills mined from defender
     catch-rate / metric telemetry (e.g. "reenact=100%, replay=100%") rather than
     transferable attack strategy. This script drops junk entries.

SAFE BY DEFAULT: dry-run unless --apply is given; every rewritten file is backed
up to <file>.bak first (never overwrites an existing .bak).

Usage:
    python migrate_storage.py --root ../outputs                  # dry-run report
    python migrate_storage.py --root ../outputs --apply          # do it
    python migrate_storage.py --root ../outputs/skills_v2 --apply # scope to one tree
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

_log = logging.getLogger("migrate_storage")

# Refusal markers (superset of ace_skill_lib._REFUSAL_MARKERS, kept local so this
# script has no import-time dependency on a configured ViviClient).
_REFUSAL_MARKERS = (
    "i cannot fulfill", "i can't fulfill", "i am unable to", "i'm unable to",
    "i cannot provide", "i can't provide", "i cannot assist", "i can't assist",
    "sorry, i cannot", "sorry, i can't", "i cannot help with", "i won't be able to",
    "regardless of the authorized", "even within an authorized",
    "from a defensive perspective", "i must decline", "i can not fulfill",
)

# memory_l4 junk: catch-rate / telemetry tokens that carry no transferable strategy.
_META_NOISE_TOKENS = (
    "caught", "catch rate", "catch_rate", "bypass rate", "=100%", "=0%",
    "confidence", "n_traj", "real image", "fake image",
)


def _block_is_refusal(block: str) -> bool:
    low = block.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_or_empty, body). Frontmatter = leading '---'…'---' block."""
    s = text.lstrip("\n")
    if not s.startswith("---"):
        return "", text
    end = s.find("\n---", 3)
    if end == -1:
        return "", text
    fm_end = s.find("\n", end + 1)
    if fm_end == -1:
        fm_end = len(s)
    return s[:fm_end + 1], s[fm_end + 1:]


def _frontmatter_for(family: str) -> str:
    name = f"forgery-{family.replace('_', '-')}-redteam"[:64]
    desc = (
        f"Strategic red-team playbook for '{family}' face-forgery attacks used to "
        f"harden an enterprise face-KYC anti-deepfake detector. Use when generating "
        f"or refining {family} attack chains, choosing operators and parameter ranges, "
        f"or avoiding detector cues known to catch {family} forgeries."
    )[:1024]
    return f"---\nname: {name}\ndescription: {desc}\n---\n\n"


def _family_from_path(skill_md: Path) -> str:
    # outputs/.../skills_v2/<family>/SKILL.md
    return skill_md.parent.name


def clean_skill_md(text: str, family: str) -> tuple[str, bool]:
    """Return (cleaned_text, changed).

    Surgical: only rewrite docs that actually contain a refusal block. Clean docs
    are left byte-for-byte untouched (frontmatter normalization is the live code's
    job on next save, not this migration's). When a refusal is present, strip the
    refusal block(s), re-apply Agent-Skills frontmatter, and reset to a clean stub
    if nothing meaningful survives.
    """
    fm, body = _split_frontmatter(text)
    blocks = body.split("\n\n")
    if not any(_block_is_refusal(b) for b in blocks):
        return text, False  # no poison → leave untouched
    kept = [b for b in blocks if b.strip() and not _block_is_refusal(b)]
    new_body = "\n\n".join(kept).strip()
    meaningful = new_body and any(
        ln.strip() and not ln.lstrip().startswith("#") and "_No experience yet._" not in ln
        for ln in new_body.splitlines()
    )
    if not meaningful:
        new_body = f"# {family} SOP\n\n_No experience yet._"
    cleaned = _frontmatter_for(family) + new_body + "\n"
    return cleaned, True


def clean_meta_skills(obj: dict) -> tuple[dict, int]:
    """Drop junk meta-skill entries. Return (cleaned_dict, n_removed)."""
    out = {}
    removed = 0
    for name, entry in obj.items():
        blob = f"{name} {entry.get('description','')} {entry.get('body','')}".lower()
        is_refusal = any(m in blob for m in _REFUSAL_MARKERS)
        is_noise = any(t in blob for t in _META_NOISE_TOKENS)
        # mostly-digits name => telemetry, not strategy
        digit_heavy = sum(c.isdigit() for c in name) >= 3
        if is_refusal or is_noise or digit_heavy:
            removed += 1
            continue
        out[name] = entry
    return out, removed


def _backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def migrate(root: Path, apply: bool) -> dict:
    stats = {"skill_md_scanned": 0, "skill_md_changed": 0,
             "meta_scanned": 0, "meta_changed": 0, "meta_entries_removed": 0}

    for skill_md in root.rglob("SKILL.md"):
        stats["skill_md_scanned"] += 1
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        family = _family_from_path(skill_md)
        cleaned, changed = clean_skill_md(text, family)
        if changed:
            stats["skill_md_changed"] += 1
            _log.info("%s SKILL.md: %s", "FIX " if apply else "would-fix", skill_md)
            if apply:
                _backup(skill_md)
                skill_md.write_text(cleaned, encoding="utf-8")

    for meta in root.rglob("meta_skills.json"):
        stats["meta_scanned"] += 1
        try:
            obj = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            _log.warning("skip unreadable %s (%s)", meta, e)
            continue
        if not isinstance(obj, dict):
            continue
        cleaned_obj, removed = clean_meta_skills(obj)
        if removed:
            stats["meta_changed"] += 1
            stats["meta_entries_removed"] += removed
            _log.info("%s meta_skills.json: -%d entries  %s",
                      "FIX " if apply else "would-fix", removed, meta)
            if apply:
                _backup(meta)
                meta.write_text(json.dumps(cleaned_obj, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir to scan recursively")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite files (default: dry-run report only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")
    stats = migrate(root, apply=args.apply)

    mode = "APPLIED" if args.apply else "DRY-RUN (no files changed; pass --apply to write)"
    print(f"\n=== migrate_storage {mode} ===")
    print(f"  SKILL.md:        {stats['skill_md_changed']}/{stats['skill_md_scanned']} need cleaning")
    print(f"  meta_skills.json {stats['meta_changed']}/{stats['meta_scanned']} files "
          f"({stats['meta_entries_removed']} junk entries)")
    if not args.apply and (stats["skill_md_changed"] or stats["meta_changed"]):
        print("  → re-run with --apply to fix (originals backed up to *.bak)")


if __name__ == "__main__":
    main()
