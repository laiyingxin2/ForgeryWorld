"""Skill library manager — best-skills compatible (xstongxue/best-skills).

Each skill is stored as a directory containing SKILL.md (YAML frontmatter + body)
plus optional reference/, examples/, scripts/ subdirs.

Schema:
    skill-name/
    ├── SKILL.md            # required
    │     ---
    │     name: <max 64 chars, lowercase + digits + hyphens>
    │     description: <max 1024 chars, third-person, contains trigger keywords>
    │     source_round: <int>
    │     source_failure_ids: [<str>]
    │     trigger_score_threshold: <float>
    │     created_at: <iso>
    │     ---
    │     <markdown body with sections like 适用条件 / 强制规则 / 失败案例 / 验证方法>
    ├── reference/          # optional detailed docs
    ├── examples/           # optional before/after pairs
    ├── scripts/            # optional utility scripts
    └── metadata.json       # tracking: applied_count, success_rate

Usage:
    mgr = SkillManager("/path/to/outputs/skills")
    skill_id = mgr.create_skill(
        name="forgery-skill-jpeg-low-quality-evasion",
        description="...trigger keywords...",
        body="# JPEG Evasion\n## 强制规则\n- ...",
        source_round=3,
        source_failure_ids=["r3_017", "r3_021"],
    )
    actives = mgr.match_active_skills(generator_config={"algo": "inswapper_128", ...})
    mgr.record_application(skill_id, success=True)
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _yaml_dump(d: dict) -> str:
    """Minimal YAML dumper for frontmatter (avoids dep on PyYAML)."""
    lines = []
    for k, v in d.items():
        if isinstance(v, list):
            if not v:
                lines.append(f"{k}: []")
            else:
                lines.append(f"{k}: [{', '.join(repr(x) if isinstance(x, str) else str(x) for x in v)}]")
        elif isinstance(v, str):
            # quote if contains special chars
            if any(c in v for c in ':\n"'):
                lines.append(f'{k}: "{v.replace(chr(34), chr(92)+chr(34))}"')
            else:
                lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-ish frontmatter. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    fm_text = text[4:end].strip()
    body = text[end+4:].lstrip("\n")
    meta = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        # parse simple types
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if inner:
                meta[k] = [x.strip().strip("'\"") for x in inner.split(",")]
            else:
                meta[k] = []
        elif v.startswith('"') and v.endswith('"'):
            meta[k] = v[1:-1]
        elif v.lower() in ("true", "false"):
            meta[k] = v.lower() == "true"
        else:
            try:
                meta[k] = int(v)
            except ValueError:
                try:
                    meta[k] = float(v)
                except ValueError:
                    meta[k] = v
    return meta, body


class SkillManager:
    NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------- creation -----------------
    def create_skill(self, name: str, description: str, body: str,
                     source_round: int,
                     source_failure_ids: Optional[list[str]] = None,
                     trigger_score_threshold: float = 0.7,
                     overwrite: bool = False) -> str:
        if not self.NAME_RE.match(name):
            raise ValueError(f"invalid skill name {name!r} (lowercase a-z, 0-9, -; max 64)")
        if len(description) > 1024:
            raise ValueError(f"description too long ({len(description)} > 1024)")
        skill_dir = self.root / name
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists() and not overwrite:
            return name  # idempotent
        skill_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": name,
            "description": description,
            "source_round": source_round,
            "source_failure_ids": source_failure_ids or [],
            "trigger_score_threshold": trigger_score_threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        content = f"---\n{_yaml_dump(meta)}\n---\n\n{body.lstrip()}"
        skill_md.write_text(content, encoding="utf-8")
        # init tracking metadata
        (skill_dir / "metadata.json").write_text(json.dumps({
            "skill_id": name,
            "version": 1,
            "source_round": source_round,
            "applied_count": 0,
            "success_after_apply_count": 0,
            "applied_success_rate": 0.0,
            "last_updated_round": source_round,
            "deprecated": False,
        }, indent=2))
        return name

    # ---------------- read -----------------
    def load_skill(self, name: str) -> dict:
        skill_md = self.root / name / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(skill_md)
        meta, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        meta["body"] = body
        # merge tracking
        meta_json = self.root / name / "metadata.json"
        if meta_json.exists():
            meta["tracking"] = json.loads(meta_json.read_text())
        return meta

    def list_skills(self, include_deprecated: bool = False) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            try:
                m = self.load_skill(d.name)
            except FileNotFoundError:
                continue
            if not include_deprecated and m.get("tracking", {}).get("deprecated"):
                continue
            out.append(m)
        return out

    # ---------------- matching -----------------
    def match_active_skills(self, generator_config: dict[str, Any]) -> list[dict]:
        """Match skills whose description triggers match the config."""
        text = " ".join(f"{k}={v}" for k, v in generator_config.items()).lower()
        matched = []
        for sk in self.list_skills():
            keywords = self._extract_triggers(sk["description"])
            if any(kw in text for kw in keywords):
                matched.append(sk)
        return matched

    @staticmethod
    def _extract_triggers(description: str) -> list[str]:
        """Heuristic: tokens after 'Triggers:' / 'Keywords:' OR all lowercase words."""
        m = re.search(r"(?:triggers|keywords|关键词)[:：]\s*(.+)", description, re.I)
        if m:
            return [t.strip().lower() for t in re.split(r"[,，、]", m.group(1)) if t.strip()]
        # fallback: all single-word lowercase tokens
        return list({w.lower() for w in re.findall(r"[a-z][a-z0-9_-]+", description.lower())})

    # ---------------- tracking -----------------
    def record_application(self, name: str, success: bool, round_id: Optional[int] = None) -> None:
        meta_json = self.root / name / "metadata.json"
        if not meta_json.exists():
            return
        m = json.loads(meta_json.read_text())
        m["applied_count"] += 1
        if success:
            m["success_after_apply_count"] += 1
        m["applied_success_rate"] = m["success_after_apply_count"] / max(1, m["applied_count"])
        if round_id is not None:
            m["last_updated_round"] = round_id
        meta_json.write_text(json.dumps(m, indent=2))

    def mark_deprecated(self, name: str, reason: str = "") -> None:
        meta_json = self.root / name / "metadata.json"
        if meta_json.exists():
            m = json.loads(meta_json.read_text())
            m["deprecated"] = True
            m["deprecated_reason"] = reason
            m["deprecated_at"] = datetime.now(timezone.utc).isoformat()
            meta_json.write_text(json.dumps(m, indent=2))

    # ---------------- prompt assembly -----------------
    def active_skills_prompt(self, generator_config: dict) -> str:
        """Format active skills' '强制规则' sections as a system-prompt addon."""
        actives = self.match_active_skills(generator_config)
        if not actives:
            return ""
        parts = ["[ACCUMULATED SKILL CONSTRAINTS — MUST FOLLOW]\n"]
        for sk in actives:
            body = sk.get("body", "")
            # extract '强制规则' (or '## Rules', etc.)
            rules = re.search(r"##\s*(?:强制规则|Rules|必须遵守|Must Follow)(.+?)(?=\n##|\Z)",
                              body, re.S | re.I)
            section = rules.group(1).strip() if rules else body[:300]
            parts.append(f"### [{sk['name']}]\n{section}\n")
        return "\n".join(parts)


# ---------------------- smoke test --------------------------------
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mgr = SkillManager(tmp)
        sk = mgr.create_skill(
            name="forgery-skill-jpeg-low-quality-evasion",
            description="应对 FakeVLM 在 JPEG 低质量样本上检出率高。Triggers: jpeg, compression, qp, gfpgan, restoration.",
            body="# JPEG Evasion\n## 适用条件\n- jpeg_qp<80\n## 强制规则\n- MUST GFPGAN before JPEG\n## 失败案例\nsee reference/",
            source_round=3,
            source_failure_ids=["r3_017", "r3_021"],
        )
        print(f"✓ Created skill: {sk}")
        loaded = mgr.load_skill(sk)
        print(f"  frontmatter: {dict((k, v) for k, v in loaded.items() if k != 'body')}")
        print(f"  body preview: {loaded['body'][:80]!r}")

        # test matching
        matched = mgr.match_active_skills({"algo": "inswapper_128", "post_process": ["jpeg_qp_60"]})
        print(f"\n✓ Matched {len(matched)} skill(s) for jpeg config")

        # record + track
        mgr.record_application(sk, success=True, round_id=4)
        mgr.record_application(sk, success=True, round_id=4)
        mgr.record_application(sk, success=False, round_id=5)
        print(f"\n✓ After 3 applications: {json.loads((mgr.root / sk / 'metadata.json').read_text())}")

        # prompt assembly
        prompt = mgr.active_skills_prompt({"algo": "x", "post_process": ["jpeg_qp_60"]})
        print(f"\n✓ Active-skill prompt:\n{prompt}")
