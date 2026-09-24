"""Post-generation checks on LLM answers — deterministic, generic, no hardcoded phrases."""
from __future__ import annotations

import re


def reconcile_list_count(answer: str) -> str:
    """If the answer states its own count near the start (e.g. 'all 29 MOOCs',
    'There are 12 courses'), and the actual bullet count in the answer exceeds
    that stated number, trim the list to match. Fully generic — works on any
    number/noun, not tied to this document or query.
    """
    bullet_pattern = re.compile(r"^\s*[\*\-•]\s+.+$", re.MULTILINE)
    bullets = bullet_pattern.findall(answer)
    if not bullets:
        return answer

    # Look for a stated count only in the text BEFORE the first bullet —
    # this is where models typically announce "here are all N X".
    first_bullet_pos = answer.find(bullets[0])
    preamble = answer[:first_bullet_pos]

    count_match = re.search(r"\b(\d+)\b", preamble)
    if not count_match:
        return answer

    stated_count = int(count_match.group(1))
    actual_count = len(bullets)

    if actual_count <= stated_count:
        return answer  # matches, or under — nothing to trim

    # Trim excess bullets from the end.
    keep = bullets[:stated_count]
    trimmed_answer = preamble + "\n".join(keep)

    # Preserve any trailing text after the bullet list (e.g. closing sentence).
    last_bullet_pos = answer.rfind(bullets[-1])
    trailing = answer[last_bullet_pos + len(bullets[-1]):]
    return trimmed_answer + trailing