from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    word_count = len(re.findall(r"\w+", cleaned, flags=re.UNICODE))
    char_estimate = max(1, len(cleaned) // 4)
    return max(1, int((word_count + char_estimate) / 2))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip() or "default")
        return self.root_dir / safe / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# User Profile\n\n## Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"-\s*([A-Za-z0-9_.-]+):\s*(.+)", line)
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        value = value.strip().strip(" .")
        if not value:
            return
        facts = self.facts(user_id)
        facts[key] = value
        body = "# User Profile\n\n## Facts\n"
        for fact_key in sorted(facts):
            body += f"- {fact_key}: {facts[fact_key]}\n"
        self.write_text(user_id, body)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join((message or "").split())
    if not text:
        return {}

    lower = text.lower()
    question_markers = ["?", "không?", "gì?", "đâu?", "như thế nào?"]
    if any(marker in lower for marker in question_markers) and not any(
        marker in lower for marker in ["mình tên", "tên mình", "hiện tại", "mình muốn", "mình thích"]
    ):
        return {}

    updates: dict[str, str] = {}

    patterns = {
        "name": [
            r"(?:mình|tôi)\s+tên\s+là\s+([^,.]+)",
            r"tên\s+(?:mình|tôi)\s+là\s+([^,.]+)",
        ],
        "location": [
            r"nơi\s+ở\s+hiện\s+tại\s+là\s+([^,.]+)",
            r"hiện\s+(?:tại\s+)?(?:mình|tôi)\s+(?:đang\s+)?ở\s+([^,.]+)",
            r"(?:mình|tôi)\s+(?:đang\s+)?ở\s+([^,.]+)",
            r"đang\s+làm\s+việc\s+ở\s+([^,.]+?)\s+(?:vài|mấy|trong)",
        ],
        "profession": [
            r"nghề\s+nghiệp\s+(?:hiện\s+tại\s+)?(?:thì\s+)?(?:vẫn\s+)?là\s+([^,.]+)",
            r"(?:đang|vẫn)\s+làm\s+([^,.]+?engineer[^,.]*)",
            r"(?:nghề|công\s+việc).*?\blà\s+([^,.]+)",
        ],
        "favorite_drink": [
            r"đồ\s+uống\s+yêu\s+thích\s+là\s+([^,.]+)",
            r"vẫn\s+uống\s+([^,.]+?)\s+như\s+cũ",
        ],
        "favorite_food": [
            r"món\s+ăn\s+yêu\s+thích\s+là\s+([^,.]+)",
        ],
        "response_style": [
            r"(?:muốn|thích)\s+(?:bạn\s+)?trả\s+lời\s+([^,.]+(?:,\s*[^,.]+){0,3})",
            r"style\s+trả\s+lời.*?(?:là|:)\s+([^,.]+(?:,\s*[^,.]+){0,3})",
            r"trả\s+lời\s+theo\s+dạng\s+([^,.]+(?:,\s*[^,.]+){0,3})",
        ],
    }

    for key, regexes in patterns.items():
        for regex in regexes:
            matches = list(re.finditer(regex, text, flags=re.IGNORECASE | re.UNICODE))
            if matches:
                value = matches[-1].group(1).strip()
                value = re.sub(r"\s+(?:nhé|giúp mình|trong giai đoạn này)$", "", value, flags=re.I)
                if key == "location" and "không phải nơi ở" in lower and "nơi ở hiện tại là" not in lower:
                    continue
                updates[key] = value
                break

    interests: list[str] = []
    interest_match = re.search(r"(?:mình|tôi)\s+thích\s+([^,.]+(?:,\s*[^,.]+){0,4})", text, flags=re.I)
    if interest_match and "trả lời" not in interest_match.group(1).lower():
        interests.append(interest_match.group(1).strip())
    if "Python" in text or "AI" in text or "MLOps" in text:
        tech = [term for term in ["Python", "AI", "MLOps", "RAG", "evaluation"] if term.lower() in lower]
        if tech:
            interests.append(", ".join(tech))
    if interests:
        updates["interests"] = "; ".join(dict.fromkeys(interests))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    selected = messages[-max_items:]
    bullets = []
    for item in selected:
        role = item.get("role", "unknown")
        content = " ".join(item.get("content", "").split())
        if len(content) > 220:
            content = content[:217].rstrip() + "..."
        bullets.append(f"- {role}: {content}")
    return "\n".join(bullets)


def _bounded_summary(text: str, max_lines: int = 10, max_chars: int = 1200) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines[-max_lines:])
    if len(compact) > max_chars:
        compact = compact[-max_chars:].lstrip()
    return compact


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread.get("summary", ""))
        total_tokens = estimate_tokens(summary) + sum(
            estimate_tokens(str(message.get("content", ""))) for message in messages
        )
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_at = max(1, len(messages) - self.keep_messages)
            older = messages[:split_at]
            recent = messages[split_at:]
            new_summary = summarize_messages(older)
            thread["summary"] = _bounded_summary((summary + "\n" + new_summary).strip())
            thread["messages"] = recent
            thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id).get("compactions", 0))
