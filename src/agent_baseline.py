from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_offline(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        answer = self._offline_response(session, message)
        session.messages.append({"role": "assistant", "content": answer})
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": None,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _offline_response(self, session: SessionState, message: str) -> str:
        lower = message.lower()
        user_text = "\n".join(item["content"] for item in session.messages if item["role"] == "user")

        def find(patterns: list[str]) -> str | None:
            import re

            for pattern in patterns:
                matches = list(re.finditer(pattern, user_text, flags=re.IGNORECASE | re.UNICODE))
                if matches:
                    return matches[-1].group(1).strip().strip(" .")
            return None

        if any(word in lower for word in ["tên", "đồ uống", "style", "nơi", "ở đâu", "nghề"]):
            facts: list[str] = []
            name = find([r"(?:mình|tôi)\s+tên\s+là\s+([^,.]+)", r"tên\s+(?:mình|tôi)\s+là\s+([^,.]+)"])
            drink = find([r"đồ\s+uống\s+yêu\s+thích\s+là\s+([^,.]+)", r"vẫn\s+uống\s+([^,.]+?)\s+như\s+cũ"])
            style = find([r"(?:muốn|thích)\s+(?:bạn\s+)?trả\s+lời\s+([^,.]+(?:,\s*[^,.]+){0,3})"])
            location = find([r"nơi\s+ở\s+hiện\s+tại\s+là\s+([^,.]+)", r"(?:mình|tôi)\s+(?:đang\s+)?ở\s+([^,.]+)"])
            profession = find([r"nghề\s+nghiệp.*?là\s+([^,.]+)", r"(?:đang|vẫn)\s+làm\s+([^,.]+?engineer[^,.]*)"])
            for label, value in [
                ("Tên", name),
                ("Đồ uống yêu thích", drink),
                ("Style trả lời", style),
                ("Nơi ở", location),
                ("Nghề nghiệp", profession),
            ]:
                if value:
                    facts.append(f"{label}: {value}")
            if facts:
                return "Mình nhớ trong thread này: " + "; ".join(facts) + "."
            return "Mình chưa có thông tin đó trong thread hiện tại."
        return "Mình đã ghi nhận trong thread này, nhưng baseline sẽ không nhớ qua thread mới."
