from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    normalized = answer.lower()
    hits = sum(1 for item in expected if item.lower() in normalized)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    score = recall_points(answer, expected)
    if len(answer.strip()) < 10:
        return min(score, 0.25)
    if "chưa" in answer.lower() or "không" in answer.lower():
        return min(score, 0.5)
    return min(1.0, score + 0.1)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    user_ids = sorted({conv["user_id"] for conv in conversations})
    initial_sizes = {
        user_id: getattr(agent, "memory_file_size", lambda _user_id: 0)(user_id)
        for user_id in user_ids
    }

    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: list[str] = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = f"{conversation['id']}-{agent_name.lower()}-train"
        thread_ids.append(thread_id)
        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, turn)

        for index, recall in enumerate(conversation.get("recall_questions", [])):
            recall_thread_id = f"{conversation['id']}-{agent_name.lower()}-recall-{index}"
            thread_ids.append(recall_thread_id)
            result = agent.reply(user_id, recall_thread_id, recall["question"])
            answer = result["answer"]
            expected = recall.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_sizes = {
        user_id: getattr(agent, "memory_file_size", lambda _user_id: 0)(user_id)
        for user_id in user_ids
    }
    memory_growth = sum(final_sizes[user_id] - initial_sizes[user_id] for user_id in user_ids)
    all_threads = set(thread_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in all_threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in all_threads),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in all_threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in table:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    lines = [fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt(row) for row in table)
    return "\n".join(lines)


def benchmark_config(config, suite_name: str, agent_name: str):
    run_dir = config.state_dir / "benchmark_runs" / f"{suite_name}-{agent_name.lower()}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, state_dir=run_dir)


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    print("## Standard Benchmark")
    standard_rows = [
        run_agent_benchmark(
            "Baseline",
            BaselineAgent(benchmark_config(config, "standard", "baseline"), force_offline=True),
            standard,
            config,
        ),
        run_agent_benchmark(
            "Advanced",
            AdvancedAgent(benchmark_config(config, "standard", "advanced"), force_offline=True),
            standard,
            config,
        ),
    ]
    print(format_rows(standard_rows))

    print("\n## Long-Context Stress Benchmark")
    stress_rows = [
        run_agent_benchmark(
            "Baseline",
            BaselineAgent(benchmark_config(config, "stress", "baseline"), force_offline=True),
            stress,
            config,
        ),
        run_agent_benchmark(
            "Advanced",
            AdvancedAgent(benchmark_config(config, "stress", "advanced"), force_offline=True),
            stress,
            config,
        ),
    ]
    print(format_rows(stress_rows))

    print(
        "\nAnalysis: Advanced recall is higher because stable user facts are persisted in User.md. "
        "It may spend more prompt tokens on short conversations because it loads profile memory, "
        "but compact memory keeps long-thread prompt processing lower by summarizing older turns. "
        "Memory growth is the trade-off: User.md improves recall, while stale or incorrect facts need cleanup policies."
    )


if __name__ == "__main__":
    main()
