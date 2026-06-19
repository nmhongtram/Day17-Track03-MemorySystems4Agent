from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    config = load_config(tmp_path)
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 80
    config.compact_keep_messages = 4
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    assert "## Facts" in store.read_text("user-1")
    path = store.write_text("user-1", "# User Profile\n\n## Facts\n- name: Dung\n")
    assert path.exists()
    assert "Dung" in store.read_text("user-1")
    assert store.edit_text("user-1", "Dung", "DungCT") is True
    assert "DungCT" in store.read_text("user-1")
    assert store.file_size("user-1") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(12):
        agent.reply(
            "u1",
            "long-thread",
            f"Tin nhắn rất dài số {index}: mình đang nói nhiều về Python, AI, MLOps và compact memory để vượt ngưỡng.",
        )
    assert agent.compaction_count("long-thread") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    learn_thread = "learn"
    baseline.reply("u2", learn_thread, "Chào bạn, mình tên là DũngCT.")
    advanced.reply("u2", learn_thread, "Chào bạn, mình tên là DũngCT.")

    baseline_answer = baseline.reply("u2", "new-thread", "Mình tên gì?")["answer"]
    advanced_answer = advanced.reply("u2", "new-thread", "Mình tên gì?")["answer"]

    assert "DũngCT" not in baseline_answer
    assert "DũngCT" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    long_message = (
        "Mình tên là DũngCT và đang làm MLOps engineer. "
        "Đây là một đoạn dài nói về benchmark, compact memory, prompt token, recall và trade-off. "
    )

    for index in range(30):
        baseline.reply("u3", "thread", f"{long_message} Lượt {index}.")
        advanced.reply("u3", "thread", f"{long_message} Lượt {index}.")

    assert advanced.compaction_count("thread") > 0
    assert advanced.prompt_token_usage("thread") < baseline.prompt_token_usage("thread")
