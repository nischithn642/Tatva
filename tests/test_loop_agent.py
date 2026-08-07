"""
Tests for the scaffolding closed-loop agent ("Antigravity Engine").

The engine advertised prompt-driven autonomous generation while
_generate_initial_workspace ignored the prompt entirely and returned a fixed 9-file
template -- the prompt only ever appeared quoted inside the generated README. These
tests pin down that the prompt now reaches the model, that the template fallback is
labelled as such, and that a language model cannot write outside the sandbox.
"""

import pytest

from scaffolding.loop_agent import LoopAgent


class FakeProvider:
    """Stands in for LLMProvider. Records what it was asked, replies with a script."""

    def __init__(self, reply: str = "", raises: Exception | None = None):
        self.reply = reply
        self.raises = raises
        self.prompts: list[str] = []

    def query(self, prompt, system_prompt, messages, model_name, api_key=None, stream_callback=None):
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return self.reply, 0.0


VALID_REPLY = """Sure, here you go:
{"files": [
  {"path": "src/main.c", "content": "int main(void){return 0;}"},
  {"path": "CMakeLists.txt", "content": "project(x C)"}
]}
"""


@pytest.mark.unit
def test_prompt_actually_reaches_the_model() -> None:
    """The user's prompt must be sent, not just quoted back in a README."""
    provider = FakeProvider(reply=VALID_REPLY)
    agent = LoopAgent(llm_provider=provider)

    files, source = agent._generate_initial_workspace(
        "build a CRC32 kernel", "RV64GC", model_name="Ollama: test"
    )

    assert source == "llm"
    assert provider.prompts, "no LLM call was made at all"
    assert "build a CRC32 kernel" in provider.prompts[0]
    assert files["src/main.c"] == "int main(void){return 0;}"


@pytest.mark.unit
def test_no_model_selected_falls_back_and_says_so() -> None:
    """Without a backend there is nothing to generate with; that must be reported."""
    agent = LoopAgent(llm_provider=FakeProvider())
    logs: list[str] = []

    files, source = agent._generate_initial_workspace("anything", "RV64GC", model_name="", log=logs.append)

    assert source == "template"
    assert "src/main.c" in files
    assert any("template" in m.lower() for m in logs)


@pytest.mark.unit
def test_unreachable_model_falls_back_and_says_so() -> None:
    """A dead Ollama server must not look like a successful generation."""
    provider = FakeProvider(raises=RuntimeError("connection refused"))
    agent = LoopAgent(llm_provider=provider)
    logs: list[str] = []

    _, source = agent._generate_initial_workspace(
        "anything", "RV64GC", model_name="Ollama: test", log=logs.append
    )

    assert source == "template"
    assert any("connection refused" in m for m in logs)


@pytest.mark.unit
def test_incomplete_llm_workspace_is_rejected() -> None:
    """A reply missing CMakeLists.txt cannot build; take the template instead."""
    provider = FakeProvider(reply='{"files": [{"path": "src/main.c", "content": "int main(void){}"}]}')
    agent = LoopAgent(llm_provider=provider)

    files, source = agent._generate_initial_workspace("x", "RV64GC", model_name="Ollama: test")

    assert source == "template"
    assert "CMakeLists.txt" in files


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "no json here", "{not json}", '{"files": []}', '{"nope": 1}'])
def test_unparseable_replies_yield_none(bad: str) -> None:
    """An empty dict would be mistaken for a valid but empty workspace."""
    assert LoopAgent._parse_workspace_payload(bad) is None


@pytest.mark.unit
def test_parser_drops_paths_that_escape_the_workspace() -> None:
    """Paths come from a language model; traversal must not survive parsing."""
    payload = """{"files": [
      {"path": "../../evil.sh", "content": "rm -rf /"},
      {"path": "C:/Windows/System32/x.dll", "content": "x"},
      {"path": "src/ok.c", "content": "ok"}
    ]}"""
    files = LoopAgent._parse_workspace_payload(payload)
    assert files == {"src/ok.c": "ok"}


@pytest.mark.unit
def test_write_refuses_to_escape_the_sandbox(tmp_path) -> None:
    """Defence in depth: even if a bad path got through, writing it must fail loudly."""
    agent = LoopAgent(llm_provider=FakeProvider())
    sandbox = tmp_path / "ws"
    sandbox.mkdir()

    with pytest.raises(ValueError, match="outside the sandbox"):
        agent._write_files_to_dir(str(sandbox), {"../escaped.txt": "nope"})

    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.unit
def test_self_correction_cannot_smuggle_in_a_traversal_path() -> None:
    """The correction path must be filtered exactly like the generation path."""
    provider = FakeProvider(reply='{"files": [{"path": "../../evil.c", "content": "bad"}]}')
    agent = LoopAgent(llm_provider=provider)
    current = {"src/main.c": "original"}

    result = agent._apply_self_correction("fix it", "Ollama: test", None, current)

    assert result == {"src/main.c": "original"}


@pytest.mark.unit
def test_self_correction_applies_valid_files() -> None:
    provider = FakeProvider(reply='{"files": [{"path": "src/main.c", "content": "fixed"}]}')
    agent = LoopAgent(llm_provider=provider)

    result = agent._apply_self_correction("fix it", "Ollama: test", None, {"src/main.c": "original"})

    assert result["src/main.c"] == "fixed"


@pytest.mark.unit
def test_template_still_records_the_prompt_and_target() -> None:
    """The fallback is a starter project; it should at least document what was asked."""
    agent = LoopAgent(llm_provider=FakeProvider())
    files = agent._builtin_workspace_template("make a matmul kernel", "RV64GCV")

    assert "make a matmul kernel" in files["README.md"]
    assert "RV64GCV" in files["README.md"]
