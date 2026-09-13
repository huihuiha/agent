"""Prompt 版本管理测试：加载、渲染、diff、追加版本。"""

from __future__ import annotations

import pytest

from llm_unify.exceptions import PromptError
from llm_unify.prompts import PromptRepository


def test_list_and_default_active_version(prompt_repo: PromptRepository):
    records = prompt_repo.list()
    assert [r.id for r in records] == ["translator"]
    record = records[0]
    assert record.active_version == 2
    assert [v.version for v in record.versions] == [1, 2]


def test_render_with_variables(prompt_repo: PromptRepository):
    item, rendered = prompt_repo.render("translator", {"target_lang": "日文", "style": "口语"})
    assert item.version == 2
    assert "日文" in rendered and "口语" in rendered


def test_render_missing_variable_raises(prompt_repo: PromptRepository):
    with pytest.raises(PromptError):
        prompt_repo.render("translator", {"target_lang": "日文"}, version=1)  # v1 模板还需要 style


def test_render_pinned_version(prompt_repo: PromptRepository):
    item, rendered = prompt_repo.render("translator", {"target_lang": "日文", "style": "正式"}, version=1)
    assert item.version == 1
    assert "把用户输入翻译成日文" in rendered


def test_unknown_version_raises(prompt_repo: PromptRepository):
    with pytest.raises(PromptError):
        prompt_repo.get("translator").version(99)


def test_diff_between_versions(prompt_repo: PromptRepository):
    diff = prompt_repo.diff("translator", 1, 2)
    text = "".join(diff)
    assert "--- translator@v1" in text and "+++ translator@v2" in text
    assert "资深译员" in text


def test_create_version_appends_and_activates(prompt_repo: PromptRepository):
    entry = prompt_repo.create_version(
        "translator", name="翻译助手", description="v3", role="system", template="v3 模板 {{ target_lang }}"
    )
    assert entry.version == 3
    record = prompt_repo.get("translator")
    assert record.active_version == 3
    assert [v.version for v in record.versions] == [1, 2, 3]
    # 历史版本不可变：仍可按 v1 渲染
    assert "把用户输入翻译成x" in prompt_repo.render("translator", {"target_lang": "x", "style": "y"}, version=1)[1]


def test_create_version_without_activate(prompt_repo: PromptRepository):
    prompt_repo.create_version(
        "translator", name="翻译助手", description="草稿", role="system", template="draft", activate=False
    )
    assert prompt_repo.get("translator").active_version == 2


def test_missing_prompt_raises(prompt_repo: PromptRepository):
    with pytest.raises(PromptError):
        prompt_repo.get("ghost")
