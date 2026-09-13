"""Prompt 版本管理：文件系统存储，一个 prompt 一个 YAML，内含多个版本。

存储布局：
    prompts/
      translator.yaml      # id: translator, versions: [v1, v2], active_version: 2

能力：
- 按 name(+version) 加载，缺省取 active 版本；
- Jinja2 沙箱渲染（StrictUndefined，变量缺失即报错，防止静默产出半成品）；
- 版本间 diff（difflib.unified_diff）；
- 新增版本（CLI / HTTP API 均可），版本号单调递增，历史版本不可变。
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from llm_unify.exceptions import PromptError


@dataclass(frozen=True)
class PromptVersion:
    version: int
    role: str
    template: str
    created_at: str = ""


@dataclass(frozen=True)
class PromptRecord:
    id: str
    name: str
    description: str
    active_version: int
    versions: list[PromptVersion]

    def version(self, number: int | None = None) -> PromptVersion:
        wanted = number if number is not None else self.active_version
        for item in self.versions:
            if item.version == wanted:
                return item
        known = [v.version for v in self.versions]
        raise PromptError(f"prompt {self.id!r} 不存在版本 {wanted}，可用版本: {known}")


class PromptRepository:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)

    # ------------------------------------------------------------------ 读写
    def _path(self, prompt_id: str) -> Path:
        return self.directory / f"{prompt_id}.yaml"

    def list(self) -> list[PromptRecord]:
        if not self.directory.is_dir():
            return []
        records = []
        for path in sorted(self.directory.glob("*.yaml")):
            try:
                records.append(self._load(path.stem))
            except PromptError:
                continue  # 跳过损坏文件，不让一个坏文件拖垮 list
        return records

    def get(self, prompt_id: str) -> PromptRecord:
        return self._load(prompt_id)

    def _load(self, prompt_id: str) -> PromptRecord:
        path = self._path(prompt_id)
        if not path.is_file():
            raise PromptError(f"prompt {prompt_id!r} 不存在（查找路径: {path}）")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise PromptError(f"prompt 文件 YAML 解析失败: {path}: {exc}") from exc

        versions = sorted(
            (
                PromptVersion(
                    version=int(item["version"]),
                    role=item.get("role", "system"),
                    template=item.get("template", ""),
                    created_at=item.get("created_at", ""),
                )
                for item in raw.get("versions", [])
            ),
            key=lambda v: v.version,
        )
        if not versions:
            raise PromptError(f"prompt 文件没有可用版本: {path}")
        active = int(raw.get("active_version", versions[-1].version))
        return PromptRecord(
            id=str(raw.get("id", prompt_id)),
            name=str(raw.get("name", prompt_id)),
            description=str(raw.get("description", "")),
            active_version=active,
            versions=versions,
        )

    def create_version(
        self,
        prompt_id: str,
        *,
        name: str,
        description: str,
        role: str,
        template: str,
        activate: bool = True,
    ) -> PromptVersion:
        """追加新版本；版本号 = max(existing) + 1，历史版本不可变。

        不可变性靠"只 INSERT 新版本、永不 UPDATE 旧版本"保证——
        因此 diff(v1, v2) 的结果永远稳定，线上引用旧版本的请求行为可复现。
        activate=False 用于保存草稿（如灰度未验证的新模板）。
        """
        try:
            record = self.get(prompt_id)
            next_version = record.versions[-1].version + 1
            versions = list(record.versions)
            display_name, display_desc = name or record.name, description or record.description
            active = next_version if activate else record.active_version
        except PromptError:
            # 首个版本：文件还不存在，从 v1 开始
            next_version, versions, display_name, display_desc = 1, [], name, description
            active = 1

        entry = PromptVersion(
            version=next_version,
            role=role,
            template=template,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": prompt_id,
            "name": display_name,
            "description": display_desc,
            "active_version": active,
            "versions": [v.__dict__ for v in [*versions, entry]],
        }
        self._path(prompt_id).write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return entry

    # ------------------------------------------------------------------ 渲染/对比
    def render(
        self, prompt_id: str, variables: dict[str, Any], version: int | None = None
    ) -> tuple[PromptVersion, str]:
        """按 name(+version) 渲染模板；缺省取 active 版本。

        StrictUndefined 保证变量缺失时直接失败而不是静默输出 "{{ var }}"——
        半成品 prompt 发给模型比报错更危险（行为不可预测且难排查）。
        沙箱环境（SandboxedEnvironment）阻止模板触达文件系统等危险操作。
        """
        record = self.get(prompt_id)
        item = record.version(version)
        try:
            rendered = self.env.from_string(item.template).render(**variables)
        except Exception as exc:
            raise PromptError(
                f"prompt {prompt_id!r} v{item.version} 渲染失败: {exc}（检查模板变量是否齐全）"
            ) from exc
        return item, rendered

    def diff(self, prompt_id: str, left: int, right: int) -> list[str]:
        """两个版本模板的 unified diff（带 @@ 块定位），供 CLI/API 展示变更范围。"""
        record = self.get(prompt_id)
        left_text = record.version(left).template.splitlines(keepends=True)
        right_text = record.version(right).template.splitlines(keepends=True)
        return list(
            difflib.unified_diff(
                left_text,
                right_text,
                fromfile=f"{prompt_id}@v{left}",
                tofile=f"{prompt_id}@v{right}",
            )
        )
