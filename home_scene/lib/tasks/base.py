"""任务公共类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    description: str
    success_note: str = ""


class HomeTask(Protocol):
    info: TaskInfo

    def reset(self, model, data) -> None: ...

    def success(self, model, data) -> bool: ...
