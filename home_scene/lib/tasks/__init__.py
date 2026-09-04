"""任务注册：已实现与规划中的 home_* 任务。"""

from __future__ import annotations

from tasks.base import HomeTask, TaskInfo
from tasks.close_cabinet import CloseCabinetTask

PLANNED_TASKS: tuple[TaskInfo, ...] = (
    TaskInfo("home_pick_cup", "从台面拿起指定颜色杯子", "掌侧稳定接触"),
    TaskInfo("home_lift_bag", "穿带承托并抬起包", "抬升+保持"),
    TaskInfo("home_close_cabinet", "关闭柜门", "铰链角接近闭合并保持"),
    TaskInfo("home_toggle_light", "开关灯（占位）", "开关状态翻转"),
)

IMPLEMENTED: dict[str, type] = {
    "home_close_cabinet": CloseCabinetTask,
}

__all__ = [
    "HomeTask",
    "TaskInfo",
    "PLANNED_TASKS",
    "IMPLEMENTED",
    "CloseCabinetTask",
    "make_task",
]


def make_task(task_id: str):
    if task_id not in IMPLEMENTED:
        raise KeyError(
            f"任务未实现: {task_id}. 已实现: {sorted(IMPLEMENTED)}. "
            f"规划中: {[t.task_id for t in PLANNED_TASKS]}"
        )
    return IMPLEMENTED[task_id]()
