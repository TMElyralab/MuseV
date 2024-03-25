from copy import deepcopy
from typing import Dict, List

from .load_template import get_template_by_name


def generate_prompts(tasks: List[Dict]) -> List[Dict]:
    new_tasks = []
    for task in tasks:
        task["origin_prompt"] = deepcopy(task["prompt"])
        # 如果prompt单元值含有模板 {}，或者 没有填写任何值（默认为空模板），则使用原prompt值
        if "{" not in task["prompt"] and len(task["prompt"]) != 0:
            new_tasks.append(task)
        else:
            template = get_template_by_name(
                template=task["prompt"], name=task.get("template_name", None)
            )
            prompts = template(task)
            if not isinstance(prompts, list) and isinstance(prompts, str):
                prompts = [prompts]
            for prompt in prompts:
                task_cp = deepcopy(task)
                task_cp["prompt"] = prompt
                new_tasks.append(task_cp)
    return new_tasks
