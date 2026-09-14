import json
import os
from types import SimpleNamespace


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(v) for v in d]
    return d

class ConfigLoader:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
        self.raw = raw
        self.model = _to_ns(raw["model"])
        self.data = _to_ns(raw["data"])
        self.training = _to_ns(raw["training"])
        self.eval = _to_ns(raw["eval"])
        self.swanlab = _to_ns(raw.get("swanlab", {}))

        # 补全数据路径为绝对路径，避免受工作目录影响
        self.data.train_path = os.path.join(self.project_root, self.data.root_dir, self.data.train_file)
        self.data.eval_path = os.path.join(self.project_root, self.data.root_dir, self.data.eval_file)
        self.data.test_path = os.path.join(self.project_root, self.data.root_dir, self.data.test_file)
        self.eval.adapter_path = os.path.join(self.project_root, self.eval.adapter_path)
        self.eval.dev_path = os.path.join(self.project_root, self.data.root_dir, self.eval.dev_file)

        # 输出目录也改成绝对路径
        self.training.output_dir = os.path.join(self.project_root, self.training.output_dir)

