
# 模仿llama factory用一个字典管理模板
TEMPLATES = {}

def register_template(name):
    def decorator(func):
        TEMPLATES[name] = func()
        return func
    return decorator

class Template:

    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def apply(self, instruction, input_text = ""):
        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

@register_template("qwen")
def register_qwen_template():
    return Template(
        system_prompt="你是生物医学命名实体识别专家。从输入句子中提取所有基因实体。每个实体单独一行，格式：实体名:GENE。如果没有基因实体，输出'无实体'。"
    )

@register_template("default")
def register_default_template() -> Template:
    return Template(
        system_prompt="You are a helpful assistant."
    )

def get_template(name):
    if name not in TEMPLATES:
        return TEMPLATES["default"]
    return TEMPLATES[name]