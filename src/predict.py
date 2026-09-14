import os
import logging
import argparse
import torch
from src.configloader import ConfigLoader
from src.datasetloader import DatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_prompt_ids(tokenizer, template, sentence, instruction):
    messages = template.apply(instruction, sentence)
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    return prompt_ids



@torch.inference_mode()
def predict_one(model, tokenizer, datasetloader, sentence, instruction,
                device, max_new_tokens=128):
    # 直接调 datasetloader 的方法，拿到 prompt_ids
    prompt_ids = datasetloader.build_prompt_ids(instruction, sentence)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = input_ids.shape[1]
    pred_text = tokenizer.decode(
        gen_out[0, input_len:], skip_special_tokens=True
    ).strip()
    return pred_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", type=str, default=None,
                   help="adapter 目录，默认用 cfg.eval.adapter_path")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="指定 checkpoint 名，如 checkpoint-2000")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--instruction", type=str,
                   default="识别出下面句子中所有基因实体")
    return p.parse_args()


def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/config.json")
    cfg = ConfigLoader(config_path)
    logger.info(f"配置文件加载成功，项目根目录: {cfg.project_root}")

    adapter_path = os.path.join(cfg.project_root, cfg.eval.adapter_path)


    model, tokenizer,_ = DatasetLoader.load_model_and_tokenizer(cfg, adapter_path)

    datasetloader = DatasetLoader(
        tokenizer=tokenizer,
        template_name=cfg.data.template_name,
        max_seq_length=cfg.data.max_seq_length,
        overwrite_cache=False,
    )
    instruction = "你是生物医学命名实体识别专家。从输入句子中提取所有基因实体。每个实体单独一行，格式：实体名:GENE。如果没有基因实体，输出'无实体'。"
    device = next(model.parameters()).device
    logger.info("=" * 60)
    logger.info(f"adapter: {adapter_path}")
    logger.info(f"template: {cfg.data.template_name}")
    logger.info(f"instruction: {instruction}")
    logger.info("=" * 60)


    while True:
        try:
            sentence = input("\n输入： ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not sentence:
            continue
        if sentence.lower() in ("q"):
            break


        pred = predict_one(
            model=model,
            tokenizer=tokenizer,
            datasetloader=datasetloader,
            sentence=sentence,
            instruction=instruction,
            device=device,
            max_new_tokens=cfg.data.max_seq_length,
        )
        print(f"实体为：\n{pred}")

if __name__ == "__main__":
    main()