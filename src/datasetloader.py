import os
import json
import logging
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modelloader import ModelLoader
from src.template import get_template
from datasets import Dataset
from src.modelloader import DTYPE_MAP

class DatasetLoader:
    def __init__(self, tokenizer, template_name, max_seq_length, overwrite_cache=False):
        self.tokenizer = tokenizer
        self.template = get_template(template_name)
        self.max_seq_length = max_seq_length
        self.overwrite_cache = overwrite_cache

    def create_json(self, data_path, out_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"文件不存在：{data_path}")

        raw_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            sentence = item["sentence"]
            # entities一项{"name": "xxx", "type": "GENE", "pos": [xx,xx]}
            entities = item["entities"]
            gene = [ent["name"] for ent in entities]

            if gene:
                output = '\n'.join([f"{name}:GENE" for name in gene])
            else:
                output = "无实体"

            sample = {
                "instruction": "识别出下面句子中所有基因实体",
                "input": sentence,
                "output": output
            }

            raw_data.append(sample)

        with open(out_path, "w", encoding="utf-8") as fw:
            for s in raw_data:
                fw.write(json.dumps(s, ensure_ascii=False) + "\n")

    def load_from_json(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"文件不存在：{data_path}")

        raw_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_data.append(json.loads(line))  # json.loads转字典

        return Dataset.from_list(raw_data)

    # 后面给dataset的map函数用，给每一条都加上text
    def format_example(self, example, add_generation_prompt = False):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # messages 是字典为元素的列表
        messages = self.template.apply(instruction, input_text)
        messages.append({"role": "assistant", "content": output})

        text = self.tokenizer.apply_chat_template(messages,
                                                      tokenize=False,
                                                      add_generation_prompt=add_generation_prompt)
        return {"text": text}


    def build_prompt_ids(self, instruction, sentence):
        messages = self.template.apply(instruction, sentence)
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        return prompt_ids



    def tokenizer_function(self, example):

        tokenized = self.tokenizer(example["text"],
                                    truncation=True,
                                    padding=False,
                                    max_length=self.max_seq_length)
        input_ids_list = tokenized["input_ids"]
        labels_list = []
        prompt_ids_list = [] #

        for idx, input_ids in enumerate(input_ids_list):
            msg_prompt = self.template.apply(example["instruction"][idx], example["input"][idx])
            prompt_str = self.tokenizer.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True)
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            labels = input_ids.copy()
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
            labels_list.append(labels)
            prompt_ids_list.append(prompt_ids)  #

        tokenized["labels"] = labels_list
        tokenized["prompt_ids"] = prompt_ids_list  #


        return tokenized

    def get_dataset(self, data_path):

        cache_path = f"{data_path}.cached"
        if not self.overwrite_cache and os.path.exists(cache_path):
            print(f"[DATA] 加载dataset来自缓存： {cache_path}")
            return Dataset.load_from_disk(cache_path)

        print(f"[DATA]  加载json文件{data_path}")
        raw_ds = self.load_from_json(data_path) # 返回dataset

        # 整合system instruction assistant为text 并加入到formatted_ds里
        # 然后只保留 text
        formatted_ds = raw_ds.map(
            self.format_example,
            batched=False,
            remove_columns=raw_ds.column_names,
            load_from_cache_file=not self.overwrite_cache,
        )
        combine_ds = Dataset.from_dict({
            "text": formatted_ds["text"],
            "instruction": raw_ds["instruction"],
            "input": raw_ds["input"]
        })
        # 由text生成 input_ids attention_mask 外加手写生成的labels
        tokenized_ds = combine_ds.map(
            self.tokenizer_function,
            batched=True,
            remove_columns=["text", "instruction", "input"],
            load_from_cache_file=not self.overwrite_cache
        )
        if not self.overwrite_cache:
            tokenized_ds.save_to_disk(cache_path)
            print(f"[DATA] dataset保存于{cache_path}")

        return tokenized_ds

    @staticmethod
    def load_model_and_tokenizer(cfg, adapter_path):

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        logger = logging.getLogger(__name__)

        base_path = cfg.model.model_name_or_path

        # 复用 ModelLoader 的量化逻辑，保证和训练时一致
        quant_config = ModelLoader.get_quantization_config(cfg)

        dtype = DTYPE_MAP.get(
            getattr(cfg.model, "torch_dtype", "bfloat16"), torch.bfloat16
        )

        if quant_config is not None:
            base = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype,
                                                        device_map=cfg.model.device_map, trust_remote_code=True,
                                                        quantization_config=quant_config)
        else:
            base = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype,
                                                        device_map=cfg.model.device_map, trust_remote_code=True)

        mode = "QLoRA" if quant_config is not None else "LoRA"
        logger.info(f"加载模式: {mode}")
        logger.info(f"base: {base_path}")
        logger.info(f"adapter: {adapter_path}")

        if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")):
            token_path = adapter_path
        else:
            token_path = base_path

        tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"tokenizer 来源: {token_path}")

        model = PeftModel.from_pretrained(base, adapter_path)  # 加上lora适配器
        model.eval()

        return model, tokenizer, quant_config is not None
