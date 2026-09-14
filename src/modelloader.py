from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

class ModelLoader:

    def __init__(self,config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_quantized = False


    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name_or_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def get_quantization_config(config):
        quantization = getattr(config.model, 'quantization', None)

        compute_dtype = DTYPE_MAP.get(
            getattr(config.model, "torch_dtype", "bfloat16"),
            torch.bfloat16,
        )

        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    def load_base_model(self, quantization_config):

        dtype_str = getattr(self.config.model, "torch_dtype", "bfloat16")
        dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)

        return AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            dtype=dtype,  # 影响非量化部分（如模型里的部分 bias，以及 LoRA） 会被后面quantization_config覆盖
            device_map=self.config.model.device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

    def apply_lora(self, model, is_quantized):

        use_gc = getattr(self.config.model, "gradient_checkpointing", False)

        if is_quantized:
            # QLoRA走，把 LayerNorm 升 fp32、开 input require grads
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=use_gc,  # 是否使用梯度检查点
            )
        elif use_gc:
            # 普通 LoRA 开 梯度检查点
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        raw_targets = getattr(self.config.model, "target_modules", None)
        if isinstance(raw_targets, str) and raw_targets:
            target_modules = [m.strip() for m in raw_targets.split(",") if m.strip()]
        elif isinstance(raw_targets, (list, tuple)):
            target_modules = list(raw_targets)
        else:
            target_modules = None

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.model.lora_rank,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def load(self):
        self.tokenizer = self.load_tokenizer()

        quantization_config = self.get_quantization_config(self.config)
        self.is_quantized = quantization_config is not None

        base_model = self.load_base_model(quantization_config)
        finetuning_type = getattr(self.config.model, 'finetuning_type', 'lora')
        if finetuning_type == "lora":

            mode = "QLoRA" if self.is_quantized else "LoRA" #依据基座是否量化判断
            print(f"[ModelLoader] 使用 {mode} "
                  f"(quantization={getattr(self.config.model, 'quantization', None)})")

            self.model = self.apply_lora(base_model, self.is_quantized)
        elif finetuning_type == "full":
            self.model = base_model
        else:
            raise ValueError(f"类型错误：{finetuning_type}")

        return self.model, self.tokenizer
