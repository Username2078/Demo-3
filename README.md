## 一.简介

---

- 使用 Pytorch HuggingFace 完成
- 基于 Qwen2.5-7B模型，分别使用LoRA QLoRA完成微调。
- 计算精确率 (Precision)、召回率 (Recall)、准确率 (Accuracy)、F1分数 作为微调评估指标
- 使用swanlab实现训练过程可视化
- 使用json文件管理超参数及文件路径
- 设置随机种子，保证训练复现


## 二.环境配置

---

- python 3.10.20
- pytorch 2.5.1
- cuda 12.1

## 三.项目结构

---

```
Demo3
├── bc2gm1                    # 存放数据集及处理后的jsonl
├── output                    # 存放训练输出adapter、tokenizer等
│   ├── qwen2.5-7b-ner
│   └── qwen2.5-7b-ner-qlora
├── src
│   ├── configloader.py       # 用于加载config.json
│   ├── datasetloader.py      # 原始数据集处理、存储、读取
│   ├── metric.py             # P R F1计算
│   ├── modelloader.py        # 模型、分词器加载
│   ├── template.py           # 模型输入格式模板
│   ├── test.py               # 测试
│   ├── trainer.py            # 仿照HF的Trainer完成的训练类
│   ├── predict.py            # 交互
│   └── train.py              # 模型训练和评估
└── .gitignore
```

## 四.数据集

---
### bc2gm数据集
> 英文数据集，总体读出为一个列表，每一项为一个字典。字典有两个键，分别是"sentence"和"entities"
> "sentence"为一个英文字符串。
> "entities"记录sentence中的实体，若无实体则为空列表，若有实体则列表元素为字典，如：{"name": "GH", "type": "GENE", "pos": [94, 96]}

示例如下：
```markdown
[{"sentence": "Comparison with alkaline phosphatases and 5 - nucleotidase", "entities": [{"name": "alkaline phosphatases", "type": "GENE", "pos": [16, 37]}]}
```

## 五.运行

---
- `train`会读取json配置，调用ModelLoader加载原始模型基座和tokenizer，然后调用DatasetLoader完成数据处理，然后构造Trainer并开始训练，在训练结束后进行评估。
- `test`读取配置并加载基座和tokenizer，借助Trainer中的函数对测试集进行测试。
- `predict`进行交互式实体识别

## 六.结果分析

---
- swanlab: 📁 View project at https://swanlab.cn/@ZhangShenLong/Demo3/v1/kthu6c/overview

### LoRA参数设置：

#### model模型配置

- 模型路径 model_name_or_path：/root/autodl-tmp/Qwen2.5-7B-Instruct
- 微调方式 finetuning_type：lora（只训练低秩适配器，冻结基座）
- LoRA 秩 lora_rank：8
- LoRA 缩放系数 lora_alpha：16.0
- LoRA dropout：0.1
- 目标模块 target_modules：q_proj,k_proj,v_proj,o_proj（只对注意力 Q/K/V/O 投影加 LoRA）
- 量化 quantization：null（未开启 4bit/8bit 量化）
- 信任远程代码 trust_remote_code：true
- 加载精度 torch_dtype：bfloat16
- 设备映射 device_map：auto

#### data数据集

- 数据目录 root_dir：bc2gm1
- 训练文件 train_file：train.json
- 验证文件 eval_file：dev.json
- 测试文件 test_file：test.json
- 指令模板 template_name：qwen
- 最大序列长度 max_seq_length：512
- 重建缓存 overwrite_cache：false（复用已有缓存）

#### training训练

- 输出目录 output_dir：output/qwen2.5-7b-ner（LoRA 适配器保存位置）
- 训练轮数 num_train_epochs：3
- 最大步数 max_steps：-1（以 epoch 为准，不设上限）
- 训练 batch 大小 per_device_train_batch_size：4
- 验证 batch 大小 per_device_eval_batch_size：4
- 梯度累积步数 gradient_accumulation_steps：1
- 学习率 learning_rate：2e-4
- 权重衰减 weight_decay：0.01
- 预热比例 warmup_ratio：0.03（前3%步数线性升温）
- 学习率调度器 lr_scheduler_type：cosine（余弦退火）
- 日志步数 logging_steps：10
- 保存步数 save_steps：200
- 最多保留 checkpoint save_total_limit：3（自动删旧）
- 保存格式 save_safetensors：true
- 是否评估 do_eval：true
- 评估策略 evaluation_strategy：epoch（每轮结束评估）
- 评估步数 eval_steps：200
- 梯度检查点 gradient_checkpointing：false（为了训快点）
- fp16：false
- bf16：true（开启 bfloat16 混合精度 模型权重、前向反向使用BF16 权重更新时用FP32）
- 随机种子 seed：42
- 数据加载进程 dataloader_num_workers：0
- 断点续训 resume_from_checkpoint：null（从头训练）

#### eval评估

- 适配器路径 adapter_path：output/qwen2.5-7b-ner
- 验证文件 dev_file：dev.json
- 推理 batch 大小 batch_size：4
- 最大新 token 数 max_new_tokens：64
- 最大输入长度 max_input_len：2048

#### swanlab

- 是否启用 enable：true
- 项目名 project：qwen2.5-7b-ner-bc2gm
- 实验名 experiment_name：lora-r8
- 日志目录 logdir：./swanlog
- 每轮评估指标 eval_every_epoch_for_metrics：true

### QLoRA

## model模型配置

- 模型路径 model_name_or_path：/root/autodl-tmp/Qwen2.5-7B-Instruct（Qwen2.5-7B 指令微调基座）
- 微调方式 finetuning_type：lora（QLoRA，冻结基座仅训练 LoRA 适配器）
- 梯度检查点 gradient_checkpointing：true（节省显存，以少量计算开销换取更低显存占用）
- LoRA 秩 lora_rank：16
- LoRA 缩放系数 lora_alpha：32.0
- LoRA dropout：0.05
- 目标模块 target_modules：q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj（注意力层 + FFN 层全部接入 LoRA）
- 量化 quantization：4bit（开启 4bit 量化，QLoRA）
- 信任远程代码 trust_remote_code：true
- 加载精度 torch_dtype：bfloat16
- 设备映射 device_map：auto（自动分配模型层至 GPU/CPU）

## data数据集

- 数据目录 root_dir：bc2gm1（BC2GM 生物医学 NER 数据集）
- 训练文件 train_file：train.json
- 验证文件 eval_file：dev.json
- 测试文件 test_file：test.json
- 指令模板 template_name：qwen（Qwen 系列 prompt 模板）
- 最大序列长度 max_seq_length：512
- 重建缓存 overwrite_cache：false（复用数据集缓存，加速加载）

## training训练

- 输出目录 output_dir：output/qwen2.5-7b-ner-qlora（LoRA 适配器保存路径）
- 训练轮数 num_train_epochs：3
- 最大步数 max_steps：-1（以 epoch 作为训练终止条件，不限制最大 step）
- 训练批次大小 per_device_train_batch_size：4
- 验证批次大小 per_device_eval_batch_size：4
- 梯度累积步数 gradient_accumulation_steps：1
- 学习率 learning_rate：2e-4
- 权重衰减 weight_decay：0.01
- 预热比例 warmup_ratio：0.03（前 3% 训练步长学习率线性上升至峰值）
- 学习率调度器 lr_scheduler_type：cosine（余弦退火策略）
- 日志打印步数 logging_steps：10
- 保存检查点步数 save_steps：200
- 最多保留检查点数量 save_total_limit：3（自动删除较早的 ckpt）
- 权重保存格式 save_safetensors：true
- 开启验证 do_eval：true
- 评估策略 evaluation_strategy：epoch（每完成一轮训练执行验证集评估）
- 评估步数 eval_steps：200
- 梯度检查点 gradient_checkpointing：true
- fp16：false
- bf16：true（启用 bfloat16 混合精度）
- 随机种子 seed：42（固定种子保证实验可复现）
- 数据加载进程数 dataloader_num_workers：0
- 断点续训 resume_from_checkpoint：null（从头开始训练，不从断点加载）
- 实验名称 experiment_name：qlora-r16

## eval评估

- 适配器路径 adapter_path：output/qwen2.5-7b-ner-qlora（test 脚本加载训练好的 LoRA 适配器）
- 验证文件 dev_file：dev.json
- 推理批次大小 batch_size：4
- 最大新生成 token max_new_tokens：64（适配 NER 短实体输出）
- 推理最大输入长度 max_input_len：2048

## swanlab

- 是否启用 enable：true
- 项目名 project：qwen2.5-7b-ner-bc2gm
- 实验名 experiment_name：qlora-r16
- 日志目录 logdir：./swanlog
- 每轮训练后评估指标 eval_every_epoch_for_metrics：true





