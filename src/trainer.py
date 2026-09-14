import logging
import shutil

import math
import json
import os
import contextlib
import time
import random
import numpy as np
import torch
from datasets import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
import swanlab
from src.metric import NERMetric

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, tokenizer, args, train_dataset, eval_dataset,
                 data_collator, compute_loss_func=None, use_bnb = False):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self.default_data_collator
        self.compute_loss_func = compute_loss_func
        self.scaler = torch.amp.GradScaler("cuda") if args.fp16 else None # 解决fp16精度梯度下溢

        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.log_history = []

        self.device = self.get_device()
        self.set_seed(args.seed)

        self.use_bf16 = args.bf16
        self.use_fp16 = args.fp16
        self.use_bnb = use_bnb

        self.ner_metric = NERMetric()

        os.makedirs(args.output_dir, exist_ok=True)


    def swanlab_init(self):
        swanlab.init(
            project="Demo3",
            name=f"{self.args.experiment_name}",
            config={
                "model_name": "Qwen2.5-7B",
                "lora_rank": 16,
                "max_new_tokens": 128,
                "eval_steps": self.args.eval_steps,
                "logging_steps": self.args.logging_steps
            }
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        no_decay_keys = ("bias", "LayerNorm.weight", "layernorm.weight", "ln_")
        decay_params, no_decay_params = [], []

        # 找出需要权重衰减的参数和不需要的参数
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in no_decay_keys):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            param_groups,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999), # 计算 mt 和 vt时的参数
            eps=1e-8,
        )

        logger.info(
            f"优化器创建完成: 可训练参数 {len(decay_params) + len(no_decay_params)} 组"
        )
        return self.optimizer


    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.default_data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            drop_last=False,
        )


    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            return DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.eval_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.eval_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )



    def create_scheduler(self,  num_training_steps,  optimizer = None,):
        optimizer = optimizer or self.optimizer
        num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)

        def lr_lambda(current_step):
            # 没预热完，返回预热进度
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # 训练进度
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            progress = min(1.0, progress)

            if self.args.lr_scheduler_type == "cosine":
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif self.args.lr_scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            elif self.args.lr_scheduler_type == "constant":
                return 1.0
            else:
                raise ValueError(f"未知 lr_scheduler_type: {self.args.lr_scheduler_type}")

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)  # 第一个参数优化器 第二个参数接收当前步数返回缩放系数
        logger.info(
            f"调度器创建完成，类型为：{self.args.lr_scheduler_type} "
            f"总步数={num_training_steps} 预热步数={num_warmup_steps}"
        )
        return self.lr_scheduler

    def load_checkpoint(self, checkpoint_path, optimizer, scheduler):
        state_file = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.log_history = state.get("log_history", [])

        opt_file = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(opt_file):
            optimizer.load_state_dict(torch.load(opt_file, map_location=self.device))

        sch_file = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(sch_file):
            scheduler.load_state_dict(torch.load(sch_file))

        logger.info(f"从 {checkpoint_path} 恢复，全局步数为：{self.global_step}")


    def training_step(self, model, inputs):
        inputs = self.prepare_inputs(inputs)

        # 设置精度
        with self.autocast_context():
            loss, _ = self.compute_loss(model, inputs)
            loss = loss / self.args.gradient_accumulation_steps
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss.detach() # 不再计算梯度 只返回一个标量


    def compute_loss(self, model, inputs):
        labels = inputs.get("labels")
        outputs = model(**inputs) # 解包input_ids attention_mask labels
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels)
        elif hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        return loss, outputs


    def prepare_inputs(self, inputs):
        return {k: self.move_to_device(v) for k, v in inputs.items()}

    def move_to_device(self, v):
        if isinstance(v, torch.Tensor):
            return v.to(self.device, non_blocking=True) # 一次搬一批，可以non blocking提高性能
        if isinstance(v, dict):
            return {k: self.move_to_device(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(self.move_to_device(x) for x in v)
        return v

    def train(self):

        self.swanlab_init()

        if self.train_dataset is None:
            raise ValueError("train_dataset 不能为空")

        args = self.args
        model = self.model



        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_epochs = math.ceil(max_steps / max(1, steps_per_epoch))
        else:
            max_steps = int(steps_per_epoch * args.num_train_epochs)
            num_epochs = math.ceil(args.num_train_epochs)

        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(num_training_steps=max_steps, optimizer=optimizer)

        # 恢复检查点
        if args.resume_from_checkpoint:
            self.load_checkpoint(args.resume_from_checkpoint, optimizer, scheduler)

        model.train()

        # 记录显存
        torch.cuda.reset_peak_memory_stats()
        self.log_gpu_memory("训练开始")


        progress_bar = tqdm(total=max_steps, desc="Training", disable=False)
        progress_bar.update(self.global_step)

        running_loss = 0.0
        loss_count = 0
        start_time = time.time()

        eval_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,  #
            num_workers=0,
        )

        for epoch in range(num_epochs):
            self.epoch = epoch

            for step, inputs in enumerate(train_dataloader):
                loss = self.training_step(model, inputs)
                running_loss += loss.item()
                loss_count += 1

                is_last_step = (step + 1 == len(train_dataloader))
                should_update = ((step + 1) % args.gradient_accumulation_steps == 0) or is_last_step

                if not should_update:
                    continue

                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)  # 优化器要更新参数了，还原被放大的梯度

                if self.scaler is not None:
                    self.scaler.step(optimizer) # 判断是否溢出，没有正常继续执行
                    self.scaler.update() # 溢出则动态调整缩放因子，如果上一步没有溢出则累计

                else:
                    optimizer.step()

                scheduler.step()  # 调整学习率 要在optimizer.step()之后
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True会释放显存，参数更新后执行一次
                self.global_step += 1
                progress_bar.update(1)  # 用于进度条

                # 日志
                avg_loss = 0
                if self.global_step % args.logging_steps == 0:
                    avg_loss = running_loss / max(1, loss_count)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    logs =  {
                        "step": self.global_step,
                        "epoch": round(epoch + (step + 1) / len(train_dataloader), 4),
                        "loss": round(avg_loss, 6),
                        "lr": lr,
                        "elapsed_sec": round(elapsed, 2),
                    }
                    self.log_history.append(logs)
                    logger.info(
                        f"[step {self.global_step}] loss={avg_loss:.4f} "
                        f"lr={lr:.2e} elapsed={elapsed:.1f}s"
                    )

                    #self.log_gpu_memory(f"step为： {self.global_step}") #记录显存

                    running_loss = 0.0
                    loss_count = 0

                # 计算训练集采样PRF1 swanlab
                if self.global_step % args.eval_steps == 0:
                    self.ner_metric.reset()
                    model.eval()

                    # 从训练dataloader取少量样本
                    with torch.no_grad(), self.autocast_context():
                        # 取少量batch
                        for idx, batch in enumerate(eval_loader):
                            if idx >= 2:  # 只取2个batch
                                break
                            inputs = self.prepare_inputs(batch)
                            gen_out = model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=128,
                                do_sample=False
                            )

                            input_len = inputs["input_ids"].shape[1]
                            pred_texts, true_texts = self.metric_decode(gen_out, input_len, batch["labels"])

                            print("pred:", pred_texts[:2])
                            print("true:", true_texts[:2])

                            self.ner_metric.update_batch(pred_texts, true_texts)
                    metric_result = self.ner_metric.compute()
                    model.train()  # 切回训练模式
                    torch.cuda.empty_cache()  #

                    train_p = metric_result["precision"]
                    train_r = metric_result["recall"]
                    train_f1 = metric_result["f1"]

                    self.log_gpu_memory(f"step为： {self.global_step}") #记录显存

                    logs_pr = {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "train/precision": train_p,
                        "train/recall": train_r,
                        "train/f1": train_f1,
                    }
                    logger.info(
                        f"===== 训练评估 {self.global_step} ====="
                        f"P={train_p:.4f}, R={train_r:.4f}, F1={train_f1:.4f}"
                    )
                    swanlab.log(logs_pr, step=self.global_step)

                # 保存检查点
                if args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    self.save_checkpoint(optimizer, scheduler, self.global_step)

            if self.global_step >= max_steps:
                break

        progress_bar.close()

        self.save_model(args.output_dir)
        self.save_trainer_state(args.output_dir)
        logger.info(f"训练完成，模型已保存到 {args.output_dir}")

        return self.log_history


    def evaluate(self, eval_dataset=None, max_new_tokens=128):

        if eval_dataset is not None:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
        else:
            eval_dataloader = self.get_eval_dataloader()

        was_training = self.model.training
        self.model.eval()
        self.ner_metric.reset()

        with torch.no_grad(), self.autocast_context():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                inputs = self.prepare_inputs(batch)

                gen_out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                input_len = inputs["input_ids"].shape[1]
                pred_texts, true_texts = self.metric_decode(
                    gen_out, input_len, batch["labels"]
                )
                self.ner_metric.update_batch(pred_texts, true_texts)

        metric_result = self.ner_metric.compute()

        if was_training:
            self.model.train()
        torch.cuda.empty_cache() #add

        metrics = {
            f"eval/precision": metric_result["precision"],
            f"eval/recall": metric_result["recall"],
            f"eval/f1": metric_result["f1"],
        }


        return metrics

    def eval_collator(self, features):
        ###
        input_seqs = [f["prompt_ids"] for f in features]
        label_seqs = [f["labels"] for f in features]

        input_max = max(len(x) for x in input_seqs)
        label_max = max(len(x) for x in label_seqs)

        input_ids, attention_mask = [], []
        for x in input_seqs:
            pad = input_max - len(x)
            input_ids.append([0] * pad + list(x))
            attention_mask.append([0] * pad + [1] * len(x))

        labels = [list(x) + [-100] * (label_max - len(x)) for x in label_seqs]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


    def metric_decode(self, gen_out, input_len, labels):
        pred_texts = self.tokenizer.batch_decode(
            gen_out[:, input_len:], skip_special_tokens=True
        )
        mask = labels != -100
        true_texts = [
            self.tokenizer.decode(labels[i][mask[i]], skip_special_tokens=True)
            for i in range(labels.size(0))
        ]
        return pred_texts, true_texts

    def save_trainer_state(self, output_dir):
        state_path = os.path.join(output_dir, "trainer_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                {"global_step": self.global_step, "epoch": self.epoch, "log_history": self.log_history},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def save_checkpoint(self, optimizer, scheduler, step):
        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.save_model(ckpt_dir)
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        state = {
            "global_step": step,
            "epoch": self.epoch,
            "log_history": self.log_history,
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        if self.args.save_total_limit > 0:
            self.rotate_checkpoints(self.args.save_total_limit)

        logger.info(f"检查点已保存到 {ckpt_dir}")

    def save_model(self, output_dir = None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(
            output_dir, safe_serialization=self.args.save_safetensors
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(f"模型已保存到 {output_dir}")


    @contextlib.contextmanager
    def autocast_context(self):
        if self.use_bnb:
            yield
            return
        if self.use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                yield
        elif self.use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                yield
        else:
            yield


    def default_data_collator(self, features):
        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]
            first = values[0]
            pad_value = -100 if key == "labels" else 0
            max_len = max(len(v) for v in values)
            padded = [v + [pad_value] * (max_len - len(v)) for v in values]
            batch[key] = torch.tensor(padded, dtype=torch.long)
        return batch

    def eval_data_collator(self, features):
        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]
            pad_value = -100 if key == "labels" else 0
            max_len = max(len(v) for v in values)
            padded = [[pad_value] * (max_len - len(v)) + list(v) for v in values]

            batch[key] = torch.tensor(padded, dtype=torch.long)
        return batch


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def get_device(self):
        """获取模型所在设备。兼容 device_map="auto" 的多卡分发"""
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first = next(iter(self.model.hf_device_map.values()))
            if isinstance(first, int):
                return torch.device(f"cuda:{first}")
            if first == "cpu":
                return torch.device("cpu")
            return torch.device(first)
        return next(self.model.parameters()).device

    def rotate_checkpoints(self, save_total_limit):
        ckpts = [
            d for d in os.listdir(self.args.output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(self.args.output_dir, d))
        ]
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
        for old in ckpts[:-save_total_limit]:
            shutil.rmtree(os.path.join(self.args.output_dir, old), ignore_errors=True)

    def log_gpu_memory(self, tag):
        if not torch.cuda.is_available():
            logger.info(f"[{tag}] CUDA不可用")
            return
        # 同步GPU，保证读数准确
        torch.cuda.synchronize()
        alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(
            f"[{tag}] 当前张量显存={alloc_gb:.2f} GB, "
            f"PyTorch缓存={reserved_gb:.2f} GB, "
            f"峰值显存={peak_gb:.2f} GB"
        )
        # 同步记录到swanlab
        swanlab.log({
            "gpu/allocated_gb": alloc_gb,
            "gpu/reserved_gb": reserved_gb,
            "gpu/peak_gb": peak_gb
        }, step=self.global_step)
