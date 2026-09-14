import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from src.configloader import ConfigLoader
from src.datasetloader import DatasetLoader
from src.modelloader import ModelLoader
from src.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config/config.json")
    cfg = ConfigLoader(config_path)
    logging.info(f"配置文件加载成功:\n{cfg}")

    loader = ModelLoader(cfg)
    model, tokenizer = loader.load()

    datasetloader = DatasetLoader(
        tokenizer=tokenizer,
        template_name=cfg.data.template_name,
        max_seq_length=cfg.data.max_seq_length,
        overwrite_cache=cfg.data.overwrite_cache,
    )
    data_path = os.path.join(cfg.project_root, cfg.data.root_dir)
    train_json = os.path.join(data_path, "train.jsonl")
    eval_json = os.path.join(data_path, "dev.jsonl")

    if not os.path.exists(train_json):
        logging.info(f"生成 train.jsonl 于：{cfg.data.train_path}")
        datasetloader.create_json(cfg.data.train_path, train_json)

    if os.path.exists(cfg.data.eval_path) and not os.path.exists(eval_json):
        logging.info(f"生成 dev.jsonl 于：{cfg.data.eval_path}")
        datasetloader.create_json(cfg.data.eval_path, eval_json)

    train_dataset = datasetloader.get_dataset(train_json)
    eval_dataset = datasetloader.get_dataset(eval_json)
    logging.info(
        f"训练集样本: {len(train_dataset)}"
        + f"，验证集样本: {len(eval_dataset)}"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=cfg.training,           # 直接把 training 命名空间丢进去
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=None,          # 使用 CustomTrainer 内置 default collator
        compute_loss_func=None,
        use_bnb=loader.is_quantized,
    )
    logs = trainer.train()
    logging.info(f"训练完成，日志条数: {len(logs)}")

    metrics = trainer.evaluate(eval_dataset)
    logging.info(
        f"验证集评估结果: "
        f"P={metrics['eval/precision']:.4f}, "
        f"R={metrics['eval/recall']:.4f}, "
        f"F1={metrics['eval/f1']:.4f}"
    )

if __name__ == "__main__":
    main()