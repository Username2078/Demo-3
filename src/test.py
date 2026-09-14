import os
import logging
import torch
from transformers import AutoTokenizer
from src.configloader import ConfigLoader
from src.trainer import Trainer
from src.datasetloader import DatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/config-qlora.json")
    cfg = ConfigLoader(config_path)
    logger.info(f"配置文件加载成功，项目根目录: {cfg.project_root}")

    adapter_path = os.path.join(cfg.project_root, cfg.eval.adapter_path)
    batch_size =  cfg.eval.batch_size
    max_new_tokens =  cfg.eval.max_new_tokens
    logger.info(f"batch_size={batch_size}, max_new_tokens={max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 复用一下里面的create json
    datasetloader = DatasetLoader(
        tokenizer=tokenizer,
        template_name=cfg.data.template_name,
        max_seq_length=cfg.data.max_seq_length,
        overwrite_cache=cfg.data.overwrite_cache,
    )

    data_path = os.path.join(cfg.project_root, cfg.data.root_dir)
    test_json = os.path.join(data_path, "test.jsonl")

    if not os.path.exists(test_json):
        logger.info(f"生成 test.jsonl 于：{test_json}")
        datasetloader.create_json(cfg.data.test_path, test_json)

    logger.info("加载 test dataset ...")
    test_dataset = datasetloader.get_dataset(test_json)
    logger.info(f"test 样本数: {len(test_dataset)}")
    logger.info(f"test columns: {test_dataset.column_names}")


    model, tokenizer, is_quantized = datasetloader.load_model_and_tokenizer(cfg, adapter_path)

    logger.info(f"当前显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    cfg.training.per_device_eval_batch_size = batch_size

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=cfg.training,
        train_dataset=None,
        eval_dataset=test_dataset,
        data_collator=None,
        compute_loss_func=None,
        use_bnb=is_quantized,
    )

    metrics = trainer.evaluate(test_dataset, max_new_tokens=max_new_tokens)

    logger.info(
        "\n===== Test 结果 =====\n"
        f"P={metrics['eval/precision']:.4f}  "
        f"R={metrics['eval/recall']:.4f}  "
        f"F1={metrics['eval/f1']:.4f}"
    )


if __name__ == "__main__":
    main()