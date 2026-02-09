import json
import os
import sys
import logging
import datetime
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BertTokenizer,  # 使用BertTokenizer
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from torch.utils.data import DataLoader
import gc
import re  # 新增：用于正则清理空格

# ===================== 日志配置 =====================
def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    log_dir = "../results/logs/bart_logs/02_model_bart"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_inference_bart_log_{timestamp}.log")
    
    log_format = "%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 屏蔽第三方库冗余日志
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ===================== 配置参数 =====================
# 数据路径
TRAIN_DATA_PATH = "../results/processed_data/train_data.json"
TEST_DATA_PATH = "../results/processed_data/test_data.json"
OUTPUT_DIR = "../results/new_bart_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "inference_results_bart.json")

# 模型配置 - 使用本地的中文BART模型（使用BertTokenizer）
MODEL_NAME_OR_PATH = "../model/bart-base-chinese"  # 本地路径

MODEL_SAVE_DIR = "../model/bart-chinese-finetuned"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 训练参数 - 针对BART和RTX 4090优化
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {DEVICE}")

# ===================== 数据加载 =====================
def load_data(file_path):
    """加载处理好的训练/测试数据"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"加载数据完成：{file_path}，共{len(data)}条样本")
        return data
    except Exception as e:
        logger.error(f"加载数据失败：{str(e)}", exc_info=True)
        raise

def prepare_input_text(sample):
    """准备输入文本：将DB编码转换为更易理解的格式"""
    db_codes = sample.get("db_codes", [])
    db_meanings = sample.get("db_code_meanings", {})
    
    # 构建更详细的输入：每个DB编码后附上它的含义
    input_parts = []
    for code in db_codes:
        meanings = db_meanings.get(code, ["未知含义"])
        # 取第一个含义作为代表
        meaning = meanings[0] if meanings else "未知含义"
        # 清理含义中的特殊字符
        meaning = meaning.replace(";", "；").replace(",", "，").replace("、", "，")
        input_parts.append(f"{code}:{meaning}")
    
    input_text = " | ".join(input_parts)
    return f"生成要求：1. 包含所有DB编码的核心含义，2. 语言流畅，与原始中文风格一致,3. 必须将所有DB编码替换为对应的中文释义（参考下方映射表），4. 长度与原始中文相近。将以下DB编码翻译成中文句子：{input_text}"

# ===================== 数据预处理 =====================
def preprocess_function(examples, tokenizer):
    """预处理函数：将文本转换为模型输入格式"""
    # 准备输入文本
    inputs = []
    for i in range(len(examples["sample_id"])):
        sample = {
            "sample_id": examples["sample_id"][i],
            "db_codes": examples["db_codes"][i],
            "db_code_meanings": examples["db_code_meanings"][i],
            "original_chinese": examples["original_chinese"][i],
            "original_db_codes_str": examples["original_db_codes_str"][i]
        }
        input_text = prepare_input_text(sample)
        inputs.append(input_text)
    
    targets = examples["original_chinese"]
    
    # 对输入进行编码 - 关键：禁用token_type_ids
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False  # BART不需要token_type_ids
    )
    
    # 对目标进行编码 - 关键：禁用token_type_ids
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False  # BART不需要token_type_ids
    )
    
    # 将标签中填充部分设为-100，这样在计算损失时会被忽略
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    # 确保没有token_type_ids
    if "token_type_ids" in model_inputs:
        del model_inputs["token_type_ids"]
    
    return model_inputs

# ===================== 模型训练与推理 =====================
def train_model(train_dataset, val_dataset, tokenizer):
    """训练模型"""
    logger.info(f"开始加载模型：{MODEL_NAME_OR_PATH}")
    
    try:
        # 加载模型
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH)
        logger.info(f"模型加载成功，模型类型：{type(model).__name__}")
        
    except Exception as e:
        logger.error(f"模型加载失败：{str(e)}")
        raise
    
    # 移动到设备
    model = model.to(DEVICE)
    logger.info(f"模型已加载到 {DEVICE}")
    
    # 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数统计：总参数={total_params:,}，可训练参数={trainable_params:,}")
    
    # 计算训练步数
    total_steps = len(train_dataset) * NUM_EPOCHS // TRAIN_BATCH_SIZE
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    # 定义训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{MODEL_SAVE_DIR}/logs",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        save_only_model=True,
        remove_unused_columns=False,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
    )
    
    # 自定义数据收集器，确保移除token_type_ids
    class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
        def __call__(self, features, return_tensors=None):
            batch = super().__call__(features, return_tensors)
            # 移除token_type_ids如果存在
            if "token_type_ids" in batch:
                del batch["token_type_ids"]
            return batch
    
    # 数据收集器
    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    
    # 定义Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    logger.info(f"开始模型训练")
    logger.info(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    logger.info(f"总步数: {total_steps} | Warmup步数: {warmup_steps}")
    logger.info(f"Batch Size: {TRAIN_BATCH_SIZE} | 学习率: {LEARNING_RATE}")
    
    try:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        
        logger.info(f"训练完成")
        
        # 安全地记录训练损失（避免格式化错误）
        train_loss = train_metrics.get('train_loss', 'N/A')
        if isinstance(train_loss, (int, float)):
            logger.info(f"训练损失: {train_loss:.4f}")
        else:
            logger.info(f"训练损失: {train_loss}")
        
        # 安全地记录评估损失（避免格式化错误）
        eval_loss = train_metrics.get('eval_loss', 'N/A')
        if isinstance(eval_loss, (int, float)):
            logger.info(f"最终评估损失: {eval_loss:.4f}")
        else:
            logger.info(f"最终评估损失: {eval_loss}")
        
        # 保存最终模型
        logger.info(f"保存模型到: {MODEL_SAVE_DIR}")
        trainer.save_model(MODEL_SAVE_DIR)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        
        # 保存训练历史
        history_path = os.path.join(MODEL_SAVE_DIR, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_metrics, f, ensure_ascii=False, indent=2)
        
        return model
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}", exc_info=True)
        # 如果训练失败，尝试使用更简单的训练方式
        logger.info("尝试使用简单训练模式...")
        return train_simple(model, train_dataset, val_dataset, tokenizer)

def train_simple(model, train_dataset, val_dataset, tokenizer):
    """简化版训练函数"""
    logger.info("使用简化训练模式")
    
    # 自定义数据收集器，确保移除token_type_ids
    from transformers import DataCollatorForSeq2Seq
    
    class SimpleDataCollator:
        def __init__(self, tokenizer, model):
            self.collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
        
        def __call__(self, features):
            batch = self.collator(features)
            # 移除token_type_ids如果存在
            if "token_type_ids" in batch:
                del batch["token_type_ids"]
            return batch
    
    data_collator = SimpleDataCollator(tokenizer, model)
    
    # 使用简单的DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=data_collator
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: 验证损失 = {avg_val_loss:.4f}")
        
        # 保存检查点
        if epoch % 2 == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_dir = os.path.join(MODEL_SAVE_DIR, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"保存检查点到: {checkpoint_dir}")
    
    # 保存最终模型
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    logger.info(f"模型保存到: {MODEL_SAVE_DIR}")
    
    return model

def generate_predictions(model, test_data, tokenizer):
    """生成预测结果"""
    logger.info("开始生成预测结果")
    
    model.eval()
    predictions = []
    
    for item in tqdm(test_data, desc="生成预测"):
        try:
            # 准备输入文本
            input_text = prepare_input_text(item)
            
            # 编码输入 - 关键：禁用token_type_ids
            inputs = tokenizer(
                input_text,
                max_length=MAX_INPUT_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False  # BART不需要token_type_ids
            ).to(DEVICE)
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.95,
                )
            
            # 解码结果
            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # ========== 核心修改：清理所有空白字符 ==========
            # 1. 移除所有半角空格、全角空格、制表符、换行符
            generated_text = re.sub(r'[\s\u3000\t\n\r]', '', generated_text)
            # 2. 移除连续的标点符号（可选，增强文本质量）
            generated_text = re.sub(r'([，。！？；：])+', r'\1', generated_text)
            # 3. 首尾去空格（兜底）
            generated_text = generated_text.strip()
            # ==============================================
            
            # 移除可能的前缀
            if generated_text.startswith("将以下DB编码翻译成中文句子："):
                generated_text = generated_text.replace("将以下DB编码翻译成中文句子：", "").strip()
            
        except Exception as e:
            logger.warning(f"样本ID {item['sample_id']} 生成失败：{str(e)}")
            generated_text = "生成失败"
        
        # 保存结果
        predictions.append({
            "sample_id": item["sample_id"],
            "db_codes": item["db_codes"],
            "original_chinese": item["original_chinese"],
            "generated_chinese": generated_text,
            "original_db_codes_str": item["original_db_codes_str"],
            "db_code_meanings": item.get("db_code_meanings", {}),
            "input_text": input_text
        })
    
    return predictions

# ===================== 主函数 =====================
def main():
    try:
        logger.info("="*60)
        logger.info("开始执行BART中文模型微调与推理流程")
        logger.info("="*60)
        
        # 设置环境变量避免tokenizer并行问题
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # 1. 加载数据
        train_data = load_data(TRAIN_DATA_PATH)
        test_data = load_data(TEST_DATA_PATH)
        
        # 2. 加载tokenizer - 使用BertTokenizer
        logger.info(f"加载Tokenizer：{MODEL_NAME_OR_PATH}")
        try:
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
            logger.info(f"Tokenizer加载成功，类型：{type(tokenizer).__name__}")
            logger.info(f"Tokenizer词汇表大小：{len(tokenizer)}")
            
        except Exception as e:
            logger.error(f"本地tokenizer加载失败：{str(e)}")
            raise
        
        # 3. 数据预处理
        logger.info("开始数据预处理")
        
        # 为训练数据添加输入文本字段
        for sample in train_data:
            sample["input_text"] = prepare_input_text(sample)
        
        for sample in test_data:
            sample["input_text"] = prepare_input_text(sample)
        
        # 转换为Dataset
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        # 划分训练集和验证集 (10% 作为验证集)
        train_val_split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        
        logger.info(f"数据集划分：训练集 {len(train_dataset)}，验证集 {len(val_dataset)}，测试集 {len(test_data)}")
        
        # 应用预处理函数
        logger.info("应用预处理函数...")
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        # 4. 训练模型
        model = train_model(train_dataset, val_dataset, tokenizer)
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 5. 生成预测
        predictions = generate_predictions(model, test_data, tokenizer)
        
        # 6. 保存预测结果
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"预测结果已保存至：{OUTPUT_PATH}，共{len(predictions)}条记录")
        
        # 7. 生成示例对比
        logger.info("\n===== 预测结果示例 =====")
        good_examples = []
        bad_examples = []
        
        for pred in predictions:
            orig_len = len(pred["original_chinese"])
            gen_len = len(pred["generated_chinese"])
            # 简单的质量判断：生成文本长度不能太短，也不能包含"生成失败"
            if gen_len > 5 and "生成失败" not in pred["generated_chinese"]:
                good_examples.append(pred)
            else:
                bad_examples.append(pred)
        
        # 展示3个好的例子
        logger.info(f"良好生成示例 ({len(good_examples)}个):")
        for i in range(min(3, len(good_examples))):
            pred = good_examples[i]
            logger.info(f"样本ID: {pred['sample_id']}")
            logger.info(f"输入: {pred['input_text'][:100]}...")
            logger.info(f"原始中文: {pred['original_chinese']}")
            logger.info(f"生成中文: {pred['generated_chinese']}")
            logger.info("-" * 50)
        
        if bad_examples:
            logger.info(f"生成质量较差的示例 ({len(bad_examples)}个):")
            for i in range(min(2, len(bad_examples))):
                pred = bad_examples[i]
                logger.info(f"样本ID: {pred['sample_id']} - 生成: {pred['generated_chinese'][:50]}...")
        
        # 8. 生成统计信息
        total_chars = sum(len(p["generated_chinese"]) for p in predictions)
        avg_chars = total_chars / len(predictions) if predictions else 0
        success_count = sum(1 for p in predictions if len(p["generated_chinese"]) > 5 and "生成失败" not in p["generated_chinese"])
        
        logger.info("\n===== 生成结果统计 =====")
        logger.info(f"总样本数: {len(predictions)}")
        logger.info(f"成功生成数: {success_count} ({success_count/len(predictions)*100:.1f}%)")
        logger.info(f"平均生成长度: {avg_chars:.1f} 字符")
        logger.info(f"输出文件: {OUTPUT_PATH}")
        
        logger.info("="*60)
        logger.info("BART中文模型微调与推理流程执行完成")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"流程执行失败：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()