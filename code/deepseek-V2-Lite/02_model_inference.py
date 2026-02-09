import json
import os
import torch
import logging
import datetime
from tqdm import tqdm
import gc
import sys

# ===================== 日志配置 =====================
def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    log_dir = "../results/logs/deepseekv2_logs/02_model_inference"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_deepseek_log_{timestamp}.log")
    
    log_format = "%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    # 屏蔽第三方库日志
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ===================== 配置参数 =====================
INPUT_DATA_PATH = "../results/processed_data/test_data.json"
OUTPUT_DIR = "../results/new_deepseekv2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "inference_results_deepseekv2.json")

# DeepSeek-V2-Lite模型配置
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
MODEL_CACHE_DIR = "/root/autodl-tmp/dongba/model/DeepSeek-V2-Lite"

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16  # 使用半精度减少显存占用

# 生成参数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
REPETITION_PENALTY = 1.1

# ===================== 数据加载 =====================
def load_test_data():
    """加载测试数据"""
    logger.info(f"加载测试数据: {INPUT_DATA_PATH}")
    if not os.path.exists(INPUT_DATA_PATH):
        raise FileNotFoundError(f"测试数据不存在: {INPUT_DATA_PATH}")
    
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"成功加载 {len(data)} 条测试样本")
    return data

# ===================== 模型加载 =====================
def load_deepseek_model():
    """加载DeepSeek-V2-Lite模型"""
    logger.info(f"开始加载DeepSeek-V2-Lite模型，路径: {MODEL_CACHE_DIR}")
    
    # 检查模型目录是否存在
    if not os.path.exists(MODEL_CACHE_DIR):
        logger.error(f"模型目录不存在: {MODEL_CACHE_DIR}")
        raise FileNotFoundError(f"模型目录不存在: {MODEL_CACHE_DIR}")
    
    try:
        # 导入必要的库
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("开始加载tokenizer...")
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE_DIR,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("开始加载模型...")
        # 加载模型 - DeepSeek-V2-Lite是MoE模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE_DIR,
            torch_dtype=TORCH_DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 确保模型在正确设备上
        if DEVICE == "cuda" and model.device.type != "cuda":
            model = model.to(DEVICE)
        
        # 设置为评估模式
        model.eval()
        
        logger.info(f"模型加载完成，设备: {model.device}")
        logger.info(f"模型类型: {type(model)}")
        logger.info(f"Tokenizer词汇表大小: {len(tokenizer)}")
        
        # 打印模型配置信息
        if hasattr(model.config, "num_experts"):
            logger.info(f"MoE专家数: {model.config.num_experts}")
        if hasattr(model.config, "num_experts_per_tok"):
            logger.info(f"每token激活专家数: {model.config.num_experts_per_tok}")
        
        # 清空显存
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        raise

# ===================== 提示词构建 =====================
def build_deepseek_prompt(sample):
    """构建DeepSeek的提示词"""
    # 提取DB编码和释义
    db_info = []
    for db_code in sample.get("db_codes", []):
        meanings = sample.get("db_code_meanings", {}).get(db_code, ["未知释义"])
        meaning = meanings[0] if meanings else "未知释义"
        db_info.append(f"{db_code}: {meaning}")
    
    db_context = "\n".join(db_info)
    
#     # DeepSeek的聊天格式
#     prompt = f"""你是一个专业的小语种翻译专家，擅长将符号编码转换为自然流畅的中文句子。
# 任务要求：将DB编码序列转换为符合语境的中文句子，要求语义准确、语言自然。

# DB编码及其含义：
# {db_context}

# 原始参考译文（仅供参考风格）：
# {sample['original_chinese']}

# 请根据上述DB编码的含义，生成一个完整、自然的中文句子。注意：
# 1. 必须涵盖所有编码的核心含义
# 2. 保持语言流畅，符合中文表达习惯
# 3. 避免直接复制编码，要将其含义自然融入句子
# 4. 输出应为单个完整句子

# 生成的中文翻译："""

    # DeepSeek的聊天格式
    prompt = f"""任务：根据DB编码释义，生成符合语境的中文文本。

DB编码及其含义：
{db_context}

原始参考译文（仅供参考风格）：
{sample['original_chinese']}

生成要求：
1. 包含所有DB编码的核心含义
2. 语言流畅，与原始中文风格一致
3. 必须将所有DB编码替换为对应的中文释义（参考下方映射表）
4. 长度与原始中文相近

生成的中文翻译（只给出翻译后的句子）："""
    
    return prompt

# ===================== 文本生成 =====================
def generate_with_deepseek(model, tokenizer, sample, max_retries=2):
    """使用DeepSeek生成文本，带重试机制"""
    # 使用局部变量而不是修改全局变量
    current_max_new_tokens = MAX_NEW_TOKENS
    prompt = build_deepseek_prompt(sample)
    
    for attempt in range(max_retries):
        try:
            # 编码输入
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # 移动到设备
            if DEVICE == "cuda":
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # 生成配置 - 移除不支持参数
            generation_kwargs = {
                "max_new_tokens": current_max_new_tokens,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": DO_SAMPLE,
                "repetition_penalty": REPETITION_PENALTY,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # 解码生成结果
            generated = tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            ).strip()
            
            # 清理生成结果
            generated = clean_generated_text(generated)
            
            logger.debug(f"样本ID {sample['sample_id']} 生成成功: {generated[:50]}...")
            return generated
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"样本ID {sample['sample_id']} 显存不足，清理缓存 (尝试 {attempt+1}/{max_retries})")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            # 尝试减少max_new_tokens（使用局部变量）
            current_max_new_tokens = max(64, current_max_new_tokens // 2)
            logger.info(f"将max_new_tokens减少到: {current_max_new_tokens}")
            continue
            
        except Exception as e:
            logger.warning(f"样本ID {sample['sample_id']} 生成失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                continue
            else:
                return f"生成失败: {str(e)[:100]}"
    
    return "生成失败: 达到最大重试次数"

def clean_generated_text(text):
    """清理生成的文本"""
    if not text:
        return "空生成"
    
    # 移除可能的特殊标记
    special_tokens = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|assistant|>", "<|system|>", "<|user|>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # 移除首尾空白和引号
    text = text.strip()
    text = text.strip('"').strip("'").strip("。").strip()
    
    # 如果文本以冒号或逗号开头，去除
    if text.startswith(("：", ":", "，", ",")):
        text = text[1:].strip()
    
    # 确保以句号结束
    if text and text[-1] not in ("。", "！", "？", ".", "!", "?"):
        text += "。"
    
    return text

# ===================== 批量推理 =====================
def batch_inference(model, tokenizer, samples):
    """批量推理"""
    logger.info(f"开始批量推理，样本数: {len(samples)}")
    
    results = []
    success_count = 0
    fail_count = 0
    
    # # 限制样本数量以避免显存溢出
    # max_samples = min(100, len(samples))  # 根据显存调整
    # if len(samples) > max_samples:
    #     logger.info(f"样本数较多({len(samples)})，限制为前 {max_samples} 个样本进行测试")
    #     samples = samples[:max_samples]
    
    for i, sample in enumerate(tqdm(samples, desc="推理进度")):
        try:
            # 生成文本
            generated_text = generate_with_deepseek(model, tokenizer, sample)
            
            # 构建结果
            result = {
                "sample_id": sample.get("sample_id", i + 1),
                "db_codes": sample.get("db_codes", []),
                "db_code_meanings": sample.get("db_code_meanings", {}),
                "original_chinese": sample.get("original_chinese", ""),
                "generated_chinese": generated_text,
                "original_db_codes_str": sample.get("original_db_codes_str", "")
            }
            
            results.append(result)
            
            # 统计
            if "生成失败" not in generated_text and generated_text != "空生成":
                success_count += 1
            else:
                fail_count += 1
                if "生成失败" in generated_text:
                    logger.warning(f"样本ID {sample['sample_id']} 生成失败: {generated_text[:50]}")
            
            # 每3个样本清理一次显存（MoE模型可能需要更频繁）
            if (i + 1) % 3 == 0 and DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                    
        except Exception as e:
            logger.error(f"样本ID {sample.get('sample_id', i+1)} 处理异常: {str(e)}")
            fail_count += 1
            
            # 添加失败记录
            results.append({
                "sample_id": sample.get("sample_id", i + 1),
                "db_codes": sample.get("db_codes", []),
                "original_chinese": sample.get("original_chinese", ""),
                "generated_chinese": f"处理异常: {str(e)[:100]}",
                "error": str(e)[:200]
            })
            
            # 清理显存
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
    
    logger.info(f"推理完成 - 成功: {success_count}, 失败: {fail_count}, 成功率: {success_count/len(samples)*100:.2f}%")
    return results

# ===================== 保存结果 =====================
def save_results(results):
    """保存推理结果"""
    output_file = OUTPUT_PATH
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"推理结果已保存至: {output_file}")
    
    # 同时保存简化的CSV格式
    try:
        import pandas as pd
        csv_file = output_file.replace(".json", ".csv")
        
        simplified_results = []
        for r in results:
            simplified_results.append({
                "sample_id": r["sample_id"],
                "original": r["original_chinese"],
                "generated": r["generated_chinese"],
                "status": "成功" if "失败" not in r["generated_chinese"] else "失败",
                "db_codes": " ".join(r["db_codes"]) if isinstance(r["db_codes"], list) else r["db_codes"]
            })
        
        df = pd.DataFrame(simplified_results)
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        logger.info(f"简化结果已保存至: {csv_file}")
    except ImportError:
        logger.warning("pandas未安装，跳过CSV保存")
    
    return output_file

# ===================== 主函数 =====================
def main():
    try:
        logger.info("=" * 60)
        logger.info("DeepSeek-V2-Lite 模型推理流程开始")
        logger.info("=" * 60)
        
        # 设置环境
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # 检查CUDA
        if DEVICE == "cuda":
            logger.info(f"CUDA可用，设备: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 1. 加载数据
        samples = load_test_data()
        
        # 2. 加载模型
        model, tokenizer = load_deepseek_model()
        
        # 3. 执行推理
        results = batch_inference(model, tokenizer, samples)
        
        # 4. 保存结果
        output_file = save_results(results)
        
        # 5. 打印示例
        logger.info("\n" + "=" * 60)
        logger.info("推理结果示例:")
        logger.info("=" * 60)
        
        success_examples = [r for r in results if "失败" not in r["generated_chinese"]]
        
        for i, result in enumerate(success_examples[:3]):
            logger.info(f"\n示例 {i+1}:")
            logger.info(f"样本ID: {result['sample_id']}")
            logger.info(f"DB编码: {', '.join(result['db_codes'][:3])}{'...' if len(result['db_codes']) > 3 else ''}")
            logger.info(f"原始中文: {result['original_chinese'][:80]}...")
            logger.info(f"生成中文: {result['generated_chinese'][:80]}...")
            logger.info("-" * 50)
        
        # 6. 统计信息
        success = len(success_examples)
        total = len(results)
        avg_len = sum(len(r["generated_chinese"]) for r in results) / total if total > 0 else 0
        
        logger.info("\n" + "=" * 60)
        logger.info("推理统计:")
        logger.info(f"总样本数: {total}")
        logger.info(f"成功生成: {success}")
        logger.info(f"生成成功率: {success/total*100:.2f}%")
        logger.info(f"平均生成长度: {avg_len:.1f} 字符")
        logger.info(f"输出文件: {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"推理流程失败: {str(e)}", exc_info=True)
        
        # 提供具体的错误信息
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"详细错误信息:\n{error_details}")
        
        raise

if __name__ == "__main__":
    main()