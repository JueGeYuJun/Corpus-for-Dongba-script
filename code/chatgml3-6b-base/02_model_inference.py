import json
import os
import torch
import logging
import datetime
from tqdm import tqdm
from modelscope import AutoModel, AutoTokenizer
from modelscope.utils.constant import DownloadMode

# ===================== 日志配置 =====================
def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    # 创建日志目录
    log_dir = "../results/logs/02_model_inference"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_log_{timestamp}.log")
    
    # 日志格式
    log_format = "%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# 初始化日志器
logger = setup_logging()


# 配置路径（与01_data_loading.py保持一致）
INPUT_DATA_PATH = "../results/processed_data"  # 步骤1输出目录
TEST_DATA_PATH = os.path.join(INPUT_DATA_PATH, "test_data.json")  # 使用测试集进行推理
# TEST_DATA_PATH = os.path.join(INPUT_DATA_PATH, "full_data.json")  # 使用测试集进行推理
OUTPUT_PATH = "../results/new_chatgml/01.20inference_results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 模型配置
MODEL_ID = "ZhipuAI/chatglm3-6b-base"  # chatglm-6B-base模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16  # RTX 4090适配半精度
# 模型缓存路径（修改为数据盘绝对路径）
MODEL_CACHE_DIR = "/autodl-tmp/dongba/model/chatglm3-6b-base"  # 替换为实际数据盘路径
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def load_processed_data():
    """加载步骤1处理的测试集数据"""
    if not os.path.exists(TEST_DATA_PATH):
        error_msg = f"未找到测试集数据，请先运行01_data_loading.py：{TEST_DATA_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"成功加载{len(data)}条测试样本（来自{TEST_DATA_PATH}）")
        return data
    except Exception as e:
        logger.error(f"加载测试数据失败：{str(e)}", exc_info=True)
        raise

def load_modelscope_model():
    """通过ModelScope加载国内镜像的ChatGLM-6B模型"""
    try:
        logger.info(f"开始加载模型：{MODEL_ID}，缓存路径：{MODEL_CACHE_DIR}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR
        )
        
        # 加载模型
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=TORCH_DTYPE,
            cache_dir=MODEL_CACHE_DIR
        ).eval()
        
        # 确保模型在目标设备上
        if DEVICE == "cuda":
            model = model.to(DEVICE)
        
        logger.info(f"模型加载完成（ModelScope国内镜像），设备：{DEVICE}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型加载失败：{str(e)}", exc_info=True)
        raise

def build_prompt(sample):
    """构造提示词"""
    # 格式化DB编码及其释义
    db_info = []
    for db_code, meanings in sample["db_code_meanings"].items():
        db_info.append(f"{db_code}：{'; '.join(meanings)}")
    db_context = "\n".join(db_info)
    
    # 提示模板
    prompt = f"""任务：根据DB编码释义，生成符合语境的中文文本。

DB编码释义：
{db_context}

原始中文参考（风格参考）：
{sample['original_chinese'][:500]}  # 避免过长文本

生成要求：
1. 包含所有DB编码的核心含义
2. 语言流畅，与原始中文风格一致
3. 必须将所有DB编码替换为对应的中文释义（参考下方映射表）
4. 长度与原始中文相近

生成结果：
"""
    return prompt

def generate_text(model, tokenizer, sample):
    """调用模型生成文本"""
    prompt = build_prompt(sample)
    
    try:
        # 模型输入处理
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(DEVICE)
        
        # 推理配置
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 提取生成结果（排除提示部分）
        generated = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()
        return generated
    except Exception as e:
        logger.error(f"样本ID {sample['sample_id']} 生成失败：{str(e)}")
        raise

def main():
    try:
        logger.info("="*50)
        logger.info("开始执行模型推理流程")
        logger.info("="*50)
        
        # 优化CUDA性能
        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("已启用CUDA性能优化")
        
        # 加载数据和模型
        samples = load_processed_data()
        model, tokenizer = load_modelscope_model()
        
        # 批量推理
        results = []
        for sample in tqdm(samples, desc="模型推理中"):
            try:
                generated = generate_text(model, tokenizer, sample)
                sample["generated_chinese"] = generated
                results.append(sample)
                if sample["sample_id"] % 10 == 0:  # 每10个样本记录一次日志
                    logger.info(f"样本ID {sample['sample_id']} 处理成功")
            except Exception as e:
                logger.error(f"样本ID {sample['sample_id']} 处理失败：{str(e)}")
                sample["generated_chinese"] = f"推理失败：{str(e)}"
                results.append(sample)
        
        # 保存结果
        output_path = os.path.join(OUTPUT_PATH, "inference_results_allcon.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        success_count = sum(1 for x in results if "推理失败" not in x["generated_chinese"])
        fail_count = len(results) - success_count
        
        logger.info(f"推理完成！结果保存至：{output_path}")
        logger.info(f"成功样本：{success_count}")
        logger.info(f"失败样本：{fail_count}")
        logger.info(f"成功率：{success_count/len(results):.2%}")
        
    except Exception as e:
        logger.error(f"推理过程发生错误：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()