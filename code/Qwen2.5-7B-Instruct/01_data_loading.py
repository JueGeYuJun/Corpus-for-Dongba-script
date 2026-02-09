import json
import os
import logging
import datetime
import random
from datasets import load_from_disk, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== 日志配置 =====================
def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    # 创建日志目录
    log_dir = "../results/logs/01_data_loading"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_loading_log_{timestamp}.log")
    
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

# 配置路径
PROCESSED_DATA_DIR = "../processed_data"
DB_CODE_DICT_PATH = os.path.join(PROCESSED_DATA_DIR, "db_code_dict.json")
ARROW_DATA_ROOT = os.path.join(PROCESSED_DATA_DIR, "db_chinese_dataset")
OUTPUT_DIR = "../results/processed_data"  # 修改为目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集划分比例
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42

def load_db_code_dict():
    """加载DB编码-释义字典"""
    try:
        with open(DB_CODE_DICT_PATH, "r", encoding="utf-8") as f:
            db_dict = json.load(f)
        
        # 格式校验：确保key为DBxxx，value为列表
        for db_code, meanings in db_dict.items():
            if not isinstance(meanings, list):
                db_dict[db_code] = [meanings]
        
        logger.info(f"DB编码字典加载完成，共{len(db_dict)}条记录，示例：{list(db_dict.items())[:2]}")
        return db_dict
    except Exception as e:
        logger.error(f"加载DB编码字典失败：{str(e)}", exc_info=True)
        raise
    
def load_full_arrow_data():
    """加载并合并train/val/test所有Arrow数据为全量数据集"""
    # 遍历所有子目录（train/val/test）
    data_dirs = [
        os.path.join(ARROW_DATA_ROOT, sub_dir)
        for sub_dir in ["train", "validation", "test"]
        if os.path.exists(os.path.join(ARROW_DATA_ROOT, sub_dir))
    ]
    
    if not data_dirs:
        error_msg = f"未找到Arrow数据目录！请检查路径：{ARROW_DATA_ROOT}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 加载并合并所有数据集
    full_datasets = []
    for dir_path in data_dirs:
        logger.info(f"加载数据目录：{dir_path}")
        try:
            ds = load_from_disk(dir_path)
            full_datasets.append(ds)
            logger.info(f"成功加载{len(ds)}条数据 from {dir_path}")
        except Exception as e:
            logger.warning(f"加载目录{dir_path}失败：{str(e)}，跳过该目录")
    
    if not full_datasets:
        raise ValueError("未成功加载任何数据集")
    
    full_dataset = concatenate_datasets(full_datasets)
    logger.info(f"合并后总样本数：{len(full_dataset)}")
    
    # 格式化全量样本
    formatted_samples = []
    for sample in tqdm(full_dataset, desc="格式化全量数据集"):
        # 字段名仍需与原数据一致：db_sequence（DB编码字符串）、chinese_translation（原始中文）

        db_codes_str = sample.get("db_sequence", "")  # 原始是空格分隔的字符串
        original_chinese = sample.get("chinese_translation", "")
        
        # 放宽过滤条件，只过滤完全为空的样本
        if not db_codes_str and not original_chinese:
            logger.warning(f"跳过完全无效样本：DB编码和中文翻译均为空")
            continue

        # 将DB编码字符串按空格分割为列表，过滤空字符串
        db_codes_list = [code.strip() for code in db_codes_str.split() if code.strip()]
        
        # 对于只有部分为空的样本，给出警告但仍保留
        if not db_codes_str:
            logger.warning(f"样本DB编码为空，中文翻译: {original_chinese[:50]}...")
        if not original_chinese:
            logger.warning(f"样本中文翻译为空，DB编码: {db_codes_str[:50]}...")

        formatted_samples.append({
            "sample_id": len(formatted_samples) + 1,  # 重新生成全局唯一sample_id
            "db_codes": db_codes_list,  # 存储为列表，而非字符串
            "original_chinese": original_chinese,
            "original_db_codes_str": db_codes_str  # 保留原始编码字符串
        })
    
    logger.info(f"格式化完成，有效样本数：{len(formatted_samples)}")
    return formatted_samples

def split_dataset(samples):
    """将数据集划分为训练集和测试集"""
    logger.info(f"开始划分数据集，训练集比例: {TRAIN_RATIO}, 测试集比例: {TEST_RATIO}")
    
    # 划分训练集和测试集
    train_samples, test_samples = train_test_split(
        samples,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED
    )
    
    logger.info(f"数据集划分完成 - 训练集: {len(train_samples)}条, 测试集: {len(test_samples)}条")
    return train_samples, test_samples
    
def main():
    try:
        logger.info("="*50)
        logger.info("开始执行数据加载与预处理流程")
        logger.info("="*50)
        
        # 加载数据
        db_code_dict = load_db_code_dict()
        full_samples = load_full_arrow_data()
        
        # 为每个样本补充DB编码的释义
        for sample in tqdm(full_samples, desc="补充DB释义"):
            db_code_meanings = {}
            for db_code in sample["db_codes"]:
                if db_code not in db_code_dict:
                    logger.warning(f"样本ID {sample['sample_id']} 包含未知DB编码: {db_code}")
                db_code_meanings[db_code] = db_code_dict.get(db_code, ["未知释义"])
            sample["db_code_meanings"] = db_code_meanings
        
        # 划分数据集
        train_samples, test_samples = split_dataset(full_samples)
        
        # 保存格式化数据
        train_path = os.path.join(OUTPUT_DIR, "train_data.json")
        test_path = os.path.join(OUTPUT_DIR, "test_data.json")
        full_path = os.path.join(OUTPUT_DIR, "full_data.json")
        
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
            
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(full_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"全量数据加载完成！")
        logger.info(f"训练集保存至: {train_path}, 共{len(train_samples)}条样本")
        logger.info(f"测试集保存至: {test_path}, 共{len(test_samples)}条样本")
        logger.info(f"全量数据保存至: {full_path}, 共{len(full_samples)}条样本")
        logger.info(f"示例样本：{full_samples[0]['sample_id']}")
        
    except Exception as e:
        logger.error(f"数据处理过程发生错误：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()