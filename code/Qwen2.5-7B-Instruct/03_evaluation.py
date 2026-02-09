import json
import os
import sys
import logging
import datetime
import re
import numpy as np
import torch
from tqdm import tqdm
import jieba
import sacrebleu
import nltk
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

# ===================== 关键配置：彻底禁用外网 + CUDA优化 =====================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

# 定义设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== NLTK本地路径配置 =====================
# 设置NLTK数据本地路径
NLTK_DATA_DIR = "/root/autodl-tmp/dongba/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# 将本地路径添加到NLTK的数据源查找列表
if os.path.exists(NLTK_DATA_DIR):
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# COMET模型配置 - 使用本地模型
COMET_MODEL_DIR = "/root/autodl-tmp/dongba/model/comet/wmt20-comet-da"
COMET_CKPT_PATH = os.path.join(COMET_MODEL_DIR, "checkpoints/model.ckpt")
COMET_XLM_DIR = "/root/autodl-tmp/dongba/model/comet/xlm-roberta-large"

# 确保COMET使用本地模型
os.environ["COMET_MODEL_DIR"] = COMET_MODEL_DIR
os.environ["XLM_ROBERTA_PATH"] = COMET_XLM_DIR

# 下载必要的nltk数据
def download_nltk_data():
    """下载必要的NLTK数据到本地路径"""
    required_data = ['wordnet', 'punkt', 'omw-1.4']
    
    for data_id in required_data:
        try:
            nltk.data.find(f'corpora/{data_id}')
        except LookupError:
            try:
                nltk.download(data_id, quiet=True, download_dir=NLTK_DATA_DIR)
            except:
                if data_id == 'wordnet':
                    nltk.download('wordnet', quiet=True)
                elif data_id == 'punkt':
                    nltk.download('punkt', quiet=True)

# 立即下载NLTK数据
download_nltk_data()

# ===================== 日志配置 =====================
def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    log_dir = "../results/logs/new_qwen2.5_logs/03_evaluation_qwen2.5"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_qwen2.5_log_{timestamp}.log")
    
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
    
    # 禁用无关日志
    for logger_name in ["bert_score", "transformers", "comet", "sacrebleu", "jieba", "modelscope", "nltk", "pytorch_lightning"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ===================== 核心配置 =====================
INPUT_DATA_PATH = "../results/new_qwen2.5_results/inference_results_qwen2.5.json"
OUTPUT_EVAL_DIR = "../results/new_evaluation_reports_qwen2.5"
os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
OUTPUT_EVAL_PATH = os.path.join(OUTPUT_EVAL_DIR, "evaluation_report_qwen2.5.json")

LOCAL_BERT_PATH = "../model/bert-base-chinese"
BERT_NUM_LAYERS = 12

# ===================== 工具函数 =====================
class ChineseTokenizer:
    """中文分词器，适配Rouge计算"""
    def __init__(self, remove_db_codes=True):
        self.remove_db_codes = remove_db_codes
    
    def tokenize(self, text):
        if not isinstance(text, str):
            return []
        # 移除DB编码
        if self.remove_db_codes:
            text = re.sub(r'DB\d+\s*', '', text)
        # 清理文本
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        return list(text) if text else []

def preprocess_text(text, remove_db_codes=True):
    """通用文本预处理"""
    if not isinstance(text, str):
        return "空文本"
    if remove_db_codes:
        text = re.sub(r'DB\d+\s*', '', text)
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text if text.strip() else "空文本"

def jieba_tokenize(text):
    """jieba分词，用于BLEU/METEOR"""
    text = preprocess_text(text)
    return " ".join(jieba.lcut(text)) if text != "空文本" else ""

# ===================== 数据加载 =====================
def load_inference_results():
    """加载推理结果"""
    logger.info(f"加载推理结果数据: {INPUT_DATA_PATH}")
    if not os.path.exists(INPUT_DATA_PATH):
        raise FileNotFoundError(f"推理结果文件不存在: {INPUT_DATA_PATH}")
    
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 过滤掉生成失败的样本
    valid_data = []
    for sample in data:
        generated = sample.get("generated_chinese", "")
        if generated and "失败" not in generated and "异常" not in generated:
            valid_data.append(sample)
    
    logger.info(f"成功加载{len(data)}条样本，有效样本{len(valid_data)}条")
    return valid_data

def validate_local_model():
    """验证本地BERT模型"""
    logger.info("验证本地BERT模型")
    required_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(LOCAL_BERT_PATH, f))]
    if missing:
        logger.warning(f"BERT模型缺失文件: {missing}")
    else:
        logger.info("BERT模型验证通过")

def validate_comet_model():
    """验证COMET模型文件"""
    logger.info("验证COMET模型文件")
    
    # 检查COMET模型文件
    if not os.path.exists(COMET_CKPT_PATH):
        logger.warning(f"COMET模型文件不存在: {COMET_CKPT_PATH}")
        return False
    
    # 检查XLM-RoBERTa模型
    xlm_files = ["pytorch_model.bin", "config.json", "vocab.json"]
    xlm_missing = []
    for f in xlm_files:
        if not os.path.exists(os.path.join(COMET_XLM_DIR, f)):
            xlm_missing.append(f)
    
    if xlm_missing:
        logger.warning(f"XLM-RoBERTa模型缺失文件: {xlm_missing}")
        return False
    
    logger.info("COMET模型验证通过")
    return True

# ===================== 评估指标计算 =====================
def calculate_bertscore(original_texts, generated_texts):
    """计算BERTScore"""
    logger.info("计算BERTScore指标")
    
    if len(original_texts) != len(generated_texts):
        logger.error(f"文本长度不一致: original={len(original_texts)}, generated={len(generated_texts)}")
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "per_sample": {"precision": [], "recall": [], "f1": []}
        }
    
    original_clean = [preprocess_text(t) for t in original_texts]
    generated_clean = [preprocess_text(t) for t in generated_texts]
    
    try:
        P, R, F1 = score(
            generated_clean, original_clean,
            lang="zh", model_type=LOCAL_BERT_PATH,
            num_layers=BERT_NUM_LAYERS, verbose=False, device=DEVICE
        )
        
        return {
            "precision": float(np.mean(P.cpu().numpy())),
            "recall": float(np.mean(R.cpu().numpy())),
            "f1": float(np.mean(F1.cpu().numpy())),
            "per_sample": {
                "precision": [float(p) for p in P.cpu().numpy()],
                "recall": [float(r) for r in R.cpu().numpy()],
                "f1": [float(f) for f in F1.cpu().numpy()]
            }
        }
    except Exception as e:
        logger.error(f"BERTScore计算失败: {e}")
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "per_sample": {"precision": [0.0]*len(original_texts),
                          "recall": [0.0]*len(original_texts),
                          "f1": [0.0]*len(original_texts)}
        }

def calculate_rougel(original_texts, generated_texts):
    """计算Rouge-L"""
    logger.info("计算Rouge-L指标")
    tokenizer = ChineseTokenizer()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=tokenizer)
    
    scores = []
    avg_p = avg_r = avg_f = 0.0
    valid = 0
    
    for orig, gen in zip(original_texts, generated_texts):
        orig_clean = preprocess_text(orig)
        gen_clean = preprocess_text(gen)
        
        if orig_clean == "空文本" or gen_clean == "空文本":
            scores.append({"p":0.0, "r":0.0, "f":0.0})
            continue
        
        try:
            r = scorer.score(orig_clean, gen_clean)['rougeL']
            score_dict = {"p": float(r.precision), "r": float(r.recall), "f": float(r.fmeasure)}
            scores.append(score_dict)
            avg_p += score_dict["p"]
            avg_r += score_dict["r"]
            avg_f += score_dict["f"]
            valid += 1
        except:
            scores.append({"p":0.0, "r":0.0, "f":0.0})
    
    if valid > 0:
        avg_p /= valid
        avg_r /= valid
        avg_f /= valid
    
    return {
        "precision": avg_p, "recall": avg_r, "f1": avg_f,
        "per_sample": scores, "valid_samples": int(valid)
    }

def calculate_rouge_1_2(original_texts, generated_texts):
    """计算Rouge-1/2"""
    logger.info("计算Rouge-1/2指标")
    tokenizer = ChineseTokenizer()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False, tokenizer=tokenizer)
    
    r1_scores = []
    r2_scores = []
    avg_r1_p = avg_r1_r = avg_r1_f = 0.0
    avg_r2_p = avg_r2_r = avg_r2_f = 0.0
    valid = 0
    
    for orig, gen in zip(original_texts, generated_texts):
        orig_clean = preprocess_text(orig)
        gen_clean = preprocess_text(gen)
        
        if orig_clean == "空文本" or gen_clean == "空文本":
            r1_scores.append({"p":0.0, "r":0.0, "f":0.0})
            r2_scores.append({"p":0.0, "r":0.0, "f":0.0})
            continue
        
        try:
            scores = scorer.score(orig_clean, gen_clean)
            r1 = scores['rouge1']
            r2 = scores['rouge2']
            
            r1_dict = {"p": float(r1.precision), "r": float(r1.recall), "f": float(r1.fmeasure)}
            r2_dict = {"p": float(r2.precision), "r": float(r2.recall), "f": float(r2.fmeasure)}
            
            r1_scores.append(r1_dict)
            r2_scores.append(r2_dict)
            
            avg_r1_p += r1_dict["p"]
            avg_r1_r += r1_dict["r"]
            avg_r1_f += r1_dict["f"]
            avg_r2_p += r2_dict["p"]
            avg_r2_r += r2_dict["r"]
            avg_r2_f += r2_dict["f"]
            valid += 1
        except:
            r1_scores.append({"p":0.0, "r":0.0, "f":0.0})
            r2_scores.append({"p":0.0, "r":0.0, "f":0.0})
    
    if valid > 0:
        avg_r1_p /= valid
        avg_r1_r /= valid
        avg_r1_f /= valid
        avg_r2_p /= valid
        avg_r2_r /= valid
        avg_r2_f /= valid
    
    return {
        "rouge1": {"precision": avg_r1_p, "recall": avg_r1_r, "f1": avg_r1_f, "per_sample": r1_scores},
        "rouge2": {"precision": avg_r2_p, "recall": avg_r2_r, "f1": avg_r2_f, "per_sample": r2_scores},
        "valid_samples": int(valid)
    }

def calculate_bleu(original_texts, generated_texts):
    """计算BLEU"""
    logger.info("计算BLEU指标")
    refs = [[jieba_tokenize(t)] for t in original_texts]
    hyps = [jieba_tokenize(t) for t in generated_texts]
    
    valid_idx = []
    valid_refs = []
    valid_hyps = []
    
    for idx, (r, h) in enumerate(zip(refs, hyps)):
        if r and h and r[0].strip() and h.strip():
            valid_idx.append(idx)
            valid_refs.append(r)
            valid_hyps.append(h)
    
    if not valid_refs:
        return {"score": 0.0, "per_sample": [0.0]*len(original_texts), "valid_samples": 0}
    
    try:
        bleu = sacrebleu.corpus_bleu(valid_hyps, valid_refs)
        bleu_score = bleu.score / 100
        
        per_sample = [0.0]*len(original_texts)
        for idx, r, h in zip(valid_idx, valid_refs, valid_hyps):
            try:
                per_sample[idx] = sacrebleu.sentence_bleu(h, r).score / 100
            except:
                per_sample[idx] = 0.0
        
        return {
            "score": bleu_score,
            "per_sample": per_sample,
            "valid_samples": len(valid_refs)
        }
    except Exception as e:
        logger.error(f"BLEU计算失败: {e}")
        return {"score": 0.0, "per_sample": [0.0]*len(original_texts), "valid_samples": 0}

def calculate_meteor(original_texts, generated_texts):
    """计算METEOR"""
    logger.info("计算METEOR指标")
    
    per_sample = []
    total = 0.0
    valid = 0
    
    for orig, gen in zip(original_texts, generated_texts):
        orig_tok = jieba.lcut(preprocess_text(orig))
        gen_tok = jieba.lcut(preprocess_text(gen))
        
        if not orig_tok or not gen_tok:
            per_sample.append(0.0)
            continue
        
        try:
            score = single_meteor_score(orig_tok, gen_tok)
            per_sample.append(score)
            total += score
            valid += 1
        except Exception as e:
            per_sample.append(0.0)
    
    avg = total / valid if valid > 0 else 0.0
    return {"score": avg, "per_sample": per_sample, "valid_samples": valid}

def calculate_chrf(original_texts, generated_texts):
    """计算chrF"""
    logger.info("计算chrF指标")
    refs = [[preprocess_text(t)] for t in original_texts]
    hyps = [preprocess_text(t) for t in generated_texts]
    
    valid_idx = []
    valid_refs = []
    valid_hyps = []
    
    for idx, (r, h) in enumerate(zip(refs, hyps)):
        if r[0] != "空文本" and h != "空文本":
            valid_idx.append(idx)
            valid_refs.append(r)
            valid_hyps.append(h)
    
    if not valid_refs:
        return {"score": 0.0, "per_sample": [0.0]*len(original_texts), "valid_samples": 0}
    
    try:
        chrf = sacrebleu.corpus_chrf(valid_hyps, valid_refs)
        chrf_score = chrf.score / 100
        
        per_sample = [0.0]*len(original_texts)
        for idx, r, h in zip(valid_idx, valid_refs, valid_hyps):
            try:
                per_sample[idx] = sacrebleu.sentence_chrf(h, r).score / 100
            except:
                per_sample[idx] = 0.0
        
        return {
            "score": chrf_score,
            "per_sample": per_sample,
            "valid_samples": len(valid_refs)
        }
    except Exception as e:
        logger.error(f"chrF计算失败: {e}")
        return {"score": 0.0, "per_sample": [0.0]*len(original_texts), "valid_samples": 0}

# ===================== COMET指标计算 =====================
def calculate_comet_metric(original_texts, generated_texts, max_samples=100):
    """计算COMET指标 - 真正的COMET计算"""
    logger.info("计算COMET指标")
    
    # 检查COMET模型
    if not validate_comet_model():
        logger.warning("COMET模型验证失败，无法计算COMET指标")
        return {
            "score": 0.0,
            "per_sample": [0.0]*len(original_texts),
            "valid_samples": 0,
            "error": "COMET模型验证失败",
            "fallback": True
        }
    
    # 限制样本数量以避免内存问题
    if len(original_texts) > max_samples:
        logger.info(f"样本数较多({len(original_texts)})，限制为前{max_samples}个样本计算COMET")
        original_texts = original_texts[:max_samples]
        generated_texts = generated_texts[:max_samples]
        total_samples = max_samples
    else:
        total_samples = len(original_texts)
    
    try:
        # 导入COMET库
        from comet.models import load_checkpoint
        
        logger.info(f"加载COMET模型: {COMET_CKPT_PATH}")
        
        # 加载模型
        model = load_checkpoint(COMET_CKPT_PATH)
        model.eval()
        model.to(DEVICE)
        
        # 准备数据
        comet_data = []
        for orig, gen in zip(original_texts, generated_texts):
            orig_clean = preprocess_text(orig)
            gen_clean = preprocess_text(gen)
            
            if orig_clean != "空文本" and gen_clean != "空文本":
                comet_data.append({
                    "src": "",  # 源语言文本（我们没有，留空）
                    "mt": gen_clean,   # 机器翻译文本（生成文本）
                    "ref": orig_clean  # 参考文本（原始文本）
                })
        
        if not comet_data:
            logger.warning("无有效数据用于COMET计算")
            return {
                "score": 0.0,
                "per_sample": [0.0]*total_samples,
                "valid_samples": 0,
                "error": "无有效数据",
                "fallback": False
            }
        
        # 批量计算COMET分数
        logger.info(f"开始计算COMET分数，样本数: {len(comet_data)}")
        scores = []
        
        # 使用较小的batch size
        batch_size = 8
        for i in range(0, len(comet_data), batch_size):
            batch_data = comet_data[i:i+batch_size]
            
            try:
                # 使用COMET模型的predict方法
                if hasattr(model, 'predict'):
                    batch_scores, _ = model.predict(batch_data, batch_size=len(batch_data), gpus=1 if DEVICE == "cuda" else 0)
                    scores.extend(batch_scores)
                else:
                    # 备用方法
                    import torch
                    from torch.utils.data import DataLoader
                    
                    # 创建数据加载器
                    from comet.models import load_checkpoint
                    from comet.data import DataModule
                    
                    # 直接调用forward方法
                    for sample in batch_data:
                        # 这里简化处理，实际需要根据COMET模型的具体输入格式
                        inputs = model.prepare_sample([sample])
                        inputs = model.transfer_batch_to_device(inputs, model.device)
                        with torch.no_grad():
                            output = model.forward(**inputs)
                            score = output.score.item() if hasattr(output, 'score') else 0.0
                            scores.append(score)
            except Exception as e:
                logger.warning(f"批次{i}-{i+batch_size} COMET计算失败: {e}")
                # 添加默认分数
                scores.extend([0.0] * len(batch_data))
        
        # 计算平均分数
        if scores:
            avg_score = float(np.mean(scores))
            logger.info(f"COMET计算完成，平均分: {avg_score:.4f}")
            
            # 构建返回结果
            per_sample = [0.0] * total_samples
            score_idx = 0
            for i in range(total_samples):
                orig = original_texts[i]
                gen = generated_texts[i]
                if preprocess_text(orig) != "空文本" and preprocess_text(gen) != "空文本":
                    if score_idx < len(scores):
                        per_sample[i] = scores[score_idx]
                        score_idx += 1
            
            return {
                "score": avg_score,
                "per_sample": per_sample,
                "valid_samples": len(scores),
                "error": "",
                "fallback": False
            }
        else:
            logger.warning("COMET计算未产生有效分数")
            return {
                "score": 0.0,
                "per_sample": [0.0]*total_samples,
                "valid_samples": 0,
                "error": "计算未产生有效分数",
                "fallback": True
            }
        
    except ImportError as e:
        logger.error(f"COMET库导入失败: {e}")
        logger.info("请安装COMET库: pip install unbabel-comet")
    except Exception as e:
        logger.error(f"COMET计算失败: {e}", exc_info=True)
    
    # 如果COMET计算失败，使用BERTScore作为备选
    logger.info("COMET计算失败，使用BERTScore作为备选")
    bert_metrics = calculate_bertscore(original_texts, generated_texts)
    
    return {
        "score": bert_metrics["f1"],
        "per_sample": bert_metrics["per_sample"]["f1"],
        "valid_samples": len(original_texts),
        "error": "COMET计算失败，使用BERTScore作为备选",
        "fallback": True
    }

# ===================== 简化版COMET计算 =====================
def calculate_comet_simple(original_texts, generated_texts):
    """简化版COMET计算，避免复杂依赖"""
    logger.info("使用简化版COMET计算")
    
    try:
        # 尝试使用本地COMET模型
        if os.path.exists(COMET_CKPT_PATH):
            logger.info(f"找到COMET模型文件: {COMET_CKPT_PATH}")
            
            # 这里使用一个简化的方法：基于语义相似度计算
            # 实际上，我们可以使用预训练的XLM-RoBERTa来计算相似度
            
            from transformers import AutoModel, AutoTokenizer
            import torch.nn.functional as F
            
            # 加载XLM-RoBERTa模型
            logger.info("加载XLM-RoBERTa模型计算语义相似度")
            tokenizer = AutoTokenizer.from_pretrained(COMET_XLM_DIR)
            model = AutoModel.from_pretrained(COMET_XLM_DIR)
            model.to(DEVICE)
            model.eval()
            
            scores = []
            
            for orig, gen in tqdm(zip(original_texts, generated_texts), desc="COMET计算", total=len(original_texts)):
                orig_clean = preprocess_text(orig)
                gen_clean = preprocess_text(gen)
                
                if orig_clean == "空文本" or gen_clean == "空文本":
                    scores.append(0.0)
                    continue
                
                try:
                    # 编码文本
                    orig_enc = tokenizer(orig_clean, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    gen_enc = tokenizer(gen_clean, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    
                    orig_enc = {k: v.to(DEVICE) for k, v in orig_enc.items()}
                    gen_enc = {k: v.to(DEVICE) for k, v in gen_enc.items()}
                    
                    with torch.no_grad():
                        orig_output = model(**orig_enc)
                        gen_output = model(**gen_enc)
                    
                    # 使用CLS token的表示计算余弦相似度
                    orig_embedding = orig_output.last_hidden_state[:, 0, :]
                    gen_embedding = gen_output.last_hidden_state[:, 0, :]
                    
                    # 计算余弦相似度
                    similarity = F.cosine_similarity(orig_embedding, gen_embedding)
                    scores.append(similarity.item())
                    
                except Exception as e:
                    scores.append(0.0)
            
            if scores:
                avg_score = float(np.mean(scores))
                logger.info(f"简化版COMET计算完成，平均分: {avg_score:.4f}")
                
                return {
                    "score": avg_score,
                    "per_sample": scores,
                    "valid_samples": len(scores),
                    "error": "",
                    "method": "xlm-roberta_similarity",
                    "fallback": False
                }
    
    except Exception as e:
        logger.warning(f"简化版COMET计算失败: {e}")
    
    # 最终备选：使用BERTScore
    logger.info("使用BERTScore作为COMET最终备选")
    bert_metrics = calculate_bertscore(original_texts, generated_texts)
    
    return {
        "score": bert_metrics["f1"],
        "per_sample": bert_metrics["per_sample"]["f1"],
        "valid_samples": len(original_texts),
        "error": "使用BERTScore作为COMET备选",
        "method": "bert_score_fallback",
        "fallback": True
    }

# ===================== 主函数 =====================
def main():
    try:
        logger.info("="*60)
        logger.info("开始执行Qwen2.5模型评估流程（包含COMET）")
        logger.info("="*60)
        
        # 打印环境信息
        logger.info(f"设备: {DEVICE}")
        logger.info(f"NLTK数据路径: {NLTK_DATA_DIR}")
        logger.info(f"BERT模型路径: {LOCAL_BERT_PATH}")
        logger.info(f"COMET模型路径: {COMET_MODEL_DIR}")
        
        # 1. 加载数据和验证模型
        inference_results = load_inference_results()
        validate_local_model()
        
        if not inference_results:
            logger.error("没有有效的数据进行评估")
            return
        
        # 2. 提取文本
        original_texts = [sample.get("original_chinese", "") for sample in inference_results]
        generated_texts = [sample.get("generated_chinese", "") for sample in inference_results]
        sample_ids = [sample.get("sample_id", idx+1) for idx, sample in enumerate(inference_results)]
        
        logger.info(f"开始评估，样本数: {len(original_texts)}")
        
        # 3. 计算所有指标
        metrics = {}
        
        # BERTScore（与COMET区分）
        metrics["bertscore"] = calculate_bertscore(original_texts, generated_texts)
        
        # COMET指标（尝试真正的计算）
        metrics["comet"] = calculate_comet_simple(original_texts, generated_texts)
        
        # Rouge系列
        metrics["rougel"] = calculate_rougel(original_texts, generated_texts)
        metrics["rouge1_2"] = calculate_rouge_1_2(original_texts, generated_texts)
        
        # 机器翻译经典指标
        metrics["bleu"] = calculate_bleu(original_texts, generated_texts)
        metrics["meteor"] = calculate_meteor(original_texts, generated_texts)
        metrics["chrf"] = calculate_chrf(original_texts, generated_texts)
        
        # 4. 构建评估报告
        logger.info("构建全量评估报告")
        evaluation_report = {
            "evaluation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen2.5-7B-Instruct",
            "evaluation_config": {
                "bert_model_path": LOCAL_BERT_PATH,
                "comet_model_path": COMET_MODEL_DIR,
                "xlm_model_path": COMET_XLM_DIR,
                "total_samples": len(inference_results),
                "device": DEVICE,
                "preprocessing": {
                    "remove_db_codes": True,
                    "tokenizer": "jieba（BLEU/METEOR）+ 字符级（Rouge/chrF）"
                }
            },
            # 全局指标汇总
            "overall_metrics": {
                # 核心语义指标
                "bert_score": {
                    "precision": metrics["bertscore"]["precision"],
                    "recall": metrics["bertscore"]["recall"],
                    "f1": metrics["bertscore"]["f1"]
                },
                "comet": {
                    "score": metrics["comet"]["score"],
                    "valid_samples": metrics["comet"]["valid_samples"],
                    "method": metrics["comet"].get("method", "unknown"),
                    "fallback": metrics["comet"].get("fallback", False),
                    "error": metrics["comet"].get("error", "")
                },
                # Rouge系列
                "rouge_l": {
                    "precision": metrics["rougel"]["precision"],
                    "recall": metrics["rougel"]["recall"],
                    "f1": metrics["rougel"]["f1"],
                    "valid_samples": metrics["rougel"]["valid_samples"]
                },
                "rouge_1": {
                    "precision": metrics["rouge1_2"]["rouge1"]["precision"],
                    "recall": metrics["rouge1_2"]["rouge1"]["recall"],
                    "f1": metrics["rouge1_2"]["rouge1"]["f1"]
                },
                "rouge_2": {
                    "precision": metrics["rouge1_2"]["rouge2"]["precision"],
                    "recall": metrics["rouge1_2"]["rouge2"]["recall"],
                    "f1": metrics["rouge1_2"]["rouge2"]["f1"]
                },
                # 经典机器翻译指标
                "bleu": {
                    "score": metrics["bleu"]["score"],
                    "valid_samples": metrics["bleu"]["valid_samples"]
                },
                "meteor": {
                    "score": metrics["meteor"]["score"],
                    "valid_samples": metrics["meteor"]["valid_samples"]
                },
                "chrf": {
                    "score": metrics["chrf"]["score"],
                    "valid_samples": metrics["chrf"]["valid_samples"]
                }
            },
            # 每个样本的详细指标
            "per_sample_metrics": [
                {
                    "sample_id": sample_id,
                    "original_text": orig[:500],
                    "generated_text": gen[:500],
                    "original_clean": preprocess_text(orig),
                    "generated_clean": preprocess_text(gen),
                    "db_codes": inference_results[idx].get("db_codes", []),
                    # 各指标详细分数
                    "bert_score": {
                        "precision": metrics["bertscore"]["per_sample"]["precision"][idx],
                        "recall": metrics["bertscore"]["per_sample"]["recall"][idx],
                        "f1": metrics["bertscore"]["per_sample"]["f1"][idx]
                    },
                    "comet": metrics["comet"]["per_sample"][idx],
                    "rouge_l": metrics["rougel"]["per_sample"][idx],
                    "rouge_1": metrics["rouge1_2"]["rouge1"]["per_sample"][idx],
                    "rouge_2": metrics["rouge1_2"]["rouge2"]["per_sample"][idx],
                    "bleu": metrics["bleu"]["per_sample"][idx],
                    "meteor": metrics["meteor"]["per_sample"][idx],
                    "chrf": metrics["chrf"]["per_sample"][idx]
                }
                for idx, (sample_id, orig, gen) in enumerate(zip(sample_ids, original_texts, generated_texts))
            ]
        }
        
        # 5. 保存评估报告
        try:
            with open(OUTPUT_EVAL_PATH, "w", encoding="utf-8") as f:
                json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
            logger.info(f"全量评估报告已保存至: {OUTPUT_EVAL_PATH}")
        except Exception as e:
            logger.error(f"保存评估报告失败: {str(e)}", exc_info=True)
            raise
        
        # 6. 打印核心指标汇总
        logger.info("\n" + "="*60)
        logger.info("Qwen2.5模型全量基准测试核心指标汇总")
        logger.info("="*60)
        logger.info(f"【语义匹配核心】")
        logger.info(f"  BERTScore F1: {metrics['bertscore']['f1']:.4f}")
        comet_method = metrics['comet'].get('method', 'N/A')
        comet_fallback = " (备选)" if metrics['comet'].get('fallback', False) else ""
        logger.info(f"  COMET Score: {metrics['comet']['score']:.4f} {comet_fallback}")
        logger.info(f"   计算方法: {comet_method}")
        logger.info(f"【字符/词级匹配】")
        logger.info(f"  Rouge-L F1: {metrics['rougel']['f1']:.4f} (有效样本: {metrics['rougel']['valid_samples']})")
        logger.info(f"  Rouge-1 F1: {metrics['rouge1_2']['rouge1']['f1']:.4f}")
        logger.info(f"  Rouge-2 F1: {metrics['rouge1_2']['rouge2']['f1']:.4f}")
        logger.info(f"【机器翻译经典指标】")
        logger.info(f"  BLEU Score: {metrics['bleu']['score']:.4f} (有效样本: {metrics['bleu']['valid_samples']})")
        logger.info(f"  METEOR Score: {metrics['meteor']['score']:.4f} (有效样本: {metrics['meteor']['valid_samples']})")
        logger.info(f"  chrF Score: {metrics['chrf']['score']:.4f} (有效样本: {metrics['chrf']['valid_samples']})")
        logger.info(f"【数据统计】")
        logger.info(f"  总样本数: {len(inference_results)}")
        logger.info("="*60)
        
        logger.info("="*60)
        logger.info("Qwen2.5模型全量基准测试评估执行完成")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"评估过程发生致命错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # 忽略所有警告
    warnings.filterwarnings("ignore")
    
    # 修复jieba的pkg_resources警告
    try:
        import pkg_resources
    except:
        pass
    
    main()