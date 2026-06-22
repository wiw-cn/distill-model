#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型蒸馏数据生成脚本 (GitHub Actions 版本)
================================================
基于 minimax-m2.5 模型，自动生成复杂领域的极其复杂的问题（含伪命题和含有错误的命题），
然后调用同一模型进行逐步推理回答，生成用于蒸馏的高质量训练数据。

特性：
- 多 API Key 轮询（避免单 key 速率限制）
- 生成一条存储一条（即时持久化，进程崩溃不丢数据）
- 429 智能退避重试
- 邮件通知（可选）
- 断点续传
"""

import json
import os
import re
import random
import time
import logging
import argparse
import smtplib
import ssl
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

try:
    from openai import OpenAI
except ImportError:
    print("正在安装 openai 库...")
    os.system("pip install openai -q")
    from openai import OpenAI

# ==================== 配置（从环境变量读取）====================

# 支持多 key：API_KEY 或 API_KEYS（逗号分隔）
API_KEY = os.environ.get("API_KEY", "")
API_KEYS_STR = os.environ.get("API_KEYS", "")
API_KEYS: List[str] = [k.strip() for k in API_KEYS_STR.split(",") if k.strip()] if API_KEYS_STR else ([API_KEY] if API_KEY else [])

BASE_URL = os.environ.get("BASE_URL", "https://ollama.com/v1")
MODEL = os.environ.get("MODEL", "minimax-m2.5")
COUNT = int(os.environ.get("COUNT", "500"))
WORKERS = int(os.environ.get("WORKERS", "5"))

# 输出目录
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "distill_output")

# 邮件通知配置（可选）
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "")

# 日志配置
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "generation.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== 领域定义 ====================
DOMAINS = {
    "logic": {
        "name": "逻辑推理",
        "subtopics": [
            "命题逻辑中的蕴含与等价关系",
            "三段论推理的有效性判断",
            "逻辑悖论（如说谎者悖论、罗素悖论）",
            "条件命题的真值判断",
            "逻辑谬误识别（如滑坡谬误、稻草人谬误、循环论证）",
            "模态逻辑中的必然与可能",
            "反事实条件推理",
            "多前提逻辑推理链",
            "逻辑等价变换与化简",
            "谓词逻辑中的量词嵌套推理"
        ]
    },
    "math": {
        "name": "数学",
        "subtopics": [
            "高等数学中的极限与连续性证明",
            "数论中的素数分布与同余问题",
            "概率论中的贝叶斯推断与条件概率",
            "抽象代数中的群论与环论证明",
            "组合数学中的递推与生成函数",
            "实分析中的测度论基础",
            "图论中的染色问题与 Ramsey 理论",
            "线性代数中的特征值与矩阵分解",
            "数学归纳法与强归纳法的复杂应用",
            "数列与级数的收敛性判断"
        ]
    },
    "code": {
        "name": "代码与算法",
        "subtopics": [
            "动态规划的状态转移方程设计",
            "图算法中的最短路径与最小生成树",
            "高级数据结构（线段树、树状数组、跳表）",
            "算法正确性证明与复杂度分析",
            "并发编程中的死锁检测与避免",
            "编译原理中的语法分析与类型推断",
            "数据库查询优化与索引策略",
            "分布式系统中的一致性协议",
            "代码中的边界条件与异常处理",
            "函数式编程中的高阶函数与柯里化"
        ]
    }
}

QUESTION_TYPES = ["true_proposition", "false_proposition", "flawed_proposition"]


# ==================== 工具函数 ====================
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def ensure_step_format(text: str) -> str:
    if "Step 1:" in text and "Step 2:" in text and "Step 3:" in text:
        return text
    return text


# ==================== 多 Key 轮询管理器 ====================
class KeyRotator:
    """多 API Key 轮询管理，自动跳过被限流的 key"""

    def __init__(self, keys: List[str], base_url: str, model: str):
        self.keys = keys
        self.base_url = base_url
        self.model = model
        self._index = 0
        self._lock = threading.Lock()
        self._cooldown: dict = {}  # key -> 解除冷却时间
        self._key_stats = {k: {"success": 0, "fail": 0, "429": 0} for k in keys}

    def get_next_key(self) -> str:
        """获取下一个可用的 key"""
        with self._lock:
            now = time.time()
            # 先尝试找不在冷却期的 key
            for _ in range(len(self.keys)):
                key = self.keys[self._index % len(self.keys)]
                self._index += 1
                cooldown_end = self._cooldown.get(key, 0)
                if now >= cooldown_end:
                    return key
            # 所有 key 都在冷却，找最快解除的
            key = min(self.keys, key=lambda k: self._cooldown.get(k, 0))
            wait = self._cooldown.get(key, 0) - now
            if wait > 0:
                logger.warning(f"所有 key 都在冷却，等待 {wait:.1f}s")
                time.sleep(wait)
            return key

    def mark_success(self, key: str):
        with self._lock:
            self._key_stats[key]["success"] += 1

    def mark_429(self, key: str, cooldown_seconds: float = 60):
        """标记 key 被 429，进入冷却期"""
        with self._lock:
            self._key_stats[key]["429"] += 1
            self._cooldown[key] = time.time() + cooldown_seconds
            logger.warning(f"Key {key[:8]}... 被429，冷却 {cooldown_seconds:.0f}s")

    def mark_fail(self, key: str):
        with self._lock:
            self._key_stats[key]["fail"] += 1

    def get_stats(self) -> dict:
        with self._lock:
            return {k[:8] + "...": v for k, v in self._key_stats.items()}


# ==================== API 客户端 ====================
class ModelClient:
    def __init__(self, key_rotator: KeyRotator, timeout: int = 120):
        self.key_rotator = key_rotator
        self.timeout = timeout
        self.max_retries = 5
        self.retry_delay = 3
        self._lock = threading.Lock()

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        last_error = None
        current_max_tokens = max_tokens
        used_keys = set()

        for attempt in range(self.max_retries):
            key = self.key_rotator.get_next_key()
            used_keys.add(key)

            try:
                client = OpenAI(api_key=key, base_url=self.key_rotator.base_url, timeout=self.timeout, max_retries=0)
                with self._lock:
                    response = client.chat.completions.create(
                        model=self.key_rotator.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
                        top_p=0.9,
                    )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length" and not content:
                    current_max_tokens = min(current_max_tokens * 2, 16384)
                    time.sleep(1)
                    continue
                if content:
                    self.key_rotator.mark_success(key)
                    return content.strip()
                return None
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # 429 速率限制：标记 key 冷却，指数退避
                if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str or "usage limit" in err_str:
                    cooldown = min(2 ** attempt * 10 + random.uniform(0, 5), 300)
                    self.key_rotator.mark_429(key, cooldown)
                    # 如果还有未尝试的 key，立即换 key
                    if len(used_keys) < len(self.key_rotator.keys):
                        logger.info(f"换 key 重试 ({len(used_keys)}/{len(self.key_rotator.keys)} 已用)")
                        continue
                    # 所有 key 都 429 了，等待
                    wait = min(2 ** attempt * 5 + random.uniform(0, 3), 300)
                    logger.warning(f"所有 key 都429，等待 {wait:.1f}s 后重试 ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                    used_keys.clear()
                    continue
                wait = self.retry_delay * (attempt + 1) + random.uniform(0, 2)
                logger.debug(f"API失败 ({attempt + 1}/{self.max_retries}): {e}, 等待{wait:.1f}s")
                time.sleep(wait)

        logger.error(f"API最终失败: {last_error}")
        return None


# ==================== 问题生成器 ====================
class QuestionGenerator:
    def __init__(self, client: ModelClient):
        self.client = client

    def generate_question(self, domain: str, question_type: str) -> Optional[dict]:
        domain_info = DOMAINS[domain]
        subtopic = random.choice(domain_info["subtopics"])
        type_desc = {
            "true_proposition": "生成一个【正确但需要复杂推理才能验证的命题】",
            "false_proposition": "生成一个【看似合理但实际错误的伪命题】，错误必须非常隐蔽",
            "flawed_proposition": "生成一个【含有微妙错误的命题】，需要仔细审查才能定位错误"
        }
        system_prompt = f"""你是{domain_info['name']}领域的出题专家。请直接生成一道关于「{subtopic}」的复杂问题。
类型要求：{type_desc[question_type]}
要求：问题极其复杂，包含具体数值/条件/约束，需要5步以上推理。用中文输出。
直接输出JSON（不要输出其他内容）：
{{"question": "问题描述"}}"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"生成{domain_info['name']}问题。只输出JSON。"}
        ]
        response = self.client.chat(messages, temperature=0.7, max_tokens=2048)
        if response:
            result = extract_json(response)
            if result and "question" in result:
                question_text = result["question"]
                if isinstance(question_text, str):
                    inner = extract_json(question_text)
                    if inner and "question" in inner:
                        question_text = inner["question"]
                return {
                    "question": question_text,
                    "question_type": question_type,
                    "domain": domain,
                    "subtopic": subtopic,
                    "difficulty": "expert",
                    "has_hidden_error": question_type != "true_proposition",
                }
            clean_text = re.sub(r'```(?:json)?\s*', '', response).strip()
            return {
                "question": clean_text,
                "question_type": question_type,
                "domain": domain,
                "subtopic": subtopic,
                "difficulty": "expert",
                "has_hidden_error": question_type != "true_proposition",
            }
        return None


# ==================== 回答生成器 ====================
class AnswerGenerator:
    def __init__(self, client: ModelClient):
        self.client = client

    def generate_answer(self, question_data: dict) -> Optional[dict]:
        question = question_data["question"]
        question_type = question_data["question_type"]
        domain = question_data["domain"]
        domain_name = DOMAINS[domain]["name"]
        type_hint = {
            "true_proposition": "该命题可能为真，请验证其正确性",
            "false_proposition": "该命题可能包含错误，请找出其中的错误",
            "flawed_proposition": "该命题可能含有隐蔽的错误，请仔细审查"
        }
        system_prompt = f"""你是{domain_name}专家。分析以下命题，按Step格式输出，必须完成全部5个Step和最终结论。
提示：{type_hint[question_type]}
格式（严格遵循）：
Step 1: 初步判断 — 命题真假初步判断+理由
Step 2: 核心概念 — 关键定义、定理、公式
Step 3: 推理分析 — 详细推导、公式、逻辑链
Step 4: 验证检验 — 多角度验证、边界条件
Step 5: 深入探讨 — 深层含义、推广、反例
最终结论：命题真/假/含错误，总结关键发现"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"分析：{question}"}
        ]
        response = self.client.chat(messages, temperature=0.3, max_tokens=8192)
        if response:
            processed = ensure_step_format(response)
            if "最终结论" not in processed or "Step 5:" not in processed:
                logger.warning("回答可能被截断，用更大预算重试")
                retry = self.client.chat(messages, temperature=0.3, max_tokens=16384)
                if retry:
                    processed = ensure_step_format(retry)
            return {
                "answer": processed,
                "question_type": question_type,
                "domain": domain,
                "model": self.key_rotator.model if hasattr(self, 'key_rotator') else "unknown",
                "timestamp": datetime.now().isoformat()
            }
        return None


# ==================== 数据生成流水线 ====================
class DistillationPipeline:
    def __init__(self, keys: List[str], base_url: str, model: str):
        self.keys = keys
        self.base_url = base_url
        self.model = model
        self.key_rotator = KeyRotator(keys, base_url, model)
        self.client = ModelClient(self.key_rotator)
        self.question_generator = QuestionGenerator(self.client)
        self.answer_generator = AnswerGenerator(self.client)
        self.stats = {
            "total": 0, "success": 0,
            "failed_generation": 0, "failed_answer": 0,
            "by_domain": {d: 0 for d in DOMAINS},
            "by_type": {t: 0 for t in QUESTION_TYPES}
        }
        self._stats_lock = threading.Lock()

    def generate_one(self, domain: str, question_type: str) -> Optional[dict]:
        with self._stats_lock:
            self.stats["total"] += 1
        question_data = self.question_generator.generate_question(domain, question_type)
        if not question_data:
            with self._stats_lock:
                self.stats["failed_generation"] += 1
            logger.warning(f"问题生成失败: {domain}, {question_type}")
            return None
        answer_data = self.answer_generator.generate_answer(question_data)
        if not answer_data:
            with self._stats_lock:
                self.stats["failed_answer"] += 1
            logger.warning(f"回答生成失败: {domain}, {question_type}")
            return None
        record = {
            "id": f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['total']:04d}",
            "question": question_data["question"],
            "question_type": question_type,
            "domain": domain,
            "domain_name": DOMAINS[domain]["name"],
            "subtopic": question_data.get("subtopic", ""),
            "difficulty": question_data.get("difficulty", "expert"),
            "has_hidden_error": question_data.get("has_hidden_error", False),
            "answer": answer_data["answer"],
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
        with self._stats_lock:
            self.stats["success"] += 1
            self.stats["by_domain"][domain] += 1
            self.stats["by_type"][question_type] += 1
        return record

    def generate_batch(self, total_count: int, output_file: str, batch_size: int = 10) -> str:
        """串行模式：生成一条，立即存储一条"""
        logger.info(f"开始串行生成 {total_count} 条，模型: {self.model}")
        existing_count = self._count_existing(output_file)
        domains = list(DOMAINS.keys())
        for i in range(existing_count, total_count):
            domain = domains[i % len(domains)]
            question_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]
            record = self.generate_one(domain, question_type)
            if record:
                # 生成一条，立即追加存储一条
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % batch_size == 0:
                logger.info(f"进度: {i + 1}/{total_count} | 成功: {self.stats['success']} | Key统计: {self.key_rotator.get_stats()}")
            time.sleep(2)
        self._save_stats(output_file)
        return output_file

    def generate_batch_concurrent(self, total_count: int, output_file: str, batch_size: int = 10, max_workers: int = 5) -> str:
        """并发模式：每个 worker 独立 client，生成一条立即存储一条"""
        logger.info(f"开始并发 {total_count} 条 (workers={max_workers})，模型: {self.model}")
        existing_count = self._count_existing(output_file)
        domains = list(DOMAINS.keys())
        completed = existing_count
        success_count = 0
        file_lock = threading.Lock()

        def worker_task(idx):
            if idx < existing_count:
                return None
            # 每个 worker 独立创建 key rotator 和 client
            worker_rotator = KeyRotator(self.keys, self.base_url, self.model)
            worker_client = ModelClient(worker_rotator)
            qgen = QuestionGenerator(worker_client)
            agen = AnswerGenerator(worker_client)
            domain = domains[idx % len(domains)]
            qtype = QUESTION_TYPES[idx % len(QUESTION_TYPES)]
            qdata = qgen.generate_question(domain, qtype)
            if not qdata:
                return {"error": "question_failed", "domain": domain, "type": qtype}
            adata = agen.generate_answer(qdata)
            if not adata:
                return {"error": "answer_failed", "domain": domain, "type": qtype}
            return {
                "id": f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:04d}",
                "question": qdata["question"],
                "question_type": qtype,
                "domain": domain,
                "domain_name": DOMAINS[domain]["name"],
                "subtopic": qdata.get("subtopic", ""),
                "difficulty": qdata.get("difficulty", "expert"),
                "has_hidden_error": qdata.get("has_hidden_error", False),
                "answer": adata["answer"],
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }

        tasks = list(range(existing_count, total_count))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_task, idx): idx for idx in tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result and "error" not in result:
                        # 生成一条，立即追加存储一条
                        with file_lock:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            success_count += 1
                    elif result and "error" in result:
                        with self._stats_lock:
                            if result["error"] == "question_failed":
                                self.stats["failed_generation"] += 1
                            else:
                                self.stats["failed_answer"] += 1
                except Exception as e:
                    logger.error(f"任务异常 (idx={idx}): {e}")
                completed += 1
                if completed % batch_size == 0:
                    logger.info(f"进度: {completed}/{total_count} | 成功: {success_count} | Key统计: {self.key_rotator.get_stats()}")
        self.stats["success"] = success_count
        self._save_stats(output_file)
        logger.info(f"完成！共 {success_count} 条新数据（总计 {existing_count + success_count} 条）")
        return output_file

    def _count_existing(self, output_file: str) -> int:
        """统计已有记录数"""
        count = 0
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            logger.info(f"加载已有记录: {count} 条")
        return count

    def _save_stats(self, output_file: str):
        """保存统计文件"""
        stats_file = output_file.replace(".jsonl", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)


# ==================== 邮件通知 ====================
def send_notification(subject: str, body: str):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL]):
        logger.info("邮件通知未配置，跳过")
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = NOTIFY_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, NOTIFY_EMAIL, msg.as_string())
        logger.info(f"邮件通知已发送至 {NOTIFY_EMAIL}")
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")


# ==================== 转换为训练格式 ====================
def convert_to_training_format(input_file: str, output_file: str):
    training_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            system_prompt = (
                f"你是{record['domain_name']}领域专家。"
                f"按Step 1到Step 5格式逐步推理，最终给出明确结论。"
            )
            training_data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": record["question"]},
                    {"role": "assistant", "content": record["answer"]}
                ],
                "metadata": {
                    "domain": record["domain"],
                    "question_type": record["question_type"],
                    "difficulty": record["difficulty"],
                    "has_hidden_error": record["has_hidden_error"],
                    "model": record["model"],
                }
            })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    logger.info(f"训练格式已保存: {output_file} ({len(training_data)} 条)")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="蒸馏数据生成 (GitHub Actions)")
    parser.add_argument("--count", type=int, default=None, help="生成条数")
    parser.add_argument("--output", type=str, default=None, help="输出文件名")
    parser.add_argument("--test", action="store_true", help="测试模式，只生成3条")
    parser.add_argument("--convert-only", type=str, default=None, help="仅转换格式")
    parser.add_argument("--concurrent", action="store_true", help="并发模式")
    parser.add_argument("--workers", type=int, default=None, help="并发线程数")
    args = parser.parse_args()

    if not API_KEYS:
        print("错误: 未设置 API_KEY 或 API_KEYS 环境变量")
        return

    logger.info(f"加载 {len(API_KEYS)} 个 API Key")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.convert_only:
        convert_to_training_format(args.convert_only, args.convert_only.replace(".jsonl", "_training.json"))
        return

    count = 3 if args.test else (args.count or COUNT)
    workers = args.workers or WORKERS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(OUTPUT_DIR, f"distill_data_{timestamp}.jsonl")

    pipeline = DistillationPipeline(API_KEYS, BASE_URL, MODEL)

    if args.concurrent:
        pipeline.generate_batch_concurrent(count, output_file, max_workers=workers)
    else:
        pipeline.generate_batch(count, output_file)

    training_file = output_file.replace(".jsonl", "_training.json")
    convert_to_training_format(output_file, training_file)

    # 发送完成通知
    data_count = pipeline._count_existing(output_file)
    notify_body = f"""蒸馏数据生成完成

模型: {MODEL}
生成数量: {data_count} 条
原始数据: {output_file}
训练格式: {training_file}
输出目录: {os.path.abspath(OUTPUT_DIR)}

文件说明:
- *.jsonl          原始数据（每行一条JSON）
- *_training.json  训练格式（标准多轮对话）
- *_stats.json     生成统计
- generation.log   运行日志
"""
    send_notification(f"[蒸馏数据] 生成完成 {data_count}条", notify_body)

    print(f"\n{'='*60}")
    print(f"原始数据: {output_file}")
    print(f"训练格式: {training_file}")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
