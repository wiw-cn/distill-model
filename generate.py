#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型蒸馏数据生成脚本 (GitHub Actions 版本)
================================================
基于 minimax-m2.5 模型，自动生成复杂领域的极其复杂的问题（含伪命题和含有错误的命题），
然后调用同一模型进行逐步推理回答，生成用于蒸馏的高质量训练数据。

配置通过环境变量传入，适合 GitHub Actions 定时运行。
"""

import json
import os
import re
import random
import time
import logging
import argparse
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from openai import OpenAI
except ImportError:
    print("正在安装 openai 库...")
    os.system("pip install openai -q")
    from openai import OpenAI

# ==================== 配置（从环境变量读取）====================
API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://ollama.com/v1")
MODEL = os.environ.get("MODEL", "minimax-m2.5")
COUNT = int(os.environ.get("COUNT", "500"))
WORKERS = int(os.environ.get("WORKERS", "5"))

# 输出目录（GitHub Actions 中通常设为仓库目录）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "distill_output")

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
        "description": "形式逻辑、命题逻辑、谓词逻辑、悖论、逻辑谬误识别",
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
        "description": "高等数学、数论、概率统计、抽象代数、拓扑学、组合数学",
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
        "description": "算法设计、数据结构、代码调试、复杂度分析、系统设计",
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
    """从文本中提取JSON对象"""
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
    """确保回答包含 Step 1-5 和最终结论"""
    if "Step 1:" in text and "Step 2:" in text and "Step 3:" in text:
        return text
    return text


# ==================== API 客户端 ====================
class ModelClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.max_retries = 5
        self.retry_delay = 3
        self._lock = threading.Lock()

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        last_error = None
        current_max_tokens = max_tokens
        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    response = self.client.chat.completions.create(
                        model=self.model,
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
                    return content.strip()
                return None
            except Exception as e:
                last_error = e
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
                "model": self.client.model,
                "timestamp": datetime.now().isoformat()
            }
        return None


# ==================== 数据生成流水线 ====================
class DistillationPipeline:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = ModelClient(api_key, base_url, model)
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
        logger.info(f"开始生成 {total_count} 条数据，模型: {self.model}")
        records = []
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            logger.info(f"加载已有记录: {len(records)} 条")
        domains = list(DOMAINS.keys())
        for i in range(total_count):
            if len(records) >= total_count:
                break
            idx = len(records)
            domain = domains[idx % len(domains)]
            question_type = QUESTION_TYPES[idx % len(QUESTION_TYPES)]
            record = self.generate_one(domain, question_type)
            if record:
                records.append(record)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (idx + 1) % batch_size == 0:
                logger.info(f"进度: {len(records)}/{total_count} | 成功: {self.stats['success']}")
            time.sleep(2)
        self._save_records(records, output_file)
        stats_file = output_file.replace(".jsonl", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"完成！共 {len(records)} 条")
        return output_file

    def generate_batch_concurrent(self, total_count: int, output_file: str, batch_size: int = 10, max_workers: int = 5) -> str:
        logger.info(f"开始并发 {total_count} 条 (workers={max_workers})，模型: {self.model}")
        records = []
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            logger.info(f"加载已有记录: {len(records)} 条")
        domains = list(DOMAINS.keys())
        completed = len(records)
        file_lock = threading.Lock()

        def worker_task(idx):
            if idx < len(records):
                return None
            client = ModelClient(self.api_key, self.base_url, self.model)
            qgen = QuestionGenerator(client)
            agen = AnswerGenerator(client)
            domain = domains[idx % len(domains)]
            qtype = QUESTION_TYPES[idx % len(QUESTION_TYPES)]
            qdata = qgen.generate_question(domain, qtype)
            if not qdata:
                with self._stats_lock:
                    self.stats["failed_generation"] += 1
                return None
            adata = agen.generate_answer(qdata)
            if not adata:
                with self._stats_lock:
                    self.stats["failed_answer"] += 1
                return None
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

        tasks = list(range(len(records), total_count))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_task, idx): idx for idx in tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    record = future.result()
                    if record:
                        with file_lock:
                            records.append(record)
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"任务异常 (idx={idx}): {e}")
                completed += 1
                if completed % batch_size == 0:
                    logger.info(f"进度: {completed}/{total_count} | 成功: {self.stats['success']}")
        self._save_records(records, output_file)
        stats_file = output_file.replace(".jsonl", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"完成！共 {len(records)} 条")
        return output_file

    def _save_records(self, records: list, output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


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

    if not API_KEY:
        print("错误: 未设置 API_KEY 环境变量")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.convert_only:
        convert_to_training_format(args.convert_only, args.convert_only.replace(".jsonl", "_training.json"))
        return

    count = 3 if args.test else (args.count or COUNT)
    workers = args.workers or WORKERS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(OUTPUT_DIR, f"distill_data_{timestamp}.jsonl")

    pipeline = DistillationPipeline(API_KEY, BASE_URL, MODEL)

    if args.concurrent:
        pipeline.generate_batch_concurrent(count, output_file, max_workers=workers)
    else:
        pipeline.generate_batch(count, output_file)

    training_file = output_file.replace(".jsonl", "_training.json")
    convert_to_training_format(output_file, training_file)

    print(f"\n{'='*60}")
    print(f"原始数据: {output_file}")
    print(f"训练格式: {training_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
