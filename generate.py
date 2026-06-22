#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型蒸馏数据生成脚本（GitHub Actions 适配版）
========================
基于商汤日日新 API，自动生成复杂领域的极其复杂的问题（含伪命题和含有错误的命题），
然后调用大模型进行逐步推理回答，生成用于蒸馏的高质量训练数据。

生成问题模型: sensenova-6.7-flash-lite
回答问题模型: deepseek-v4-flash

所有敏感配置均通过环境变量注入，适合在 CI/CD 中运行。
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

# 移除不必要的并发导入，Actions 中推荐串行运行
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading

try:
    from openai import OpenAI
except ImportError:
    print("正在安装 openai 库...")
    os.system("pip install openai --break-system-packages -q")
    from openai import OpenAI

# ==================== 配置（从环境变量读取） ====================
# 安全提示：不要在代码中硬编码 API 密钥，请通过 GitHub Secrets 设置
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("请设置环境变量 API_KEY，例如：export API_KEY='sk-xxx'")

BASE_URL = os.environ.get("BASE_URL", "https://token.sensenova.cn/v1")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "sensenova-6.7-flash-lite")
ANSWERER_MODEL = os.environ.get("ANSWERER_MODEL", "deepseek-v4-flash")

# 输出目录（Actions 中建议使用 ./output）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")

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

# 问题类型
QUESTION_TYPES = ["true_proposition", "false_proposition", "flawed_proposition"]


# ==================== 工具函数 ====================
def extract_json(text: str) -> Optional[dict]:
    """从文本中提取JSON对象，支持多种包裹格式"""
    if not text:
        return None

    # 方法1: 尝试直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 方法2: 提取 ```json ... ``` 中的内容
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 方法3: 提取第一个 { ... } 块
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # 方法4: 找到第一个 { 和最后一个 } 之间的内容
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def ensure_step_format(text: str) -> str:
    """确保回答按 Step 1/2/3/4/5 格式包含至少5步推理"""
    if "Step 1:" in text and "Step 2:" in text and "Step 3:" in text:
        return text
    # 如果缺少Step格式，直接返回原文（依赖prompt约束）
    return text


# ==================== API 客户端 ====================
class ModelClient:
    """统一的API调用客户端，串行模式（适配 Actions）"""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.max_retries = 5
        self.retry_delay = 3
        # 移除线程锁（串行运行不需要）
        # self._lock = threading.Lock()

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 4096, extra_body: dict = None) -> Optional[str]:
        """发送聊天请求，带重试机制，自动增大token限制"""
        last_error = None
        current_max_tokens = max_tokens
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                    "top_p": 0.9,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                # 直接调用，无需线程锁
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # 如果因为token限制被截断且content为空，增大token限制重试
                if finish_reason == "length" and not content:
                    current_max_tokens = min(current_max_tokens * 2, 16384)
                    logger.debug(f"输出被截断且内容为空，增大max_tokens到{current_max_tokens}重试")
                    time.sleep(1)
                    continue

                if content:
                    return content.strip()
                return None
            except Exception as e:
                last_error = e
                wait = self.retry_delay * (attempt + 1) + random.uniform(0, 2)
                logger.debug(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, 等待 {wait:.1f}s")
                time.sleep(wait)
        logger.error(f"API调用最终失败 (模型: {self.model}): {last_error}")
        return None


# ==================== 问题生成器 ====================
class QuestionGenerator:
    """生成复杂领域问题，包含伪命题和含有错误的命题"""

    def __init__(self, client: ModelClient):
        self.client = client

    def generate_question(self, domain: str, question_type: str) -> Optional[dict]:
        """生成单个问题"""
        domain_info = DOMAINS[domain]
        subtopic = random.choice(domain_info["subtopics"])

        type_desc = {
            "true_proposition": "生成一个【正确但需要复杂推理才能验证的命题】",
            "false_proposition": "生成一个【看似合理但实际错误的伪命题】，错误必须非常隐蔽",
            "flawed_proposition": "生成一个【含有微妙错误的命题】，需要仔细审查才能定位错误"
        }

        system_prompt = f"""你是{domain_info["name"]}领域的出题专家。请直接生成一道关于「{subtopic}」的复杂问题。

类型要求：{type_desc[question_type]}

要求：问题极其复杂，包含具体数值/条件/约束，需要5步以上推理。用中文输出。

直接输出JSON（不要输出其他内容）：
{{"question": "问题描述"}}"""

        user_prompt = f"生成{domain_info['name']}问题。只输出JSON。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat(messages, temperature=0.7, max_tokens=4096, extra_body={"reasoning_effort": "none"})
        if response:
            result = extract_json(response)
            if result and "question" in result:
                question_text = result["question"]
                # 清理嵌套JSON：如果question值本身是JSON字符串，提取其中的question
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
                    "expected_reasoning_steps": 5
                }
            else:
                # JSON解析失败，直接用原始文本作为问题
                logger.debug(f"JSON解析失败，使用原始文本。domain={domain}, type={question_type}")
                clean_text = re.sub(r'```(?:json)?\s*', '', response).strip()
                return {
                    "question": clean_text,
                    "question_type": question_type,
                    "domain": domain,
                    "subtopic": subtopic,
                    "difficulty": "expert",
                    "has_hidden_error": question_type != "true_proposition",
                    "expected_reasoning_steps": 5
                }
        return None


# ==================== 回答生成器 ====================
class AnswerGenerator:
    """调用大模型生成包含思考标签的逐步推理回答"""

    def __init__(self, client: ModelClient):
        self.client = client

    def generate_answer(self, question_data: dict) -> Optional[dict]:
        """生成推理回答"""
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

        response = self.client.chat(messages, temperature=0.3, max_tokens=8192, extra_body={"reasoning_effort": "none"})
        if response:
            processed_response = ensure_step_format(response)
            # 检测是否截断：缺少后面的Step或最终结论
            if "最终结论" not in processed_response or "Step 5:" not in processed_response:
                logger.warning("回答可能被截断，尝试用更大预算重试")
                retry_resp = self.client.chat(messages, temperature=0.3, max_tokens=16384, extra_body={"reasoning_effort": "none"})
                if retry_resp:
                    processed_response = ensure_step_format(retry_resp)
            return {
                "answer": processed_response,
                "question_type": question_type,
                "domain": domain,
                "model": self.client.model,
                "timestamp": datetime.now().isoformat()
            }
        return None


# ==================== 数据生成流水线（串行版） ====================
class DistillationPipeline:
    """蒸馏数据生成流水线（适配 GitHub Actions 串行运行）"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.generator_client = ModelClient(api_key, base_url, GENERATOR_MODEL)
        self.answerer_client = ModelClient(api_key, base_url, ANSWERER_MODEL)
        self.question_generator = QuestionGenerator(self.generator_client)
        self.answer_generator = AnswerGenerator(self.answerer_client)

        self.stats = {
            "total": 0,
            "success": 0,
            "failed_generation": 0,
            "failed_answer": 0,
            "by_domain": {d: 0 for d in DOMAINS},
            "by_type": {t: 0 for t in QUESTION_TYPES}
        }
        # 串行无需锁
        # self._stats_lock = threading.Lock()
        # self._records_lock = threading.Lock()

    def generate_one(self, domain: str, question_type: str) -> Optional[dict]:
        """生成一条完整的蒸馏数据"""
        self.stats["total"] += 1

        # 1. 生成问题
        question_data = self.question_generator.generate_question(domain, question_type)
        if not question_data:
            self.stats["failed_generation"] += 1
            logger.warning(f"问题生成失败: domain={domain}, type={question_type}")
            return None

        # 2. 生成回答
        answer_data = self.answer_generator.generate_answer(question_data)
        if not answer_data:
            self.stats["failed_answer"] += 1
            logger.warning(f"回答生成失败: domain={domain}, type={question_type}")
            return None

        # 3. 组装数据
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
            "generator_model": GENERATOR_MODEL,
            "answerer_model": ANSWERER_MODEL,
            "timestamp": datetime.now().isoformat()
        }

        self.stats["success"] += 1
        self.stats["by_domain"][domain] += 1
        self.stats["by_type"][question_type] += 1

        return record

    def generate_batch(self, total_count: int, output_file: str, batch_size: int = 10) -> str:
        """批量生成蒸馏数据（串行模式，稳定可靠）"""
        logger.info(f"开始生成 {total_count} 条蒸馏数据...")
        logger.info(f"生成模型: {GENERATOR_MODEL}")
        logger.info(f"回答模型: {ANSWERER_MODEL}")

        records = []
        domains = list(DOMAINS.keys())

        # 如果输出文件已存在，加载已有记录（支持断点续传）
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            logger.info(f"加载已有记录: {len(records)} 条")

        for i in range(total_count):
            if len(records) >= total_count:
                break

            # 均匀分配领域和类型
            idx = len(records)
            domain = domains[idx % len(domains)]
            question_type = QUESTION_TYPES[idx % len(QUESTION_TYPES)]

            record = self.generate_one(domain, question_type)
            if record:
                records.append(record)
                # 每条成功后立即追加保存
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 定期打印进度
            if (idx + 1) % batch_size == 0:
                logger.info(
                    f"进度: {len(records)}/{total_count} | "
                    f"成功: {self.stats['success']} | "
                    f"失败(问题): {self.stats['failed_generation']} | "
                    f"失败(回答): {self.stats['failed_answer']}"
                )

            # 请求间隔（避免速率限制）
            time.sleep(3)

        # 最终保存（完整覆盖）
        self._save_records(records, output_file)

        # 保存统计信息
        stats_file = output_file.replace(".jsonl", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        logger.info(f"数据生成完成！共生成 {len(records)} 条数据")
        logger.info(f"数据文件: {output_file}")
        logger.info(f"统计文件: {stats_file}")

        return output_file

    def _save_records(self, records: list, output_file: str):
        """保存记录到文件"""
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== 转换为训练格式 ====================
def convert_to_training_format(input_file: str, output_file: str):
    """将JSONL数据转换为标准的训练格式（多轮对话格式）"""
    training_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            system_prompt = (
                f"你是一个{record['domain_name']}领域的专家。"
                f"请对给定的命题/问题进行深度分析。"
                f"按照Step 1到Step 5的格式逐步推理，最终给出明确结论。"
            )

            sample = {
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
                    "generator_model": record["generator_model"],
                    "answerer_model": record["answerer_model"]
                }
            }
            training_data.append(sample)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f"训练格式数据已保存: {output_file} ({len(training_data)} 条)")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="大模型蒸馏数据生成工具")
    parser.add_argument("--count", type=int, default=500, help="生成数据条数")
    parser.add_argument("--output", type=str, default=None, help="输出文件名")
    parser.add_argument("--test", action="store_true", help="测试模式，只生成3条数据")
    parser.add_argument("--convert-only", type=str, default=None, help="仅转换已有数据为训练格式")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.convert_only:
        convert_to_training_format(
            args.convert_only,
            args.convert_only.replace(".jsonl", "_training.json")
        )
        return

    count = 3 if args.test else args.count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(OUTPUT_DIR, f"distill_data_{timestamp}.jsonl")

    pipeline = DistillationPipeline(API_KEY, BASE_URL)
    pipeline.generate_batch(count, output_file)

    # 转换为训练格式
    training_file = output_file.replace(".jsonl", "_training.json")
    convert_to_training_format(output_file, training_file)

    print(f"\n{'='*60}")
    print(f"生成完成！")
    print(f"原始数据: {output_file}")
    print(f"训练格式: {training_file}")
    print(f"日志文件: {os.path.join(OUTPUT_DIR, 'generation.log')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
