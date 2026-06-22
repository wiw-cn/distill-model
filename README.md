# 大模型蒸馏数据生成工具 (GitHub Actions 版)

基于 minimax-m2.5 模型，自动生成复杂领域的极其复杂的问题（含伪命题和含有错误的命题），然后调用同一模型进行逐步推理回答，生成用于蒸馏的高质量训练数据。

## 快速开始

### 1. Fork 本仓库

### 2. 配置 Secrets

在仓库 Settings -> Secrets and variables -> Actions 中添加：

**必需：**

| Secret | 说明 | 示例 |
|--------|------|------|
| `API_KEY` | API密钥 | `f3a3095d0b4d...` |
| `BASE_URL` | API基础地址 | `https://ollama.com/v1` |
| `MODEL` | 模型名称 | `minimax-m2.5` |

**可选：**

| Secret | 说明 | 默认值 |
|--------|------|--------|
| `COUNT` | 每次生成条数 | `500` |
| `WORKERS` | 并发数 | `5` |
| `SMTP_HOST` | 邮件服务器地址 | 空（不发送邮件） |
| `SMTP_PORT` | 邮件服务器端口 | `587` |
| `SMTP_USER` | 发件人邮箱 | 空 |
| `SMTP_PASS` | 发件人邮箱密码/授权码 | 空 |
| `NOTIFY_EMAIL` | 收件人邮箱 | 空 |

### 3. 手动触发

进入 Actions 页面，选择 `Generate Distillation Data`，点击 `Run workflow`。

### 4. 自动定时运行

默认每天凌晨2点UTC自动运行，可在 `.github/workflows/generate.yml` 中修改 cron 表达式。

### 5. 查看结果

生成完成后：
- **文件位置**: 仓库 `data/` 目录下
- **邮件通知**: 如果配置了 SMTP，会收到包含文件路径和数量的邮件
- **GitHub Actions 日志**: Actions 页面可查看实时进度

## 本地运行

```bash
# 安装依赖
pip install openai

# 测试模式（生成3条）
API_KEY=xxx BASE_URL=https://ollama.com/v1 MODEL=minimax-m2.5 python distill_generate.py --test

# 生成500条（并发模式）
API_KEY=xxx BASE_URL=https://ollama.com/v1 MODEL=minimax-m2.5 python distill_generate.py --concurrent --workers 5

# 带邮件通知
API_KEY=xxx ... SMTP_HOST=smtp.gmail.com SMTP_USER=you@gmail.com SMTP_PASS=xxx NOTIFY_EMAIL=you@gmail.com python distill_generate.py --concurrent
```

## 输出文件

所有文件保存在 `OUTPUT_DIR` 指定的目录（默认 `distill_output/`）：

| 文件 | 说明 |
|------|------|
| `distill_data_YYYYMMDD_HHMMSS.jsonl` | 原始数据（JSONL格式，每行一条） |
| `distill_data_YYYYMMDD_HHMMSS_training.json` | 训练格式（标准多轮对话） |
| `distill_data_YYYYMMDD_HHMMSS_stats.json` | 生成统计 |
| `generation.log` | 运行日志 |

## 数据格式

### JSONL 原始数据
```json
{
  "id": "distill_20260622_014156_0001",
  "question": "问题描述...",
  "question_type": "false_proposition",
  "domain": "math",
  "domain_name": "数学",
  "subtopic": "数论中的素数分布与同余问题",
  "difficulty": "expert",
  "has_hidden_error": true,
  "answer": "Step 1: ...\nStep 2: ...\n...\n最终结论：...",
  "model": "minimax-m2.5",
  "timestamp": "2026-06-22T01:41:56"
}
```

### 训练格式
```json
{
  "messages": [
    {"role": "system", "content": "你是数学领域专家。按Step 1到Step 5格式逐步推理，最终给出明确结论。"},
    {"role": "user", "content": "问题描述..."},
    {"role": "assistant", "content": "Step 1: ...\nStep 2: ...\n...\n最终结论：..."}
  ],
  "metadata": {
    "domain": "math",
    "question_type": "false_proposition",
    "difficulty": "expert",
    "has_hidden_error": true,
    "model": "minimax-m2.5"
  }
}
```

## 回答格式

所有回答严格遵循以下 Step 格式：

```
Step 1: 初步判断 — 命题真假初步判断+理由
Step 2: 核心概念 — 关键定义、定理、公式
Step 3: 推理分析 — 详细推导、公式、逻辑链
Step 4: 验证检验 — 多角度验证、边界条件
Step 5: 深入探讨 — 深层含义、推广、反例
最终结论：命题真/假/含错误，总结关键发现
```

## 覆盖领域

- **逻辑推理**: 命题逻辑、谓词逻辑、悖论、逻辑谬误识别
- **数学**: 高等数学、数论、概率统计、抽象代数、组合数学
- **代码与算法**: 算法设计、数据结构、代码调试、复杂度分析

## 问题类型

- `true_proposition`: 正确但需要复杂推理才能验证的命题
- `false_proposition`: 看似合理但实际错误的伪命题
- `flawed_proposition`: 含有微妙错误的命题

## 特性

- **环境变量配置**: 所有参数通过环境变量传入，适合 CI/CD
- **429 智能重试**: 遇到速率限制自动指数退避等待（5s -> 10s -> 20s -> 40s...）
- **邮件通知**: 生成完成后自动发送邮件（配置可选）
- **断点续传**: 支持中断后恢复，不重复生成
- **并发模式**: 多线程加速生成
- **截断检测**: 自动检测不完整回答并用更大预算重试
- **增量保存**: 每条数据成功后立即写入文件
