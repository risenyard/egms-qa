```text
 _____ _____ ___  ___ _____        _____  ___
|  ___|  __ \|  \/  |/  ___|      |  _  |/ _ \
| |__ | |  \/| .  . |\ `--. ______| | | / /_\ \
|  __|| | __ | |\/| | `--. \______| | | |  _  |
| |___| |_\ \| |  | |/\__/ /      \ \/' / | | |
\____/ \____/\_|  |_/\____/        \_/\_\_| |_/
```

# EGMS-QA

*[English](README.md) · [中文](README.zh-CN.md)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-blue.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](pyproject.toml)
[![Stars](https://img.shields.io/github/stars/risenyard/egms-qa?style=flat)](https://github.com/risenyard/egms-qa/stargazers)
[![Forks](https://img.shields.io/github/forks/risenyard/egms-qa?style=flat)](https://github.com/risenyard/egms-qa/network/members)

面向[欧洲地面运动服务](https://egms.land.copernicus.eu/)(EGMS)的持久散射体形变
时间序列自然语言问答系统。

EGMS-QA 读取每个 7 km 瓦片(tile)内的 EGMS 点位时间序列,将其编码为一组固定长度的
token,再由一个宿主语言模型用自然语言回答监测类问题,并对超出范围的问题给出经过校准
的拒答。

```
瓦片(点数可变,294 步历史)
   → 冻结的 EGMS 编码器 + 8×8 池化 → 65 × 256 token
   → 两层投影器 → 前缀 ; "Question: …\nAnswer:" → 冻结宿主 LLM + LoRA → 答案
```

## 三个模块

| 模块 | 目录 | 内容 |
|---|---|---|
| **encoder** | [`src/egms_encoder/`](src/egms_encoder/) | 冻结的瓦片表示与 token 提取 |
| **qa_construction** | [`src/egms_qa/qa_construction/`](src/egms_qa/qa_construction/) | 78 个任务的定义与问答对记录 |
| **translator** | [`src/egms_qa/translator/`](src/egms_qa/translator/) | 投影器 + LoRA 在宿主 LLM 上的训练与评测 |

每个模块都有各自的 README。任务体系(78 个任务,A/B/C/D/S/X 六族)与数据集
datasheet 记录在 [`src/egms_qa/qa_construction/README.md`](src/egms_qa/qa_construction/README.md)。

## 结果(留出测试集)

冻结编码器对被掩盖区间的重建 RMSE 为 1.510 mm,接近源产品的残差噪声。四个宿主模型
(Qwen、Gemma、Llama、Mistral)以相同配方训练后,数值任务平均 R² 最高达 0.778,
分类任务平衡准确率最高达 0.777,对超范围问题接近满分拒答;在打乱 token 的对照下退化
到接近随机——说明答案确实依赖于所给瓦片。用
`python -m egms_qa.translator.summarize_results` 可重新生成完整结果表。

## 安装

```bash
pip install -e .                 # 核心(表示 + QA 渲染)
pip install -e ".[translator]"   # + 宿主 LLM 训练/评测
pip install -e ".[tasks]"        # + 任务参考值计算
```

需要 Python ≥ 3.10。编码器 token 提取与翻译器训练/评测需要 GPU。

## 数据

代码在本仓库;重数据在数据发布包中:

- 编码器 checkpoint 与 1 万瓦片的 token 缓存,
- 问答对记录与各任务族的参考值表,
- 四个训练好的翻译器 adapter(每个宿主模型一个)。

**下载:** _<数据发布链接——待补充>_。放置位置:

```
data/encoder/checkpoint/encoder.pt
data/encoder/tokens/encoder_tokens_10k.pt
outputs/qa/…            # QA 记录 + 标签
outputs/tasks/…         # 各任务族参考值表
outputs/runs/<key>/best/   # 训练好的 adapter(qwen | gemma | llama | mistral)
```

路径均可通过环境变量 `EGMS_QA_ROOT`、`EGMS_QA_DATA`、`EGMS_QA_OUTPUTS`、
`EGMS_ENCODER_HOME` 覆盖(见 [`src/egms_qa/paths.py`](src/egms_qa/paths.py))。
原始 EGMS Level-3 产品为 Copernicus 数据,本发布不再转发;见
[`data/METADATA.md`](data/METADATA.md)。

## 复现

```bash
# 1. token:下载缓存,或从原始瓦片存储提取
python -m egms_encoder.extract_tokens --output-dir outputs/tokens

# 2. 任务标签 + QA 记录(或直接下载)
python -m egms_qa.qa_construction.build_probe_labels
python -m egms_qa.qa_construction.generate_qa --out-dir outputs/qa

# 3. 训练并评测一个宿主模型
python -m egms_qa.translator.train --qwen-path Qwen/Qwen3.5-9B \
    --token-cache data/encoder/tokens/encoder_tokens_10k.pt --output-dir outputs/runs/qwen
python -m egms_qa.translator.evaluate --adapter-dir outputs/runs/qwen/best --split test

# 4. 四模型汇总报告
python -m egms_qa.translator.summarize_results
```

`pytest` 运行答案提取器的测试。

## 许可

代码以 MIT 许可发布([`LICENSE`](LICENSE));数据集与模型权重以 CC-BY-4.0 发布
([`DATA_LICENSE`](DATA_LICENSE))。
