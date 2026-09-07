# EGMS-QA

[English](README.md) · [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data terms](https://img.shields.io/badge/data-CC%20BY%204.0%20%2B%20CLMS-blue.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](pyproject.toml)
[![CI](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/release-v1.0.0-green.svg)](CHANGELOG.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EGMS--QA-yellow)](https://huggingface.co/collections/risenyard/egms-qa)

EGMS-QA 支持基于欧洲地面运动服务（EGMS）位移时间序列的自然语言问答。
Encoder 从每个 7 km 瓦片中提取点位表示，并池化为 65 个 token。
QA construction 定义监测任务、计算参考值，并生成问题与参考答案记录。
Translator 通过投影器与 LoRA 适配宿主语言模型，使其根据冻结的瓦片表示回答问题。

![EGMS-QA 总体框架](docs/assets/egms-framework.png)

## 代码与模型

| 模块 | 指南 | 发布产物 |
|---|---|---|
| Encoder | [编码瓦片与复现预训练](src/egms_encoder/README.md) | [权重、归一化参数与训练配方](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [生成问答与查看任务目录](src/egms_qa/qa_construction/README.md) | [瓦片、tokens、标签、QA split 与参考值表](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [训练与评测宿主语言模型](src/egms_qa/translator/README.md) | [四套投影器与 LoRA 模型](https://huggingface.co/risenyard/egms-qa-translator) |

GitHub 提供代码，Hugging Face 提供数据、权重与训练配方，集中于
[EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa)。
各模块指南包含数据安装、训练、评测和输出文件说明。

## 安装

编码一个发布瓦片，检查安装是否可用：

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --max-tiles 1 --output-dir outputs/tokens
```

这一 encoder smoke test 下载所需文件，并在 `outputs/tokens/` 写出 tokens 与
元数据。环境要求 Python 3.10 或更高版本。少量 encoder 检查可使用 CPU，完整
集合建议使用 GPU。

要运行问答评测，先安装 `pip install -e '.[translator]'`，再按
[发布模型评测指南](src/egms_qa/translator/README.md#evaluate-a-released-model)操作。
Translator 训练与评测需要 CUDA。数据集包含预计算 tokens，可直接用于这些流程。

## 评测

编码器评测衡量掩码区间的重建误差。Translator 评测分别衡量数值答案、类别答案，
以及对超范围问题的拒答。

| 模型 | 评测 | 报告结果 |
|---|---|---|
| EGMS-QA Encoder | 掩码重建 | 1,000 个留出瓦片上的 RMSE 为 1.510 mm |
| Mistral translator | 数值答案 | 平均 R² 为 0.778 |
| Llama translator | 类别答案 | 平均平衡准确率为 0.777 |

Translator 协议采用 1,000 个测试瓦片上的 71 个任务，分别对 29 个数值任务的 R²、
28 个分类任务和 14 个拒答任务的平衡准确率取宏平均。表中的 Mistral 和 Llama
分别取得四个版本中这两类答案的最高平均指标。完整结果与协议设置见
[模型卡](https://huggingface.co/risenyard/egms-qa-translator)。

## 数据集

数据集包含 10,000 个可直接输入模型的瓦片，位移数组形状为 `[N,294]`，固定
train/validation/test 划分为 8,000/1,000/1,000。训练使用完整的 78 个任务。
[QA 构建指南](src/egms_qa/qa_construction/README.md)提供任务定义和记录说明。

## 适用范围

新数据集合需满足[编码器输入要求](src/egms_encoder/README.md#input-requirements)。
使用发布的冻结 checkpoint 时保留其配套 normalization；在另一语料上训练新
encoder 时，仅用新语料的训练集拟合 normalization。官方产品下载与数据准备
需要单独的工作流。

EGMS-QA 描述已观测的形变历史。答案不用于确定成因、预测未来运动或认证结构安全。

## 引用

软件引用信息见 [CITATION.cff](CITATION.cff)。使用源测量数据的研究还应引用
[来源说明](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md)
中标明的 EGMS 产品。

## 许可

代码采用 [MIT License](LICENSE)，EGMS-QA 创建的数据与模型产物采用 CC-BY-4.0。
重打包的 EGMS Level-3 Ortho Vertical 测量保留 Copernicus Land Monitoring Service
的来源标注与修改说明要求，详见 [DATA_LICENSE](DATA_LICENSE)。
