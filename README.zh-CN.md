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


## 模块

三个模块指南分别提供数据安装、训练与评测命令。
[任务实现索引](src/egms_qa/qa_construction/tasks/README.md)列出各任务组的依赖和重建范围。
对应的数据、权重与配方集中于
[EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa)。

| 模块 | 代码与指南 | Hugging Face 发布内容 |
|---|---|---|
| Encoder | [预训练与 token 提取](src/egms_encoder/README.md) | [编码器权重、配置与归一化参数](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [QA 构建与复现](src/egms_qa/qa_construction/README.md) | [源瓦片、tokens、标签、参考值表与 QA 记录](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [语言模型适配与评测](src/egms_qa/translator/README.md) | [Qwen、Gemma、Llama 和 Mistral 四个版本](https://huggingface.co/risenyard/egms-qa-translator) |

## 安装

需要 Python 3.10 或更高版本，Translator 工作流需要 CUDA。
按[环境配置指南](docs/environment.md)安装基础包与可选依赖，
再进入上方模块指南准备数据、运行示例。

## 数据集

发布数据包含 EGMS Level-3 Ortho Vertical 产品 2019–2023 参考期内的
10,000 个重叠 7 km 瓦片。每个瓦片以 `[N,294]` 数组存储位移历史，固定瓦片级
划分为 8,000 个训练瓦片、1,000 个验证瓦片和 1,000 个测试瓦片。

QA construction 定义 78 个任务，覆盖观测质量、运动特征、空间组织、时间动态、
表示属性和拒答边界。参考值表与标签提供生成问答记录所需的目标值。

发布数据集包含源瓦片、预计算 encoder tokens、参考值表、标签和 QA split，
支持三个模块的直接使用与复现。
[数据集卡片](https://huggingface.co/datasets/risenyard/egms-qa-dataset)
说明文件布局和数据契约。

## 评测

编码器评测衡量掩码区间的重建误差。Translator 评测分别衡量数值答案、类别答案，
以及对超范围问题的拒答。

Translator 报告采用 1,000 个测试瓦片上的 71 个任务，分别对 29 个数值任务的 R²、
28 个分类任务和 14 个拒答任务的平衡准确率取宏平均。

| 模型 | 评测 | 报告结果 |
|---|---|---|
| EGMS-QA Encoder | 掩码重建 | 1,000 个留出瓦片上的 RMSE 为 1.510 mm |
| Mistral translator | 数值答案 | 平均 R² 为 0.778 |
| Llama translator | 类别答案 | 平均平衡准确率为 0.777 |

表中的 Mistral 和 Llama 分别取得四个版本中对应答案类型的最高平均指标。
完整结果与协议设置见 [Encoder](https://huggingface.co/risenyard/egms-qa-encoder)
和 [Translator](https://huggingface.co/risenyard/egms-qa-translator) 模型卡。

## 适用范围

代码使用已准备好的 EGMS-QA 瓦片。新数据集合需满足
[编码器输入要求](https://huggingface.co/risenyard/egms-qa-encoder#input-requirements)，包括位移单位、
分量、时间采样与坐标几何。使用发布的冻结 checkpoint 时保留其配套 normalization；
在另一语料上训练新 encoder 时，仅用新语料的训练集拟合 normalization。
官方产品下载与数据准备需要单独的工作流。

EGMS-QA 描述已观测的形变历史。答案不用于确定成因、预测未来运动或认证结构安全。

## 引用

软件引用信息见 [CITATION.cff](CITATION.cff)。使用源测量数据的研究还应引用
[来源说明](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md)
中标明的 EGMS 产品。

## 许可

代码采用 [MIT License](LICENSE)，EGMS-QA 创建的数据与模型产物采用 CC-BY-4.0。
重打包的 EGMS 测量保留 Copernicus Land Monitoring Service 的来源标注与修改说明
要求，详见 [DATA_LICENSE](DATA_LICENSE)。
