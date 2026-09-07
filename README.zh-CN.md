# EGMS-QA

[English](README.md) · [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data terms](https://img.shields.io/badge/data-CC%20BY%204.0%20%2B%20CLMS-blue.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](pyproject.toml)
[![CI](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/release-v1.0.0-green.svg)](CHANGELOG.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EGMS--QA-yellow)](https://huggingface.co/collections/risenyard/egms-qa)

EGMS-QA 基于欧洲地面运动服务（EGMS）的位移时间序列回答监测问题。冻结的编码器
将每个 7 km 瓦片表示为 65 个 token，投影器与经 LoRA 适配的语言模型据此生成数值
答案、类别答案，或对超出支持范围的问题给出拒答。

![EGMS-QA 总体框架](docs/assets/egms-framework.png)

## 使用入口

| 模块 | 指南 | 发布产物 |
|---|---|---|
| Encoder | [编码瓦片与复现预训练](src/egms_encoder/README.md) | [权重、归一化参数与训练配方](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [生成问答与查看任务目录](src/egms_qa/qa_construction/README.md) | [瓦片、tokens、标签、QA split 与参考值表](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [训练与评测宿主语言模型](src/egms_qa/translator/README.md) | [四套投影器与 LoRA 模型](https://huggingface.co/risenyard/egms-qa-translator) |

GitHub 提供代码，Hugging Face 提供数据、权重与训练配方。三个发布仓库汇集于
[EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa)。

## 快速开始

安装代码，并编码一个发布瓦片：

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --max-tiles 1 --output-dir outputs/tokens
```

提取器将所需文件下载到 HF 缓存，并在 `outputs/tokens/` 写出 tokens 与元数据。
删除 `--max-tiles 1` 可处理完整数据集。环境要求 Python 3.10 或更高版本。
少量 encoder 检查可使用 CPU，完整集合建议使用 GPU；translator 训练与评测需要 CUDA。

按实际工作流安装可选依赖：

```bash
pip install -e '.[translator]'   # 宿主模型训练与评测
pip install -e '.[tasks]'        # 任务参考值计算
```

## 评测发布的 translator

安装 `.[translator]` 后，下载数据集与一个模型版本，将数据集安装到代码使用的运行路径：

```bash
hf download risenyard/egms-qa-dataset --repo-type dataset \
    --local-dir release/egms-qa-dataset
python -m egms_qa.release audit --release-dir release/egms-qa-dataset
python -m egms_qa.release install \
    --release-dir release/egms-qa-dataset --target-root .
hf download risenyard/egms-qa-translator \
    --include 'qwen/*' --include 'evaluation_config.json' \
    --local-dir outputs/runs
python -m egms_qa.reproduce evaluate \
    --variant-dir outputs/runs/qwen \
    --evaluation-config outputs/runs/evaluation_config.json \
    --output-dir outputs/evaluation/qwen
```

数据集包含预计算的 token 缓存，可直接用于评测。
将下载参数和运行路径中的 `qwen` 换为 `gemma`、`llama` 或 `mistral` 即可选择其他
模型。加 `--dry-run` 可在加载宿主模型前查看评测命令。

安装器创建 `data/` 与 `outputs/` 下的链接，不复制瓦片存储。所有命令均从仓库
根目录运行。[Encoder 指南](src/egms_encoder/README.md)另提供仅安装编码器输入的流程。

## 复现训练与 QA 构建

安装数据集后，下载编码器配方并启动预训练：

```bash
hf download risenyard/egms-qa-encoder --include '*.json' \
    --local-dir data/encoder/checkpoint
python -m egms_encoder.pretrain \
    --output-dir outputs/encoder_pretrain --device cuda:0
```

使用已下载模型版本的完整配方，从固定 revision 的宿主基座模型训练 translator：

```bash
python -m egms_qa.reproduce translator \
    --variant-dir outputs/runs/qwen --output-dir outputs/training/qwen
```

配方定义各训练阶段及其 checkpoint 衔接。加 `--dry-run` 可查看全部训练命令。
评测新训练的模型时，将 `--variant-dir` 指向最后一个阶段的 `best/` 目录。

[QA 构建指南](src/egms_qa/qa_construction/README.md)说明参考值表、标签与自然语言
记录之间的关系，并提供将生成结果写入独立目录的命令。

## 评测结果

编码器评测衡量掩码区间的重建误差。Translator 评测分别衡量数值答案、类别答案，
以及对超范围问题的拒答。

| 评测 | 报告结果 |
|---|---|
| Encoder 掩码重建 | 1,000 个留出瓦片上的 RMSE 为 1.510 mm |
| Translator 数值答案 | 模型平均 R² 最高为 0.778 |
| Translator 类别答案 | 模型平均平衡准确率最高为 0.777 |

Translator 报告采用 1,000 个测试瓦片上的 71 个任务，分别对 29 个数值任务的 R²、
28 个分类任务和 14 个拒答任务的平衡准确率取宏平均。训练保留完整的 78 个任务。
各模型结果与协议设置见 [Encoder](https://huggingface.co/risenyard/egms-qa-encoder)
和 [Translator](https://huggingface.co/risenyard/egms-qa-translator) 模型卡。

完成四个 translator 版本的评测后，汇总结果：

```bash
python -m egms_qa.translator.summarize_results \
    --evaluation-root outputs/evaluation
```

## 数据要求与适用范围

数据集包含 10,000 个可直接输入模型的瓦片，位移数组形状为 `[N,294]`，固定
train/validation/test 划分为 8,000/1,000/1,000。存储窗口 `[0,294)` 对应源数据准备
时间轴的 `[8,302)`，原始索引偏移与预处理定义保存在 data config 中。

新数据集合需满足编码器的输入契约，包括单位、位移分量、时间采样与坐标几何。
使用发布的冻结 checkpoint 时保留其配套 normalization；在另一语料上训练新
encoder 时，仅用新语料的训练集拟合 normalization。官方产品下载、源格式转换
和时间窗口选择需要单独的数据准备流程。

EGMS-QA 描述已观测的形变历史。答案不用于确定成因、预测未来运动或认证结构安全。

## 许可与来源

代码采用 [MIT License](LICENSE)，EGMS-QA 创建的数据与模型产物采用 CC-BY-4.0。
源瓦片是 EGMS Level-3 Ortho Vertical 产品经过选择与重打包的衍生物，保留
Copernicus Land Monitoring Service 的来源标注和修改说明要求。详见
[DATA_LICENSE](DATA_LICENSE) 与
[数据来源说明](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md)。
