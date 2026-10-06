# FedCAFE

异构大语言模型联邦微调框架：客户端持有不同架构的模型（OPT / GPT-2），通过 LoRA + 跨注意力 Adapter 与服务器交换特征，在不共享本地数据的前提下完成双向知识蒸馏微调。

## 环境依赖

Python 3.9 及以上。

```bash
pip install torch transformers peft datasets fire tqdm numpy pandas
```

## 数据准备

原始数据集为 [alpaca_data](https://huggingface.co/datasets/tatsu-lab/alpaca)（23 MB）、[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)（13 MB）和 MMLU 测试集，已随仓库提供，无需下载。

```bash
python client_data_allocation_dolly.py   # 生成全局测试集、全局训练集与 5 份客户端划分,其中一份作为公共数据集public_training.json
python sample_public_subset.py           # 从 public_training.json 采样共享数据
```

**关于 `num_clients`**：划分脚本中设为 `5`，是为了额外生成一份**共享数据集**。因此划分脚本用 5 个客户端，而实际实验用 4 个。共享数据供服务器全局训练与客户端 Adapter 训练使用，仓库中已按上述方式提供对应文件。

## 运行

先按上一步生成数据，并配置好基础模型（默认路径需按本机情况修改）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `llama_path` | Llama-2-7b-hf | 服务器基座模型 |
| `opt_model_path` | opt-1.3b | OPT 客户端模型 |
| `model_path` | gpt2-xl | GPT-2 客户端模型 |

```bash
python heter_main.py \
  --num_clients 4 \
  --num_opt_clients 4 \
  --num_communication_rounds 3 \
  --data_path ./data
```

`num_clients=4` 且 `num_opt_clients=4` 表示 4 个客户端均为 OPT 模型；若设为 2，则 2 个 OPT + 2 个 GPT-2。训练产物保存在 `Heter-gpt2-dolly-homo/4/`（下划线分隔的客户端输出与 `server_output/`）下。

每轮通信依次为：客户端本地自训练 → 客户端 Adapter 训练并聚合 → 服务器全局训练 → 服务器蒸馏回客户端。

## 说明

- `heter_main.py` 中的模型路径默认写在脚本里，运行前请改为自己的本地路径。
- 仓库已包含划分好的 `data/4` 与 `data/8`，可直接开始训练，无需重新生成。
- 本仓库目前未附开源许可证。
