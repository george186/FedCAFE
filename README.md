# FedCAFE

异构大语言模型联邦微调框架：客户端持有不同架构的模型（OPT / GPT-2），通过 LoRA + 跨注意力 Adapter 与服务器交换特征，在不共享本地数据的前提下完成指令微调。

## 环境依赖

Python 3.9 及以上，需 NVIDIA GPU（代码中大量使用 `.to('cuda')`）。

```bash
pip install torch transformers peft datasets fire tqdm numpy pandas
```

## 项目结构

```
heter_main.py                     训练主入口（联邦训练循环）
load_dataset.py                   Dolly / Alpaca 数据集封装
fed_utils/
  client.py                       客户端：本地 LoRA 训练、Adapter 训练
  server.py                       服务器：全局训练、知识蒸馏（v1/v2 为变体）
  adapter.py                      跨注意力 Adapter 定义
  agg_adapter.py                  同类型客户端 Adapter 聚合
  client_participation_scheduling.py  客户端采样
client_data_allocation_dolly.py   按类别均匀划分 Dolly 数据
client_data_allocation_alpaca.py  按 IID 均匀划分 Alpaca 数据
public_subset.py                  采样共享数据集（target_size=2000）
sample_public_subset.py           采样共享数据集（target_size=1000）
GlobalModel_generated.py          生成/推理脚本（改编自 text-generation-webui）
cpoy_adapter.py                   Adapter 文件复制工具
data/{4,8}/                       已划分好的联邦数据
alpaca_data.json / databricks-dolly-15k.json / mmlu_test_14042.jsonl   原始数据集
```

## 数据准备

原始数据集为 [alpaca_data](https://huggingface.co/datasets/tatsu-lab/alpaca)（23 MB）、[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)（13 MB）和 MMLU 测试集，已随仓库提供，无需下载。

```bash
python client_data_allocation_dolly.py   # 在 data/5/ 下生成全局测试集、全局训练集与 5 份客户端划分
python sample_public_subset.py           # 从 public_training.json 采样共享数据
```

`client_data_allocation_dolly.py` 的 `num_clients` 改为 `4` 或 `8` 并重新运行，即可得到 `data/4/`、`data/8/` 对应的划分。共享数据单独采样得到（`public_subset.py` 采样 2000 条，`sample_public_subset.py` 采样 1000 条），需命名为 `public_training_1000.json` 后再训练，因为 `heter_main.py` 读取的正是该文件名。

**关于 `num_clients`**：划分脚本中设为 `5`，是为了额外生成一份**共享数据集**，即 `4 + 1`——4 份作为 4 个客户端的本地私有数据，另 1 份用于服务器侧共享数据。因此划分脚本用 5 个客户端，而实际实验用 4 个。共享数据供服务器全局训练与客户端 Adapter 训练使用，仓库中已按上述方式提供对应文件。

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
- `GlobalModel_generated.py` 依赖 text-generation-webui 的 `utils` 模块，单独运行时需自行提供。
- 本仓库目前未附开源许可证。
