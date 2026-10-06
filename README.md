# FedCAFE

A federated fine-tuning framework for heterogeneous large language models: clients hold models of different architectures (OPT / GPT-2), exchange features with the server through LoRA plus cross-attention adapters, and complete bidirectional knowledge distillation fine-tuning without sharing any local data.

## Requirements

Python 3.9 or later.

```bash
pip install torch transformers peft datasets fire tqdm numpy pandas
```

## Data Preparation

The raw datasets are [alpaca_data](https://huggingface.co/datasets/tatsu-lab/alpaca) (23 MB), [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (13 MB), and the MMLU test set. They are already included in this repository, so no download is needed.

```bash
python client_data_allocation_dolly.py   # generate the global test set, the global training set, and 5 client partitions, one of which serves as the public dataset public_training.json
python sample_public_subset.py           # sample the shared data from public_training.json
```

**Note on `num_clients`**: the partitioning script sets it to `5` in order to produce one extra **shared dataset**. The partitioning script therefore works with 5 clients, while the actual experiments use 4. The shared data is used for the server's global training and for client adapter training, and the corresponding files are provided in this repository as described above.

## Usage

Generate the data as described above and configure the base models (the default paths must be adapted to your own machine):

| Argument | Default | Description |
| --- | --- | --- |
| `llama_path` | Llama-2-7b-hf | Server base model |
| `opt_model_path` | opt-1.3b | OPT client model |
| `model_path` | gpt2-xl | GPT-2 client model |

```bash
python heter_main.py \
  --num_clients 4 \
  --num_opt_clients 4 \
  --num_communication_rounds 3 \
  --data_path ./data
```

`num_clients=4` together with `num_opt_clients=4` means that all 4 clients are OPT models; setting it to `2` gives 2 OPT plus 2 GPT-2 clients. Training artifacts are written to `Heter-gpt2-dolly-homo/4/` (the per-client outputs, separated by underscores, and `server_output/`).

Each communication round proceeds as follows: local client self-training → client adapter training and aggregation → server global training → server distillation back to the clients.

## Notes

- The model paths in `heter_main.py` are hard-coded defaults; change them to your own local paths before running.
- The repository already contains the partitioned `data/4` and `data/8`, so training can start directly without regenerating them.
- This repository currently ships without an open-source license.
