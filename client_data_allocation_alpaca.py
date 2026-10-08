import sys
import pandas as pd
import numpy as np
import random
import os
import json

num_clients = 5
global_test_size = 100 

np.random.seed(42)
random.seed(42)

print("正在加载 Alpaca 数据集...")
df = pd.read_json("alpaca_data.json", orient='records')

print("=" * 50)
print(f"原始数据总样本数: {len(df)}")
print("=" * 50)

sampled_df = df.sample(n=global_test_size, random_state=42)
remaining_df = df.drop(index=sampled_df.index)

sampled_df = sampled_df.reset_index(drop=True)
remaining_df = remaining_df.reset_index(drop=True)

data_path = os.path.join("data_alpaca", str(num_clients))
os.makedirs(data_path, exist_ok=True)

with open(os.path.join(data_path, "global_training.json"), 'w', encoding='utf-8') as outfile:
    json.dump(remaining_df.to_dict(orient='records'), outfile, ensure_ascii=False)

with open(os.path.join(data_path, "global_test.json"), 'w', encoding='utf-8') as outfile:
    json.dump(sampled_df.to_dict(orient='records'), outfile, ensure_ascii=False)

print("\n使用完全随机均匀分配策略 (IID)...")

indices = remaining_df.index.tolist()
np.random.shuffle(indices)

idx_partition = np.array_split(indices, num_clients)
idx_partition =[idx.tolist() for idx in idx_partition]

print("\n分配结果验证:")
print("-" * 50)

total_assigned = 0
for client_id, idx in enumerate(idx_partition):
    num_samples = len(idx)
    total_assigned += num_samples
    print(f"客户端 {client_id}: 分配了 {num_samples} 个样本")
    
    sub_df = remaining_df.loc[idx].reset_index(drop=True)
    
    with open(os.path.join(data_path, f"local_training_{client_id}.json"), 'w', encoding='utf-8') as outfile:
        json.dump(sub_df.to_dict(orient='records'), outfile, ensure_ascii=False)

print("-" * 50)
print("全局统计:")
print(f"分配的总样本数: {total_assigned} (原始可用: {len(remaining_df)})")
print(f"样本利用率: {(total_assigned/len(remaining_df))*100:.2f}%")
print("\n✅ Alpaca 数据集完全均匀分配完成!")