import os
import torch
from collections import defaultdict

def aggregate_adapters(clients):

    print("\n[Federated Averaging] Server is aggregating Up-Adapters by client type...")

    type_groups = defaultdict(list)
    for client in clients:
        type_groups[client.model_type].append(client)

    for mtype, group in type_groups.items():
        print(f"  -> Aggregating {len(group)} clients of type: {mtype}")

        base_adapter_path = os.path.join(group[0].local_output_dir, "adapter.bin")
        global_state_dict = torch.load(base_adapter_path, map_location="cpu")

        for i in range(1, len(group)):
            client_adapter_path = os.path.join(group[i].local_output_dir, "adapter.bin")
            client_state_dict = torch.load(client_adapter_path, map_location="cpu")

            for key in global_state_dict.keys():
                global_state_dict[key] += client_state_dict[key]

        num_clients_in_group = len(group)
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.div(global_state_dict[key], num_clients_in_group)

        for client in group:
            target_path = os.path.join(client.local_output_dir, "adapter.bin")
            torch.save(global_state_dict, target_path)
            print(f"     [Client {client.client_id} ({mtype})] Adapter updated.")

    print("[Federated Averaging] All Up-Adapters successfully aggregated and distributed by type.\n")
