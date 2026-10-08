import os
import gc
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import get_peft_model
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments
from fed_utils.adapter import CrossAttentionAdapter, ServerCrossAttentionAdapter

class GlobalDistillTrainer(Trainer):
    def __init__(self, *args, temp_kl=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_kl = temp_kl

        self._custom_metrics = {
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "distill_steps": 0, 
            "total_steps": 0    
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        agg_feat = inputs.pop("agg_feat")
        labels = inputs.pop("labels")
        use_distill = inputs.pop("use_distill")
        attention_mask = inputs.get("attention_mask")

        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss

        if use_distill.item() == 1:
            lm_head = model.base_model.lm_head if hasattr(model, "base_model") else model.lm_head

            with torch.no_grad():
                agg_feat = agg_feat.to(model.device)
                if agg_feat.dtype == torch.float32:
                    agg_feat = agg_feat.to(torch.float16)
                agg_logits = lm_head(agg_feat)

            valid_mask = (attention_mask == 1)
            v_s_logits = outputs.logits[valid_mask]
            v_t_logits = agg_logits[valid_mask]

            temp_kl = self.temp_kl
            kl_loss = F.kl_div(
                F.log_softmax(v_s_logits / temp_kl, dim=-1),
                F.softmax(v_t_logits / temp_kl, dim=-1),
                reduction='batchmean'
            )* (temp_kl ** 2)

            loss = 0.8 * outputs.loss + 0.2 * kl_loss

            self._custom_metrics["ce_loss"] = self._custom_metrics["ce_loss"] + outputs.loss.item() * 0.8
            self._custom_metrics["kl_loss"] = self._custom_metrics["kl_loss"] + kl_loss.item() * 0.2
            self._custom_metrics["distill_steps"] += 1
        else:
            loss = ce_loss

            self._custom_metrics["ce_loss"] += outputs.loss.item()

        self._custom_metrics["total_steps"] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):
        total = self._custom_metrics["total_steps"]

        if total > 0:
            avg_ce = self._custom_metrics["ce_loss"] / total
            logs["ce_loss"] = round(avg_ce, 4)

            if self._custom_metrics["distill_steps"] > 0:
                avg_kl = self._custom_metrics["kl_loss"] / self._custom_metrics["distill_steps"]
                logs["kl_loss"] = round(avg_kl, 4)

            self._custom_metrics = {
                "ce_loss": 0.0,
                "kl_loss": 0.0,
                "distill_steps": 0,
                "total_steps": 0
            }
        super().log(logs, *args, **kwargs)

class ClientDistillTrainer(Trainer):
    def __init__(self, *args, server_node=None, client_node=None, temp_kl=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_node = server_node
        self.client_node = client_node
        self.temp_kl = temp_kl
        self.distill_count = 0      
        self.self_train_count = 0   

        self._custom_metrics = {
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "distill_steps": 0, 
            "total_steps": 0    
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        server_device = self.server_node.model.device

        target_feat = inputs.pop("target_feat").to(server_device)
        target_loss = inputs.pop("target_loss").item() # 取出标量 Loss
        labels = inputs.pop("labels")

        outputs = model(**inputs, labels=labels, output_hidden_states=True)
        client_ce_loss = outputs.loss

        if target_loss < client_ce_loss.item():
            self.distill_count += 1

            client_lm_head = model.base_model.lm_head if hasattr(model, "base_model") else model.lm_head
            with torch.no_grad(): 
                t_logits = client_lm_head(target_feat.to(client_lm_head.weight.dtype)).to(torch.float32)

            s_logits = outputs.logits

            valid_mask = (inputs.get("attention_mask") == 1)
            v_s_logits = s_logits[valid_mask]
            v_t_logits = t_logits[valid_mask]

            temp_kl = self.temp_kl
            loss_kl = F.kl_div(
                F.log_softmax(v_s_logits / temp_kl, dim=-1),
                F.softmax(v_t_logits / temp_kl, dim=-1),
                reduction='batchmean'
            )* (temp_kl ** 2)

            loss = 0.8 * client_ce_loss + 0.2 * loss_kl

            self._custom_metrics["ce_loss"] += client_ce_loss.item() * 0.7
            self._custom_metrics["kl_loss"] += loss_kl.item() * 0.3
            self._custom_metrics["distill_steps"] += 1
        else:
            self.self_train_count += 1
            loss = client_ce_loss

            self._custom_metrics["ce_loss"] += client_ce_loss.item()

        self._custom_metrics["total_steps"] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):
        total = self._custom_metrics["total_steps"]

        if total > 0:
            avg_ce = self._custom_metrics["ce_loss"] / total
            logs["ce_loss"] = round(avg_ce, 4)

            if self._custom_metrics["distill_steps"] > 0:
                avg_kl = self._custom_metrics["kl_loss"] / self._custom_metrics["distill_steps"]
                logs["kl_loss"] = round(avg_kl, 4)

            self._custom_metrics = {
                "ce_loss": 0.0,
                "kl_loss": 0.0,
                "distill_steps": 0,
                "total_steps": 0
            }

        super().log(logs, *args, **kwargs)


def contrastive_loss(student_feat, teacher_feat, attention_mask, temperature=0.07):
    """InfoNCE Loss"""
    batch_size, seq_len, dim = student_feat.shape

    s_flat = student_feat.view(-1, dim)
    t_flat = teacher_feat.view(-1, dim)

    mask_flat = attention_mask.view(-1)

    active_indices = torch.nonzero(mask_flat).squeeze()

    if active_indices.numel() == 0:
        return torch.tensor(0.0, device=student_feat.device, requires_grad=True)

    s_active = s_flat[active_indices]
    t_active = t_flat[active_indices]

    s_emb = F.normalize(s_active, dim=-1)
    t_emb = F.normalize(t_active, dim=-1)
    logits = torch.matmul(s_emb, t_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)

def smooth_l1_feature_loss(student_feat, teacher_feat, attention_mask, beta=1.0):
    """
    Smooth L1 Feature Alignment Loss
    """

    student_feat = student_feat.to(torch.float32)
    teacher_feat = teacher_feat.to(torch.float32)

    dim = student_feat.shape[-1]
    s_flat = student_feat.view(-1, dim)
    t_flat = teacher_feat.view(-1, dim)
    mask_flat = attention_mask.view(-1)

    active_indices = torch.nonzero(mask_flat).squeeze()

    if active_indices.numel() == 0:
        return torch.tensor(0.0, device=student_feat.device, requires_grad=True)

    s_active = s_flat[active_indices]
    t_active = t_flat[active_indices]

    loss = F.smooth_l1_loss(s_active, t_active, beta=beta, reduction='mean')

    return loss

class GeneralServer:
    def __init__(self, model_path, output_dir, peft_config, opt_model_path, gpt2_model_path):
        self.base_model =  AutoModelForCausalLM.from_pretrained(
                                                                    model_path,
                                                                    torch_dtype=torch.float16,
                                                                    device_map="cpu",
                                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"
        # self.tokenizer.padding_side = "right"
        self.peft_config = peft_config
        self.model = get_peft_model(self.base_model, peft_config)

        self.server_dim = self.base_model.config.hidden_size

        temp_opt = AutoModelForCausalLM.from_pretrained(opt_model_path)
        opt_dim = temp_opt.config.hidden_size
        self.client_lm_heads = {
            "opt": temp_opt.lm_head.to("cuda"),
        }
        for param in self.client_lm_heads["opt"].parameters():
            param.requires_grad = False
        del temp_opt

        temp_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_path)
        gpt2_dim = temp_gpt2.config.hidden_size
        self.client_lm_heads["gpt2"] = temp_gpt2.lm_head.to("cuda")
        for param in self.client_lm_heads["gpt2"].parameters():
            param.requires_grad = False
        del temp_gpt2

        self.down_adapters = {
            "opt": ServerCrossAttentionAdapter(self.server_dim, opt_dim, num_heads=4).to("cuda"),
            "gpt2": ServerCrossAttentionAdapter(self.server_dim, gpt2_dim, num_heads=4).to("cuda"),
        }

        # self.down_adapters = {
        #     "opt": CrossAttentionAdapter(self.server_dim, opt_dim, bottleneck_ratio=4).to("cuda"),
        #     "gpt2": CrossAttentionAdapter(self.server_dim, gpt2_dim, bottleneck_ratio=4).to("cuda"),
        # }

        for mtype, adapter in self.down_adapters.items():
            total_params = sum(p.numel() for p in adapter.parameters())
            trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            print(f"[{mtype}] Down-Adapter 总参数量: {total_params:,} ({total_params/1e6:.2f}M), 可训练: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        self.output_dir = output_dir
        self.features = {}
        self._register_hook()
        self.local_output_dir = os.path.join(self.output_dir, "server_output")
        self.base_model.gradient_checkpointing_enable() 
        self.base_model.enable_input_require_grads()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features['out'] = output
        try:
            layer = self.model.base_model.model.model.norm
        except:
            layer = [m for n, m in self.model.named_modules() if ".norm" in n][-1]
        layer.register_forward_hook(hook_fn)

    def tokenize(self, prompt, add_eos_token=True):
        max_len = 511
        result = self.tokenizer(prompt,
                                   truncation=True,
                                   max_length=max_len,
                                   padding=False,)

        for i in range(len(result["input_ids"])):
            result["input_ids"][i].append(self.tokenizer.eos_token_id)
            result["attention_mask"][i].append(1)

        result = self.tokenizer.pad(result, padding=True, return_tensors='pt').to('cuda')


        labels = result['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        result['labels'] = labels

        return result

    def get_features(self, input_ids, attention_mask):
        self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return self.features['out']

    # --- Step 3: Global Aggregation (Public Data) ---
    def global_train(self, dataset, batch_size, epochs, learn_rate, temp_kl, clients):
        print(f"[Server] Step 3: Global Training on Public Data...")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_batches = len(dataloader)

        server_losses = []
        cached_server_queries = []

        client_types = list(set(c.model_type for c in clients))

        self.best_client_losses = {t: [float('inf')] * num_batches for t in client_types}
        self.best_client_raw_feats = {t: [None] * num_batches for t in client_types}  # raw features (client space)

        global_best_aligned_feats = [None] * num_batches
        global_best_losses = [float('inf')] * num_batches
        global_best_types = [None] * num_batches

        print("  -> Phase 1: Pre-computing Server Queries...")
        self.model.eval()

        for texts in tqdm(dataloader):
            s_enc = self.tokenize(texts)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=s_enc['input_ids'],
                    attention_mask=s_enc['attention_mask'],
                    labels=s_enc['labels']
                )
                server_losses.append(outputs.loss.item())

                query = self.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                cached_server_queries.append(query.cpu())

        print("  -> Phase 2: Compute Best Client per Type...")
        lm_head = self.model.base_model.lm_head if hasattr(self.model, "base_model") else self.model.lm_head
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        for client in clients:
            print(f"     Processing {client.client_id} ({client.model_type})...")
            client.load_state()
            client.to_gpu()
            client.model.eval()
            client.adapter.eval()

            for i, texts in enumerate(tqdm(dataloader)):
                query_gpu = cached_server_queries[i].to("cuda")

                c_enc = client.tokenize(texts)
                s_enc = self.tokenize(texts)

                with torch.no_grad():
                    client.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'])
                    raw_feat = client.features['out'].detach()

                    aligned_feat = client.adapter(raw_feat, query_gpu, c_enc['attention_mask'])
                    translated_logits = lm_head(aligned_feat.to(lm_head.weight.dtype)).to(torch.float32)

                    shift_logits = translated_logits[..., :-1, :].contiguous()
                    shift_labels = s_enc['labels'][..., 1:].contiguous().to(translated_logits.device)

                    translated_ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

                    mtype = client.model_type
                    if translated_ce_loss < self.best_client_losses[mtype][i]:
                        self.best_client_losses[mtype][i] = translated_ce_loss

                        self.best_client_raw_feats[mtype][i] = raw_feat.cpu()

                    if translated_ce_loss < global_best_losses[i]:
                        global_best_losses[i] = translated_ce_loss
                        global_best_types[i] = mtype
                        if aligned_feat.dtype == torch.float32:
                            aligned_feat = aligned_feat.to(torch.float16)
                        global_best_aligned_feats[i] = aligned_feat.cpu()

            client.unload()

        del cached_server_queries
        gc.collect()

        print("  -> Phase 3: Filtering Batches for Distillation...")
        precomputed_batches = []
        distill_count = 0
        self_train_count = 0

        for i, texts in enumerate(tqdm(dataloader)):
            s_enc = self.tokenize(texts)
            if global_best_losses[i] < server_losses[i]:
                precomputed_batches.append({
                    "input_ids": s_enc["input_ids"],
                    "attention_mask": s_enc["attention_mask"],
                    "labels": s_enc["labels"],
                    "agg_feat": global_best_aligned_feats[i],
                    "use_distill": torch.tensor(1)
                })
                distill_count += 1
            else:
                precomputed_batches.append({
                    "input_ids": s_enc["input_ids"],
                    "attention_mask": s_enc["attention_mask"],
                    "labels": s_enc["labels"],
                    "agg_feat": torch.zeros(1),
                    "use_distill": torch.tensor(0)
                })
                self_train_count += 1

        print(f"    [Status] Total Batches: {num_batches}, Distill: {distill_count}, Self-Train: {self_train_count}")

        if len(precomputed_batches) == 0:
            print("     [Warning] No clients outperformed the server on this public dataset. Skipping training.")
            return

        print("  -> Phase 4: Server Training (Fast)...")
        self.model.train()

        class PrecomputedBatchDataset(Dataset):
            def __len__(self): return len(precomputed_batches)
            def __getitem__(self, idx): return precomputed_batches[idx]

        train_dataset = PrecomputedBatchDataset()
        def batch_identity_collator(features):
            batch = features[0]
            return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            dataloader_pin_memory=False,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=learn_rate,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            output_dir=self.local_output_dir,
            save_total_limit=0,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            dataloader_drop_last=False,
            gradient_checkpointing=False,
        )

        trainer = GlobalDistillTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=batch_identity_collator,
            temp_kl=temp_kl
        )

        trainer.train()

        print("  -> Phase 5: Re-evaluating Server & Constructing Global Best Pool...")
        self.model.eval()

        self.new_server_losses = {t: [] for t in client_types}
        self.new_server_feats =[]

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for texts in tqdm(dataloader):
            s_enc = self.tokenize(texts)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=s_enc['input_ids'],
                    attention_mask=s_enc['attention_mask'],
                    labels=s_enc['labels']
                )

                feat = self.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                self.new_server_feats.append(feat.cpu())

        del global_best_aligned_feats, precomputed_batches, trainer, server_losses
        gc.collect()
        torch.cuda.empty_cache()

        for mtype in client_types:
            type_clients = [c for c in clients if c.model_type == mtype]
            if len(type_clients) == 0:
                continue
            self.train_down_adapter(dataset, batch_size, 1, 1e-4, temp_kl, type_clients[0], temp_nce=0.07, model_type=mtype)

        self.translated_server_feats = {t: [None] * num_batches for t in client_types}
        for mtype in client_types:
            type_clients = [c for c in clients if c.model_type == mtype]
            if len(type_clients) == 0:
                continue

            self.down_adapters[mtype].eval()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            client_lm_head = self.client_lm_heads[mtype]

            for i, texts in enumerate(tqdm(dataloader, desc=f"Evaluating & Routing ({mtype})")):
                c_enc = type_clients[0].tokenize(texts)
                s_enc = self.tokenize(texts)
                labels = c_enc['labels'].to("cuda")

                if self.best_client_raw_feats[mtype][i] is None:
                    self.new_server_losses[mtype].append(float('inf'))
                    continue

                with torch.no_grad():
                    new_s_feat = self.new_server_feats[i].to("cuda").to(torch.float32)
                    q_feat = self.best_client_raw_feats[mtype][i].to("cuda").to(torch.float32)

                translated_s_feat = self.down_adapters[mtype](
                        new_s_feat,
                        q_feat,
                        s_enc['attention_mask'].to("cuda")
                    )

                self.translated_server_feats[mtype][i] = translated_s_feat.detach().cpu()

                s_logits = client_lm_head(translated_s_feat.to(client_lm_head.weight.dtype)).to(torch.float32)
                shift_s_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                translated_loss = loss_fct(shift_s_logits.view(-1, shift_s_logits.size(-1)), shift_labels.view(-1)).item()
                self.new_server_losses[mtype].append(translated_loss)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()



    # --- Step 4: Distill to Clients (Public Data) ---
    def distill_to_clients(self, dataset, batch_size, epochs ,learn_rate, temp_kl, clients):
        self.model.eval()
        for mtype in self.down_adapters:
            self.down_adapters[mtype].eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for client in clients:
            print(f"[Server] Step 4: Distilling back to Client{client.client_id} ({client.model_type})")
            client.model.train() # Update LoRA
            client.adapter.eval()

            mtype = client.model_type

            precomputed_batches =[]
            for i, texts in enumerate(tqdm(dataloader)):
                c_enc = client.tokenize(texts)
                s_enc = self.tokenize(texts)

                has_same_type_best = (self.best_client_raw_feats[mtype][i] is not None)

                if not has_same_type_best:
                    server_choose = True
                elif self.new_server_losses[mtype][i] < self.best_client_losses[mtype][i]:
                    server_choose = True
                else:
                    server_choose = False

                if server_choose:
                    target_feat = self.translated_server_feats[mtype][i].clone()
                    target_loss = self.new_server_losses[mtype][i]

                else:
                    target_feat = self.best_client_raw_feats[mtype][i].clone()
                    target_loss = self.best_client_losses[mtype][i]

                precomputed_batches.append({
                    "input_ids": c_enc["input_ids"],
                    "attention_mask": c_enc["attention_mask"],
                    "labels": c_enc["labels"],
                    "target_feat": target_feat,
                    "target_loss": torch.tensor(target_loss),
                })

            class PrecomputedBatchDataset(Dataset):
                def __len__(self): return len(precomputed_batches)
                def __getitem__(self, idx): return precomputed_batches[idx]

            def batch_identity_collator(features):
                batch = features[0]
                return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            training_args = TrainingArguments(
                per_device_train_batch_size=1,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                num_train_epochs=epochs,
                learning_rate=learn_rate,
                lr_scheduler_type="cosine",
                output_dir=client.local_output_dir,
                fp16=True,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                dataloader_pin_memory=False,
            )

            trainer = ClientDistillTrainer(
                model=client.model,
                args=training_args,
                train_dataset=PrecomputedBatchDataset(),
                data_collator=batch_identity_collator,
                server_node=self,
                client_node=client,
                temp_kl=temp_kl
            )
            trainer.train()

            total_steps = trainer.distill_count + trainer.self_train_count
            if total_steps > 0:
                distill_ratio = (trainer.distill_count / total_steps) * 100
                self_ratio = (trainer.self_train_count / total_steps) * 100
                print(f"\n[{client.client_id} ({mtype})] Distillation Statistics:")
                print(f"  -> Total Forward Batches: {total_steps}")
                print(f"  -> Server Taught Client : {trainer.distill_count} times ({distill_ratio:.2f}%)")
                print(f"  -> Client Self-Trained  : {trainer.self_train_count} times ({self_ratio:.2f}%)\n")

            del trainer, precomputed_batches
            client.unload()
            gc.collect()
            torch.cuda.empty_cache()

    def train_down_adapter(self, dataset, batch_size, epochs, learn_rate, temp_kl, client, temp_nce, model_type):
        print(f" Step 3.5: Training Down-Adapter for {model_type}...")
        self.model.eval()
        down_adapter = self.down_adapters[model_type]
        client_lm_head = self.client_lm_heads[model_type]
        down_adapter.train()

        optimizer = torch.optim.AdamW(down_adapter.parameters(), lr=learn_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


        for ep in range(epochs):
            total_loss = 0
            batch_count = 0
            epoch_loss = 0
            skip_count = 0
            batch_pbar = tqdm(dataloader, desc=f"[{model_type}] Epoch {ep+1}/{epochs}", leave=False)

            for i, texts in enumerate(batch_pbar):
                if self.best_client_raw_feats[model_type][i] is None:
                    continue

                s_enc = self.tokenize(texts)
                c_enc = client.tokenize(texts)

                server_feat = self.new_server_feats[i].to("cuda").to(torch.float32)
                target_raw_client_feat = self.best_client_raw_feats[model_type][i].to("cuda").to(torch.float32)

                translated_client_feat = down_adapter(
                    server_feat,
                    target_raw_client_feat,
                    s_enc['attention_mask'].to("cuda")
                )

                s_logits = client_lm_head(translated_client_feat.to(client_lm_head.weight.dtype)).to(torch.float32)

                with torch.no_grad():
                    t_logits = client_lm_head(target_raw_client_feat.to(client_lm_head.weight.dtype)).to(torch.float32)

                valid_mask = (c_enc['attention_mask'] == 1)
                v_s_logits = s_logits[valid_mask]
                v_t_logits = t_logits[valid_mask]

                loss_kl = F.kl_div(F.log_softmax(v_s_logits / temp_kl, dim=-1), F.softmax(v_t_logits / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl

                loss_nce = smooth_l1_feature_loss(translated_client_feat, target_raw_client_feat, c_enc['attention_mask'], beta=1.0)

                shift_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = c_enc['labels'][..., 1:].contiguous()
                loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

                loss = 0.5 * loss_ce + 0.2 * loss_kl + 0.3 * loss_nce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                epoch_loss += loss.item()

                if (i + 1) % 30 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"\n [{model_type}] Epoch {ep+1}, Batch {i+1}: Average loss = {avg_loss:.4f}")

                    total_loss = 0
                    batch_count = 0

                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce_loss': f'{(loss_ce.item())*0.5:.4f}',
                    'kl_loss': f'{(loss_kl.item())*0.2:.4f}',
                    'nce_loss': f'{(loss_nce.item())*0.3:.4f}',
                })

            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\n[{model_type}] Epoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")
            print(f"\nSkipped batches: {skip_count}")

        del dataloader
        del optimizer
        gc.collect()


    def to_gpu(self):
        """将模型移动到 GPU"""
        print("Server Moving to GPU...")
        self.model.to("cuda")
        for adapter in self.down_adapters.values():
            adapter.to("cuda")
        torch.cuda.empty_cache()

    def unload(self):
        """将模型移回 CPU 并清理显存"""
        print("Server Unloading to CPU...")
        self.model.to("cpu")

        self.features = {}
        torch.cuda.empty_cache()
        gc.collect()

    def save_state(self):
        """保存 LoRA 和 Adapter 参数到磁盘"""
        if not os.path.exists(self.local_output_dir):
            os.makedirs(self.local_output_dir)

        for mtype, adapter in self.down_adapters.items():
            save_path = os.path.join(self.local_output_dir, f"adapter_{mtype}.bin")
            torch.save(adapter.state_dict(), save_path)
            print(f"Server [{mtype}] Adapter saved to {save_path}")

    def load_state(self):
        """从磁盘加载参数"""

        for mtype, adapter in self.down_adapters.items():
            adapter_path = os.path.join(self.local_output_dir, f"adapter_{mtype}.bin")
            if os.path.exists(adapter_path):
                print(f"Server [{mtype}] adapter loaded.")
                adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))

        print("Server State loaded.")

    def merge_model(self):
        print("Server Merging LoRA and resetting adapter...")

        self.base_model = self.model.merge_and_unload().to("cpu")
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.to("cuda")

        self._register_hook()

        self.features = {}

        print("Server Ready for next train.")

    def merge_save(self):
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(
                                    '/root/autodl-tmp/final',
                                    load_in_8bit=False,
                                    torch_dtype=torch.float16)
        print('finish save')
        
        del merged_model
