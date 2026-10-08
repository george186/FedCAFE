import os
import gc
from torch.utils.data import DataLoader, Dataset
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import get_peft_model
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments
import itertools
from fed_utils.adapter import CrossAttentionAdapter, ClientCrossAttentionAdapter

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

class GeneralClient:
    def __init__(self, client_id, model_path, output_dir, peft_config, server_dim, model_type="gpt2" ):
        self.client_id = client_id
        self.model_type = model_type  # "opt" or "gpt2"
        self.base_model =  AutoModelForCausalLM.from_pretrained(
                                                                    model_path,
                                                                    # load_in_8bit=False,
                                                                    torch_dtype=torch.float32,
                                                                    device_map="cpu",
                                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.peft_config = peft_config
        if self.model_type == "opt":
            self.tokenizer.pad_token_id = (0)
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"
        self.model = get_peft_model(self.base_model, peft_config)

        self.client_dim = self.base_model.config.hidden_size
        self.adapter = ClientCrossAttentionAdapter(self.client_dim, server_dim, num_heads=4).to("cpu")
        # self.adapter = CrossAttentionAdapter(self.client_dim, server_dim, bottleneck_ratio=4).to("cpu")

        total_params = sum(p.numel() for p in self.adapter.parameters())
        print(f"总参数量: {total_params:,}")
        print(f"总参数量（百万）: {total_params/1e6:.2f}M")

        trainable_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable_params:,}")
        print(f"可训练参数（百万）: {trainable_params/1e6:.2f}M")

        self.output_dir = output_dir
        self.features = {}
        self._register_hook()
        self.local_output_dir = os.path.join(self.output_dir, "local_output_{}".format(self.client_id))
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads() 

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features['out'] = output
        if self.model_type == "opt":
            # OPT: final_layer_norm in decoder
            try:
                layer = self.model.base_model.model.model.decoder.final_layer_norm
            except:
                layer = [m for n, m in self.model.named_modules() if "final_layer_norm" in n][-1]
        else:
            # GPT2: transformer.ln_f
            try:
                layer = self.model.base_model.transformer.ln_f
            except:
                layer = [m for n, m in self.model.named_modules() if "final_layer_norm" in n][-1]
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
    
    # --- Step 1: Local Training (Private Data) ---
    def local_train(self, dataset, epochs, batch_size, learn_rate):
        print(f"[{self.client_id}] Step 1: Local Training on Private Data...")
        self.model.train()
        self.adapter.eval()

        train_args = TrainingArguments(
            per_device_train_batch_size=batch_size, 
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
            report_to="none", 
        )


        def custom_data_collator(features):
            if isinstance(features[0], dict) and "text" in features[0]:
                texts = [f["text"] for f in features]
            elif isinstance(features[0], str):
                texts = features
            else:
                texts =[str(f) for f in features]

            batch_enc = self.tokenize(texts)
            
            return {
                "input_ids": batch_enc["input_ids"],
                "attention_mask": batch_enc["attention_mask"],
                "labels": batch_enc["labels"]
            }
        
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=dataset,
            data_collator=custom_data_collator,
        )

        trainer.train()

        del trainer
        torch.cuda.empty_cache()
        gc.collect()


    def train_adapter(self, dataset, batch_size, epochs, learn_rate, temp_nce, temp_kl, server_node):
        print(f"[{self.client_id}] Step 2: Training Adapter on Public Data...")
        self.model.eval()   
        self.adapter.train()
        server_node.model.eval()
        
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=learn_rate)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        server_head = server_node.model.base_model.lm_head
        total_steps = len(dataloader) * epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for ep in range(epochs):
            total_loss = 0
            batch_count = 0
            epoch_loss = 0
            skip_count = 0
            batch_pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs}", leave=False)

            for i, texts in enumerate(batch_pbar):
                # Server (Teacher)
                with torch.no_grad():
                    s_enc = server_node.tokenize(texts)
                    server_mask = s_enc['attention_mask']

                    s_outputs = server_node.model(
                        input_ids=s_enc['input_ids'], 
                        attention_mask=s_enc['attention_mask'], 
                        labels=s_enc['labels']
                    )
                    teacher_logits = s_outputs.logits
                    teacher_feat = server_node.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                
                # Client (Student)
                c_enc = self.tokenize(texts)
                c_mask = c_enc['attention_mask']

                with torch.no_grad():
                    st_output = self.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'], labels=c_enc['labels'])                  
                    raw_client_feat = self.features['out'].detach()
            
                # Adapter Forward
                aligned_feat = self.adapter(raw_client_feat, teacher_feat, c_enc['attention_mask'])

                aligned_feat = aligned_feat.to(torch.float16)

                # Loss Calculation
                student_logits = server_head(aligned_feat)

                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = s_enc['labels'][..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                valid_mask = (server_mask == 1)
                v_s_logits = student_logits[valid_mask]
                v_t_logits = teacher_logits[valid_mask]

                loss_kl = F.kl_div(F.log_softmax(v_s_logits / temp_kl, dim=-1), F.softmax(v_t_logits / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl

                loss_nce = smooth_l1_feature_loss(aligned_feat, teacher_feat, server_mask, beta=1.0)
                
                loss = 0.5 * loss_ce + 0.2 * loss_kl + 0.3 * loss_nce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                batch_count += 1
                epoch_loss += loss.item()
                
                if (i + 1) % 30 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"\n Epoch {ep+1}, Batch {i+1}: Average loss = {avg_loss:.4f}")
                    
                    total_loss = 0
                    batch_count = 0

                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce_loss': f'{(loss_ce.item())*0.5:.4f}',
                    'kl_loss': f'{(loss_kl.item())*0.2:.4f}',
                    'nce_loss': f'{(loss_nce.item())*0.3:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")
            print(f"\nSkipped batches: {skip_count}")

        del dataloader
        del optimizer
        gc.collect()

    def to_gpu(self):
        """将模型移动到 GPU"""
        print(f"[{self.client_id}] Moving to GPU...")
        self.model.to("cuda")
        self.adapter.to("cuda")
        torch.cuda.empty_cache()

    def unload(self):
        """将模型移回 CPU 并清理显存"""
        print(f"[{self.client_id}] Unloading to CPU...")
        for param in self.model.parameters():
            param.grad = None
        
        for param in self.adapter.parameters():
            param.grad = None
        self.model.to("cpu")
        self.adapter.to("cpu")
        
        self.features = {} 
        torch.cuda.empty_cache()
        gc.collect()

    def save_state(self):
        """保存 LoRA 和 Adapter 参数到磁盘"""
        if not os.path.exists(self.local_output_dir):
            os.makedirs(self.local_output_dir)
            
        torch.save(self.adapter.state_dict(), os.path.join(self.local_output_dir, "adapter.bin"))
        print(f"[{self.client_id}] State saved to {self.local_output_dir}")


    def load_state(self):
        """从磁盘加载参数"""
        adapter_path = os.path.join(self.local_output_dir, "adapter.bin")
         
        if os.path.exists(adapter_path):
            print(f"[{self.client_id}] adapter loaded.")
            self.adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        
            
        print(f"[{self.client_id}] State loaded.")

    def merge_model(self):
        """合并 LoRA 到基础模型"""
        print(f"[Client {self.client_id}] Merging LoRA and resetting adapter...")


        self.base_model = self.model.merge_and_unload().to("cpu")

        self.model = get_peft_model(self.base_model, self.peft_config)

        self._register_hook()

        self.features = {}
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[Client {self.client_id}] Ready for next train.")
    
    def merge_save(self):
        merged_model = self.model.merge_and_unload()

        merged_model.save_pretrained(
                                    '/root/autodl-tmp/final_client{}'.format(self.client_id),
                                    load_in_8bit=False)
        print('finish save')
        
        del merged_model
