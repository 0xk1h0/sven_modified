import os
import abc
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from sven.model import save_model, parallelize_model, load_model
from sven.dataset import PrefixDataset, TextPromptDataset
from sven.utils import set_seed
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
import gc

def token_weighted_loss(loss_type, inputs, targets, weights):
    """Calculate weighted loss for tokens with consistent device placement"""
    # Ensure all inputs are on the same device as the model inputs
    device = inputs.device
    targets = targets.to(device)
    weights = weights.to(device)

    if loss_type == 'cross_entropy':
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    elif loss_type == 'nll':
        loss_fct = torch.nn.NLLLoss(reduction='none')
    elif loss_type == 'kl':
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # Move loss function to correct device
    if hasattr(loss_fct, 'to'):
        loss_fct = loss_fct.to(device)

    loss = loss_fct(inputs, targets)
    if loss_type == 'kl':
        loss = loss.sum(dim=1)
    loss = loss[weights != 0]
    return loss.mean()

class TrainerBase:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.model = None
        self.dataset = None
        self.input_device = None
        
        # Initialize gpu_base_idx for the trainer
        self.gpu_base_idx = 1  # Always start from GPU 1 since we're avoiding GPU 0
        
        # Let CUDA handle device assignment
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:128,"
            "expandable_segments:True"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def load_dataset(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def step(self, batch):
        raise NotImplementedError()

    def save(self, path, step, epoch, optimizer, scheduler):
        if not os.path.exists(path):
            os.makedirs(path)
        save_model(self.model, path, self.args)
        self.tokenizer.save_pretrained(path)
        step_file = os.path.join(path, 'step_file.txt')
        with open(step_file, 'w') as f:
            f.write(str(step)+'\n')
        epoch_file = os.path.join(path, 'epoch_file.txt')
        with open(epoch_file, 'w') as f:
            f.write(str(epoch)+'\n')
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, 'scheduler.pt'))

    def add_to_loss_dict(self, acc_loss_dict, loss_dict):
        for key, val in loss_dict.items():
            if key not in acc_loss_dict:
                acc_loss_dict[key] = 0.0
            acc_loss_dict[key] += val

    def report_loss_dict(self, loss_dict, steps):
        ss = []
        for key, val in loss_dict.items():
            if key == 'kl_loss':
                r = 8
            else:
                r = 4
            ss.append(f'{key}: {round(val/steps, r)}')
        return ', '.join(ss)

    def run(self):
        try:
            self.load_model()
            self.load_dataset()

            self.args.logger.info(f'Training args {self.args}')

            batch_size = 1
            train_sampler = RandomSampler(self.dataset)
            train_dataloader = DataLoader(
                self.dataset, 
                sampler=train_sampler, 
                batch_size=batch_size,
                pin_memory=True,  # Enable pinned memory
                drop_last=True
            )

            total_samples = len(self.dataset)
            batch_size = batch_size * self.args.grad_acc_steps
            total_steps = total_samples // batch_size * self.args.num_train_epochs

            # Memory optimization for optimizer
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [], 'weight_decay': self.args.weight_decay},
                {'params': [], 'weight_decay': 0.0}
            ]

            # Group parameters by device to optimize memory
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Determine parameter group
                if not any(nd in name for nd in no_decay):
                    group_idx = 0
                else:
                    group_idx = 1
                    
                optimizer_grouped_parameters[group_idx]['params'].append(param)
            
            # Initialize optimizer with memory optimizations
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
                # foreach=True,  # Enable more efficient iteration
                # capturable=True  # Enable kernel fusion
            )
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                num_training_steps=total_steps)

            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.args.logger.info('***** Running training *****')
            self.args.logger.info('  Num samples = %d', total_samples)
            self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
            self.args.logger.info('  Batch size= 1')
            self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
            self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
            self.args.logger.info('  Total optimization steps = %d', total_steps)
            self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
            self.args.logger.info('  Num parameters = %d', num_params)
            self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)
            # self.args.logger.info('  Fraction of trainable parameters = %s', str(round(num_trainable_params/num_params*100, 4)))

            # Prepare for distributed training
            try:
                # Enable gradient checkpointing if available
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    
                if hasattr(self.model, 'enable_input_require_grads'):
                    self.model.enable_input_require_grads()
                
                # Handle device placement based on model type
                if hasattr(self.model, 'device_map'):
                    # Get the device map from args or create a default one
                    device_map = getattr(self.args, 'device_map', None)
                    if device_map is None:
                        # Create default device map
                        num_layers = getattr(self.model.config, 'num_hidden_layers', 32)
                        device_map = {
                            'model.embed_tokens': f'cuda:{self.gpu_base_idx}',
                            'model.norm': f'cuda:{self.gpu_base_idx}',
                            'lm_head': f'cuda:{self.gpu_base_idx}'
                        }
                        
                        # Distribute layers across available GPUs
                        layers_per_gpu = num_layers // (self.args.n_gpu - 1)
                        for i in range(num_layers):
                            gpu_idx = self.gpu_base_idx + 1 + (i // layers_per_gpu)
                            if gpu_idx >= self.gpu_base_idx + self.args.n_gpu:
                                gpu_idx = self.gpu_base_idx + 1 + ((i // layers_per_gpu) % (self.args.n_gpu - 1))
                            device_map[f'model.layers.{i}'] = f'cuda:{gpu_idx}'
                    
                    self.args.device_map = device_map
                    
                    # Move model components according to device map
                    for name, device in device_map.items():
                        try:
                            module = self.model
                            for part in name.split('.'):
                                if hasattr(module, part):
                                    module = getattr(module, part)
                            if hasattr(module, 'to'):
                                module.to(device)
                        except Exception as e:
                            self.args.logger.warning(f"Error moving {name} to {device}: {str(e)}")
                            continue

                # Ensure embeddings are on primary GPU
                if hasattr(self.model, 'get_input_embeddings'):
                    embeddings = self.model.get_input_embeddings()
                    if embeddings is not None:
                        embeddings.to(f'cuda:{self.gpu_base_idx}')
                
                # Prepare with accelerator
                prepared = self.accelerator.prepare(
                    optimizer,
                    train_dataloader,
                    scheduler
                )
                optimizer, train_dataloader, scheduler = prepared
                
                # Force memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                self.args.logger.error(f"Error during model preparation: {str(e)}")
                # Log memory info for available GPUs
                for i in range(self.gpu_base_idx, self.gpu_base_idx + self.args.n_gpu):
                    try:
                        if torch.cuda.device_count() > i:
                            free_mem, total_mem = torch.cuda.mem_get_info(i)
                            self.args.logger.error(f"GPU {i} memory: free={free_mem/1024**3:.2f}GB, total={total_mem/1024**3:.2f}GB")
                    except RuntimeError:
                        continue
                raise

            global_step, acc_loss_dict = 0, OrderedDict()
            set_seed(self.args)
            self.model.train()
            for idx in range(self.args.num_train_epochs):
                for step, batch in enumerate(train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        loss, loss_dict = self.step(batch)
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1

                        if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                            reported_loss = self.report_loss_dict(acc_loss_dict, self.args.logging_steps)
                            self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s', idx+1, int(self.args.num_train_epochs), global_step, total_steps, reported_loss)
                            acc_loss_dict.clear()

                if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
                    self.model.eval()
                    with torch.no_grad():
                        reported_eval_loss = self.do_eval()
                    self.model.train()
                    self.args.logger.info('val epoch %s: %s', idx+1, reported_eval_loss)
                    output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                    last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                    self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                    self.save(output_dir, global_step, idx+1, None, None)
                    self.save(last_output_dir, global_step, idx+1, None, None)

            if (idx+1) % self.args.save_epochs != 0:
                self.model.eval()
                with torch.no_grad():
                    reported_eval_loss = self.do_eval()
                self.args.logger.info('final eval loss: %s', reported_eval_loss)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir, global_step, idx+1, None, None)
                self.save(last_output_dir, global_step, self.args.num_train_epochs, None, None)
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.args.logger.error(f"GPU OOM error: {str(e)}")
                self.log_memory_stats()
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            
    def log_memory_stats(self):
        """Helper method to log memory statistics"""
        for i in range(self.gpu_base_idx, self.gpu_base_idx + self.args.n_gpu):
            try:
                if torch.cuda.device_count() > i:
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    self.args.logger.error(
                        f"GPU {i} memory: free={free_mem/1024**3:.2f}GB, "
                        f"total={total_mem/1024**3:.2f}GB, "
                        f"used={(total_mem-free_mem)/1024**3:.2f}GB"
                    )
            except RuntimeError:
                continue

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = OrderedDict()
        for batch in val_dataloader:
            loss, loss_dict = self.step(batch)
            self.add_to_loss_dict(acc_loss_dict, loss_dict)
        return self.report_loss_dict(acc_loss_dict, len(val_dataloader))

def get_logits_from_lm(model, inputs, control_ids):
    """Modified to ensure consistent device placement and modern autocast usage"""
    try:
        # Get target device from model's embeddings
        if hasattr(model, 'get_input_embeddings'):
            target_device = model.get_input_embeddings().weight.device
        else:
            target_device = next(model.parameters()).device

        # Remove full model transfer to avoid out-of-memory (model is already distributed)
        # model = model.to(target_device)

        # Prepare inputs
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(target_device)
            attention_mask = torch.ones_like(input_ids, device=target_device)
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=target_device).unsqueeze(0)
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'use_cache': False
            }
        else:
            model_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()}

        # Add control_ids if provided
        if control_ids is not None:
            model_inputs['control_ids'] = control_ids.to(target_device)

        # Forward pass with updated autocast
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # Ensure rotary embeddings are on correct device
            if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                model.model.rotary_emb = model.model.rotary_emb.to(target_device)
                if hasattr(model.model.rotary_emb, 'inv_freq'):
                    model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(target_device)
                    
            outputs = model(**model_inputs)
            
        # Process outputs
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        logits = logits.to(target_device)
        label_probs = torch.softmax(logits, dim=-1)
        
        return logits, label_probs
        
    except Exception as e:
        print(f"Error in get_logits_from_lm: {str(e)}")
        print(f"Model device map: {model.device_map if hasattr(model, 'device_map') else 'No device map'}")
        print(f"Input device: {inputs.device if isinstance(inputs, torch.Tensor) else 'dict'}")
        if hasattr(model, 'get_input_embeddings'):
            print(f"Embedding device: {model.get_input_embeddings().weight.device}")
            
        # Debug rotary embeddings
        if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
            print(f"Rotary embedding device: {model.model.rotary_emb.inv_freq.device}")
        raise

class PrefixTrainer(TrainerBase):
    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)

    def load_model(self):
        """Modified to handle device placement better"""
        try:
            self.tokenizer, self.model, self.input_device = load_model('prefix', self.args.pretrain_dir, True, self.args)
            
            # Ensure prefix parameters are properly initialized
            if hasattr(self.model, 'prefix_params'):
                for n, p in self.model.named_parameters():
                    if n.startswith('prefix_params'):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                        
                # Move prefix parameters to appropriate device
                device = f'cuda:{self.args.n_gpu - 1}'  # Use last GPU
                self.model.prefix_params = self.model.prefix_params.to(device)
                
            self.model.train()
            
        except Exception as e:
            self.args.logger.error(f"Error in load_model: {str(e)}")
            raise

    def load_dataset(self):
        self.dataset = PrefixDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = PrefixDataset(self.args, self.tokenizer, 'val')

    def step(self, batch):
        """Modified step function to ensure device consistency"""
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        
        # Get primary device from model
        primary_device = next(self.model.parameters()).device
        
        # Move batch tensors to primary device
        inputs = inputs.to(primary_device, non_blocking=True)
        weights = weights.to(primary_device, non_blocking=True)
        control_ids = control_ids.to(primary_device, non_blocking=True)
        
        # Remove batch dimension for processing
        inputs = inputs.squeeze(0)
        weights = weights.squeeze(0)
        shift_inputs = inputs[..., 1:].clone()
        shift_weights = weights[..., 1:].clone()
        
        # Ensure model is in training mode
        self.model.train()
        
        # Force rotary embeddings to correct device if present
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(primary_device)
            if hasattr(self.model.model.rotary_emb, 'inv_freq'):
                self.model.model.rotary_emb.inv_freq = self.model.model.rotary_emb.inv_freq.to(primary_device)
        
        # Get logits with proper device placement
        correct_logits, correct_label_probs = get_logits_from_lm(
            self.model, 
            inputs.unsqueeze(0), 
            control_ids
        )
        
        # Process logits
        correct_logits = correct_logits.squeeze(0)
        correct_logits = correct_logits[:-1]  # Remove last position
        
        # Compute losses with device-aware token_weighted_loss
        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio
        return_dict['lm_loss'] = lm_loss.item()

        if self.args.contrastive_loss_ratio != 0 or self.args.kl_loss_ratio != 0:
            # Rest of the code...
            # Ensure all tensors are on the same device when computing additional losses
            incorrect_control_ids = -1 * (control_ids - 1)
            incorrect_logits, incorrect_label_probs = get_logits_from_lm(self.model, inputs.unsqueeze(0), incorrect_control_ids)
            incorrect_logits = incorrect_logits.squeeze(0).to(primary_device)
            incorrect_logits = incorrect_logits[:-1]

            # Enable gradients for incorrect logits if needed
            if not incorrect_logits.requires_grad:
                incorrect_logits.requires_grad_(True)
            
            contrastive_loss = 0
            if self.args.contrastive_loss_ratio != 0:
                correct_label_probs = correct_label_probs.squeeze(0)[:-1]  # Remove batch dim and last position
                incorrect_label_probs = incorrect_label_probs.squeeze(0)[:-1]  # Remove batch dim and last position
                
                # Stack probabilities for correct and incorrect along new dimension
                contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
                contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
                contrastive_log_probs = torch.log(contrastive_probs)
                
                # Create labels and replicate for each choice
                contrastive_labels = torch.zeros(shift_inputs.size(0) * 2, dtype=torch.long, device=self.input_device)
                # First half should be 0 (correct), second half should be 1 (incorrect)
                contrastive_labels[shift_inputs.size(0):] = 1
                
                # Reshape log probs to [batch*2, vocab_size]
                contrastive_log_probs = contrastive_log_probs.view(-1, contrastive_log_probs.size(-1))
                
                # Replicate weights for both choices
                contrastive_weights = shift_weights.repeat(2)
                
                # Ensure contrastive log probs have gradients
                if not contrastive_log_probs.requires_grad:
                    contrastive_log_probs.requires_grad_(True)
                
                contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, contrastive_weights)
                contrastive_loss *= self.args.contrastive_loss_ratio / 100
                return_dict['contrastive_loss'] = contrastive_loss.item()

            kl_loss = 0
            if self.args.kl_loss_ratio != 0:
                correct_log_probs = F.log_softmax(correct_logits, dim=-1)
                
                # Ensure correct log probs have gradients
                if not correct_log_probs.requires_grad:
                    correct_log_probs.requires_grad_(True)
                    
                self.model.eval()
                with torch.no_grad():
                    ref_logits, _ = get_logits_from_lm(self.model, inputs.unsqueeze(0), None)
                self.model.train()
                
                ref_logits = ref_logits.squeeze(0)
                ref_logits = ref_logits[:-1]  # Remove last position
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                kl_loss += token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1-shift_weights)
                incorrect_log_probs = F.log_softmax(incorrect_logits, dim=-1)
                
                # Ensure incorrect log probs have gradients
                if not incorrect_log_probs.requires_grad:
                    incorrect_log_probs.requires_grad_(True)
                    
                kl_loss += token_weighted_loss('kl', incorrect_log_probs, ref_log_probs, 1-shift_weights)
                kl_loss = kl_loss * self.args.kl_loss_ratio / 1000
                return_dict['kl_loss'] = kl_loss.item()

        # Final loss aggregation
        loss = lm_loss + contrastive_loss + kl_loss
        return_dict['loss'] = loss.item()
        
        return loss, return_dict

class TextPromptTrainer(TrainerBase):

    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.pretrain_dir, True, self.args)
        self.model.train()

    def load_dataset(self):
        self.dataset = TextPromptDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = TextPromptDataset(self.args, self.tokenizer, 'val')

    def step(self, batch):
        inputs, labels= batch
        inputs = inputs.to(self.input_device)
        labels = labels.to(self.input_device)
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        return loss, {'loss': loss.item()}
