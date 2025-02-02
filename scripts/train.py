import os
import torch
import logging
import argparse
import gc
from accelerate import Accelerator, DistributedDataParallelKwargs

from sven.trainer import PrefixTrainer, TextPromptTrainer
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import MODEL_DIRS

# Configure CUDA memory management and set available GPUs
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--output_dir', type=str, default='../trained/')
    parser.add_argument('--model_type', type=str, default='prefix')
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--vul_type', type=str, default=None)

    parser.add_argument('--n_prefix_token', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--kl_loss_ratio', type=int, default=None) # will be divided by 1000
    parser.add_argument('--learning_rate', type=float, default=None)

    parser.add_argument('--contrastive_loss_ratio', type=int, default=400) # will be divided by 100
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=8) #2
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diff_level', type=str, choices=['prog', 'line', 'char', 'mix'], default='mix')
    parser.add_argument('--lm_loss_ratio', type=int, default=1)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    # Add quantization arguments
    parser.add_argument('--quantize', action='store_true', help='Enable 8-bit quantization')
    parser.add_argument('--quantize_4bit', action='store_true', help='Enable 4-bit quantization')

    # Add accelerate-specific arguments
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    
    # Update distributed training arguments
    parser.add_argument('--n_gpu', type=int, default=7)
    parser.add_argument('--num_processes', type=int, default=None)  # Added this line
    parser.add_argument('--local_rank', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Set num_processes after parsing
    if not hasattr(args, 'num_processes') or args.num_processes is None:
        args.num_processes = args.n_gpu

    if args.pretrain_dir is None:
        args.pretrain_dir = '2b'
    if args.pretrain_dir in MODEL_DIRS:
        args.pretrain_dir = MODEL_DIRS[args.pretrain_dir]

    if args.pretrain_dir.startswith('Salesforce/codegen25-'):
        if args.n_prefix_token is None:
            args.n_prefix_token = 12
        if args.num_train_epochs is None:
            args.num_train_epochs = 5
        if args.kl_loss_ratio is None:
            args.kl_loss_ratio = 2000
        # Adjust batch size and gradient accumulation for memory
        args.grad_acc_steps = args.grad_acc_steps * 2
        # Add CodeGen25 specific settings
        args.trust_remote_code = True
        args.use_fast_tokenizer = False  # Use slow tokenizer for better compatibility
        args.bf16 = True  # Use bfloat16 for better stability
        # Adjust settings specifically for CodeGen 25B
        args.mixed_precision = 'bf16'  # Force bf16 for CodeGen 25B
        args.gradient_accumulation_steps = max(64, args.grad_acc_steps)
        args.gradient_checkpointing = True
        args.max_num_tokens = min(args.max_num_tokens, 512)
        # Configure quantization
        if args.quantize_4bit:
            args.load_in_4bit = True
            args.bnb_4bit_compute_dtype = 'bfloat16'
            args.bnb_4bit_use_double_quant = True
            args.bnb_4bit_quant_type = 'nf4'

    if args.pretrain_dir.startswith('LLM360/Crystal'):
        # Adjust settings for Crystal model
        args.n_prefix_token = args.n_prefix_token or 12
        args.num_train_epochs = args.num_train_epochs or 5
        args.kl_loss_ratio = args.kl_loss_ratio or 2000
        args.gradient_accumulation_steps = max(4, args.grad_acc_steps)
        args.gradient_checkpointing = True
        args.mixed_precision = 'bf16'
        
        # Optimize memory usage
        args.max_num_tokens = min(args.max_num_tokens, 768)
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1

    # Add device_placement strategy for CodeLlama
    if args.pretrain_dir.startswith('codellama/CodeLlama-'):
        args.mixed_precision = 'bf16'
        args.gradient_accumulation_steps = max(32, args.grad_acc_steps)
        args.gradient_checkpointing = True
        args.max_grad_norm = 0.5
        args.max_num_tokens = min(args.max_num_tokens, 512)
        
        # Enhanced memory optimization settings
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.max_memory = {i: "15GiB" for i in range(1, 8)}  # Reduce per-GPU memory limit
        
        # Optimize 4-bit quantization settings
        if args.quantize_4bit:
            args.load_in_4bit = True
            args.use_4bit_quant = True
            args.bnb_4bit_compute_dtype = 'bfloat16'
            args.bnb_4bit_use_double_quant = True
            args.bnb_4bit_quant_type = 'nf4'
            
            # More balanced device mapping
            num_layers = 32
            base_gpu = 1
            layers_per_gpu = max(1, num_layers // (args.n_gpu - 1))  # Ensure at least 1 layer per GPU
            
            # Put embeddings and small components on GPU 1
            args.device_map = {
                'model.embed_tokens': f'cuda:{base_gpu}',
                'model.norm': f'cuda:{base_gpu}',
                'lm_head': f'cuda:{base_gpu}',
            }
            
            # Distribute layers more evenly starting from GPU 2
            for i in range(num_layers):
                gpu_idx = base_gpu + 1 + (i // layers_per_gpu)
                gpu_idx = min(gpu_idx, base_gpu + args.n_gpu - 1)
                args.device_map[f'model.layers.{i}'] = f'cuda:{gpu_idx}'
            
            # Set memory limits per GPU
            args.max_memory = {i: "15GiB" for i in range(base_gpu, base_gpu + args.n_gpu)}
            args.max_memory[base_gpu] = "10GiB"  # Less memory for GPU 1 (embeddings)

            # Update CodeLlama specific settings
            if args.pretrain_dir.startswith('codellama/CodeLlama-'):
                # ...existing code...
                
                if args.quantize_4bit:
                    # Calculate available GPUs
                    num_gpus = torch.cuda.device_count() - 1  # Exclude GPU 0
                    base_gpu = 1  # Start from GPU 1
                    
                    # Distribute model components
                    args.device_map = {
                        'model.embed_tokens': f'cuda:{base_gpu}',
                        'model.norm': f'cuda:{base_gpu}',
                        'lm_head': f'cuda:{base_gpu}',
                        'rotary_emb': f'cuda:{base_gpu}'  # Ensure rotary embeddings are on same device
                    }
                    
                    # Distribute 32 layers across available GPUs
                    layers_per_gpu = max(1, 32 // num_gpus)
                    for i in range(32):
                        gpu_idx = base_gpu + (i // layers_per_gpu)
                        if gpu_idx >= torch.cuda.device_count():
                            gpu_idx = base_gpu + (gpu_idx % num_gpus)
                        args.device_map[f'model.layers.{i}'] = f'cuda:{gpu_idx}'
                    
                    # Set conservative memory limits
                    args.max_memory = {}
                    for i in range(base_gpu, base_gpu + num_gpus):
                        args.max_memory[i] = "10GiB" if i == base_gpu else "15GiB"

            # Update CodeLlama specific settings
            if args.pretrain_dir.startswith('codellama/CodeLlama-'):
                # ...existing code...
                
                if args.quantize_4bit:
                    # Calculate available GPUs
                    num_gpus = min(7, torch.cuda.device_count() - 1)  # Use max 7 GPUs, starting from GPU 1
                    base_gpu = 1
                    
                    # Keep embeddings and essential components on GPU 1
                    args.device_map = {
                        'model.embed_tokens': f'cuda:{base_gpu}',
                        'model.norm': f'cuda:{base_gpu}',
                        'lm_head': f'cuda:{base_gpu}',
                    }
                    
                    # Distribute layers across GPUs 2-7 more evenly
                    layers_per_gpu = max(1, 32 // (num_gpus - 1))  # -1 for GPU 1 reserved for embeddings
                    for i in range(32):
                        gpu_idx = base_gpu + 1 + (i // layers_per_gpu)
                        if gpu_idx >= torch.cuda.device_count():
                            gpu_idx = base_gpu + 1 + ((i // layers_per_gpu) % (num_gpus - 1))
                        args.device_map[f'model.layers.{i}'] = f'cuda:{gpu_idx}'
                    
                    # Conservative memory limits
                    args.max_memory = {i: "10GiB" if i == base_gpu else "15GiB" 
                                     for i in range(base_gpu, base_gpu + num_gpus)}
                    
                    # Enable additional memory optimizations
                    args.gradient_checkpointing = True
                    args.max_grad_norm = 0.5
                    args.gradient_accumulation_steps = max(16, args.grad_acc_steps)

    if args.n_prefix_token is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.n_prefix_token = 5
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.n_prefix_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen2-1B_P':
            args.n_prefix_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.n_prefix_token = 12
        elif args.pretrain_dir == 'Salesforce/codegen2-7B_P':
            args.n_prefix_token = 12
        elif args.pretrain_dir == 'Salesforce/codegen25-7b-multi_P':
            args.n_prefix_token = 12
        elif args.pretrain_dir == 'LLM360/Crystal':
            args.n_prefix_token = 12
        elif args.pretrain_dir == 'codellama/CodeLlama-7b-hf':
            args.n_prefix_token = 12
        elif args.pretrain_dir == 'Salesforce/codet5p-6b':
            args.n_prefix_token = 12

            # 
        else:
            assert False

            # Salesforce/codet5p-6b

    if args.num_train_epochs is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.num_train_epochs = 8
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen2-1B_P':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen2-7B_P':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen25-7b-multi_P':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'LLM360/Crystal':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'codellama/CodeLlama-7b-hf':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codet5p-6b':
            args.num_train_epochs = 5
        else:
            assert False

    if args.kl_loss_ratio is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen2-1B_P':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.kl_loss_ratio = 2000
        elif args.pretrain_dir == 'Salesforce/codegen2-7B_P':
            args.kl_loss_ratio = 2000
        elif args.pretrain_dir == 'Salesforce/codegen25-7b-multi_P':
            args.kl_loss_ratio = 2000
        elif args.pretrain_dir == 'LLM360/Crystal':
            args.kl_loss_ratio = 2000
        elif args.pretrain_dir == 'codellama/CodeLlama-7b-hf':
            args.kl_loss_ratio = 2000
        elif args.pretrain_dir == 'Salesforce/codet5p-6b':
            args.kl_loss_ratio = 2000
        else:
            assert False

    if args.model_type == 'prefix':
        if args.learning_rate is None:
            args.learning_rate = 1e-2

        if args.contrastive_loss_ratio == 0:
            args.learning_rate = 5e-2
            args.grad_acc_steps = args.grad_acc_steps * 2

        if args.model_type == 'prefix' and args.diff_level in ('prog', 'line'):
            args.learning_rate = 1e-3
    elif args.model_type == 'text':
        args.learning_rate = 5e-5

    # Add memory optimization settings
    if args.pretrain_dir.startswith('Salesforce/codet5p-6b'):
        # Optimize settings for CodeT5p-6B
        args.mixed_precision = 'bf16'
        args.gradient_accumulation_steps = max(32, args.grad_acc_steps)
        args.gradient_checkpointing = True
        args.max_grad_norm = 0.5
        args.max_num_tokens = min(args.max_num_tokens, 512)
        
        # Configure quantization
        if args.quantize:
            args.load_in_8bit = True
            args.use_8bit_quant = True
            args.device_map = {"": "auto"}  # Let accelerate handle device mapping
            
        # Adjust batch sizes
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        
        # Set learning rate
        if args.learning_rate is None:
            args.learning_rate = 5e-5

    args.output_dir = os.path.join(args.output_dir, args.output_name)
    return args

def main():
    args = get_args()
    
    # Ensure CUDA is properly configured
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    # Set primary device
    torch.cuda.set_device(1)  # Use GPU 1 as primary
    
    # Initialize accelerator with optimized settings
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=True,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        device_placement=True,
        split_batches=True,
        cpu=False
    )
    
    # Update device settings
    args.device = 'cuda:1'  # Start from GPU 1
    args.n_gpu = 7  # Using GPUs 1-7
    args.gpu_base_idx = 1  # Start from GPU 1
    
    # Early memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    
    if hasattr(torch.cuda, 'memory_summary'):
        logging.info(f"Initial CUDA memory: {torch.cuda.memory_summary()}")
    
    set_devices(args)
    set_seed(args)

    try:
        if args.model_type == 'prefix':
            trainer = PrefixTrainer(args, accelerator)
        elif args.model_type == 'text':
            trainer = TextPromptTrainer(args, accelerator)
        else:
            raise NotImplementedError()

        trainer.run()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA OOM: {str(e)}")
            if hasattr(torch.cuda, 'memory_summary'):
                logging.error(f"CUDA memory at error: {torch.cuda.memory_summary()}")
        raise
    finally:
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()