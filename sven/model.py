import os
import torch
import logging as python_logging  # Rename to avoid confusion
from transformers import AutoTokenizer, AutoConfig, logging as transformers_logging
from typing import Optional, Tuple, Union, List
from transformers import AutoTokenizer, AutoConfig, logging, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from sven.hf import CodeGenForCausalLM, XGLMForCausalLM, GPT2LMHeadCustomModel, GPT2CustomConfig
from transformers import BitsAndBytesConfig
import gc

# Set CUDA_VISIBLE_DEVICES to use GPUs 1 to 7
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"

def parallelize_model(model, args):
    """Updated parallelization function with automatic device mapping"""
    if args.n_gpu > 1:
        if hasattr(model, 'parallelize'):
            # Let the model handle its own parallelization
            model.parallelize()
        else:
            # Use DataParallel with available devices
            model = torch.nn.DataParallel(model)
            
        # Get primary device from model
        input_device = next(model.parameters()).device
    else:
        # Single GPU case - let PyTorch choose the device
        model = model.to('cuda')
        input_device = next(model.parameters()).device
            
    return input_device

class CodeGenPrefixCausalLM(CodeGenForCausalLM):
    def __init__(self, config):
        # Map LLaMA config attributes
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers
        
        config.n_embd = hidden_size 
        config.n_head = num_attention_heads
        config.n_layer = num_hidden_layers
        
        super().__init__(config)

        self.n_embed_per_head = hidden_size // num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(num_hidden_layers):
                for _ in range(2):
                    param_size = (num_attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        device = self.prefix_params[0].device  # Ensure all tensors are on the same device
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx].to(device))
                val = self.dropout(self.prefix_params[val_idx].to(device))
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        device = self.prefix_params[0].device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if head_mask is not None:
            head_mask = head_mask.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if labels is not None:
            labels = labels.to(device)
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class IncoderPrefixLM(XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.d_model // config.attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_layers):
                for _ in range(2):
                    param_size = (config.attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        device = self.prefix_params[0].device  # Ensure all tensors are on the same device
        for i in range(self.config.num_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx].to(device))
                val = self.dropout(self.prefix_params[val_idx].to(device))
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": None,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        device = self.prefix_params[0].device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if head_mask is not None:
            head_mask = head_mask.to(device)
        if cross_attn_head_mask is not None:
            cross_attn_head_mask = cross_attn_head_mask.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if labels is not None:
            labels = labels.to(device)
        return super().forward(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class T5PrefixModel(PreTrainedModel):
    base_model_prefix = 't5'
    config_class = AutoConfig
    
    @staticmethod
    def _get_config_attr(config, attr_name, default=None):
        """Safely get config attribute with fallback for nested configs"""
        if hasattr(config, attr_name):
            return getattr(config, attr_name)
        # Handle CodeT5p's nested config structure
        if hasattr(config, 'encoder') and hasattr(config.encoder, attr_name):
            return getattr(config.encoder, attr_name)
        if hasattr(config, 'decoder') and hasattr(config.decoder, attr_name):
            return getattr(config.decoder, attr_name)
        return default
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.device_map = None
        self.main_device = "cuda"  # Default device
        
        # Get model dimensions with fallbacks for different config structures
        hidden_size = (self._get_config_attr(config, 'd_model') or 
                      self._get_config_attr(config, 'hidden_size'))
        num_attention_heads = (self._get_config_attr(config, 'num_heads') or 
                             self._get_config_attr(config, 'num_attention_heads'))
        num_layers = (self._get_config_attr(config, 'num_layers') or 
                     self._get_config_attr(config, 'num_hidden_layers'))
        
        if not all([hidden_size, num_attention_heads, num_layers]):
            raise ValueError(f"Could not determine model dimensions from config: {config}")
        
        self.n_embed_per_head = hidden_size // num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(num_layers * 2):  # For both encoder and decoder
                for _ in range(2):
                    param_size = (num_attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Enhanced pretrained loading with quantization support"""
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        
        # Setup quantization and device mapping
        kwargs.update({
            'low_cpu_mem_usage': True,
            'trust_remote_code': trust_remote_code,
            'device_map': "auto",
            'torch_dtype': torch.bfloat16,
            # 'quantization_config': {
            #     'load_in_8bit': True,
            #     'load_in_4bit': False,  # Can be enabled for even more memory savings
            #     'bnb_4bit_quant_type': 'nf4',
            #     'bnb_4bit_compute_dtype': torch.bfloat16,
            #     'bnb_4bit_use_double_quant': True,
            # }
        })
        
        if not config:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        
        model = cls(config)
        
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with quantization
            model.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                quantization_config=quantization_config,
                **kwargs
            )
            
            # Keep prefix parameters in full precision on first GPU
            model.prefix_params = model.prefix_params.to('cuda')
            
        except Exception as e:
            print(f"Error during model loading: {str(e)}")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            raise
            
        return model

    def parallelize(self, device_map=None):
        """Updated parallelization with automatic device mapping"""
        if device_map is None:
            device_map = "auto"
        
        try:
            # Let the accelerator handle the device mapping
            if not hasattr(self.model, 'is_parallelized'):
                self.model.is_parallelized = True
                if hasattr(self.model, "parallelize"):
                    self.model.parallelize(device_map)
                # Keep prefix parameters on first GPU for efficiency
                # self.prefix_params = self.prefix_params.to('cuda:0')
        except Exception as e:
            print(f"Error during parallelization: {str(e)}")
            raise
        return self

    def get_input_embeddings(self):
        """Gets the input embeddings from the wrapped model"""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Sets the input embeddings in the wrapped model"""
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Gets the output embeddings from the wrapped model"""
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings in the wrapped model"""
        self.model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: int):
        """Memory-efficient token embedding resizing"""
        try:
            # Clear cache before resizing
            torch.cuda.empty_cache()
            
            if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
                # Do encoder and decoder resizing separately with memory cleanup
                try:
                    self.model.encoder.resize_token_embeddings(new_num_tokens)
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Try moving encoder to CPU temporarily
                        self.model.encoder.cpu()
                        self.model.encoder.resize_token_embeddings(new_num_tokens)
                        self.model.encoder.to(self.main_device)
                        torch.cuda.empty_cache()
                
                try:
                    self.model.decoder.resize_token_embeddings(new_num_tokens)
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Try moving decoder to CPU temporarily
                        self.model.decoder.cpu()
                        self.model.decoder.resize_token_embeddings(new_num_tokens)
                        self.model.decoder.to(self.main_device)
                        torch.cuda.empty_cache()
            else:
                try:
                    self.model.resize_token_embeddings(new_num_tokens)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Try CPU fallback for whole model
                        self.model.cpu()
                        self.model.resize_token_embeddings(new_num_tokens)
                        self.model.to(self.main_device)
                        torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during token embedding resize: {str(e)}")
            raise
            
        return self

    def _get_decoder_start_token_id(self):
        """Get decoder start token ID with proper fallbacks"""
        # First check the initialized model
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'decoder_start_token_id') and \
               self.model.config.decoder_start_token_id is not None:
                return self.model.config.decoder_start_token_id
            if hasattr(self.model.config, 'bos_token_id') and \
               self.model.config.bos_token_id is not None:
                return self.model.config.bos_token_id
        
        # Then check our own config
        if hasattr(self.config, 'decoder_start_token_id') and \
           self.config.decoder_start_token_id is not None:
            return self.config.decoder_start_token_id
        if hasattr(self.config, 'bos_token_id') and \
           self.config.bos_token_id is not None:
            return self.config.bos_token_id
        
        # Fallback to 0 if nothing else works
        return 0

    def _get_pad_token_id(self):
        """Get pad token ID with proper fallbacks"""
        # First check the initialized model
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'pad_token_id') and \
               self.model.config.pad_token_id is not None:
                return self.model.config.pad_token_id
        
        # Then check our own config
        if hasattr(self.config, 'pad_token_id') and \
           self.config.pad_token_id is not None:
            return self.config.pad_token_id
            
        # Default to eos_token_id if available
        if hasattr(self.model, 'config') and \
           hasattr(self.model.config, 'eos_token_id') and \
           self.model.config.eos_token_id is not None:
            return self.model.config.eos_token_id
            
        # Fallback to decoder_start_token_id
        return self._get_decoder_start_token_id()

    def _prepare_decoder_input_ids(self, input_ids):
        """Create decoder input IDs from input IDs by shifting right and prepending start token"""
        pad_token_id = self._get_pad_token_id()
        decoder_start_token_id = self._get_decoder_start_token_id()
        
        if pad_token_id is None or decoder_start_token_id is None:
            raise ValueError(
                "Make sure that the model has the pad_token_id and decoder_start_token_id attributes set properly"
            )
        
        # Shift right by padding with pad token
        shifted = torch.full_like(input_ids, pad_token_id)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        # Set first token to decoder_start_token_id
        shifted[:, 0] = decoder_start_token_id
        return shifted

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # Fixed type hint here
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """Modified forward pass with automatic device handling"""
        # Get current devices
        enc_device = next(self.model.encoder.parameters()).device
        dec_device = next(self.model.decoder.parameters()).device
        proj_device = self.model.enc_to_dec_proj.weight.device if hasattr(self.model, 'enc_to_dec_proj') else dec_device
        
        # Process inputs
        model_inputs = {}
        
        # Filter out unexpected arguments
        expected_args = [
            'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask',
            'head_mask', 'decoder_head_mask', 'encoder_outputs', 'past_key_values',
            'inputs_embeds', 'decoder_inputs_embeds', 'labels', 'use_cache',
            'output_attentions', 'output_hidden_states', 'return_dict'
        ]
        
        # Only pass expected arguments to the model
        for arg in expected_args:
            if arg in locals() and locals()[arg] is not None:
                model_inputs[arg] = locals()[arg]
        
        # Move tensors to appropriate devices
        if input_ids is not None:
            model_inputs['input_ids'] = input_ids.to(enc_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(enc_device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(enc_device)
        
        if decoder_input_ids is not None:
            model_inputs['decoder_input_ids'] = decoder_input_ids.to(dec_device)
        elif input_ids is not None:
            model_inputs['decoder_input_ids'] = self._prepare_decoder_input_ids(input_ids).to(dec_device)
            
        # ... Rest of the device handling code remains unchanged ...
        
        # Remove control_ids and any other unexpected kwargs before passing to model
        if 'control_ids' in kwargs:
            del kwargs['control_ids']
            
        # Add remaining expected kwargs
        for k, v in kwargs.items():
            if k in expected_args:
                model_inputs[k] = v
        
        outputs = self.model(**model_inputs)
        
        # Convert logits to probabilities if needed
        if hasattr(outputs, 'logits') and not hasattr(outputs, 'label_probs'):
            outputs.label_probs = torch.softmax(outputs.logits, dim=-1)
            
        return outputs

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for T5PrefixModel"""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        else:
            python_logging.warning(
                "gradient_checkpointing_enable method not found in base model"
            )

class SantaPrefixLM(GPT2LMHeadCustomModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                # mha
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class CrystalCodePrefixLM(PreTrainedModel):
    base_model_prefix = 'crystal'
    _no_split_modules = ['crystal']  # Add this line to support device mapping
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize base model with trust_remote_code
        self.crystal = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True
        )
        
        self.n_embed_per_head = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        
        # Initialize prefix parameters
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):
                    param_size = (config.num_attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(
                        torch.zeros(param_size, dtype=torch.bfloat16),  # Use bfloat16 by default
                        requires_grad=True
                    )
                    self.prefix_params.append(param)
                    
        self.dropout = torch.nn.Dropout(config.prefix_dropout)
        
        # Initialize device tracking
        self._primary_device = None
        self._device_map = {}

    def to(self, device):
        """Enhanced device management"""
        if isinstance(device, str):
            device = torch.device(device)
            
        self._primary_device = device
        
        # Move base model
        self.crystal = self.crystal.to(device)
        
        # Move prefix parameters
        self.prefix_params = torch.nn.ParameterList([
            param.to(device) for param in self.prefix_params
        ])
        
        # Update device map
        self._device_map.clear()
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                self._device_map[name] = module.weight.device
                
        return self

    def forward(self, *args, **kwargs):
        device = next(self.parameters()).device
        
        # Handle both positional and keyword arguments
        if len(args) > 0 and isinstance(args[0], dict):
            inputs = args[0]
        else:
            inputs = kwargs
            
        # Remove control_ids from inputs if present
        if 'control_ids' in inputs:
            control_ids = inputs.pop('control_ids')
        
        # Ensure we have either input_ids or inputs_embeds
        if 'input_ids' not in inputs and 'inputs_embeds' not in inputs:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        return self.crystal(**inputs)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        device = next(self.parameters()).device
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # Ensure input_ids are on correct device
        input_ids = input_ids.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
            
        # Move past states to correct device
        if past is not None:
            past = [(p[0].to(device), p[1].to(device)) for p in past]

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def get_input_embeddings(self):
        return self.crystal.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.crystal.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.crystal.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.crystal.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: int):
        self.crystal.resize_token_embeddings(new_num_tokens)
        return self

    def get_past_from_prefix(self, control_ids):
        past = list()
        device = self.prefix_params[0].device
        for i in range(self.config.num_hidden_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx].to(device))
                val = self.dropout(self.prefix_params[val_idx].to(device))
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past
    
    def parallelize(self, device_map):
        """Memory-optimized parallelization with corrected device mapping"""
        try:
            # Move embeddings to first available GPU (GPU 1)
            if hasattr(self.crystal, 'transformer'):
                for attr in ['wte', 'wpe']:
                    if hasattr(self.crystal.transformer, attr):
                        if getattr(self.crystal.transformer, attr) is not None:
                            setattr(self.crystal.transformer, attr, getattr(self.crystal.transformer, attr).to('cuda'))
            
            # Move LM head to first available GPU (GPU 1)
            if hasattr(self.crystal, 'lm_head') and self.crystal.lm_head is not None:
                self.crystal.lm_head = self.crystal.lm_head.to('cuda')
            
            # Distribute transformer layers
            if hasattr(self.crystal, 'transformer') and hasattr(self.crystal.transformer, 'h'):
                for i, layer in enumerate(self.crystal.transformer.h):
                    gpu_idx = device_map.get(i, 1)  # Default to GPU 1 if not in map
                    try:
                        layer.to(f'cuda')
                    except RuntimeError as e:
                        print(f"Warning: Could not move layer {i} to GPU {gpu_idx}, using GPU 1")
                        layer.to('cuda')
            else:
                self.crystal.parallelize(device_map)
            
            # Distribute prefix parameters between GPUs 1 and 2
            for i, param in enumerate(self.prefix_params):
                gpu_idx = 1 + (i % 2)  # Alternate between GPUs 1 and 2
                self.prefix_params[i] = param.to(f'cuda')
                
            # Force memory cleanup
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error during parallelization: {str(e)}")
            raise
            
        return self

from huggingface_hub.file_download import hf_hub_download

def hf_hub_download_compatible(*args, **kwargs):
    try:
        return hf_hub_download(*args, **kwargs)
    except TypeError as e:
        if "exist_ok" in str(e):
            # Remove the incompatible argument and retry
            kwargs.pop('exist_ok', None)
            return hf_hub_download(*args, **kwargs)
        else:
            raise e

def model_from_pretrained(lm_path, model_type, config, **kwargs):
    # Check if model directory exists and contains config.json
    # config_path = os.path.join(lm_path, 'config.json')
    # if not os.path.exists(config_path):
    #     raise FileNotFoundError(f"Configuration file not found in model directory: {config_path}")

    kwargs = dict()
    
    # Read base model path if this is a saved prefix model
    if os.path.exists(os.path.join(lm_path, 'lm.txt')):
        with open(os.path.join(lm_path, 'lm.txt')) as f:
            base_model_path = f.read().strip()
            lm_path = base_model_path
        
        # Load config from original path and add prefix model attributes
        config = AutoConfig.from_pretrained(lm_path, trust_remote_code=True)
        config.n_control = 2  # Default value for prefix models
        config.n_prefix_token = 8  # Default value
        config.prefix_dropout = 0.0  # Default value
        
    if lm_path.startswith('Salesforce/codegen-'):
        if model_type == 'lm':
            model_class = CodeGenForCausalLM
        elif model_type == 'prefix':
            model_class = CodeGenPrefixCausalLM
            # Ensure config has required prefix attributes
            if not hasattr(config, 'n_control'):
                config.n_control = 2
            if not hasattr(config, 'n_prefix_token'):
                config.n_prefix_token = 8
            if not hasattr(config, 'prefix_dropout'):
                config.prefix_dropout = 0.0
        else:
            assert False

    elif lm_path.startswith('Salesforce/codet5p-6b'):
        if model_type == 'lm':
            model_class = AutoModelForSeq2SeqLM
            kwargs.update({'trust_remote_code': True})
        elif model_type == 'prefix':
            # model_class = T5PrefixModel
            model_class = AutoModelForSeq2SeqLM
            kwargs.update({'trust_remote_code': True})
            # No need to add trust_remote_code here since it's handled in from_pretrained
            if config is None:
                config = AutoConfig.from_pretrained(lm_path, trust_remote_code=True)
        else:
            assert False
            
        if config is None:
            model = model_class.from_pretrained(lm_path, **kwargs)
        else:
            try:
                model = model_class.from_pretrained(lm_path, config=config, **kwargs)
            except ValueError as e:
                print(f"Model state dict integrity check failed: {e}")
                model = model_class.from_pretrained(lm_path, trust_remote_code=True)
        return model

    elif lm_path.startswith('LLM360/Crystal'):
        # Remove device_map from kwargs for distributed training
        if model_type == 'lm':
            kwargs.update({
                'torch_dtype': torch.float16,
                'trust_remote_code': True
            })
            model_class = AutoModelForCausalLM
        elif model_type == 'prefix':
            kwargs.update({
                'torch_dtype': torch.float16,
                'trust_remote_code': True,  # Ensure this is set
                'config': config if config is not None else None
            })
            model_class = CrystalCodePrefixLM
        else:
            assert False
            
        # Add trust_remote_code=True to config if not present
        if config is not None and not hasattr(config, 'trust_remote_code'):
            config.trust_remote_code = True
            
        return model_class.from_pretrained(lm_path, **kwargs)

    elif lm_path.startswith('Salesforce/codegen25-'):
        if model_type == 'lm':
            model_class = AutoModelForCausalLM
        elif model_type == 'prefix':
            model_class = AutoModelForCausalLM
        else:
            assert False
            
        kwargs.update({
            'device_map': "balanced",  # Changed from 'auto' to 'balanced'
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True
        })
        
        if 'quantization_config' in kwargs and kwargs['quantization_config']:
            kwargs['quantization_config'] = kwargs['quantization_config']
            
        # Special handling for prefix model
        if model_type == 'prefix':
            kwargs['device_map'] = None  # Don't split prefix model across devices

    elif lm_path.startswith('Salesforce/codegen2-'):
        if model_type == 'lm':
            model_class = CodeGenForCausalLM
        elif model_type == 'prefix':
            model_class = CodeGenPrefixCausalLM
        else:
            assert False

    elif lm_path.startswith('codellama/CodeLlama-'):
        if model_type == 'lm':
            model_class = AutoModelForCausalLM
        elif model_type == 'prefix':
            model_class = AutoModelForCausalLM
        else:
            assert False

    elif lm_path.startswith('Salesforce/codegen25-'):
        if model_type == 'lm':
            model_class = AutoModelForCausalLM
        elif model_type == 'prefix':
            model_class = AutoModelForCausalLM
        else:
            assert False
            
        kwargs.update({
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True
        })
        
        # Add quantization config if provided
        if 'quantization_config' in kwargs and kwargs['quantization_config']:
            kwargs['quantization_config'] = kwargs['quantization_config']
    
    elif lm_path.startswith('facebook/incoder-'):
        if config is not None:
            config.attention_dropout = 0.0
            config.dropout = 0.0
        if model_type == 'lm':
            model_class = XGLMForCausalLM
        elif model_type == 'prefix':
            model_class = IncoderPrefixLM
        else:
            assert False
    elif lm_path == 'bigcode/santacoder':
        kwargs['revision'] = 'mha'
        if config is not None:
            config.attn_pdrop = 0.0
            config.embd_pdrop = 0.0
            config.resid_pdrop = 0.0
        if model_type == 'lm':
            model_class = GPT2LMHeadCustomModel
        elif model_type == 'prefix':
            model_class = SantaPrefixLM
        else:
            assert False
    # Add new branch for loading saved prefix models
    elif os.path.exists(os.path.join(lm_path, 'pytorch_model.bin')):
        if model_type == 'prefix':
            return load_prefix_model(lm_path, None)  # Pass None for args since we have kwargs
        else:
            assert False
            
    else:
        python_logging.error(f"Unsupported model path prefix: {lm_path}")
        assert False

    if config is None:
        model = model_class.from_pretrained(lm_path, **kwargs)
    else:
        try:
            model = model_class.from_pretrained(lm_path, **kwargs, config=config)
        except ValueError as e:
            print(f"Model state dict integrity check failed: {e}")
            model = model_class.from_pretrained(lm_path, trust_remote_code=True)

    return model

def config_from_pretrained(lm_path, path):
    if lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    else:
        return AutoConfig.from_pretrained(path, trust_remote_code=True)

def save_model(model, path, args):
    if type(model) in (T5PrefixModel, AutoModelForSeq2SeqLM, CrystalCodePrefixLM, AutoModelForCausalLM, CodeGenPrefixCausalLM, IncoderPrefixLM, SantaPrefixLM):
        assert args.pretrain_dir.startswith('LLM360/Crystal') or args.pretrain_dir.startswith('Salesforce/codet5p-6b') or args.pretrain_dir.startswith('codellama/CodeLlama-7b-hf') or args.pretrain_dir.startswith('Salesforce/codegen-') or args.pretrain_dir.startswith('Salesforce/codegen2-') or args.pretrain_dir.startswith('Salesforce/codegen25-') or args.pretrain_dir.startswith('facebook/incoder-') or args.pretrain_dir == 'bigcode/santacoder'
        config_file = os.path.join(path)
        model.config.save_pretrained(config_file)
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = model.prefix_params.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, prefix_file)
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file, 'w') as f:
            f.write(args.pretrain_dir)
    else:
        model.save_pretrained(path)

def load_prefix_model(path, args):
    """Load a saved prefix model"""
    try:
        # Get base model path
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file) as f:
            lm_path = f.read().strip()
            
        # Load config and create model
        config = config_from_pretrained(lm_path, path)
        model = T5PrefixModel(config)
        
        # Load base model
        model.model = AutoModelForSeq2SeqLM.from_pretrained(
            lm_path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load prefix parameters
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = torch.load(prefix_file, map_location='cpu')
        model.prefix_params.load_state_dict(state_dict)
        
        return model
        
    except Exception as e:
        python_logging.error(f"Error loading prefix model: {str(e)}")
        # python_logging.error(traceback.format_exc())
        raise

def load_model(model_type, path, is_training, args):
    """Updated load_model function with improved device management"""
    transformers_logging.set_verbosity_error()
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.bos_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Configure loading kwargs
        load_kwargs = {
            'device_map': 'balanced_low_0',  # Use balanced mapping avoiding GPU 0
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        # Load model and move to appropriate device
        config = config_from_pretrained(path, path)
        model = model_from_pretrained(
            path, 
            model_type, 
            config, 
            **load_kwargs
        )
        
        # Ensure rotary embeddings are properly placed
        if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
            device = next(model.parameters()).device
            model.model.rotary_emb = model.model.rotary_emb.to(device)
            if hasattr(model.model.rotary_emb, 'inv_freq'):
                model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
        
        # Get input device from embeddings
        input_device = model.get_input_embeddings().weight.device if hasattr(model, 'get_input_embeddings') else next(model.parameters()).device
        
        return tokenizer, model, input_device
        
    except Exception as e:
        python_logging.error(f"Error loading model: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        raise