import os
import re
import abc
import torch
import numpy as np
from vllm import LLM, SamplingParams

from sven.model import CodeGenPrefixCausalLM, load_model
from sven.constant import PROMPTS
from sven.utils import try_parse

class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.use_vllm = getattr(args, 'use_vllm', False)
        self.load_model()

    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def sample(self, file_context, func_context, control, lang):
        raise NotImplementedError()

    def truncate(self, completion, lang):
        if lang == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                last_comment_str = '\n    #'
                if last_comment_str in completion:
                    completion = completion[:completion.rfind(last_comment_str)]
        elif lang == 'c':
            if '\n}' in completion:
                completion = completion[:completion.find('\n}')+2]
            else:
                last_comment_strs = ['\n    //', '\n    /*']
                for last_comment_str in last_comment_strs:
                    if last_comment_str in completion:
                        completion = completion[:completion.rfind(last_comment_str)]
                        completion = completion.rstrip() + '\n}'

            lines = completion.split('\n')
            final_lines = []
            for line in lines:
                if '->name = "' in line: continue
                final_lines.append(line)
            completion = '\n'.join(final_lines)
        else:
            raise NotImplementedError()

        return completion

    def process_completions(self, input_src, input_ids_len, gen_output, lang):
        tokens = gen_output[:, input_ids_len:, ...]
        completions = self.tokenizer.batch_decode(tokens)

        output_srcs, output_ids = [], []
        dup_srcs, non_parsed_srcs = [], []
        for i, completion in enumerate(completions):
            if self.tokenizer.eos_token in completion:
                completion = completion[:completion.find(self.tokenizer.eos_token)]
            completion = self.truncate(completion, lang)
            completion_len = len(self.tokenizer.encode(completion))
            output_src = input_src + completion
            output_src = output_src.rstrip() + '\n'
            if output_src in output_srcs:
                dup_srcs.append(output_src)
            elif try_parse(output_src, lang) != 0:
                non_parsed_srcs.append(output_src)
            else:
                output_srcs.append(output_src)
                output_ids.append((gen_output[i][:input_ids_len].tolist(), gen_output[i][input_ids_len:input_ids_len+completion_len].tolist()))

        return output_srcs, output_ids, dup_srcs, non_parsed_srcs

    def get_vllm_sampling_params(self):
        """Get vLLM sampling parameters"""
        return SamplingParams(
            n=self.args.num_gen,
            temperature=self.args.temp,
            top_p=self.args.top_p,
            max_tokens=self.args.max_gen_len,
            use_beam_search=False
        )

class LMEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.model_dir, False, self.args)
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        input_src = file_context + func_context
        input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]
        gen_output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            # return_dict_in_generate=True,
            # output_scores=True,
        )
        return self.process_completions(input_src, input_ids_len, gen_output, lang)

class PrefixEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def _get_vllm_model_type(self, model_path):
        """Map unsupported model types to compatible vLLM architectures"""
        if 'codegen' in model_path.lower():
            return 'gpt2'  # CodeGen models are GPT2-based
        elif 'codet5' in model_path.lower():
            return 'bart'  # CodeT5 models are BART-based
        return None

    def load_model(self):
        if self.use_vllm:
            try:
                # Try to determine model architecture for vLLM
                model_type = self._get_vllm_model_type(self.args.model_dir)
                if model_type:
                    self.llm = LLM(
                        model=self.args.model_dir,
                        tensor_parallel_size=self.args.tensor_parallel_size,
                        trust_remote_code=True,
                        dtype="float16",
                        model_type=model_type  # Specify model type for vLLM
                    )
                else:
                    # Fallback to standard loading if model type not supported
                    print("Model type not supported by vLLM, falling back to standard loading")
                    self.use_vllm = False
                    self.tokenizer, self.model, self.input_device = load_model(
                        'prefix', self.args.model_dir, False, self.args
                    )
                    self.model.eval()
                    return
                    
                self.tokenizer = self.llm.get_tokenizer()
            except Exception as e:
                print(f"vLLM initialization failed: {str(e)}, falling back to standard loading")
                self.use_vllm = False
                self.tokenizer, self.model, self.input_device = load_model(
                    'prefix', self.args.model_dir, False, self.args
                )
                self.model.eval()
        else:
            self.tokenizer, self.model, self.input_device = load_model(
                'prefix', self.args.model_dir, False, self.args
            )
            self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        return self.sample_prefix(file_context, func_context, control, lang)

    def sample_prefix(self, file_context, func_context, control, lang):
        input_src = file_context + func_context
        
        try:
            if self.use_vllm:
                # Use vLLM for generation with prefix control
                sampling_params = self.get_vllm_sampling_params()
                prefix_tokens = None
                if hasattr(self, 'model') and hasattr(self.model, 'get_past_from_prefix'):
                    # Get prefix tokens if available
                    prefix_tokens = self.model.get_past_from_prefix([control])
                
                outputs = self.llm.generate(
                    [input_src], 
                    sampling_params,
                    prefix_tokens=prefix_tokens  # Pass prefix tokens if available
                )
                
                # Convert vLLM outputs to expected format
                gen_output = []
                input_ids_len = len(self.tokenizer.encode(input_src))
                
                for output in outputs:
                    tokens = output.outputs[0].token_ids
                    gen_output.append(tokens)
                
                gen_output = torch.tensor(gen_output)
                
            else:
                # Use standard generation
                input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                input_ids_len = input_ids.shape[1]

                gen_kwargs = {
                    'input_ids': input_ids,
                    'do_sample': True,
                    'num_return_sequences': self.args.num_gen,
                    'temperature': self.args.temp,
                    'max_new_tokens': self.args.max_gen_len,
                    'top_p': self.args.top_p,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'use_cache': True,
                    'control_id': control,
                }

                with torch.no_grad():
                    gen_output = self.model.generate(**gen_kwargs)
                gen_output = gen_output.to(device)

            return self.process_completions(input_src, input_ids_len, gen_output, lang)
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return [], [], [], []

class TextPromptEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.model_dir, False, self.args)
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        if lang == 'py':
            input_src = file_context + '# ' + PROMPTS[control] + func_context
        elif lang == 'c':
            input_src = file_context + '// ' + PROMPTS[control] + func_context
        else:
            raise NotImplementedError()
        input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]
        gen_output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.process_completions(input_src, input_ids_len, gen_output, lang)
