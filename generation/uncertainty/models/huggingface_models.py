"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter
import torch
import numpy as np

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
from scipy.stats import entropy

from .base_model import BaseModel
from .base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        self.triggered_index = None
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                # match = stop in generation
                match_pos = generation.find(stop)
                if match_pos != -1:
                    match_token_len = len(self.tokenizer.encode(generation[:match_pos]))
                    self.triggered_index = self.initial_length + match_token_len
                    match = True
                else:
                    match = False
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                if torch.equal(stop, input_ids[0][-len(stop):]):
                    self.triggered_index = input_ids.shape[1]
                    match = True
                else:
                    match = False
            else:
                raise
            if match:
                return True
        return False



class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        print(model_name)
        
        if max_new_tokens is None:
            raise ValueError('max_new_tokens must be set.')
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        if 'llama' in model_name.lower():
            if model_name.endswith('-8bit') or '70b' in model_name.lower():
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="auto",
                token_type_ids=None)
            
            if ('7b' in model_name.lower() or '13b' in model_name.lower() \
                or '8b' in model_name.lower()) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto",
                    **kwargs)
                
            
        elif 'mistral' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            if model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-4bit')]
            else:
                kwargs = {}
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                low_cpu_mem_usage=True,
                **kwargs,
            )

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]

    def tokenization(self, input_data):
        inputs = self.tokenizer(input_data, return_tensors="pt").input_ids[0]
        return inputs

    def predict(self, input_data, temperature, return_layers):
        # Implement prediction.
        inputs = self.tokenizer(input_data, return_tensors="pt").to(self.model.device)
        input_len = inputs['input_ids'].shape[-1] # shape: 1, input_token_length

        pad_token_id = self.tokenizer.eos_token_id
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaSub(
                stops=self.stop_sequences,
                tokenizer=self.tokenizer,
                initial_length=len(inputs['input_ids'][0]))
            stopping = StoppingCriteriaList([stopping_criteria])
        else:
            stopping_criteria = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.2,
                # stopping_criteria=stopping_criteria,
                stopping_criteria=stopping,
                pad_token_id=pad_token_id,
            )

        # outputs.keys() = ['sequences', 'scores', 'hidden_states', 'past_key_values']        
        # generate answers
        cut_token = outputs.sequences[0][input_len:]
        if self.stop_sequences is not None:
            if stopping_criteria.triggered_index:
                cutoff_idx = stopping_criteria.triggered_index
                cut_token = outputs.sequences[0][input_len:cutoff_idx]
        
        sliced_answer = self.tokenizer.decode(
            cut_token, skip_special_tokens=True).strip()
        

        n_generated = len(cut_token)
        if n_generated == 0:
            print("Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.")
            print("=="*50)
            # sliced_answer, cut_token, log_likelihoods, scores_entropy, hidden_input, hidden_output = 0,0,0,0,0,0
            return 0,0,0,0,0,0
        
        else:

            # Get tokens' embeddings.


            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states 
            
            hidden_states = hidden[:n_generated]
            # extract hidden_states from higher layers

            n_layers = len(hidden_states[0])
            min_layers = int(return_layers.split(',')[0])
            interval = int(return_layers.split(',')[1])
            extracted_layers = np.arange(min_layers,n_layers,interval).tolist()

            hidden_output = []  
            for i in range(len(hidden_states)):
                if i == 0:
                    hidden_input = torch.cat([hidden_states[0][n] for n in extracted_layers]) 
                    hidden_input = hidden_input.transpose(0,1) 
                    hidden_output.append(hidden_input[-1])
                    hidden_input = hidden_input[:-1]
                else:
                    hidden_ = torch.cat([hidden_states[i][n][:,-1,:] for n in extracted_layers]) 
                    hidden_output.append(hidden_)
            hidden_output = torch.stack(hidden_output) #[n_tokens, n_layers, hidden_dim]
            print(f"embedding's shape: hidden_input: {hidden_input.shape}, hidden_output: {hidden_output.shape}")
            assert hidden_output.shape[1] == len(extracted_layers) and hidden_output.shape[0] == n_generated, 'Hidden states shape does not match'

            # Get log_likelihoods.

            scores = torch.cat(outputs["scores"][:n_generated], dim=0).to('cpu') # (n_generated_tokens, vocab_size)
            scores_softmax = torch.nn.functional.softmax(scores, dim=-1) # (n_generated_tokens, vocab_size)
            scores_entropy = entropy(scores_softmax, axis=-1) # (n_generated_tokens)
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True).to('cpu')
            # Transition_scores.shape = (bs, n_generated_tokens)

            log_likelihoods = [score.item() for score in transition_scores[0]]
            if len(log_likelihoods) == 1:
                print('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                log_likelihoods = log_likelihoods[:n_generated]

            if len(log_likelihoods) == self.max_new_tokens:
                print('Generation interrupted by max_token limit.')

            if len(log_likelihoods) == 0:
                raise ValueError

            return sliced_answer, cut_token, log_likelihoods, scores_entropy, hidden_input, hidden_output