import itertools
import random
import time
from functools import partial
from pathlib import Path
from typing import List, Union, Optional

import datasets
import einops
import numpy as np
import pandas as pd
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm
import transformer_lens
import transformer_lens.utils as utils
from fancy_einsum import einsum
from jaxtyping import Float, Int
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from IPython.display import HTML
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
) 




class glitchprober:
    
    def __init__(self, 
             model_name:str, 
             gemma = 0.1
             ):
        
        self.model_name = model_name
        self.gemma = gemma
        if 'Llama-2-7b-chat' in self.model_name:
            self.trlens_model_type = "meta-llama/Llama-2-7b-chat-hf"
            self.model_type = 'Llama-2-7b-chat'
            self.Key_layers = range(19, 29)
            self.layer_num = 32
            self.k_1 = 25
            self.b_1 = 0.5
            self.k_2 = 2
            self.b_2 = 0
        elif 'Mistral-7B-Instruct-v0.1' in self.model_name:
            self.trlens_model_type = "mistralai/Mistral-7B-Instruct-v0.1"
            self.model_type = 'Mistral-7B-Instruct-v0.1'
            self.Key_layers = range(19, 29)
            self.layer_num = 32
            self.k_1 = 25
            self.b_1 = 0.5
            self.k_2 = 2
            self.b_2 = 0
        elif 'Qwen-7B-Chat' in self.model_name:
            self.trlens_model_type = "Qwen/Qwen-7B"
            self.model_type = 'Qwen-7B-Chat'
            self.Key_layers = range(19, 29)
            self.layer_num = 32
            self.k_1 = 25
            self.b_1 = 0.5
            self.k_2 = 2
            self.b_2 = 0
        elif 'gemma-2b-it' in self.model_name:
            self.trlens_model_type = "google/gemma-2b-it"
            self.model_type = 'gemma-2b-it'
            self.Key_layers = range(5, 15)
            self.layer_num = 18
            self.k_1 = 25
            self.b_1 = 0.5
            self.k_2 = 2
            self.b_2 = 0
        elif 'Yi-6B-Chat' in self.model_name:
            self.trlens_model_type = "01-ai/Yi-6B-Chat"
            self.model_type = 'Yi-6B-Chat'
            self.Key_layers = range(19, 29)
            self.layer_num = 32
            self.k_1 = 25
            self.b_1 = 0.5
            self.k_2 = 2
            self.b_2 = 0
            

#################################################################################################### 

# Insert any model you want to test here

# Remember to adjust parameters in Transformer_lens.HookedTransformer.from_pretrained()

# Some models may cause errors from Transformer_lens.HookedTransformer, make sure that your model is supported by Transformer_lens or put in right parameters in HookedTransformer.from_pretrained() 

####################################################################################################            
            
        else:
            warnings.warn('Unsupported Model! Try to Processed as Llama!')
            self.model_type = "meta-llama/Llama-2-7b-chat-hf"
            
    
    def __model_to_cuda(self, num, tot_num):
        device = torch.device('cuda:' + str(num))
        try:
            self.model = self.model.to(device)
            return 'cuda:' + str(num)
        except:
            if num == tot_num - 1:
                self.model = self.model.to('cpu')
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                warnings.warn('GPU is not available!')
                return 'cpu'
            else:
                self.model = self.model.to('cpu')
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                self.model = self.model.to('cpu')
                return self.__model_to_cuda(num + 1, tot_num)
                
                
    def load_model(self):
        
        if 'Qwen' in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
            self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        elif 'Llama' in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.hf_model = LlamaForCausalLM.from_pretrained(self.model_name)
        elif 'Yi' in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.vocab_size = self.tokenizer.vocab_size
        self.model = HookedTransformer.from_pretrained(self.trlens_model_type, 
                                          hf_model = self.hf_model, 
                                          device='cpu', 
                                          fold_ln=False, 
                                          center_writing_weights=False, 
                                          center_unembed=False, 
                                          tokenizer = self.tokenizer
                                         )
        pynvml.nvmlInit()
        gpu_device_count = pynvml.nvmlDeviceGetCount()
        if torch.cuda.is_available():
            self.device = torch.device(self.__model_to_cuda(0, gpu_device_count))
        self.model.eval()
                            
                            
    
    def repetive_judge(self, token_id):
        token = self.tokenizer.decode([token_id])
        string_to_repeat = token
        if 'Yi' in self.model_name:
#             token = self.tokenizer.decode([token_id])
            content = f"Can you repeat the character '{token}' and return back to me?"
            messages = [{"role": "user", "content": content}]
            k = len(content) + 38
            input_ids = self.tokenizer.apply_chat_template(
                    conversation=messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors='pt'
                )
            output_ids = model.generate(input_ids, temperature=0, verbose=False, return_type='str')
            response = output_ids[k:]
            if token in response or token.upper() in response.upper():
                return True
            else:
                return False

        else:
            if 'Llama-2'in self.model_name:
                text1 = "Question: Can you repeat the string '"
                text2 = "and return back to me?\nAnswer: Here is the repeated string:\n"
                tokens1 = torch.tensor(self.tokenizer.encode(text1))
                tokens2 = torch.tensor(self.tokenizer.encode(text2))
                tokens = torch.cat((tokens1, torch.tensor([token_id]), torch.tensor([29915]), tokens2[1:]), dim=0).to(self.device)

            else:
                text1 = "Question: Can you repeat the string '"
                text2 = "' and return back to me?\nAnswer: Here is the repeated string:\n"
                tokens1 = torch.tensor(self.tokenizer.encode(text1))
                tokens2 = torch.tensor(self.tokenizer.encode(text2))
                tokens = torch.cat((tokens1, torch.tensor([token_id]), tokens2[1:]), dim=0).to(self.device)
            text = f"Question: Can you repeat the string '{token}' and return back to me?\n\nAnswer: Here is the repeated string:"
            k = len(text)
            tokens = torch.unsqueeze(tokens, dim=0)

#             response = self.__generate_response(text, tokens)
            response = self.model.generate(tokens, max_new_tokens=10, temperature=0, verbose=False, return_type='str')[k:]
            if string_to_repeat in response or string_to_repeat.upper() in response.upper():
                return True
            else:
                return False
            
            

    def pca_tensor(dataMat, topNfeat=9999999):
        dataMat_np = dataMat.numpy()
        n_components=topNfeat if topNfeat is not None else dataMat.shape[1]
        pca = PCA(n_components = n_components)
        lowDDataMat_np = pca.fit_transform(dataMat_np)
        lowDDataMat = torch.tensor(lowDDataMat_np, dtype=dataMat.dtype)
        return lowDDataMat

            
    def detect_glitch_token(self,
                 P_pca_rate = 75,
                 SVM_para_C = 0.5,
                 SVM_para_degree = 3):
        self.pca_rate = P_pca_rate
        self.SVM_para_C = SVM_para_C
        self.SVM_para_degree = SVM_para_degree

        token_data = torch.load('caches/' + self.model_type + '-token_data-pca75.pt')
        token_ids = np.array(torch.load('caches/' + self.model_type + '-token_ids.pt'))
#         print(type(token_ids))


#         shuffle
        indices = torch.randperm(len(token_ids)).numpy()
        shuffled_token_ids = token_ids[indices]
        shuffled_token_data = token_data[indices]

        train_size = int(len(token_ids) * self.gemma)
        predict_size = len(token_ids) - train_size
        train_tokens = shuffled_token_ids[:train_size]
        predict_tokens = shuffled_token_ids[train_size:]
        train_data, predict_data = shuffled_token_data.split([train_size, predict_size])
        train_labels = []
        for t in train_tokens:
            if self.repetive_judge(t):
                train_labels.append(0)
            else:
                train_labels.append(1)
        print(f"Train Data Shape = {train_data.shape} Predict Data Shape = {predict_data.shape}")
        
        
        predict_model = SVC(C=self.SVM_para_C, kernel='poly', degree=self.SVM_para_degree, class_weight='balanced', probability=True)
        predict_model.fit(train_data, train_labels)
        predictions = predict_model.predict(predict_data)

#         Post Process
        start_time = time.time()
        modified_count = 0 
        print(f"Total Predictions:{len(predictions)}")

        for idx, pred in enumerate(predictions):
            if pred == 1:
                if self.repetive_judge(predict_tokens[idx]):
                    predictions[idx] = 0
                    modified_count += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Post Process Modified count:{modified_count} Execution Time: {execution_time:.2f} 秒")

        predict_labels = []
        df = pd.read_csv('GroundTruth/' + self.model_type + '-glitch-tokens.csv')
        glitch_tokens = df['index'].tolist()
        for t in predict_tokens:
            if t in glitch_tokens:
                predict_labels.append(1)
            else:
                predict_labels.append(0)
        accuracy = accuracy_score(predict_labels, predictions)
        
        print(f"Performance with C={self.SVM_para_C} and degree={self.SVM_para_degree}:")
        print(f"Accuracy: {accuracy}")
        print(classification_report(predict_labels, predictions))


    def fix_glitch_token(self):
        df = pd.read_csv('GroundTruth/'+ self.model_type + '-glitch-tokens.csv')
        glitch_tokens = df['index'].tolist()
        g_len = len(glitch_tokens)
        sample_size = g_len

        mlp_pre_layer_dim = torch.load('caches/' + self.model_type + '-fix_mlp_pre_layer_dim.pt')
        mlp_pre_linear_layer_dim = torch.load('caches/' + self.model_type + '-fix_mlp_pre_linear_layer_dim.pt')
        mlp_pre_layer_dim_non = torch.load('caches/' + self.model_type + '-fix_mlp_pre_layer_dim_non.pt')
        mlp_pre_linear_layer_dim_non = torch.load('caches/' + self.model_type + '-fix_mlp_pre_linear_layer_dim_non.pt')

        mlp_pre_indices = torch.load('caches/' + self.model_type + '-fix_mlp_pre_indices.pt')
        mlp_pre_linear_indices = torch.load('caches/' + self.model_type + '-fix_mlp_pre_linear_indices.pt')
        mlp_pre_indices_non = torch.load('caches/' + self.model_type + '-fix_mlp_pre_indices_non.pt')
        mlp_pre_linear_indices_non = torch.load('caches/' + self.model_type + '-fix_mlp_pre_linear_indices_non.pt')
        def extract_activations(caches, layer_dims, key):
            activations = []
            for layer in range(self.Key_layers):
                neuron_indices = layer_dims[1][layer_dims[0] == layer]
                if neuron_indices.size > 0:
                    layer_activations = []
                    for sample in caches:
                        sample_layer_data = sample[layer][key] 
                        selected_neurons = [sample_layer_data[neuron] for neuron in neuron_indices] 
                        layer_activations.append(selected_neurons)
                    activations.append(np.array(layer_activations))
            return activations

        def compute_activation_difference(normal_activations, glitch_activations):
            differences = []
            layer = layer_start
            for normal_layer, glitch_layer in zip(normal_activations, glitch_activations):
                normal_mean = np.mean(normal_layer, axis=0)
                # print(f"layer = {layer} normal_mean = {normal_mean}")
                glitch_mean = np.mean(glitch_layer, axis=0)
                # print(f"layer = {layer} glitch_mean = {glitch_mean}")
                differences.append(np.mean(normal_mean - glitch_mean))
                layer+=1
            return np.mean(differences)
 
        def compute_activation_non_difference(normal_activations, glitch_activations):
            differences = []
            for normal_layer, glitch_layer in zip(normal_activations, glitch_activations):
                normal_mean = np.mean(normal_layer, axis=0)
                glitch_mean = np.mean(glitch_layer, axis=0)
                differences.append(np.mean(glitch_mean / normal_mean))
            return np.mean(differences)

        def determine_beta(difference):
            return max(0, min(3, self.b_1 + difference * self.k_1))
        def determine_alpha(difference):
            return min(5, max(2, self.b_2 + abs(difference) * self.k_2))


        average_difference_mlp_pre = torch.load('caches/' + self.model_type + '-fix_average_difference_mlp_pre.pt')
        average_difference_mlp_pre_linear = torch.load('caches/' + self.model_type + '-fix_average_difference_mlp_pre_linear.pt')
        average_difference_mlp_pre_non = torch.load('caches/' + self.model_type + '-fix_average_difference_mlp_pre_non.pt')
        average_difference_mlp_pre_linear_non = torch.load('caches/' + self.model_type + '-fix_average_difference_mlp_pre_linear_non.pt')

        alpha1 = determine_alpha(average_difference_mlp_pre_non)
        print("Computed alpha1:", alpha1)
        alpha2 = determine_alpha(average_difference_mlp_pre_linear_non)
        print("Computed alpha2:", alpha2)
        beta1 = determine_beta(average_difference_mlp_pre)
        print("Computed beta1:", beta1)
        beta2 = determine_beta(average_difference_mlp_pre_linear)
        print("Computed beta2:", beta2)


        def generate_fix_response(tokens, times, max_new_tokens=10):
            # alpha = 5
            # beta = 2 
            start_layer = 19
            end_layer = 28

            def mlp_pre_hook_layer(
                value: Float[torch.Tensor, "batch pos d_mlp"],
                hook: HookPoint,
                layer: int
            ) -> Float[torch.Tensor, "batch pos d_mlp"]:
                #print(f"Shape of the value tensor: {value.shape}")
                array_ = np.array(value[0][-1].cpu())
                glitch_dim = []
                while True:
                    if array_[array_.argmax()] > 1:
                        glitch_dim.append(array_.argmax())
                        array_[array_.argmax()] = -1000
                    else:
                        break
                if layer <= max(mlp_pre_layer_dim_non[0]):
                    current_layer_indices1 = mlp_pre_layer_dim_non[1][mlp_pre_indices_non[layer]:mlp_pre_indices_non[layer+1]]
                    glitch_more = list(set(glitch_dim).intersection(current_layer_indices1))
#                     / alpha 1
                    value[:, -1, glitch_more]  /= alpha1

#                 Calculate the difference set to find indices not in glitch_dim
                if layer <= max(mlp_pre_layer_dim[0]):
                    current_layer_indices2 = mlp_pre_layer_dim[1][mlp_pre_indices[layer]:mlp_pre_indices[layer+1]]
                    normal_more = list(set(current_layer_indices2).difference(glitch_dim))
#                     + beta 1
                    value[:, -1, mlp_pre_layer_dim[1][mlp_pre_indices[layer]:mlp_pre_indices[layer+1]]]  = torch.abs(value[:, -1, mlp_pre_layer_dim[1][mlp_pre_indices[layer]:mlp_pre_indices[layer+1]]]) + beta1       

                return value

            def mlp_pre_linear_hook_layer(
                value: Float[torch.Tensor, "batch pos d_mlp"],
                hook: HookPoint,
                layer: int
            ) -> Float[torch.Tensor, "batch pos d_mlp"]:
                array_ = np.array(value[0][-1].cpu())
                glitch_dim = []
                while True:
                    if array_[array_.argmax()] > 1:
                        glitch_dim.append(array_.argmax())
                        array_[array_.argmax()] = -1000
                    else:
                        break
                #print(layer)
#                 Calculate the difference set to find indices not in glitch_dim
                if layer <= max(mlp_pre_linear_layer_dim_non[0]):  
                    current_layer_linear_indices1 = mlp_pre_linear_layer_dim_non[1][mlp_pre_linear_indices_non[layer]:mlp_pre_linear_indices_non[layer+1]]
                    glitch_more = list(set(glitch_dim).intersection(current_layer_linear_indices1))
#                     / alpha2
                    value[:, -1, glitch_more] /= alpha2 
                if layer <= max(mlp_pre_linear_layer_dim[0]): 
                    current_layer_indices2 = mlp_pre_linear_layer_dim[1][mlp_pre_linear_indices[layer]:mlp_pre_linear_indices[layer+1]]
                    normal_more = list(set(current_layer_indices2).difference(glitch_dim))
#                     + beta2
                    value[:, -1, mlp_pre_linear_layer_dim[1][mlp_pre_linear_indices[layer]:mlp_pre_linear_indices[layer+1]]] = torch.abs(value[:, -1, mlp_pre_linear_layer_dim[1][mlp_pre_linear_indices[layer]:mlp_pre_linear_indices[layer+1]]]) + beta2
                return value
            logits, cache = self.model.run_with_cache(tokens)
            response_tokens = []
            fwd_hooks = []
            func_list = []
            for layer in self.Key_layers:
                temp_hook_fn = partial(mlp_pre_hook_layer, layer=layer)
                fwd_hooks.append((f'blocks.{layer}.mlp.hook_pre', temp_hook_fn))
                temp_hook_fn_ = partial(mlp_pre_linear_hook_layer, layer=layer)
                fwd_hooks.append((f'blocks.{layer}.mlp.hook_pre_linear', temp_hook_fn_))

#             Re-run the modified logits using a hook
            for i in range(max_new_tokens):
                aug_logits = self.model.run_with_hooks(tokens, 
                                 return_type='logits',
                                 fwd_hooks=fwd_hooks)[0][-1]

                tokens = torch.cat((tokens[0], torch.tensor([aug_logits.argmax()]).to(device)),dim=0)
                tokens = torch.unsqueeze(tokens, dim=0)
                response_tokens.append(aug_logits.argmax())
                if aug_logits.argmax() == 2:
                    break
            return self.model.to_string(torch.tensor(response_tokens))

        remain_glitch = {'index':[], 'token':[]}
        path = self.model_type + '-glitch_fix-new.csv'
        for token_id in glitch_tokens:
            token = self.tokenizer.decode([token_id])
            token_id = torch.tensor([token_id])
            text1 = "Question: Can you repeat the string '"
            text2 = "and return back to me?\n\nAnswer: Here is the repeated string:\n\n"
            tokens1 = self.model.to_tokens(text1)[0]
            tokens2 = self.model.to_tokens(text2)[0]
            tokens = torch.cat((tokens1, token_id.to(device), torch.tensor([29915]).to(self.device), tokens2[1:]), dim=0)
            tokens = torch.unsqueeze(tokens, dim=0)
            response = generate_fix_response(tokens, times = 5, max_new_tokens=10)

            if token in response or token.upper() in response.upper():
                None
            else:
                remain_glitch['index'].append(token_id)
                remain_glitch['token'].append(token)
            if token_id%100 == 0:
#                     print(token_id, len(remain_glitch['index']))
                df_ = pd.DataFrame(remain_glitch)
                df_.to_csv(path, mode='w', escapechar=',')

        df_ = pd.DataFrame(remain_glitch)
        df_.to_csv(path, escapechar=',')
        Repaired = len(df_['index'].tolist()) - len(glitch_tokens)
        print(f"Repaired glitch token: {Repaired}")
        print(f"Repaired glitch token: {Repaired / len(glitch_tokens)}.4f")

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

 
            
            
            
            