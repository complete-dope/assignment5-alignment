# this is my test grpo thing here I will build out from the lecture-17 on how the training works

# https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_17.json&step=82

import os
import sys
from typing import Callable
import math
import torch 
import torch.nn as nn 

from torch.nn import functional as F
from torch.nn.functional import softmax
from einops import einsum, rearrange, repeat



def simple_task():
    '''
    # task : sorting n numbers of a model
    
    prompt : [1,0,2]
    response : [0,1,2]

    '''
    pass


def sort_distance_reward(list_1:list[int], list_2:list[int]):
    # so list1 is the original list and list 2 is the one that we need to check against
    assert len(list_1) == len(list_2)

    return sum(1 for x,y in zip(list_2, sorted(list_1)) if x==y )


def sort_inclusion_ordering_rewarding(list_1, list_2):

    assert len(list_1) == len(list_2) ,'length mismatch of the lists '

    # find the no. of values that are matching from both the lists (that is if one of the list contains a number x , so I need to see if the secondlist is also containing that same number or not )


    inclusion_reward = sum(1 for x in list_1 if x in list_2) # this tells what sort of reward do we need to include in this 
    
    ordering_reward = sum(1 for x,y in zip(list_1, list_1[1:]) if x<=y ) # this check whether the 2 adjacent ones are in sorted order or not 

    return inclusion_reward + ordering_reward


def simple_model():

    '''
    assume fixed prompt and response length and captures per position params 
    captures positional information with separate per position parameter
    decode each position in the response independently 
    '''

    model = Model(embedding_dim = 10, vocab_size = 3, prompt_length = 3, response_length = 3)

    prompts = torch.tensor(data = [[1,0,2]]) # batch,pos

     
    pass


class Model(nn.Module):
    def __init__(self,embedding_dim:int, vocab_size:int, prompt_length:int, response_length :int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # so this the embedding table that requires grad = True  by default as we need to update that also along with our model 

        # encode the weights per position (this is a just a mock useless model )
        self.encode_weights = nn.Parameter(torch.randn(prompt_length, embedding_dim, embedding_dim) / torch.sqrt(embedding_dim)) ## this becomes, ((L,D,D) / (D**0.5)) # this is position wise encoding matrix

        # decode the weights per position
        self.decode_weights = nn.Parameter(torch.randn(response_length, embedding_dim, embedding_dim) / torch.sqrt(embedding_dim))

    def forward(self, prompts):
        embeddings = self.embedding(prompts)

        # Transform using per prompt position matrix, collapse into one vector
        encoded = einsum(embeddings, self.encode_weights, "batch pos dim1, pos dim1 dim2 -> batch dim2")

        # Turn into one vector per response position
        decoded = einsum(encoded, self.decode_weights, "batch dim2, pos dim2 dim1 -> batch pos dim1")

        #find out logits 
        logits = einsum(decoded, self.embedding.weight , 'batch pos dim1 , vocab dim1 -> batch pos vocab')

        return logits


from torch.nn import functional as F
from torch.nn.functional import softmax
from einops import einsum, rearrange, repeat

def generate_responses(prompts: torch.Tensor, model: Model, num_responses: int) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
    Returns:
        generated responses: int[batch trial pos]
    Example (batch_size = 3, prompt_length = 3, num_responses = 2, response_length = 4)
    p1 p1 p1 r1 r1 r1 r1
             r2 r2 r2 r2
    p2 p2 p2 r3 r3 r3 r3
             r4 r4 r4 r4
    p3 p3 p3 r5 r5 r5 r5
             r6 r6 r6 r6
    """
    logits = model(prompts)  # [batch pos vocab]
    batch_size = prompts.shape[0]


    # Sample num_responses (independently) for each [batch pos]
    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    
    flattened_responses = torch.multinomial(softmax(flattened_logits, dim=-1), num_samples=num_responses, replacement=True)  # [batch pos trial]
    
    responses = rearrange(flattened_responses, "(batch pos) trial -> batch trial pos", batch=batch_size)
    
    return responses

def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, model: Model) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        log_probs (float[batch trial pos]) under the model
    """
    # Compute log prob of responses under model
    logits = model(prompts)  # [batch pos vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch pos vocab]
    # Replicate to align with responses
    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", trial=num_responses)  # [batch trial pos vocab]
    # Index into log_probs using responses
    log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)  # [batch trial pos]
    return log_probs



def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:  # @inspect rewards
    """
    Args:
        rewards (float[batch trial])
    Returns:
        deltas (float[batch trial]) which are advantage-like quantities for updating
    """
    if mode == "rewards":
        return rewards
    if mode == "centered_rewards":
        # Compute mean over all the responses (trial) for each prompt (batch)
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        return centered_rewards
    if mode == "normalized_rewards":
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        std_rewards = rewards.std(dim=-1, keepdim=True)  # @inspect std_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        normalized_rewards = centered_rewards / (std_rewards + 1e-5)  # @inspect normalized_rewards
        return normalized_rewards
    if mode == "max_rewards":
        # Zero out any reward that isn't the maximum for each batch
        max_rewards = rewards.max(dim=-1, keepdim=True)[0]
        max_rewards = torch.where(rewards == max_rewards, rewards, torch.zeros_like(rewards))
        return max_rewards
    raise ValueError(f"Unknown mode: {mode}")




