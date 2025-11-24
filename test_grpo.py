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

    torch.manual_seed(10)
    responses = generate_responses(prompts)

    rewards = compute_reward(prompt=prompts, responses=responses, reward_fn = sort_inclusion_ordering_rewarding) # list[list[int]]

    # now find the delta ( that is how much to take these rewards ... how much of these rewards to take)
    # this delta term is called as temporal difference error aka advantage
    # delta is element of suprise , okay reward is this but how much to take for that reward ?

    deltas = compute_deltas(rewards, mode = 'normalized_rewards') # this is the one being used by grpo paper also paper also and is very easy to maintain 


    # compute log probs for these responses
    log_probs = compute_log_probs(prompts, responses, model) 

    loss = compute_loss(log_probs , deltas, mode ='naive')

    freezing_parameters()


def freezing_parameters():
    # Motivation: in GRPO you'll see ratios: p(a | s) / p_old(a | s)
    # When you're optimizing, it is important to freeze and not differentiate through p_old
    w = torch.tensor(2., requires_grad=True)
    p = torch.nn.Sigmoid()(w)
    p_old = torch.nn.Sigmoid()(w)
    ratio = p / p_old
    ratio.backward()
    grad = w.grad  # @inspect grad
    
    # Do it properly:
    w = torch.tensor(2., requires_grad=True)
    p = torch.nn.Sigmoid()(w)
    with torch.no_grad():  # Important: treat p_old as a constant!
        p_old = torch.nn.Sigmoid()(w)
    ratio = p / p_old
    ratio.backward()
    grad = w.grad  # @inspect grad


def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean() # multiply then together and then find the mean out from this 

        # deltas get broadcasted to each position, this is done for the outcome based estimation
        # for process rewarding the deltas are broadcasted till the new process (step) 

    if mode == "unclipped":
        ratios = log_probs / old_log_probs  # [batch trial]
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()
    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = log_probs / old_log_probs  # [batch trial]
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()
    raise ValueError(f"Unknown mode: {mode}")



def sort_inclusion_ordering_reward(prompt: list[int], response: list[int]) -> float:  # @inspect prompt, @inspect response
    """
    Return how close response is to ground_truth = sorted(prompt).
    """
    assert len(prompt) == len(response)
    # Give one point for each token in the prompt that shows up in the response
    inclusion_reward = sum(1 for x in prompt if x in response)  # @inspect inclusion_reward
    # Give one point for each adjacent pair in response that's sorted
    ordering_reward = sum(1 for x, y in zip(response, response[1:]) if x <= y)  # @inspect ordering_reward
    return inclusion_reward + ordering_reward

def compute_reward(prompts: torch.Tensor, responses: torch.Tensor, reward_fn: Callable[[list[int], list[int]], float]) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        rewards (float[batch trial])
    """
    batch_size, num_responses, _ = responses.shape
    rewards = torch.empty(batch_size, num_responses, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(num_responses):
            rewards[i, j] = reward_fn(prompts[i, :], responses[i, j, :])
    return rewards



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




