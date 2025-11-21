# making / creating a math baseline model using gsm8k model 
# https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl

TRAIN_DATASET_LINK = 'data/gsm8k/train.jsonl'

import os 
import sys
import ast
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Callable, List
from vllm import LLM, SamplingParams
from utils.utils import save_to_jsonl_file_in_streaming_manner

train_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), TRAIN_DATASET_LINK)

# read a jsonl file line-by-line
with open(train_file_path, 'r') as f:
    data =f.readlines()


# prompt r1-zero 
def prompt_template(val:str):
    prompt = '''
    A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: %s
    Assistant: <think>
    '''
    return prompt % val

dataset = []
answer_points = []
for data_point in data:
    data_point = ast.literal_eval(data_point) 
    question = data_point.get('question')
    answer = data_point.get('answer')
    data_point = prompt_template(question)
    dataset.append(data_point)

# dataset has the whole dataset now to use for the model !!


sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

MODEL_ID = 'Qwen/Qwen2.5-Math-1.5B'

# Create an LLM.
llm = LLM(model=MODEL_ID, max_num_seqs=1, enforce_eager=True)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    pass


# loop 
for idx,prompt in enumerate(dataset):
    output = llm.generate(prompt, sampling_params)
    print(f'{idx}th output is {output}')
    
    # now evaluate and save 
    eval_answer = evaluate_vllm()

    store_response = {'question':prompt , 'output':output, 'ground_truth':answer_points[idx], 'eval':eval_answer} 

    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/math_baseline_output.jsonl')
    save_to_jsonl_file_in_streaming_manner(file_path = log_path, data = store_response, to_continue = True)
    



