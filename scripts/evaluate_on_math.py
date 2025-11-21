# evaluate a model on math dataset

from vllm import LLM , SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

MODEL_ID = 'Qwen/Qwen2.5-Math-1.5B'

# Create an LLM.
llm = LLM(model=MODEL_ID,max_model_len=200, max_num_seqs=1,enforce_eager=True)


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)


# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



'''
Outputs :

Prompt: 'Hello, my name is', Generated text: " Kailas. I'm having trouble proving the following statement: A ⊕ B = A ∧ ¬B ∧ ¬(¬A ∧ B). I've attached my work, but I'm struggling to progress further. To prove the given statement, we can start by expanding the left-hand side using the definition of symmetric difference, which is A ⊕ B = (A ∪ B) - (A ∩ B). This can be further broken down into (A ∪ B) ∩ (¬A ∪ ¬B)."

Prompt: 'The president of the United States is', Generated text: ' a single-authored publication. Over the years, many of his books have been co-authored with mathematicians. Consider the following scenarios:'

Prompt: 'The capital of France is', Generated text: ' invested 200019,482 euros. Express 200019,482 in scientific notation.'

Prompt: 'The future o   f AI is', Generated text: ' dominated by transforming the "Turing-complete" problem domain into a "human-understandable" and physically realizable and understandable task domain.'

'''