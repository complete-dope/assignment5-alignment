Lecture 15

Having expert demonstrations , and how to adapt to it ( obvious answer is gradient descent) , non-obvious part (that is being used today) , 

Post training the dataset really matters 

FLAN dataset

Question : textual paragraph question , e2e ( entry dataset )
Answer : summary / One word 


Alpaca : response are in natural language and in a better way and are less divergence 

OpenAssistant : This is a nice dataset , created by internet enthusiasts

Length is still a issue in these types of datasets and are looked as a complexity of a task 

# Evals
In all these eval methods we see human / LLM bias getting creeped in and LLM and humans have a huge preference for lists and LLM have huge effect for the length (they love length)  !


## Preparing your Post-Training dataset 

### SFT 

Common pitfalls: high quality datasets and it should have lot of citations and references and should be super detailed 
So examples like these teaches the model to learn about citations and length , and also its teaching model to hallucinate (if citation is not there go out and produce something / hallucinate out of thin air)

Model doesnt have knowledge of answering a question you force it to answer it (using your pretrained dataset), this makes it learn knowledge in abstract sense and also learn other aspects of I just need to make thing up to type check what the response looks like

If the SFT data is more advanced to what the model can actually do , there is a risk of teaching the model of this shortcut behaviour , instead of teaching the right behaviour 

This is a reason why on-Policy RL is a important thing to do 

You have to be really really careful about distillation data and human annotation data 

Optimizing models at SFT level to not hallucinate is quite challenging 

500 examples are all required to make safety jails for a model ( eg: how to kill someone vs how to kill a python process)


### RLHF 

Here we am not trying to matching with some distribution, its not trying to match with some distribution  
Annotate data for Reward model then make that model learn, then use RL to improve the policy model 

PairWise Feedback Data
Input, output1 , output2 , then choose which one is better 

SO I need to tell the model what are my maximisation process is and what the rewards are 

objective is defined as how we will do these things, 
policy here refers to as the model and we are trying to optimize reward of a policy
reward refers to what we are sending to the model 


PPO is complex so people shifted to DPO that is quite easy to do and understand also  
DPO is '<see-maths>'

Most of the settings in RL is just defined on the specific settings

Overfitting on rewarded model, mode-collapse / entropy 

