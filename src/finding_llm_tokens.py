from llama_cpp import Llama
import json

mistral_model_path = "AdaptableLLM/src/mistral-7b-instruct-v0.2.Q6_K.gguf"

llm = Llama(mistral_model_path)

# Define the prompt string
prompts = ['Hello there',  'How are you doing on this fine evening?', 'Have you ever been to a christmas party?']

def streamed_tokens():
    output = ""
    # Tokenize the prompt using the same tokenizer as the framework
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8", errors="ignore")))

    response = llm(prompt, temperature=0, max_tokens=256, stream=True)
    for res in response:
        if res["choices"][0]["text"]:
            text = res["choices"][0]["text"]
            output += text
            print(text, end="", flush=True)
            if res["choices"][0]["finish_reason"] == "stop":
                break
    
    
    completion_tokens = len(llm.tokenize(output.encode("utf-8", errors="ignore")))
    results = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens 
    }
    print(results)
    return results


def non_streamed_tokens():
    response = llm(prompt, temperature=0, max_tokens=256)
    print(response["choices"][0]["text"])
    print(response["usage"])
    return response["usage"]


token_comparison = {}
for i, prompt in enumerate(prompts):
    streamed = streamed_tokens() # hacky way of calculating
    non_streamed = non_streamed_tokens() # accurate (recording from the llama library)
    token_comparison[i] = { 'streamed': streamed, 'non_streamed': non_streamed }

with open('token_comparison.json', "w") as file:
    json.dump(token_comparison, file, indent=4)