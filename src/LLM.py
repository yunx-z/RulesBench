import openai
import os
import json

MODEL2PRICE = {
        "gpt-4o" : {
            "input" : 2.5 / 1e6,
            "output" : 10 / 1e6,
            },
        "gpt-4o-mini" : {
            "input" : 0.15 / 1e6,
            "output" : 0.6 / 1e6,
            },
        "o1-mini" : {
            "input" : 3 / 1e6,
            "output" : 12 / 1e6,
            },
        "o3-mini" : {
            "input" : 1.1 / 1e6,
            "output" : 4.4 / 1e6,
            },
        "o1-preview" : {
            "input" : 15 / 1e6,
            "output" : 60 / 1e6,
            },
        "o1" : {
            "input" : 15 / 1e6,
            "output" : 60 / 1e6,
            },
	}

openai_api_key = os.getenv('OPENAI_API_KEY1')
openai_api_base = os.getenv('AZURE_OPENAI_ENDPOINT1')
openai_client = openai.AzureOpenAI(
	azure_endpoint=openai_api_base,
	api_key=openai_api_key,
	api_version="2024-12-01-preview",
	)

def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens, num_sample_tokens, thought=None):
    """ Log the prompt and completion to a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{prompt}")
        if thought:
            f.write(f"\n==================={model} thought ({max_tokens_to_sample})=====================\n")
            f.write(f"{thought}")
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")

    LOG_DIR = os.getenv("LOG_DIR", "logs/")
    cost_file = os.path.join(LOG_DIR, "api_cost.json")
    content = dict()
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as reader:
            content = json.load(reader)

    curr_cost = num_prompt_tokens * MODEL2PRICE[model]["input"] + num_sample_tokens * MODEL2PRICE[model]["output"] 
    with open(cost_file, 'w') as writer:
        updated_content = {
                "total_cost" : content.get("total_cost", 0) + curr_cost,
                "total_num_prompt_tokens" : content.get("total_num_prompt_tokens", 0) + num_prompt_tokens,
                "total_num_sample_tokens" : content.get("total_num_sample_tokens", 0) + num_sample_tokens,
                }
        json.dump(updated_content, writer, indent=2)

def complete_text_openai(prompt, stop_sequences=[], model="gpt-4o-mini", max_tokens_to_sample=4000, temperature=0.5, log_file="logs/run.log", **kwargs):
    """ Call the OpenAI API to complete a prompt."""
    if "o1" in model.lower() or "o3" in model.lower():
        raw_request = {
              "model": model,
              "max_completion_tokens": 64000 if model.lower() == "o1-mini" else 100000,
              **kwargs
        }
        if model.lower() in ["o1", "o3-mini"]:
            raw_request["reasoning_effort"] = "medium"
    else:
        raw_request = {
              "model": model,
              "temperature": temperature,
              "max_tokens": max_tokens_to_sample,
              "stop": stop_sequences or None,  # API doesn't like empty list
              **kwargs
        }

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(**{"messages": messages,**raw_request})
    completion = response.choices[0].message.content
    usage = response.usage

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens=usage.prompt_tokens, num_sample_tokens=usage.completion_tokens)
    return completion

if __name__ == "__main__":
    prompt = "Hello World!"
    response = complete_text_openai(prompt, model="gpt-4o-mini")
    print(response)
