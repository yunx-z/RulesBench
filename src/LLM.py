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
        "gemini-2.0-pro-exp-02-05" : {
            "input" : 0,
            "output" : 0,
            },
	}

openai_api_key = os.getenv('OPENAI_API_KEY1')
openai_api_base = os.getenv('AZURE_OPENAI_ENDPOINT1')
openai_client = openai.AzureOpenAI(
	azure_endpoint=openai_api_base,
	api_key=openai_api_key,
	api_version="2024-12-01-preview",
	)

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
vertexai.init(project=os.environ["GCP_PROJECT_ID"], location="us-central1")

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

def complete_text_gemini(prompt, stop_sequences=[], model="gemini-pro", max_tokens_to_sample = 8000, temperature=0.5, log_file=None, **kwargs):
    """ Call the gemini API to complete a prompt."""
    # Load the model
    _model = "gemini-2.0-pro-exp-02-05" if model == "gemini-exp-1206" else model
    gemini_model = GenerativeModel(_model)
    # Query the model
    parameters = {
            "temperature": temperature,
            "max_output_tokens": max_tokens_to_sample,
            "stop_sequences": stop_sequences,
            **kwargs
        }
    safety_settings = {
            harm_category: SafetySetting.HarmBlockThreshold(SafetySetting.HarmBlockThreshold.BLOCK_NONE)
            for harm_category in iter(HarmCategory)
        }
    response = gemini_model.generate_content( [prompt], generation_config=parameters, safety_settings=safety_settings)
    if "thinking" in model:
        print(response)
        thought = response.candidates[0].content.parts[0].text
        completion = response.candidates[0].content.parts[1].text
    else:
        thought = None
        completion = response.text
    num_prompt_tokens = response.usage_metadata.prompt_token_count
    num_sample_tokens = response.usage_metadata.candidates_token_count
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens, num_sample_tokens, thought=thought)
    return completion


MAX_RETRIES=10
WAIT_TIME=60
def complete_text(prompt, model, log_file=None, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    retry = 0
    error_msg = None
    while retry < MAX_RETRIES:
        retry += 1
        try:
            if model.startswith("gemini"):
                completion = complete_text_gemini(prompt, log_file=log_file, model=model, **kwargs)
            else:
                # use OpenAI API
                completion = complete_text_openai(prompt, log_file=log_file, model=model, **kwargs)
            return completion
        except Exception as e:
            print(f"{model} attempt {retry} failed!")
            error_msg = str(e)
            time.sleep(WAIT_TIME)

    raise ValueError(f"MAX_RETRIES exceeded: {error_msg}")



if __name__ == "__main__":
    prompt = "Hello World!"
    response = complete_text(prompt, model="gemini-2.0-pro-exp-02-05")
    print(response)
