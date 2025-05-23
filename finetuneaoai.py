# Run preliminary checks

import json
import os
import requests
import tiktoken
import numpy as np
from collections import defaultdict
from openai import AzureOpenAI
from IPython.display import clear_output
import time

encoding = tiktoken.get_encoding("o200k_base") # default encoding for gpt-4o models. This requires the latest version of tiktoken to be installed.

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),
  api_version = "2025-02-01-preview"  
)

def Checkdata():
    # Check if the required environment variables are set
    try:
        # Load the training set
        with open('training_set.jsonl', 'r', encoding='utf-8') as f:
            training_dataset = [json.loads(line) for line in f]

        # Training dataset stats
        print("Number of examples in training set:", len(training_dataset))
        print("First example in training set:")
        for message in training_dataset[0]["messages"]:
            print(message)

        # Load the validation set
        with open('validation_set.jsonl', 'r', encoding='utf-8') as f:
            validation_dataset = [json.loads(line) for line in f]

        # Validation dataset stats
        print("\nNumber of examples in validation set:", len(validation_dataset))
        print("First example in validation set:")
        for message in validation_dataset[0]["messages"]:
            print(message)
        

    except AssertionError as e:
        print(f"Environment variable check failed: {e}")
        return


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

def processtoken():
    files = ['training_set.jsonl', 'validation_set.jsonl']

    for file in files:
        print(f"Processing file: {file}")
        with open(file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        total_tokens = []
        assistant_tokens = []

        for ex in dataset:
            messages = ex.get("messages", {})
            total_tokens.append(num_tokens_from_messages(messages))
            assistant_tokens.append(num_assistant_tokens_from_messages(messages))

        print_distribution(total_tokens, "total tokens")
        print_distribution(assistant_tokens, "assistant tokens")
        print('*' * 50)

def uploadfinetunefiles():
    training_file_name = 'training_set.jsonl'
    validation_file_name = 'validation_set.jsonl'

    # Upload the training and validation dataset files to Azure OpenAI with the SDK.

    training_response = client.files.create(
        file = open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id

    validation_response = client.files.create(
        file = open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)

def deploymodel():
    # Deploy fine-tuned model


    token = os.getenv("TEMP_AUTH_TOKEN")
    subscription = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME")
    model_deployment_name = "gpt-4o-mini-2024-07-18-ft" # Custom deployment name you chose for your fine-tuning model

    deploy_params = {'api-version': "2024-10-01"} # Control plane API version
    deploy_headers = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'application/json'}

    deploy_data = {
        "sku": {"name": "standard", "capacity": 1},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": "<YOUR_FINE_TUNED_MODEL>", #retrieve this value from the previous call, it will look like gpt-4o-mini-2024-07-18.ft-0e208cf33a6a466994aff31a08aba678
                "version": "1"
            }
        }
    }
    deploy_data = json.dumps(deploy_data)

    request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'

    print('Creating a new deployment...')

    r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data)

    print(r)
    print(r.reason)
    print(r.json())

def deploytest():
    returntxt  = ""
    response = client.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18-ft", # model = "Custom deployment name you chose for your fine-tuning model"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": "Do other Azure services support this too?"}
        ]
    )

    print(response.choices[0].message.content)

    returntxt = response.choices[0].message.content
    return returntxt

def main():
    # Check if the required environment variables are set
    Checkdata()

    # Process the training and validation datasets
    processtoken()

    # Upload the training and validation datasets to Azure OpenAI
    training_file_name = 'training_set.jsonl'
    validation_file_name = 'validation_set.jsonl'

    # Upload the training and validation dataset files to Azure OpenAI with the SDK.

    training_response = client.files.create(
        file = open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id

    validation_response = client.files.create(
        file = open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)
    # Submit fine-tuning training job

    response = client.fine_tuning.jobs.create(
        training_file = training_file_id,
        validation_file = validation_file_id,
        model = "gpt-4o-mini-2024-07-18", # Enter base model name. Note that in Azure OpenAI the model name contains dashes and cannot contain dot/period characters.
        seed = 105 # seed parameter controls reproducibility of the fine-tuning job. If no seed is specified one will be generated automatically.
    )

    job_id = response.id

    # You can use the job ID to monitor the status of the fine-tuning job.
    # The fine-tuning job will take some time to start and complete.

    print("Job ID:", response.id)
    print("Status:", response.status)
    print(response.model_dump_json(indent=2))

    # Monitor the fine-tuning job status
    start_time = time.time()

    # Get the status of our fine-tuning job.
    response = client.fine_tuning.jobs.retrieve(job_id)

    status = response.status

    # If the job isn't done yet, poll it every 10 seconds.
    while status not in ["succeeded", "failed"]:
        time.sleep(10)

        response = client.fine_tuning.jobs.retrieve(job_id)
        print(response.model_dump_json(indent=2))
        print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
        status = response.status
        print(f'Status: {status}')
        clear_output(wait=True)

    print(f'Fine-tuning job {job_id} finished with status: {status}')

    # List all fine-tuning jobs for this resource.
    print('Checking other fine-tune jobs for this resource.')
    response = client.fine_tuning.jobs.list()
    print(f'Found {len(response.data)} fine-tune jobs.')

    response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
    print(response.model_dump_json(indent=2))

    response = client.fine_tuning.jobs.checkpoints.list(job_id)
    print(response.model_dump_json(indent=2))

    # Retrieve fine_tuned_model name

    response = client.fine_tuning.jobs.retrieve(job_id)

    print(response.model_dump_json(indent=2))
    fine_tuned_model = response.fine_tuned_model

    # for deployment
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune?tabs=command-line

if __name__ == "__main__":
    main()