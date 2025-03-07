import os
import time
import math
import traceback
from openai import OpenAI
import random
import concurrent.futures
from tqdm import tqdm_notebook

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

error_args = []
working_args = []

def call_openai_api(api_call_fn, args_list, max_retries=5, backoff_factor=2):
    """
    Call any OpenAI API for a list of arguments in parallel with retries and exponential backoff.

    Args:
        api_call_fn (function): The function to call for each set of arguments.
        args_list (list): A list of tuples or lists, where each tuple or list contains the arguments to be passed to api_call_fn.
        max_retries (int): The maximum number of retries in case of API errors.
        backoff_factor (float): The factor by which the waiting time between retries should increase.

    Yields:
        The results of each API call, as returned by api_call_fn.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for args in args_list:
            future = executor.submit(call_openai_api_single, api_call_fn, args, max_retries, backoff_factor)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            yield result


def call_openai_api_single(api_call_fn, call_fn_args, max_retries=5, backoff_factor=2):
    global error_args
    result = None
    retries = 0
    while retries < max_retries:
        try:
            result = api_call_fn(**call_fn_args)
            break
        except Exception as e:
            print(f"API error: ")
            print(traceback.format_exc())
            error_args.append((call_fn_args, e))
            retries += 1
            if retries < max_retries:
                backoff_time = (2 ** retries) * backoff_factor + random.uniform(0, 1)
                print(f"Waiting {backoff_time:.2f} seconds before retrying...")
                time.sleep(backoff_time)
            else:
                print(f"Max retries ({max_retries}) exceeded, giving up.")
                raise Exception('Max Retries Exceeded Exception')
                # return call_fn_args
    return result


def call_openai_api_batch(api_call_fn, args_list, batch_size=1, max_retries=5, backoff_factor=2):
    """
    Call any OpenAI API for a list of arguments in parallel with retries and exponential backoff.

    Args:
        api_call_fn (function): The function to call for each set of arguments.
        args_list (list): A list of tuples or lists, where each tuple or list contains the arguments to be passed to api_call_fn.
        batch_size (int): The number of arguments to process in each batch.
        max_retries (int): The maximum number of retries in case of API errors.
        backoff_factor (float): The factor by which the waiting time between retries should increase.

    Returns:
        The results of each API call, as returned by api_call_fn, in order of pr_id.
    """
    results = {}
    num_batches = math.ceil(len(args_list)/batch_size)
    ctr = 1
    end_chr = '\r'
    for i in tqdm_notebook(range(0, len(args_list), batch_size)):
        if ctr == num_batches:
            end_chr = '\n'
        print(f"Processing batch {ctr} of {num_batches} | {len(args_list)} prompts for batch size {batch_size}", end=end_chr)
        batch_args = args_list[i:i+batch_size]
        for result in call_openai_api(api_call_fn, batch_args, max_retries, backoff_factor):
            pr_id = result[0]
            results[pr_id] = result[1:]
        ctr += 1
    return [(ar['pr_id'], results[ar['pr_id']]) for ar in args_list]


def chat_api_call(pr_id, messages, model, config, print_context=False):
    if print_context:
        f"""
        ___________________________________________________________
        Chat API Call
        pr_id: {pr_id}
        model: {model}
        config: {config}
        messages: 
            system_message:
                ```
                {messages[0]['content']}
                ```
            user_message:
                ```
                {messages[1]['content']}
                ```
        """

    response = client.chat.completions.create(
                    model=model,
                    response_format=config[model]['response_format'],
                    messages=messages,
                    temperature=config[model]['temperature'],
                    max_tokens=config[model]['max_tokens'],
                    top_p=config[model]['top_p'],
                    frequency_penalty=config[model]['frequency_penalty'],
                    presence_penalty=config[model]['presence_penalty'],
                    stop=config[model]['stop'],
                    seed=config[model]['seed'],
                    n=config[model]['n'] if 'n' in config[model] else 1,
                )
    
    return pr_id, response.choices


def embedding_api_call(pr_id, data, embedding_model, dimensions):
    if embedding_model == "text-embedding-3-small":
        embedding = client.embeddings.create(
                        input=data, 
                        model=embedding_model,
                    )
    else:
        embedding = client.embeddings.create(
                        input=data, 
                        model=embedding_model,
                        dimensions=dimensions
                    )

    embedding = [e.embedding for e in embedding.data]
    return pr_id, embedding, data


# Convenience function to call embeddings api for a list of prompts
def embed(prompts, embedding_model="text-embedding-3-small", dimensions=1024, batch_size=500):
    args_list = [dict(pr_id=pr_id,
                      data=prompt,
                      dimensions=dimensions,
                      embedding_model=embedding_model)
                 for pr_id, prompt in enumerate(prompts)]

    return call_openai_api_batch(api_call_fn=embedding_api_call,
                                 args_list=args_list,
                                 batch_size=batch_size,
                                 max_retries=15,
                                 backoff_factor=2)


# Convenience function to call chat api for a list of prompts
def chat(system_message, prompts, model="gpt-3.5-turbo", config=None, batch_size=500):
    if config is None:
        config = {
            "gpt-3.5-turbo": {
                'temperature': 0,
                'max_tokens': 300,
                'top_p': 1,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'response_format': { "type": "json_object" }, # 'text' or 'json_object'
                'stop': None,
                'seed': 42,
            }
        }

    # form the messages
    def form_message(system_message, content):
        return [{"role": "system", "content": system_message},
                {"role": "user", "content": content}]

    args_list = [dict(pr_id=pr_id,
                      messages=form_message(system_message, prompt),
                      model=model,
                      config=config)
                 for pr_id, prompt in enumerate(prompts)]

    return call_openai_api_batch(api_call_fn=chat_api_call,
                                 args_list=args_list,
                                 batch_size=batch_size,
                                 max_retries=5,
                                 backoff_factor=2)


if __name__ == "__main__":
    # Example usage for the OpenAI completions API
    prompts = ["Make a list of all the things you need to do today.",
               "Best time to go to the gym ?",
               "Imagine you are a teacher. Write a letter to the parents of one of your students who has been misbehaving.",
               "What is the best way to learn a new language?",
            ]

    model = "gpt-3.5-turbo-0125"
    embedding_model = "text-embedding-3-small"
    n = 1

    # api_call_fn = openai.Completion.create
    api_call_fn = embedding_api_call
    args_list = [dict(pr_id=pr_id, data=prompt, embedding_model=embedding_model) for pr_id, prompt in enumerate(prompts)]

    print("EMBEDDINGS: ", args_list)
    for result in call_openai_api(api_call_fn, args_list, max_retries=5, backoff_factor=2):
        # process results here, e.g. print them
        pr_id, embedding, prompt = result
        print(pr_id, len(embedding[0]), prompt)
        print("--------")
        
    print("BATCHED EMBEDDINGS: ", args_list)
    results = call_openai_api_batch(api_call_fn, args_list, batch_size=3, max_retries=5, backoff_factor=2)
    for result in results:
        # print(result)
        # process results here, e.g. print them
        pr_id, (embedding, prompt) = result
        print(pr_id, len(embedding[0]), prompt)
        print("--------")
        
    
    # api_call_fn = openai.Completion.create
    api_call_fn = chat_api_call
    chat_config = {
        "gpt-3.5-turbo-0125": {
            'temperature': 0,
            'max_tokens': 300,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'stop': None,
            'response_format': { "type": "json_object" }, # 'text' or 'json_object',
            'seed': 42, 
        }
    }

    def form_message(system_message, content):
        return [{"role": "system", "content": system_message},
                {"role": "user", "content": content}]


    system_message = \
"""You are an Alien pretending to be a human. 
You are having a conversation with a human. 
The human is asking you questions and you are answering them. 
You are trying to be as alien-like as possible.
Answer in JSON format.
"""

    args_list = [dict(pr_id=pr_id,
                      messages=form_message(system_message, prompt),
                      model=model,
                      config=chat_config)
                 for pr_id, prompt in enumerate(prompts)]

    print("COMPLETIONS: ", args_list, "\n")
    for result in call_openai_api(api_call_fn, args_list, max_retries=5, backoff_factor=2):
        # process results here, e.g. print them
        pr_id, choices = result
        completions = choices[0].message.content
        print(completions)
        print("--------")


    # Chat function example
    chat(
        system_message="You are the world expert on Pokémon. JSON output format",
        prompts=["Gotta catch 'em all!", "What's your favorite Pokémon?", "What's the best Pokémon game?"],
        model="gpt-3.5-turbo",
        config={
            "gpt-4-1106-preview": {
                'temperature': 0,
                'max_tokens': 300,
                'top_p': 1,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'response_format': { "type": "json_object" }, # 'text' or 'json_object'
                'stop': None,
                'seed': 42,
            },
            "gpt-3.5-turbo": {
                'temperature': 0,
                'max_tokens': 300,
                'top_p': 1,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'response_format': { "type": "json_object" }, # 'text' or 'json_object'
                'stop': None,
                'seed': 42,
            }
        }
    )
