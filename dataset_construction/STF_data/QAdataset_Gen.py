import os
from openai import OpenAI
import json
from tqdm import tqdm
import itertools
import random

def llm_api(query):
    response = llm_api_qwen(query)
    return response

def llm_api_qwen(query):
    api_key = ""
    client = OpenAI(
        api_key=api_key,
        base_url="",
    )

    completion = client.chat.completions.create(
        model="",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
    )
    response = completion.choices[0].message.content
    print(response)
    return response

def json_data_extract(response):
    json_start_index = response.find("```json")

    if json_start_index != -1:
        json_start_index += len("```json")
        json_end_index = response.find("```", json_start_index)

        if json_end_index != -1:
            json_data = response[json_start_index:json_end_index]
            print(json_data)
        else:
            print("Ending ``` not found")
            return None
    else:
        print("```json not found")
        return None
    return json_data

existing_data = []
train_data = {}

def get_example(task_count, combo_APIName):
    global existing_data, train_data
    with open('prompt_QAdatasetGen.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
        dataset_description = prompts['dataset_description']
    
    prompts["what you need to do"]["design_requirement"] = prompts["what you need to do"][
        "design_requirement"].format(API_NAME=combo_APIName)
    print(prompts["what you need to do"]["design_requirement"])
    prompts = json.dumps(prompts, indent=4, ensure_ascii=False)
    conversation = llm_api(prompts)
    
    print(f"LLM response: {conversation}")
    try:
        conversation_json = json.loads(conversation)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return
    
    with open('prompt_system.json', 'r', encoding='utf-8') as f:
        system = json.load(f)
    print(f"dataset_description: {dataset_description}")
    system["value"] = system["value"].format(dataset_description=dataset_description)
    print(system["value"])

    conversation_json.insert(0, system)

    new_id = f"en_dataset01_{task_count}"
    new_instance = {
        "id": new_id,
        "conversations": conversation_json
    }
    train_data[f"en_Chemical_item_{task_count}"] = {
        "cnt": 1,
        "instance": [new_instance]
    }

def traverse_api_Gen():
    with open('SFT_dataset_en/SFT_API-name_00.json', 'r', encoding='utf-8') as f:
        APIset = json.load(f)
    categories = list(APIset.keys())

    total_apiComb = 100
    single_api_tasks = 5
    task_count = 0
    
    with tqdm(total=single_api_tasks * total_apiComb, desc="Generating Progress") as pbar:
        for _ in range(total_apiComb):
            selected_categories = random.sample(categories, 4)
            selected_apis = []
            for category in selected_categories:
                api = random.choice(APIset[category])
                selected_apis.append(api)
            
            print(f"Processing APIs: {selected_apis}")

            for _ in range(single_api_tasks):
                get_example(task_count, selected_apis)
                task_count += 1
                pbar.update(1)
                pbar.set_postfix({"Current Task": task_count})
            
            with open('SFT_dataset_en/SFT_QA_cash_4api.json', 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=4, ensure_ascii=False)

def test_api_Gen():
    combo = ["<API>G.in_degree(nbunch=None, weight=None)</API>", "<API>nx.weakly_connected_components(G)</API>"]
    total_tasks = 2

    task_count = 0
    with tqdm(total=total_tasks, desc="Generating Progress") as pbar:
        for _ in range(total_tasks):
            get_example(task_count, combo)
            task_count += 1
            pbar.update(1)
            pbar.set_postfix({"Current Task": task_count})
    
    print(existing_data)
    print(train_data)
    with open('SFT_dataset_en/SFT_QA_Chemical.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    traverse_api_Gen()