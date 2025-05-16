from openai import OpenAI
import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx

# Read the GEXF file
gexf_file_path = ''  # Replace with your GEXF file path
G = nx.read_gexf(gexf_file_path)
lastStepResult = {}
the_specification_of_lastStepResult = {"FirstStepNotheStepResult": ""}

def llm_api(query):
    # print(f"Actual parameters passed to llm_api: {query}")
    response = llm_api_qwen(query)
    return response
def llm_api_qwen(query):
    api_key = ""
    # Debug information: Print API key
    # print(f"API Key: {api_key}")
    client = OpenAI(
        # If the environment variable is not configured, please replace the next line with your Bailian API Key: api_key="sk-xxx",
        api_key=api_key,
        base_url="",
    )

    completion = client.chat.completions.create(
        model="",  # Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
    )
    response = completion.choices[0].message.content
    # print(response)
    return response

def llm_api_gpt(query):
    api_key = ""
    # Debug information: Print API key
    # print(f"API Key: {api_key}")
    client = OpenAI(
        api_key=api_key,
        base_url="",
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
    )
    response = completion.choices[0].message.content
    # print(response)
    return response

def json_data_extract(response):
    # Find the position of ```json
    json_start_index = response.find("```json")
    if json_start_index != -1:
        json_start_index += len("```json")
        json_end_index = response.find("```", json_start_index)
        if json_end_index != -1:
            json_data = response[json_start_index:json_end_index]
            return json_data
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        return None
def python_data_extract(response):
    print(f"Actual parameters passed to python_data_extract: {response}")
    # Step 1: Find the position of ```python
    python_start_index = response.find("```python")

    # If ```python is found, continue execution
    if python_start_index != -1:
        # Step 2: Start searching for the next ``` position after ```python
        python_start_index += len("```python")  # Move after ```json
        python_end_index = response.find("```", python_start_index)
        # If the closing ``` is found, extract the JSON data
        if python_end_index != -1:
            python_data = response[python_start_index:python_end_index]
            # print(python_data)
            return python_data
        else:
            print("The closing ``` was not found when extracting Python code")
            return None
    else:
        return response

# 1. Load the model and tokenizer
model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.chat_template = '''
{% for message in messages %}
{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'function' %}
<|im_start|>function {{ message['name'] }}
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
<|im_start|>assistant
'''.strip()
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

def call_function_api(question: str, newresponse: str) -> str:
    # Call llm_api to fake the result
    try:
        with open('./promptCodeGen_cache_en.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        # Replace API_NAME with the combined list
        prompts["input"]["user_question"] = prompts[
        "input"]["user_question"].format(QUESTION=question)
        prompts["input"]["the_specification_of_lastStepResult"] = the_specification_of_lastStepResult
        # prompts["input"]["the_specification_of_lastStepResult"] = prompts[
        # "input"]["the_specification_of_lastStepResult"].format(FirstStepNotheStepResult=the_specification_of_lastStepResult)
        prompts["input"]["this_step_demand"] = prompts[
        "input"]["this_step_demand"].format(DEMAND=newresponse)
        prompts = json.dumps(prompts, indent=4, ensure_ascii=False)
        result = llm_api(prompts)
        # print("\033[96m", result, "\033[0m")
        result = python_data_extract(result)
        print("\033[36m", result, "\033[0m")
        return result
    except Exception as e:
        print(f"\033[31mError occurred in call_function_api: {e}\033[0m")
        return f"Simulated response"
    
def step_judge_score(question: str, thought: str, api_call: str) -> str:
    try:
        with open('./prompt_stepScore.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        # Replace API_NAME with the combined list
        prompts["input"]["user_question"] = prompts[
        "input"]["user_question"].format(QUESTION=question)
        prompts["input"]["thought"] = prompts[
        "input"]["thought"].format(THOUGHT=thought)
        prompts["input"]["api"] = prompts[
        "input"]["api"].format(API_NAME=api_call)
        prompts = json.dumps(prompts, indent=4, ensure_ascii=False)
        result = llm_api(prompts)
        # print("\033[96m", result, "\033[0m")
        result = json_data_extract(result)
        print("\033[35m", result, "\033[0m")
        return result
    except Exception as e:
        print(f"\033[31mError occurred in step_judge_score: {e}\033[0m")
        return f"Simulated response: Executed {api_call}"
def result_judge_score() -> str:
    try:
        with open('./prompt_resultScore.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        # Extract all content except for the system
        print("\033[35m", dialogue, "\033[0m")
        solve_log = [item for item in dialogue if item.get('role') != 'system']
        # Replace
        prompts["input"]= prompts[
        "input"].format(SOLVE_LOG=solve_log)
        prompts = json.dumps(prompts, indent=4, ensure_ascii=False)
        result = llm_api(prompts)
        # print("\033[96m", result, "\033[0m")
        result = json_data_extract(result)
        print("\033[35m", result, "\033[0m")
        return result
    except Exception as e:
        print(f"\033[31mError occurred in result_judge_score: {e}\033[0m")
# 3. Extract the <API>...</API> segment
def extract_api_call(text: str) -> str:
    # Find the <API>...</API> part after Action: 
    action_match = re.search(r"Action: <API>(.*?)</API>", text)
    return action_match.group(1).strip() if action_match else None

def extract_thought(text: str) -> str:
    # Find the text after Thought: until Action: or the end of the string
    thought_match = re.search(r"Thought: (.*?)(?=Action:|$)", text, re.DOTALL)
    return thought_match.group(1).strip() if thought_match else None

def extract_user_content(dialogue):
    user_contents = []
    for item in dialogue:
        if not isinstance(item, dict):
            print(f"Non-dictionary type item: {item}")
        elif "role" in item and item["role"] == "user":
            user_contents.append(item["content"])
    return user_contents

def extract_answer(text: str) -> str:
    # Find the text after Answer: until the end of the string
    answer_match = re.search(r"Answer: (.*)$", text, re.DOTALL)
    return answer_match.group(1).strip() if answer_match else None

# 4. Main generation function that supports automatic API recognition and injecting function responses
def generate_with_api_tool(dialogue: list, file_counter_list: list, max_steps=8, max_loops=3):
    original_dialogue = dialogue.copy()  # Save the original dialogue list
    i = 0
    j = 0
    step_score_list = []
    
    # Define the parent folder path
    parent_folder = 'dialogue_history'
    while i < max_steps:
        # Construct the Chat template input
        # print("\033[35mConverted input text:", dialogue, "\033[0m")
        input_text = tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
        # print("\033[32mConverted input text:", input_text, "\033[0m")
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate a complete assistant response segment
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id
        )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"\033[31mRound {i + 1}, decoded full output: {full_output}\033[0m")
        # new_response = full_output.replace(input_text, "")
        # new_response = full_output[len(input_text):].strip()
        split_point = full_output.rfind("assistant")
        if split_point != -1:
            new_response = full_output[split_point + len("assistant"):].strip()
        else:
            new_response = "(Assistant reply content not found)"

        user_questions = extract_user_content(dialogue)
        if i == 0:
           print(f"\033[32m==User questions: {user_questions}==\033[0m")
        print(f"\033[33m======Round {i + 1}, new response after removing input text: ======\n\033[0m\033[37m{new_response}\033[0m")

        dialogue.append({"role": "assistant", "content": new_response})

        
        thouht_text = extract_thought(new_response)
        # Check if there is an <API> call
        api_call = extract_api_call(new_response)
        
        
        if api_call:
            if "Finish->answer" in api_call:
                answer_text = extract_answer(new_response)
                # print(f"\033[34m======Answer extracted in Round {i + 1}: {answer_text}\033[0m")
                result_score = result_judge_score()
                # Print the type of result_score
                print(f"The type of result_score is: {type(result_score)}")
                result_score = json.loads(result_score)
                dialogue.append(result_score)
                dialogue.append(step_score_list)
                # print(f"\033[34m======Score extracted in Round {i + 1}: {result_score}\033[0m")
                # result_scope = result_judge_score(user_questions, thouht_text, api_call)
                print(f"\033[42m\033[95mRound {i + 1}, end the dialogue\033[0m")
                # Ensure the parent folder exists
                if not os.path.exists(parent_folder):
                    os.makedirs(parent_folder)
                # Get the value of file_counter from the list
                file_counter = file_counter_list[0]
                # Save the dialogue to a JSON file
                file_name = f'{parent_folder}/dialogue_history_{file_counter}.json'
                try:
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(dialogue, f, ensure_ascii=False, indent=4)
                    print(f"\033[32mDialogue history has been successfully saved to {file_name}\033[0m")
                    # After successful saving, increment the sequence counter by 1
                    file_counter_list[0] += 1
                except Exception as e:
                    print(f"\033[31mError occurred while saving dialogue history: {e}\033[0m")
                break
            elif "Finish->give_up_and_restart" in api_call and j < max_loops:
                j += 1
                print(f"\033[35m===========Loop {j + 1} ===========\n\033[0m")
                i = 0  # Reset the loop counter
                dialogue = original_dialogue.copy()  # Reset the dialogue list
                continue

            else:
                step_score = step_judge_score(user_questions, thouht_text, api_call)
                step_score = json.loads(step_score)
                step_key = f"step_{i + 1}"
                step_score = {step_key: step_score["apiResult"]}
                step_score_list.append(step_score)

                # dialogue.append(step_score)
                response_funcGen = call_function_api(user_questions, new_response)

                local_namespace = {}

                # Dynamically execute the generated code
                try:
                    exec(response_funcGen, globals(), local_namespace)  # Convert the code string to an executable object
                except Exception as e:
                    error_msg = f"Code generation failed: {str(e)}"
                    print(f"\033[31m{error_msg}\033[m")
                    func_response = {
                        "error": True,
                        "message": error_msg,
                        "advice": "Please try to describe your problem again"
                    }

                # Assume local_namespace has only one function
                func_name, func = next(iter(local_namespace.items()), (None, None))
                if func_name and callable(func):
                    max_retries = 3  # Maximum number of retries
                    for attempt in range(max_retries):
                        try:
                            func_response = func()
                            break  # Exit the retry loop if execution is successful
                        except Exception as e:
                            func_response = {  # Structure containing error information
                                "error": True,
                                "message": f"Execution failed: {str(e)}",
                                "advice": "Please check if the input parameters or problem description are correct"
                            }
                            print(f"\033[31mFunction execution failed (Retry {attempt + 1}):\033[m", str(e))
                            # Regenerate the function with error information
                            response_funcGen = call_function_api(user_questions, new_response)
                            local_namespace = {}
                            try:
                                exec(response_funcGen, globals(), local_namespace)
                                func_name, func = next(iter(local_namespace.items()), (None, None))
                            except Exception as new_e:
                                error_msg = f"Failed to regenerate code: {str(new_e)}"
                                print(f"\033[31m{error_msg}\033[m")
                                func_response = {
                                    "error": True,
                                    "message": error_msg,
                                    "advice": "Please try to describe your problem again"
                                }
                                break
                    else:
                        print("\033[31mMaximum number of retries reached, give up execution\033[m")
                else:
                    func_response = {
                        "error": True,
                        "message": "Callable function not found",
                        "advice": "Please try to describe your problem again"
                    }
            print(f"\033[45mTool call result: {func_response}\033[0m")

            try:
                # Try to parse func_response as a JSON object
                # stepresult = json.loads(func_response)
                lastStepResult = func_response["thisStepResult"]
                the_specification_of_lastStepResult = func_response["the_specification_of_thisStepResult"]
                if "the_description_of_content" in the_specification_of_lastStepResult:
                    value = the_specification_of_lastStepResult["the_description_of_content"]
                    if isinstance(value, str):
                        the_specification_of_lastStepResult["the_description_of_content"] = value.replace("thisStepResult", "lastStepResult")

                if "how_to_use" in the_specification_of_lastStepResult:
                    value = the_specification_of_lastStepResult["how_to_use"]
                    if isinstance(value, str):
                        the_specification_of_lastStepResult["how_to_use"] = value.replace("thisStepResult", "lastStepResult")
                print(f"\033[42mDescription of the previous step result: {the_specification_of_lastStepResult}\033[0m")
            except json.JSONDecodeError as e:
                # If parsing fails, print an error message and provide suggestions
                print(f"\033[31mError occurred while parsing tool call result as JSON: {e}. Please check if the content of func_response is in valid JSON format.\033[0m")
                stepresult = None

            dialogue.append({
                "role": "tool",
                "content": the_specification_of_lastStepResult
            })
            # dialogue.append(step_score)
        else:
            break  # No API call, exit the loop
        i += 1

    return dialogue

# 5. Test usage
if __name__ == "__main__":
    dialogue = [
        {
                "role": "system",
                "content": "You are AutoGPT, you can use many tools(functions) from NetworkX to do the graphic analysis.\nFirst I will give you the task description and the graph dateset description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:\nThought:...\nAction:<API>...</API>\n\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.In the last step, the format should be: \nThought:...\nAction:<API>Finish->answer</API>\nAnswer:...\nor\nThought:...\nAction:<API>Finish->give_up_and_restart</API>\nRemember: \n1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say \"I give up and restart\".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet's Begin!\nTask description: You should use functions to help handle the real time user querys. \nRemember:\n1.ALWAYS call <API>Finish</API> function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function <API>Finish->give_up_and_restart</API>.\nYou have access of the following tools:\n   \"Basic Graph Properties\": [\n        \"<API>G.number_of_nodes()</API>\",\n        \"<API>G.number_of_edges()</API>\",\n        \"<API>G.has_node(n)</API>\",\n        \"<API>G.has_edge(u, v)</API>\",\n        \"<API>G.degree(nbunch=None, weight=None)</API>\",\n        \"<API>G.in_degree(nbunch=None, weight=None)</API>\",\n        \"<API>G.out_degree(nbunch=None, weight=None)</API>\",\n        \"<API>G.nodes()</API>\",\n        \"<API>G.edges()</API>\",\n        \"<API>G.get_edge_data(u, v, default=None)</API>\"\n    ],\n    \"Centrality Metrics\": [\n        \"<API>nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)</API>\",\n        \"<API>nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)</API>\",\n        \"<API>nx.degree_centrality(G)</API>\",\n        \"<API>nx.eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight='weight')</API>\",\n        \"<API>nx.harmonic_centrality(G, nbunch=None, distance=None)</API>\",\n        \"<API>nx.load_centrality(G, normalized=True, weight=None)</API>\",\n        \"<API>nx.percolation_centrality(G, attribute=None, k=None, runs=100, seed=None)</API>\",\n        \"<API>nx.second_order_centrality(G, weight=None)</API>\",\n        \"<API>nx.subgraph_centrality(G)</API>\"\n    ],\n    \"Connectivity and Components\": [\n        \"<API>nx.strongly_connected_components(G)</API>\",\n        \"<API>nx.weakly_connected_components(G)</API>\",\n        \"<API>nx.number_strongly_connected_components(G)</API>\",\n        \"<API>nx.number_weakly_connected_components(G)</API>\",\n        \"<API>nx.algorithms.connectivity.articulation_points(G)</API>\",\n        \"<API>nx.algorithms.connectivity.bridge_connected_components(G)</API>\",\n        \"<API>nx.algorithms.connectivity.bridges(G)</API>\",\n        \"<API>nx.algorithms.connectivity.k_edge_components(G, k=None)</API>\",\n        \"<API>nx.algorithms.connectivity.k_node_components(G, k=None)</API>\",\n        \"<API>nx.algorithms.connectivity.node_connectivity(G, s=None, t=None, flow_func=None)</API>\",\n        \"<API>nx.algorithms.connectivity.edge_connectivity(G, s=None, t=None, flow_func=None)</API>\"\n    ],\n    \"Shortest Paths and Distances\": [\n        \"<API>nx.all_pairs_shortest_path(G, cutoff=None, weight=None)</API>\",\n        \"<API>nx.all_pairs_shortest_path_length(G, cutoff=None, weight=None)</API>\",\n        \"<API>nx.algorithms.shortest_paths.unweighted.breadth_first_search(G, source, cutoff=None)</API>\",\n        \"<API>nx.dijkstra_path(G, source, target, weight='weight')</API>\",\n        \"<API>nx.dijkstra_path_length(G, source, target, weight='weight')</API>\",\n        \"<API>nx.floyd_warshall(G, weight='weight')</API>\"\n    ],\n    \"Clustering and Communities\": [\n        \"<API>nx.average_clustering(G, weight=None, nodes=None, mode='original')</API>\",\n        \"<API>nx.clustering(G, weight=None, nodes=None)</API>\",\n        \"<API>nx.algorithms.clustering.generalized_degree(G, nodes=None, weight=None)</API>\",\n        \"<API>nx.transitivity(G)</API>\",\n        \"<API>nx.triangles(G, nodes=None)</API>\",\n        \"<API>nx.algorithms.community.label_propagation.label_propagation_communities(G)</API>\",\n        \"<API>nx.algorithms.community.louvain_communities(G, weight='weight', resolution=1, threshold=1e-07, seed=None)</API>\"\n    ],\n    \"Flow Algorithm\": [\n        \"<API>nx.algorithms.flow.boykov_kolmogorov.min_cut(G, s, t, capacity='capacity', residual=None, value_only=False)</API>\",\n        \"<API>nx.algorithms.flow.dinic.min_cut(G, s, t, capacity='capacity', residual=None, value_only=False)</API>\",\n        \"<API>nx.algorithms.flow.edmonds_karp.min_cut(G, s, t, capacity='capacity', residual=None, value_only=False)</API>\",\n        \"<API>nx.algorithms.flow.ford_fulkerson.min_cut(G, s, t, capacity='capacity', residual=None, value_only=False)</API>\",\n        \"<API>nx.minimum_cut(G, s, t, capacity='capacity', flow_func=None, residual=None, value_only=False)</API>\"\n    ],\n    \"Cycle Detection\": [\n        \"<API>nx.simple_cycles(G)</API>\",\n        \"<API>nx.has_cycle(G, source=None)</API>\",\n        \"<API>nx.find_cycle</API>\",\n        \"<API>nx.cycle_basis</API>（cycle_basis 仅适用于无向图）\"\n    ],\n    \"Topological Sorting\": [\n        \"<API>nx.topological_sort(G)</API>\",\n        \"<API>nx.is_directed_acyclic_graph(G)</API>\",\n        \"<API>nx.all_topological_sorts</API>\",\n        \"<API>nx.topological_generations</API>\"\n    ], \n    \"Final\": [\n        \"<API>Finish->answer</API>\",\n        \"<API>Finish->give_up_and_restart</API>\"\n    ]\n"
            },
            {
                "role": "user",
                "content": "Help me find the two largest cycles in this transfer graph (note that this is a multi-directed graph, you need to use simple_cycles to find cycles), then find the node with the largest outgoing amount in each of these two cycles, and finally tell me how many transfers these two nodes with the largest outgoing amounts have received.\nBegin!"
            }
    ]

    # 初始化 file_counter 列表
    file_counter_list = [1]
    for round_num in range(1):
        
        
        print(f"\n\n=== Question {round_num + 1} ====== Question {round_num + 1} ====== Question {round_num + 1} ===\n")
        result = generate_with_api_tool(dialogue, file_counter_list, max_steps=8, max_loops=3)

        # Generate new user content
        query = """
        The description of a dataset is as follows:

{
  "dataset_description": {
    "name": "cash_flow_graph.gexf",
    "type": "MultiDirected graph with weights and dates",
    "content": "The fund transfer data of a specific group of people. Directed edge A->B means that A has transferred funds to B. The graph construction operation is: G = nx.MultiDiGraph()\n G.add_edge(sender, receiver, weight=amount, date=transfer_date), where sender and receiver are the sender and receiver of the transfer, amount is the amount, and transfer_date is the date. The integer type is used to store nodes when constructing the graph."
  }
}

Please design a question based on this dataset that can be solved using methods provided in networkX.
        \n## IMPORTANT!!!!!
Your output only needs to be a question.
Do not include any other content, such as explanations, background information, or other context. Just output the question itself.
        \n\n
        Output example:
        'Help me find the two largest cycles in this transfer graph, then find the node with the largest outgoing amount in each of these two cycles, and finally tell me how many transfers these two nodes with the largest outgoing amounts have received.\nBegin!'
        \n\n
        """
        new_user_content = llm_api(query)

        # Only keep system and user content in dialogue
        dialogue = [item for item in dialogue if isinstance(item, dict) and "role" in item and item["role"] in ["system", "user"]] 

        # Replace user content in dialogue
        for item in dialogue:
            if item["role"] == "user":
                item["content"] = new_user_content
                break
