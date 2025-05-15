import json


def slice_and_expand_messages(data):
    sliced_data = []

    for item in data:
        conversations = item["conversations"]
        temp_conversations = []  # Temporarily store the current fragment
        for conv in conversations:
            temp_conversations.append(conv)
            if conv["role"] == "assistant":
                # Each time an assistant is encountered, store the current conversations fragment
                sliced_data.append({"conversations": list(temp_conversations)})

    return sliced_data




with open("swift_dataset/swift_SFT_QA_Agent_cash.json", "r", encoding="utf-8") as f:
    data = json.load(f)


expanded_data = slice_and_expand_messages(data)


with open("swift_dataset/swift_SFT_QA_Agent_cash_multiStep.json", "w", encoding="utf-8") as f:
    json.dump(expanded_data, f, ensure_ascii=False, indent=4)

print("Dataset slicing and expansion completed, results saved to swift_SFT_QA_multiStep.json")