import json
from utils.utils import load_jsonl
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
import re

dataset = "scientific"
dataset_full_name = "Industrial_and_Scientific"

dataset_path = f"dataset/{dataset}/"

train_path = f"{dataset_path}{dataset}.train.jsonl"
test_path = f"{dataset_path}{dataset}.test.jsonl"
testtest_path = f"{dataset_path}{dataset}.testtest.jsonl"
valid_path = f"{dataset_path}{dataset}.valid.jsonl"

meta_path = f"{dataset_path}meta_{dataset_full_name}.jsonl"

processed_train_path = f"{dataset_path}{dataset}.train.processed.jsonl"
processed_test_path = f"{dataset_path}{dataset}.test.processed.jsonl"
processed_testtest_path = f"{dataset_path}{dataset}.testtest.processed.jsonl"
processed_valid_path = f"{dataset_path}{dataset}.valid.processed.jsonl"

if __name__ == "__main__":
    print("Dataset name:", dataset)
    print("Dataset full name:", dataset_full_name)
    print("Dataset path:", dataset_path)
    print("Metadata path:", meta_path)
    print("Train path:", train_path)
    print("Test path:", test_path)
    print("Valid path:", valid_path)

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    testtest_data = load_jsonl(testtest_path)
    valid_data = load_jsonl(valid_path)
    
    # Load meta_data into a Pandas DataFrame
    meta_data_list = load_jsonl(meta_path)

    meta_df = pd.DataFrame(meta_data_list)
    meta_df = meta_df.set_index("parent_asin")
    print("Meta data loaded into Pandas DataFrame and indexed by 'parent_asin'.")
    print("Meta data shape:", meta_df.shape)

    print("Train data length:", len(train_data))
    print("Test data length:", len(test_data))
    print("Valid data length:", len(valid_data))

    sampling_params = SamplingParams(temperature=0.6, 
                                    top_p=0.95,
                                    max_tokens=1024,)
    llm = LLM(model="Qwen/Qwen3-4B")
    tokenizer = llm.get_tokenizer()

    def process_file(input_file, output_path, meta_lookup_df, llm, tokenizer):
        with open(output_path, "w") as f:
            messages_list = []
            new_items = []
            for item in tqdm(input_file, dynamic_ncols=True, desc=f"Processing {output_path}"):
                new_item = item.copy() # Start with a copy of the original item
                new_items.append(new_item)

                prompt = "The user's browsing history is: \n"

                for i, item_asin in enumerate(item["inter_history"]):

                    meta_row = meta_lookup_df.loc[item_asin]
                    title = meta_row["title"] if "title" in meta_row else None
                    main_category = meta_row["main_category"] if "main_category" in meta_row else None
                    categories = meta_row["categories"] if "categories" in meta_row else None
                    features = meta_row["features"] if "features" in meta_row else None
                    description = meta_row["description"] if "description" in meta_row else None

                    prompt += f"{i + 1}. {{\n"
                    prompt += f"    \"title\": \"{title}\",\n"
                    prompt += f"    \"main_category\": \"{main_category}\",\n"
                    prompt += f"    \"categories\": {categories},\n"
                    prompt += f"    \"features\": {features},\n"
                    prompt += f"    \"description\": \"{description}\"\n"
                    prompt += "}\n"
                    prompt += "Remember to use the <interest></interest> tags for your output.\n"

                messages_list.append([
                    {
                        "role": "system", "content": "You are an expert in recommending goods. You need to predict the user's interest based on the user's browse history. Please provide exactly 5 most likely interest categories. Each line should be wrapped in <interest></interest> tags, with one interest per line inside the tags. The interests should be relevant to the user's history and not too generic. The interests should be in English."
                    },
                    {
                        "role": "user", "content": prompt
                    }
                ])

            prompts = tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                # print(f"Prompt: {prompt}, Generated text: {generated_text}")

                # Extract interests from the generated text
                interests_matches = re.findall(r"<interest>(.*?)</interest>", generated_text)
                interests = "\\n".join(interests_matches)
                
                invalid_count = 0

                if interests == "":
                    invalid_count += 1
                    # print(f"Warning: No valid interests found for item {i} in {output_path}.")
                    # # Extract and print content after </think> tag if it exists
                    # think_match = re.search(r"</think>(.*?)$", generated_text, re.DOTALL)
                    # if think_match:
                    #     print(f"Content after </think> tag: {think_match.group(1).strip()}")
                    # # Print the full generated text to debug
                    # print(f"Generated text: {generated_text}")
                
                print(f"Total invalid interests found: {invalid_count}")

                new_items[i]["interests"] = interests

                f.write(json.dumps(new_items[i]) + "\n")
            
    process_file(testtest_data, processed_testtest_path, meta_df, llm, tokenizer)
    process_file(valid_data, processed_valid_path, meta_df, llm, tokenizer)
    process_file(test_data, processed_test_path, meta_df, llm, tokenizer)
    process_file(train_data, processed_train_path, meta_df, llm, tokenizer)