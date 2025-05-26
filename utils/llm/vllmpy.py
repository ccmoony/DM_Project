from vllm import LLM, SamplingParams

messages_list = [
    [
        {
            "role": "system", "content": "You are an expert in recommending goods. You need to predict the user's interest based on the user's browse history. Please provide exactly 5 most likely interest categories. Your entire output should be wrapped in <interest></interest> tags, with one interest per line inside the tags. The interests should be relevant to the user's history and not too generic. The interests should be in English."
        }, 
        {
            "role": "user", "content": """The user's browsing history is: 
1. {
    "title": "Carlisle FoodService Products 1077108 StorPlus Round Lid, Forest Green, (For 2 to 4-quart Storage Containers)",
    "main_category": "Industrial & Scientific",
    "features": ["Tight double-sealing lids", "Easy opening design", "date indicators", "Fits 2-4qt StorPlus round storage container"],
    "description": "Round Lid fits 2-4qt StorPlus round storage containers. Tight double-sealing lids protect against spills and extend freshness...
}
2. {
    "title": "OSHA Notice Signs - Do Not Open Door Must Be Opened from The Inside Sign | Extremely Durable Made in The USA Signs or Heavy Duty Vinyl Label | Protect Your Warehouse & Business",
    "main_category": "Office Products",
    "categories": ["Industrial & Scientific", "Occupational Health & Safety Products", "Safety Signs & Signals", "Signs"],
    "features": ["EXTREMELY DURABLE: All our OSHA safety products are made from commercial grade materials...", "SIZE & SPECS: 10\" X 7\" Vinyl Decal...", "PROTECT YOUR BUSINESS: These signs help protect your business from legal issues...", "VIBRANT & VISIBLE"],
    "description": "Make Sure You & Your Business are OSHA & ANSI Compliant. Is your business or work space 100% covered and up to code with current signage and labels? ..."
}
    """
        }
    ] for _ in range(100)
]
sampling_params = SamplingParams(temperature=0.8, 
                                 top_p=0.95,
                                 max_tokens=1024,)

llm = LLM(model="Qwen/Qwen3-8B")
tokenizer = llm.get_tokenizer()
prompts = tokenizer.apply_chat_template(
    messages_list,
    tokenize=False,
    add_generation_prompt=True
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}, Generated text: {generated_text}")
