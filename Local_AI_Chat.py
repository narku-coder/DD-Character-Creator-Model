from transformers import pipeline
import torch

print("Downloading and loading the model... (This will take a few minutes the first time!)")

# 1. Initialize the Hugging Face Pipeline
# We are using TinyLlama because it is extremely fast and lightweight for local machines.
# It has enough world knowledge to answer basic trivia and logic questions.
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16, # Compresses the math to save your computer's memory
    device_map="auto" # Automatically uses your GPU if you have one, or defaults to CPU
)

print("\nModel loaded successfully! Type 'quit' to exit.\n")

# 2. The Chat Loop
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['quit', 'exit']:
        break

    # 3. Format the prompt using the model's specific chat template
    # Instead of our custom <USER> tags, we use the standardized Hugging Face format
    messages = [
        {"role": "system", "content": "You are a helpful, accurate, and concise AI assistant."},
        {"role": "user", "content": user_input},
    ]
    
    # Let the tokenizer apply the correct structural tokens for this specific model
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Generate the response
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7, # Controls creativity (lower = more factual, higher = more creative)
        top_k=50,
        top_p=0.95
    )
    
    # 5. Clean up and print the output
    generated_text = outputs[0]["generated_text"]
    # Slice off the prompt so we only see the assistant's new response
    final_answer = generated_text[len(prompt):]
    
    print(f"\nAssistant: {final_answer}\n")