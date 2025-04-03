'''
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-03-31 13:39:34
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-04-02 18:32:32
FilePath: /SiriusxLLM/python/chat.py
Description: 
'''
from modelscope import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
device = "cuda"  # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "你是一个有用的AI助手。回答要简洁、专业、有帮助。"}
]

def apply_chat_template(conversation, tokenize=False, add_generation_prompt=False):
    """
    Applies a chat template to a conversation history.
    
    Args:
        conversation (list): List of message dicts with "role" and "content"
        tokenize (bool): Whether to tokenize the result (not implemented in this pure Python version)
        add_generation_prompt (bool): Whether to add a prompt to indicate the start of the assistant's turn
    
    Returns:
        str: Formatted conversation string
    """
    # Qwen1.5 chat template uses the following format:
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # How are you?<|im_end|>
    # <|im_start|>assistant
    # I'm fine.<|im_end|>
    
    formatted_messages = []
    
    for message in conversation:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            formatted_messages.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            formatted_messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # Add generation prompt if needed
    if add_generation_prompt:
        formatted_messages.append("<|im_start|>assistant\n")
    
    return "\n".join(formatted_messages)

def chat():
    print("Assistant: Hello! How can I help you today? (Type 'quit' to end)")
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Assistant: Goodbye!")
            break
            
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate model input
        text = apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generate response
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids)[0]
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response})
        
        # Print response
        print(f"Assistant: {response}")

# Start the chat
chat()