from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Chatbot: Hello! I am your AI assistant. Type 'exit' to quit.")

chat_history_ids = None

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,
        no_repeat_ngram_size=3
    )

    bot_response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if bot_response.strip() == "":
        bot_response = "I'm not sure how to respond to that. Can you rephrase?"

    print("Chatbot:", bot_response)


#Output:
# (venv) PS C:\Users\LENOVO\OneDrive\Desktop\InnomaticsAssignment\NLP_Project\chatbot-transformers> python chatbot.py
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████| 293/293 [00:00<00:00, 8345.03it/s] 
# Chatbot: Hello! I am your AI assistant. Type 'exit' to quit.
# You: Hello
# Chatbot: I'm going to go get a beer
# You: Ok Great , can you tell me about you , who are you and what can you do for me
# Chatbot: I can't I'm still a student
# You: exit
# Chatbot: Goodbye!