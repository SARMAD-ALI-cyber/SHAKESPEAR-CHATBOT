import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import flask

model_path = 'shakespear_gpt2.pth'
model_state_dict = torch.load(model_path)

# Initialize the GPT-2 model and load the state dictionary
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(model_state_dict)
model.eval()

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token







# Generate Response Function
def generate_response(question):
    # Encode the question using the tokenizer
    input_ids = tokenizer.encode(question + "<|question|>", add_special_tokens=False, padding=True, truncation=True, return_tensors="pt")
    # Generate the answer using the model
    attention_mask = torch.ones_like(input_ids)
    sample_output = model.generate(input_ids, do_sample=True, max_length=100, top_k=20, top_p=1.0, attention_mask=attention_mask)

    # Decode the generated answer using the tokenizer
    answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)

    # Split the generated answer into individual sentences
    sentences = answer.split(". ")

    # Look for the sentence that contains the answer to the question
    for sentence in sentences:
        if question in sentence:
            return sentence

    # If no sentence contains the answer, return the full generated answer
    return answer


# Asking Questions
question = "what's the weather going to be like?"
response = generate_response(question)
print(f"{question}\n {response}")