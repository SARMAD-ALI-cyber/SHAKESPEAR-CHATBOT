import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import streamlit.components.v1 as components

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
    input_ids = tokenizer.encode(question + "$", add_special_tokens=False, padding=True, truncation=True, return_tensors="pt")
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

# Streamlit UI
st.set_page_config(
    page_title="Shakespearian Bot",
    page_icon="✒️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Shakespearian Bot 📜")
user_input = st.text_input("Ask a question:")
if user_input:
    bot_response = generate_response(user_input)
    index = bot_response.find("$")

    if index != -1:
        bot_response = bot_response[index + 1:]
    st.markdown(f"**Bot:** {bot_response}")

# Add a Shakespearean background image
st.markdown(
    """<style>
    .stApp {
        background-image: url('https://www.thoughtco.com/thmb/khVHRCkBEe7nGt8N5do6L2o2VqA=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-184986309-5a1b7e7989eacc003779d5a3.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        color: black;
    }
    </style>""",
    unsafe_allow_html=True,
)
