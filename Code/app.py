import streamlit as st
import torch
import yaml
from transformers import AutoTokenizer, AutoConfig
from model import CustomModel
from utils import Config, round_to_half


def load_model(cfg, checkpoint_path):
    backbone_config = AutoConfig.from_pretrained(cfg.model.backbone_name, output_hidden_states=True)
    model = CustomModel(cfg, backbone_config)
    model.to(device='cpu')
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()
    return model


def make_prediction(cfg, model, input_text):
    inputs = cfg.tokenizer.encode_plus(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output = model(inputs)
    predictions = output.cpu().numpy()[0]  # Example conversion, adjust as needed
    return predictions


# Streamlit UI
st.title("Text Analysis with Machine Learning")

with open("config.yaml", 'r') as yaml_file:
    cfg_dict = yaml.safe_load(yaml_file)
    cfg = Config(cfg_dict)

# Model selection
model_names = ['deberta_base_meanpooling', 'deberta_base_lstmpooling']  # Replace with your actual model names
selected_model_name = st.selectbox("Select a Model", model_names)
model_dict = {'deberta_base_meanpooling': 'saved_checkpoints/microsoft_18_0.4923.pth',
              'deberta_base_lstmpooling': 'saved_checkpoints/microsoft_lstm_7_0.4879.pth',
              }
cfg.model.pooling = 'lstm' if selected_model_name == 'deberta_base_lstmpooling' else 'mean'
tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
cfg.tokenizer = tokenizer
model = load_model(cfg, model_dict[selected_model_name])

# Text input
input_text = st.text_area("Input Text", "Type or paste text here...")

# Button to make predictions
if st.button("Analyze Text"):
    if input_text:
        predictions = make_prediction(cfg, model, input_text)
        predictions = [round_to_half(num) for num in predictions]
        # Display predictions
        st.write("Predicted :")
        labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        for label, score in zip(labels, predictions):
            st.write(f"{label}: {score:.3f}")
    else:
        st.write("Please input some text to analyze.")

# Run the Streamlit app by executing `streamlit run your_script_name.py`
