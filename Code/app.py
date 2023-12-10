import streamlit as st
import pandas as pd
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

with open("test_config.yaml", 'r') as yaml_file:
    cfg_dict = yaml.safe_load(yaml_file)
    cfg = Config(cfg_dict)

# Model selection
model_options = ['bert-base-uncased', 'electra-base-discriminator',
                 'roberta-large', 'deberta-v3-base', 'deberta-v3-large', ]
pooling_options = ['mean', 'lstm', 'concat', 'conv1d']

selected_model = st.selectbox("Select model", ['Select model'] + model_options)

# Conditional display of the second dropdown based on the model selection
if selected_model != 'Select model':
    selected_pooling = st.selectbox("Select pooling", ['Select pooling'] + pooling_options)

    ckp_path = f"ckp_app/{selected_model}_{selected_pooling}.pth"

    # Conditional display of the "Next" button based on the pooling selection
    if selected_pooling != 'Select pooling':
        if st.button("Next"):
            # Use a session state to store the selection and proceed to text input
            st.session_state['selected_model'] = selected_model
            st.session_state['selected_pooling'] = selected_pooling
            st.session_state['proceed_to_text_input'] = True

        if pooling_options == 'lstm':
            cfg.model.pooling = selected_pooling
        elif pooling_options == 'concat':
            cfg.model.pooling = selected_pooling
        elif pooling_options == 'conv1d':
            cfg.model.pooling = selected_pooling
        else:
            cfg.model.pooling = selected_pooling

        if selected_model == 'bert-base-uncased':
            cfg.model.backbone_name = 'bert-base-uncased'
        elif selected_model == 'deberta-v3-base' or selected_model == 'deberta-v3-large':
            cfg.model.backbone_name = 'microsoft/' + selected_model
        elif selected_model == 'electra-base-discriminator':
            cfg.model.backbone_name = 'google/' + selected_model
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
        cfg.tokenizer = tokenizer
        model = load_model(cfg, ckp_path)

# Conditional display of text input box based on the "Next" button press
if st.session_state.get('proceed_to_text_input', False):
    user_input = st.text_area("Enter your text here:")

# input_text = st.text_area("Input Text", "Type or paste text here...")

    # Button to make predictions
    if st.button("Analyze Text"):
        # st.write(f"Model: {st.session_state['selected_model']}")
        # st.write(f"Pooling: {st.session_state['selected_pooling']}")
        predictions = make_prediction(cfg, model, user_input)
        predictions = [round(round_to_half(num), 1) for num in predictions]
        # Display predictions
        st.write("Predicted :")
        labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        data = {
            'Labels': labels,
            'Scores': predictions
        }
        results_df = pd.DataFrame(data)

        # Display the DataFrame as a table in Streamlit
        st.table(results_df)
    # else:
    #     st.write("Please input some text to analyze.")

# Run the Streamlit app by executing `streamlit run your_script_name.py`
