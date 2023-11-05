#Installations
#Installation
#pip install streamlit transformers
#pip install scipy
#pip install transformers[torch] --upgrade

#Importations
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import streamlit as st

# Requirements
model_path = f"MarthaK-Coder/test_trainer"
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    
    # Format output dict of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }
    
    return scores

# Streamlit app
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter a text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    if user_input:
        scores = sentiment_analysis(user_input)
        st.write("Sentiment Scores:")
        for label, score in scores.items():
            st.write(f"{label}: {score:.3f}")
    else:
        st.warning("Please enter a text for analysis.")