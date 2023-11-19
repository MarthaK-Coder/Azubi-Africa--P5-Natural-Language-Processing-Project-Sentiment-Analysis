# Azubi-Africa--P5-Natural-Language-Processing-Project-Sentiment-Analysis
This project demonstrates the process of fine-tuning a sentiment analysis model using Hugging Face's Transformers library and deploying it alongside a sentiment analysis app on Hugging Face.

## Setup
### 1. Virtual Environment
Create and activate a virtual environment for the project:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 2. Install Dependecies
 Install the required Python packages using the provided requirements.txt file:
 pip install -r requirements.txt

## Fine-tuning with Hugging Face
### 3. Fine-tune Model
Fine-tune a sentiment analysis model using the Hugging Face Transformers library. Use your dataset and adjust training parameters as needed.

### 4. Save Model
Save the fine-tuned model and associated tokenizer:
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

## Hosting on Hugging Face
### 5. Push to Hugging Face
Push the fine-tuned model and tokenizer to Hugging Face for easy sharing and deployment:
transformers-cli login
transformers-cli repo create your-username/fine-tuned-model
transformers-cli upload fine_tuned_model
Sentiment Analysis App

### 6. Create App
Develop a sentiment analysis app using a web framework like Streamlit. Use the fine-tuned model and tokenizer to perform sentiment analysis on user input.

### 7. Push App to Hugging Face
Host the sentiment analysis app on Hugging Face:

transformers-cli repo create your-username/sentiment-analysis-app
transformers-cli upload sentiment-analysis-app

## Conclusion
Your sentiment analysis model and app are now hosted on Hugging Face, making them accessible and shareable with the community.

Feel free to customize and extend the project based on your requirements.
Note: Adjust the paths, model names, and usernames according to your project specifications.

