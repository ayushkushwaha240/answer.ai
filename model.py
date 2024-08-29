import requests
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
import numpy as np
import tensorflow as tf
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('hf_key')
HEADERS = {"Authorization": f"Bearer {api_key}"}
ranker_model = "sentence-transformers/all-MiniLM-L6-v2"
Question_answer = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

# Define the function to query the Hugging Face Inference API
def query_huggingface_api(model_name, payload):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def load_models():
    # Initialize the API wrappers with specific configurations
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_result=5, doc_content_chars_max=10000)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    
    arxiv_api_wrapper = ArxivAPIWrapper()
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
    
    tools = [wiki_tool, arxiv_tool]
    
    return {
        "tools": tools,
        # Models are no longer loaded locally, so we don't return them here
    }

def get_responses_from_tools(question, tools):
    # Collect responses from all tools
    responses = []
    for tool in tools:
        result = tool.run(question)
        if result and "No good" not in result:
            responses.append(result)  # Append the result string to the list
    return responses

def rank_responses(question, responses):
    
    payload = {
        "inputs": {
            "source_sentence": question,
            "sentences": list(responses)
        }
    }
    # Encode the question and the responses using the Hugging Face API
    result = query_huggingface_api(ranker_model, payload)
    
    ranked_responses = {score: response for score, response in zip(result, responses)}
    return ranked_responses

def process_ranked_responses(question, tools, score_threshold=0.3):
    # Get responses from tools
    responses = get_responses_from_tools(question, tools)
    if not responses:
        return None
    # Rank the responses
    ranked_responses = rank_responses(question, responses)
    # Filter responses by score threshold
    responses_above_threshold = {score: response for score, response in ranked_responses.items() if score > score_threshold}
    
    # Combine selected texts into a single string
    if responses_above_threshold:
        return responses_above_threshold  # Remove trailing newlines
    else:
        return None

def answer(question, tools, score_threshold=0.50):
    # Process responses and rank them
    context = process_ranked_responses(question, tools, score_threshold)
    if not context:
        return "No context found."
    context = " ".join(context.values())
    print("context ",context)
    # Get the answer using the QA model via Hugging Face Inference API
    result = query_huggingface_api("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", {
        "inputs": {
            "question": question,
            "context": context
        }
    })
    return result["answer"]
