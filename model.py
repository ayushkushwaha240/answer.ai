from transformers import pipeline
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf

def load_models():
    # Load the QA pipeline and the sentence transformer model
    question_answerer = pipeline("question-answering", model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    ranker = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize the API wrappers with specific configurations
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_result=5, doc_content_chars_max=10000)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    
    arxiv_api_wrapper = ArxivAPIWrapper()
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
    
    tools = [wiki_tool, arxiv_tool]
    
    return {
        "qa": question_answerer,
        "tools": tools,
        "ranker": ranker
    }

def get_responses_from_tools(question, tools):
    # Collect responses from all tools
    responses = []
    for tool in tools:
        result = tool.run(question)
        if result and "No good" not in result:
            responses.append(result)  # Append the result string to the list
    return responses

def rank_responses(question, responses, ranker):
    # Encode the question and the responses
    question_embedding = ranker.encode([question])
    response_embeddings = ranker.encode(responses)

    # Compute cosine similarity between the question and each response
    cosine_similarities = -tf.keras.losses.cosine_similarity(question_embedding, response_embeddings)
    cosine_similarities = cosine_similarities.numpy().tolist()

    output = {
        cosine_similarities[i]: responses[i]
        for i in range(len(responses))
    }
    # Sort the responses by their score in descending order
    sorted_output = dict(sorted(output.items(), key=lambda item: item[0], reverse=True))
    return sorted_output


def process_ranked_responses(question, tools, ranker, score_threshold=0.3):
    # Get responses from tools
    responses = get_responses_from_tools(question, tools)
    if not responses:
        return None
    
    # Rank the responses
    ranked_responses = rank_responses(question, responses, ranker)
    # Filter responses by score threshold
    responses_above_threshold = ""
    for score,text in ranked_responses.items():
        if score >= score_threshold:
            responses_above_threshold += text + "\n\n"  # Adding newline for separation
    
    # Combine selected texts into a single string
    if responses_above_threshold:
        combined_context = "".join(responses_above_threshold)
        return combined_context
    else:
        return None


def answer(question, tools, question_answerer, ranker, score_threshold=0.50):
    # Process responses and rank them
    context = process_ranked_responses(question, tools, ranker, score_threshold)
    # print(context)
    # print(type(context))
    if not context:
        return "No context found."
    
    # Get the answer using the QA pipeline
    # print(question)
    # print(type(question))
    result = question_answerer(question=question, context=context)
    
    
    # Handle the case where the result might be a list or a dictionary
    if isinstance(result, list):
        if result:  # Check if the list is not empty
            result = result[0]
        else:
            return "No answer found."
    
    # Check if the result is a dictionary and contains the 'answer' key
    if isinstance(result, dict) and 'answer' in result:
        return result['answer']
    
    # If the result is not in the expected format, return a default message
    return "No valid answer found."


