# run this file to make inference requests: python qdrant/inference.py
# modify QUERY and COLLECTION_NAME as needed
import os, sys
from wsgiref import types
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pprint
from qdrant.client import get_client, get_reranker
import os
import datetime
import time
# import retrieval pipeline as absolute path using importlib
from qdrant.pipeline import retrieval_pipeline
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# ------ TODO ------
# modify query to ask what you want
QUERY = "What does the New York times say is the best wireless headphones?"
COLLECTION_NAME = "production_data"
RESULT_COUNT = 10 # top n results to returnc

client = get_client()
reranker_model = get_reranker()
# request qdrant with vector
start = time.time()
print("Searching for similar documents in Qdrant...")
ranked_results = retrieval_pipeline(QUERY, COLLECTION_NAME, RESULT_COUNT, client, reranker_model)
end = time.time()
print(f"Search completed in {end - start:.2f} seconds.")

print("\n--- Top Search Results ---")
out_dir = os.path.join(os.getcwd(), "qdrant", "out")
os.makedirs(out_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(out_dir, f"{timestamp}_qdrant_inference.txt")
latency_path = os.path.join(out_dir, f"{timestamp}_qdrant_latency.txt")


with open(out_path, "a") as f:
    for i, result in enumerate(ranked_results):
        f.write(f"Result {i}:\n")
        f.write(pprint.pformat(result))
        f.write("\n\n")

with open(latency_path, "w") as f:
    f.write(f"Query: {QUERY}\n")
    f.write(f"Latency: {end - start:.2f} seconds\n")
    
print(f"Results saved to: {out_path}")

print("\n Retrieval Output:")
context_texts = [res[0].payload["source_file"] for res in ranked_results]
for i, text in enumerate(context_texts):
    print(f"Context {i+1}:\n{text}\n")


prompt = """
You are a RAG system. Your task is to create a concise response in 3-5 lines based on the context provided. 
The user query is given by the <user_query> tag and the relevant contexts are given with the <context> tag. 
Only use the given context to ground your responses, do not use outside references or knowledge to answer the user query.
"""

user_query = """
<user_query>
{query}
</user_query>   
<context>
{contexts}
</context>
""".format(query=QUERY, contexts="\n".join(context_texts))

client = genai.Client()
prompt = "Explain how AI works in a few words"
response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=prompt),
    contents=user_query,
)
print("\n LLM Response:")
print(response.text)

## LLM as a Judge 
judge_prompt = """
You are a strict judge. Your task is to evaluate the response based on the user query and the provided context in a likert scale from 1 to 5.
1 - Poor: The response is irrelevant or incorrect based on the context.
2 - Fair: The response has some relevance but lacks accuracy or completeness.
3 - Good: The response is generally accurate but may miss some details from the context.
4 - Very Good: The response is accurate and covers most of the important details from the context.
5 - Excellent: The response is highly accurate, comprehensive, and directly addresses the user query using only the information from the context.

"""
judge_query = """<user_query>
{query}
</user_query>
<context>
{contexts}
</context>
<response>
{response}
</response>
""".format(query=QUERY, contexts="\n".join(context_texts), response=response.text)

judge_response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=judge_prompt),
    contents=judge_query,
)
print("\nJudge's Evaluation:")
print(judge_response.text)
