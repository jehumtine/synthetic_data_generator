from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import PyPDF2
from transformers import AutoTokenizer
import requests
from pathlib import Path
from tqdm import tqdm
import re
import json


os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-falcon")
history = {'internal': [], 'visible': []}
command = ""
template = """{question} \
        Task:You are an API that converts bodies of text into a single question and answer into a JSON format. Each JSON " \
          "contains a single question with a single answer. Only respond with the JSON and no additional text.
          \n."""
prompt = PromptTemplate(template=template, input_variables=["question"])
model_path = '/home/jehu/.local/share/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin'
folder_path = '/home/jehu/Documents/law data/data'
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=model_path, callbacks=callbacks,verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)


def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def read_documents(folder_path):
    combined_text = ""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print("The File path is"+ file_path)
        if filename.endswith(".pdf"):
            text = read_pdf(file_path)
            combined_text += text + "\n\n"
    return combined_text

def tokenize(text):
    enc = tokenizer.encode(text)
    return enc

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def is_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False


def submit_to_llm(chunk, retries=3):
    for i in range(retries):
        try:
            print(i)
            print("th try to communicate to llm")
            response = llm_chain.run(chunk.strip())
            # Extract JSON string from between back-ticks
            if is_json(response):
                print(response)
                return json.loads(response)
            else:
                match = re.search(r'`(.*?)`', response, re.S)
                if match and is_json(match.group(1)):
                    print(f"Attempt {i + 1} failed. Retrying...")
                    return json.loads(match.group(1))  # assuming you want to return the JSON data
                else:
                    print("Request failed:")
                    print(response)
        except requests.exceptions.RequestException as e:
            continue
    print("Max retries exceeded. Skipping this chunk.")
    return None

print("Extracting Texts From PDF........")
text = read_documents(folder_path)
tokens = tokenize(text)

token_chunks = list(chunks(tokens, 256))
print("Done Tokenizing........")
responses = []
q=0
for chunk in token_chunks:
    q=q+1
    print(q)
    response = submit_to_llm(tokenizer.decode(chunk))
    if response is not None:
        #responses.append(response)
        with open('responses.json', 'a') as f:
            if q > 1:  # Add a comma before writing new JSON, except for the first one
                f.write(",\n")
            json.dump(response, f)
    else:
        print("Response is NON")                

# Write responses to a JSON file
##with open('responses.json', 'w') as f:
##    json.dump(responses, f)