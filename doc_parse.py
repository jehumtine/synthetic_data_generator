import PyPDF2
from transformers import AutoTokenizer
import json
import requests
import re


HOST = 'localhost:5000'
URI = f'http://{HOST}/query_ai'
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-falcon")
history = {'internal': [], 'visible': []}
command = ""

def run(user_input, history):
    print("initialising request......")
    request = {
        'user_input': user_input,
        'history': history,
    }
    print("sending request....")
    response = requests.post(' http://127.0.0.1:5000/query_ai', json=request)
    print("request sent....")
    result = response.json()['body']
    print("response received....")
    return result

def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text

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

def submit_to_api(chunk, retries=3):
    for i in range(retries):
        try:
            print(i)
            print("th try to communicate to llm")
            response = run(command + chunk.strip(), history)
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
text = extract_text_from_pdf('/home/jehu/Desktop/projs/power/read/(Synthese Library 220) Gordon Pask (auth.), Gertrudis van de Vijver (eds.) - New Perspectives on Cybernetics_ Self-Organization, Autonomy and Connectionism-Springer Netherlands (1992).pdf')
tokens = tokenize(text)

token_chunks = list(chunks(tokens, 256))
print("Done Tokenizing........")
responses = []
q=0
for chunk in token_chunks:
    q=q+1
    print(q)
    response = submit_to_api(tokenizer.decode(chunk))
    if response is not None:
        responses.append(response)
    else:
        print("Response is NON")

# Write responses to a JSON file
with open('responses.json', 'w') as f:
    json.dump(responses, f)