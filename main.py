from flask import Flask,request
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__)
model_path = '/home/jehu/.local/share/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin'
template = """{question} \
        Task:You are an API that converts bodies of text into a single question and answer into a JSON format. Each JSON " \
          "contains a single question with a single answer. Only respond with the JSON and no additional text. 
          \n."""
callbacks = [StreamingStdOutCallbackHandler()]
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = GPT4All(model=model_path, callbacks=callbacks,verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)


@app.route("/query_ai", methods = ['POST'])
def query_ai():
    content_type = request.headers.get('Content-Type')
    query = None
    print("API Accesed")
    if(content_type == 'application/json'):
        json_payload = request.get_json()
        query = json_payload['user_input']
        print("The query is"+query)
    else:
        return 'Content-Type not supported'
    print("-----------Running Query-------------")
    resp =llm_chain.run(query)
    return{
        'statusCode': 500,
        'body' : resp,
    }