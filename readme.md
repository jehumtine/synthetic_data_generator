## Overview
This script is designed to convert bodies of text into a question and answer JSON format using the GPT-4 language model. The process involves extracting text from PDF files, tokenizing the text, generating questions and answers, and then saving the results in a JSON file.
## Prerequisites
Python (version X.X.X)
Required Python packages: langchain, PyPDF2, transformers, requests, pathlib, tqdm
Setup
Clone this repository to your local machine.
Install the required packages by running: pip install -r requirements.txt
Obtain an API token from Hugging Face Hub and set it as an environment variable:

export HUGGINGFACEHUB_API_TOKEN='your_api_token'

## Usage
Place your PDF files in the specified folder (folder_path) that you want to process.
Run the script: python convert_text_to_qa.py
The script will perform the following steps:
Extract text from PDF files.
Tokenize the extracted text.
Generate questions and answers using the GPT-4 language model.
Save the generated Q&A pairs in a JSON file named responses.json.
## Note
You can modify the model_path, folder_path, and other parameters in the script as needed.
The script processes the text in chunks to manage memory usage. You can adjust the chunk size (256 in the example) based on your system's capabilities.
## License


## Contact
If you have any questions or suggestions, feel free to contact me at jmtine.ndhlovu@cs.unza.zm

