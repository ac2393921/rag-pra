import pandas as pd
from datasets import load_dataset
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import GPT4All

loader = TextLoader('medical_data.txt', encoding="utf-8")
index = VectorstoreIndexCreator(embedding= HuggingFaceEmbeddings()).from_loaders([loader])

llm_path = './model/ggml-gpt4all-j-v1.3-groovy.bin'  # replace with your desired local file path
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=llm_path, callbacks=callbacks, verbose=True, backend='gptj')

results = index.vectorstore.similarity_search("what is the solution for soar throat", k=4)
context = "\n".join([document.page_content for document in results])
print(f"{context}")
