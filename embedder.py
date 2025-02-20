from langchain.embeddings import HuggingFaceBgeEmbeddings
import json 
import os   
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

data_directory = "./new_datasets/microlabs_usa"
embedding_directory = "./content/chroma_db"

embedding_db = None

def embed():

    print("\nCalculating Embeddings\n")
    jq_schema = 'to_entries | map(.key + ": " + .value) | join("\\n")'
    # Load the text from the data directory
    loader=DirectoryLoader(data_directory,
                        glob="*.json",
                        loader_cls=JSONLoader,
                        loader_kwargs={"jq_schema": jq_schema}
                        )

    documents=loader.load()
    if not documents:
        print("No documents found.")
        return
    # Split the data into chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)

    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)

    # Load the huggingface embedding model
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        #model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    embedding_db = Chroma.from_documents(chunks, embedding_model, persist_directory=embedding_directory)

    print("Embeddings completed")
embed()