import io
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
import pickle
from sentence_transformers import SentenceTransformer, util
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from transformers import AutoTokenizer
import transformers
import torch
from scipy.spatial import distance
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import xformers
import accelerate
import numpy as np

# pdf parser
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# vector db
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

# prompt
from langchain.prompts import PromptTemplate

# task
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.document_loaders import PyPDFLoader
from urllib.request import urlretrieve
import os
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.prompts import PromptTemplate  # prompt engeenering

from langchain.memory import ConversationBufferMemory

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from typing import Dict

from langchain.chains import ConversationalRetrievalChain

# Initialize FastAPI
from fastapi import FastAPI, HTTPException, Body

app = FastAPI()

# Define your question-answering function
@app.post("/qa")
async def question_answering(input_data: Dict[str, str]):
    try:
        # Extract the text input from the request
        text_input = input_data.get("input_text", "")

        # Perform question-answering here using your existing code
        query = text_input
        os.makedirs("datasets", exist_ok=True)
        files = [
            "https://www.nice.org.uk/guidance/ng192/resources/caesarean-birth-pdf-66142078788805",
            "https://www.nice.org.uk/guidance/ng50/resources/cirrhosis-in-over-16s-assessment-and-management-pdf-1837506577093",
        ]
        for url in files:
            file_path = os.path.join("datasets", url.rpartition("/")[2])
            urlretrieve(url, file_path)
        loader = PyPDFDirectoryLoader("./datasets/")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        len(docs)

        avg_doc_length = lambda documents: sum(
            [len(doc.page_content) for doc in documents]
        ) // len(documents)
        avg_char_count_pre = avg_doc_length(documents)
        avg_char_count_post = avg_doc_length(docs)
        print(
            f"Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters."
        )
        print(
            f"After the split we have {len(docs)} documents more than the original {len(documents)}."
        )
        print(
            f"Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters."
        )

        # embeddings:

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}

        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        sample_embedding = np.array(embeddings.embed_query(docs[0].page_content))
        print("Size of the embedding: ", sample_embedding.shape)

        ### vectorstore:

        start = time.time()

        vectorstore_faiss = FAISS.from_documents(
            docs,
            embeddings,
        )

        wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

        exec_time = time.time() - start

        print(f"total embedding vector store time: {exec_time}")

        ### prompt:

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # memory part:

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        ### QA model:

        llm = HuggingFacePipeline.from_model_id(
            model_id="bigscience/bloom-1b7",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 200},
            device=-1,
        )

        ## conversation chain
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            memory=memory,
        )


        ### first time a new chat starts we begin with

        chat_history = []
        result = qa({"question": query, "chat_history": chat_history})

        # Extract the answer from the result
        answer = result['answer']

        # Format the answer as needed
        formatted_answer = {
            "question": query,
            "answer": answer,
        }

        # Serialize the answer as JSON using json.dumps()
        json_answer = json.dumps(formatted_answer)

        # Return the JSON-serialized answer
        return json_answer

    except Exception as e:
        error_message = {"error": str(e)}
        return error_message

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

