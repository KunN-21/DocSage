
import streamlit as st
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma

from .azure_doc_parser import parse_document

def process_pdf_to_retriever(updated_file):
    
    documents, parser_used, tmp_file_path = parse_document(updated_file)

    #Chunking
    semantic_splitter = SemanticChunker(
        embeddings= st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type= "percentile",
        breakpoint_threshold_amount= 95,
        min_chunk_size= 500,
        add_start_index=True
    )
    docs = semantic_splitter.split_documents(documents)

    #Vector Retriever
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=st.session_state.embeddings,
        persist_directory="./data/chroma_db"
    )
    vector_retriever = vector_db.as_retriever(search_type = "similarity",
                                       search_kwargs = {"k":5})
    
    #BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    #Hybrid Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    return ensemble_retriever, len(docs), parser_used, tmp_file_path