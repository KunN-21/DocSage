
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import config


def _create_openai_llm(
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):

    kwargs = {
        "model": model,
        "api_key": config.OPENAI_API_KEY,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Chỉ set base_url nếu có (OpenRouter, etc.)
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL

    return ChatOpenAI(**kwargs)

def _create_ollama_llm(
    model: str,
    temperature: float = 0.1,
    num_ctx: int = 4096,
    **kwargs,
    ):
    base_url = config.ollama_url()
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        reasoning=False,
        **kwargs,
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm(provider: str = None):
    provider = provider or config.LLM_PROVIDER
    if provider == "openai":
        if not config.openai_available():
            raise ValueError("OPENAI_API_KEY chưa được cấu hình trong .env")
        return _create_openai_llm(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            max_tokens=512,
        )
    elif provider == "ollama":
        return _create_ollama_llm(
            model=config.OLLAMA_MODEL,
            temperature=0.1,
            num_ctx=4096,
            num_predict=512,
            top_p=0.9,
        )
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Dùng 'ollama' hoặc 'openai'.")
    
@st.cache_resource
def load_fast_llm(provider: str = None):
    provider = provider or config.LLM_PROVIDER
    if provider == "openai":
        if not config.openai_available():
            raise ValueError("OPENAI_API_KEY chưa được cấu hình trong .env")
        return _create_openai_llm(
            model=config.OPENAI_FAST_MODEL,
            temperature=0.3,
            max_tokens=256,
        )
    elif provider == "ollama":
        return _create_ollama_llm(
            model=config.OLLAMA_FAST_MODEL,
            temperature=0.3,
            num_ctx=2048,
        )
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Dùng 'ollama' hoặc 'openai'.")