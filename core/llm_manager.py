
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter

from .config import config


def _create_openrouter_llm(
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
    return ChatOpenRouter(
        model=model,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_API_BASE or None,
        temperature=temperature,
        max_tokens=max_tokens,
    )

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
    if provider == "openrouter":
        if not config.openrouter_available():
            raise ValueError("OPENROUTER_API_KEY chưa được cấu hình trong .env")
        return _create_openrouter_llm(
            model=config.OPENROUTER_MODEL,
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
        raise ValueError(f"Unknown LLM provider: '{provider}'. Dùng 'ollama' hoặc 'openrouter'.")
    
@st.cache_resource
def load_fast_llm(provider: str = None):
    provider = provider or config.LLM_PROVIDER
    if provider == "openrouter":
        if not config.openrouter_available():
            raise ValueError("OPENROUTER_API_KEY chưa được cấu hình trong .env")
        return _create_openrouter_llm(
            model=config.OPENROUTER_FAST_MODEL,
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
        raise ValueError(f"Unknown LLM provider: '{provider}'. Dùng 'ollama' hoặc 'openrouter'.")