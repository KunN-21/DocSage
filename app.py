import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from core.document_loader import process_pdf_to_retriever
from core.llm_manager import load_embeddings, load_fast_llm, load_llm
from core.rag_chains import build_conversational_rag_chain

from core.config import config

# INIT
st.set_page_config(page_title= "PDF RAG Assistant", layout= "wide")
st.title("PDF RAG Assistant")

st.markdown("""
**Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**

**Cách sử dụng đơn giản:**
1. **Upload PDF** -> Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
2. **Đặt câu hỏi** -> Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức
---
""")

# Init Chat-history
msgs = StreamlitChatMessageHistory(key= "chat_history")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Xin chào! Tôi có thể giúp gì cho bạn với tài liệu PDF này?")

# Init Session
if "current_provider" not in st.session_state:
    st.session_state.current_provider = config.LLM_PROVIDER
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "debug_events" not in st.session_state:
    st.session_state.debug_events = []

# SIDEBAR
with st.sidebar:
    st.header("Cấu hình")
    if config.azure_available():
        st.success("Azure Document Intelligence: Sẵn sàng")
    else:
        st.warning("Azure DI: Chưa cấu hình (chỉ hỗ trợ PDF)")
    st.divider()

    available_provider = ['ollama']
    if config.openrouter_available():
        available_provider.append('openrouter')

    #Select Box
    select_provider = st.selectbox(
        label= "LLM Provider",
        options= available_provider,
        index= available_provider.index(st.session_state.current_provider) if st.session_state.current_provider in available_provider else 0,
    )

    #Detect change
    if select_provider != st.session_state.current_provider:
        st.session_state.current_provider = select_provider
        st.session_state.models_loaded = False
        st.cache_resource.clear()
        st.rerun()

    provider = st.session_state.current_provider
    if provider == "openrouter":
        st.caption(f"Model: {config.OPENROUTER_MODEL}")
        st.caption(f"Fast model: {config.OPENROUTER_FAST_MODEL}")
    else:
        st.caption(f"Model: {config.OLLAMA_MODEL}")
        st.caption(f"Fast model: {config.OLLAMA_FAST_MODEL}")


# Load model
if not st.session_state.models_loaded:
    st.info("Đang tải model...")
    provider = st.session_state.current_provider
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm(provider)
    st.session_state.fast_llm = load_fast_llm(provider)
    st.session_state.models_loaded = True
    st.success("Model đã sẵn sàng!")
    st.rerun()

# File Upload
uploaded_file = st.file_uploader(
    "Upload file", type=["pdf", "jpg", "jpeg", "png", "tiff"]
)

if uploaded_file and st.button("Xử lý tài liệu"):
    with st.spinner("Đang xử lý tài liệu"):
        retriever, num_chunks, parser_used, tmp_file_path= process_pdf_to_retriever(uploaded_file)
        st.session_state.rag_chain = build_conversational_rag_chain(
            llm = st.session_state.llm,
            fast_llm= st.session_state.fast_llm,
            retriever= retriever,
            msgs= msgs,
            debug_sink= st.session_state.debug_events,
        )
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        st.success(f"Hoàn thành! {num_chunks} chunks")
        st.info(f"Parser: **{parser_used.upper()}**")

# CHAT
if st.session_state.rag_chain:
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    
    if question := st.chat_input("Đặt câu hỏi:"):
        st.chat_message("human").write(question)

        with st.chat_message("ai"):
            with st.spinner("Đang suy nghĩ..."):
                st.session_state.debug_events.clear()
                
                rag_config = {"configurable": {"session_id": "any_session_id"}}
                response = st.session_state.rag_chain.invoke(
                    {"input": question},
                    config=rag_config
                )

                st.sidebar.header("Debug Trace:")
                if len(st.session_state.debug_events) == 0:
                    st.sidebar.caption("Chưa có debug trace cho lượt hỏi này.")
                else:
                    for event in st.session_state.debug_events:
                        stage = event.get("stage")
                        text = event.get("text", "")
                        if stage == "rephrase":
                            st.sidebar.info(f"**RePhrase tạo lại câu hỏi:**\n_{text}_")
                        elif stage == "hyde":
                            st.sidebar.warning(f"**HyDE tạo đáp án ảo hỗ trợ retriever:**\n_{text}_")
                        else:
                            st.sidebar.write(text)

                clean_answer = response.split("Answers:")[1].strip() if "Answers:" in response else response.strip()
                st.write(clean_answer)