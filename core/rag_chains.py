from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

def build_conversational_rag_chain(llm, fast_llm, retriever, msgs, debug_sink=None):

    def emit_debug(stage, text):
        # Collect debug data for rendering in Streamlit after chain execution.
        if isinstance(debug_sink, list):
            debug_sink.append({"stage": stage, "text": text})
        return text

    #DEBUG
    def debug_rephrase(text):
        return emit_debug("rephrase", text)

    def debug_hyde(text):
        return emit_debug("hyde", text)
    
    #RePhrase
    contextualize_q_system_prompt = (
        "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, "
        "Hãy TẠO LẠI CÂU HỎI của người dùng (bổ sung ngữ nghĩa, dữ liệu,...) cho câu hỏi"
        "Tuyệt đối KHÔNG TRẢ LỜI CÂU HỎI, chỉ viết lại câu hỏi nó nếu câu hỏi quá ngắn hoặc thiếu nội dung, nếu câu hỏi đã đủ nghĩa hãy giữ nguyên câu hỏi"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("placeholder","{chat_history}"),
        ("human", "{input}")
    ])

    rephrase_chain = RunnableBranch(
        (
            lambda x: len(x.get("chat_history", [])) > 0,
            contextualize_q_prompt 
            | fast_llm
            | StrOutputParser()
            | RunnableLambda(debug_rephrase)
        ),

        lambda x: x["input"]
        | RunnableLambda(debug_rephrase)
    )

    #HyDE
    hyde_prompt = ChatPromptTemplate.from_template(
        "Viết một đoạn văn bản (1-2 câu) giả vờ chứa câu trả lời cho câu hỏi sau. "
        "Dùng từ ngữ chuyên ngành. Không cần giải thích.\n\nCâu hỏi: {question}"
    )

    hyde_chain = (
        {"question": RunnablePassthrough()}
        |hyde_prompt
        |fast_llm
        |StrOutputParser()
        |RunnableLambda(debug_hyde)
    )

    # QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một trợ lý AI phân tích tài liệu chuyên nghiệp. Dựa vào các thông tin dưới đây, hãy trả lời câu hỏi của người dùng.\n\nNgữ cảnh:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    #RAG CHAIN
    rag_chain = (
        RunnablePassthrough.assign(
            context = (rephrase_chain | hyde_chain | retriever | format_docs)
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key= "input",
        history_messages_key= "chat_history"
    )


    return conversational_rag_chain