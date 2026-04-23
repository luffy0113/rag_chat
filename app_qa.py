"""
RAG 问答前端：基于 streamlit 的多轮对话界面
"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from file_history_store import get_history
from rag import RAGService


DEFAULT_SESSION_ID = "default_session"


st.set_page_config(page_title="RAG 知识库问答")
st.title("RAG 知识库问答")


@st.cache_resource
def get_rag_service() -> RAGService:
    """初始化 RAG 服务，整个 streamlit 生命周期内只建一次"""
    return RAGService()

def load_messages_from_history(session_id: str) -> list[dict]:
    """把文件里的历史消息转成 streamlit 可直接渲染的格式"""
    history = get_history(session_id)
    result = []
    for m in history.messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            result.append({"role": "assistant", "content": m.content})
    return result


rag = get_rag_service()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = DEFAULT_SESSION_ID

if "messages" not in st.session_state:
    st.session_state["messages"] = load_messages_from_history(
        st.session_state["session_id"]
    )


with st.sidebar:
    st.header("会话设置")

    new_sid = st.text_input("会话 ID", value=st.session_state["session_id"])
    if new_sid and new_sid != st.session_state["session_id"]:
        st.session_state["session_id"] = new_sid
        st.session_state["messages"] = load_messages_from_history(new_sid)
        st.rerun()

    st.caption(f"当前会话：`{st.session_state['session_id']}`")
    st.caption(f"共 {len(st.session_state['messages'])} 条消息")

    if st.button("清空当前会话历史"):
        get_history(st.session_state["session_id"]).clear()
        st.session_state["messages"] = []
        st.rerun()


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


question = st.chat_input("请输入你的问题")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            stream = rag.query_stream(
                question=question,
                session_id=st.session_state["session_id"],
            )
        answer = st.write_stream(stream)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
