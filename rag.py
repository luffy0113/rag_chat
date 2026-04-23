"""
RAG 服务：检索 + 生成 + 多轮对话历史
"""
from operator import itemgetter

from dotenv import load_dotenv

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

import config_data as config
from file_history_store import get_history
from vector_stores import VectorStoreService

from langchain_core.globals import set_debug, set_verbose

load_dotenv()

set_debug(True)

SYSTEM_PROMPT = """你是一个知识库问答助手。请严格遵守以下规则：
1. 优先且严格根据【参考资料】的内容回答用户问题，不要编造资料中没有的事实。
2. 如果资料中没有相关信息，先明确回答"根据已有资料无法回答这个问题，以下回答不一定准确"，然后再基于常识给出你的推理，并标注"以下为推测"。
3. 结合上文对话历史理解用户的最新提问（例如"他""那个"指代的是谁）。
4. 回答要简洁、条理清晰。"""

USER_PROMPT = """【参考资料】
{context}

【用户问题】
{question}"""


def _debug_tap(label: str):
    """在链里插一个'探针'：原样透传数据，顺便打印出来"""

    def _tap(x):
        print(f"\n========== [{label}] ==========")
        print(x)
        print(f"========== [/{label}] ==========\n")
        return x  # ← 关键：原样返回，不影响下游

    return RunnableLambda(_tap)

class RAGService(object):

    def __init__(self):
        self.embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        self.llm = ChatTongyi(model=config.chat_model_name)
        self.vector_service = VectorStoreService(self.embedding)
        self.retriever = self.vector_service.get_retriever()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", USER_PROMPT),
        ])

        base_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | self.retriever | self._format_docs
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        self.chain = RunnableWithMessageHistory(
            base_chain,
            get_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str, session_id: str) -> str:
        """根据用户提问 + 会话历史，检索向量库并生成回答"""
        return self.chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

    def query_stream(self, question: str, session_id: str):
        """流式版本：边生成边吐字符串片段。

        RunnableWithMessageHistory 在流完整消费完后会自动写历史；
        这里额外累加所有 chunk 做兜底，确保哪怕自动机制出问题，
        历史也能手动补写。
        """
        chunks = []
        stream = self.chain.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        for chunk in stream:
            chunks.append(chunk)
            yield chunk

        full_answer = "".join(chunks)
        print(f"[query_stream] 流结束，聚合答案长度={len(full_answer)}")

        history = get_history(session_id)
        last_two = history.messages[-2:] if len(history.messages) >= 2 else []
        already_saved = (
            len(last_two) == 2
            and isinstance(last_two[0], HumanMessage)
            and last_two[0].content == question
            and isinstance(last_two[1], AIMessage)
            and last_two[1].content == full_answer
        )
        if not already_saved:
            print("[query_stream] 自动保存未生效，手动补写历史")
            history.add_messages([
                HumanMessage(content=question),
                AIMessage(content=full_answer),
            ])


if __name__ == '__main__':
    rag = RAGService()
    sid = "test_session"
    print(rag.query("周杰伦是谁", session_id=sid))
    print("—" * 40)
    print(rag.query("他的代表作有哪些", session_id=sid))
