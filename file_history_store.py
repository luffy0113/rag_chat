"""
文件形式的聊天历史存储：按 session_id 一会话一文件，持久化为本地 JSON。
"""
import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import config_data as config


def get_history(session_id: str) -> "FileChatMessageHistory":
    """根据 session_id 返回一个对应的历史存储对象"""
    return FileChatMessageHistory(
        session_id=session_id,
        storage_path=config.chat_history_storage_path,
    )


class FileChatMessageHistory(BaseChatMessageHistory):
    """将聊天消息按会话持久化到本地 JSON 文件"""

    def __init__(self, session_id: str, storage_path: str):
        """
        :param session_id: 会话唯一标识，用作文件名
        :param storage_path: 所有会话文件所在的文件夹路径
        """
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(storage_path, f"{session_id}.json")

        os.makedirs(self.storage_path, exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """在历史末尾追加一批新消息，并同步写回文件"""
        all_messages = list(self.messages)
        all_messages.extend(messages)

        data = [message_to_dict(m) for m in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        """读出当前会话的所有历史消息，文件不存在则返回空列表"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        """清空当前会话的历史"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
