"""
知识库
"""
import hashlib
import os
from datetime import datetime

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from dotenv import load_dotenv


load_dotenv()

def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了
        return false 就是没处理 true就是处理过
    """

    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding = 'utf-8').close()
        return False
    else:
        for line in open(config.md5_path, "r", encoding = 'utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False


def save_md5(md5_str: str):
    """将传入的md5字符串， 记录到文件内保存"""
    with open(config.md5_path, "a", encoding = 'utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""

    """讲字符串转换为bytes字节数组"""
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()

    return md5_hex



class KnowledgeBaseService(object):

    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma (
            collection_name = config.collection_name,
            embedding_function = DashScopeEmbeddings(model = "text-embedding-v4"),
            persist_directory = config.persist_directory,
        )  # 向量存储的示例Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,  # 每个文本块的最大长度(字符数)
            chunk_overlap = config.chunk_overlap, # 相邻块之间的重叠字符数,保留上下文连续性
            length_function = config.length_function,   # 长度计算函数,默认 len 按字符数算
            separators = config.separators,  # 分隔符优先级:先按段落切,依次降级到换行、句号、问号……最后按字符硬切
            is_separator_regex = config.is_separator_regex,   # 分隔符按普通字符串匹配,不作正则解析
        )

    def upload_by_str(self, data: str, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.chunk_size:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            # 2025-01-01 10:00:00
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小曹",
        }

        self.chroma.add_texts(
            # iterable -> list \ tuple
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        save_md5(md5_hex)
        return "[成功] 内容已经成功载入向量库"

if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰轮", filename="testfile")
    print(r)