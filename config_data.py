from langchain_community.embeddings import DashScopeEmbeddings

md5_path = "./md5.txt"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# RecursiveCharacterTextSplitter
chunk_size = 1000                                                   # 每个文本块的最大长度(字符数)
chunk_overlap = 100                                               # 相邻块之间的重叠字符数,保留上下文连续性
length_function = len                                               # 长度计算函数,默认按字符数算;要按 token 切可换成 tiktoken 编码函数
separators = ["\n\n", "\n", "。", "!", "?", ",", "、", " ", ""]   # 分隔符优先级:先按段落切,依次降级到换行、句号、问号……最后按字符硬切
is_separator_regex = False

# 分隔符按普通字符串匹配,不作正则解析

# 相似度检索阈值
similarity_threshold = 2

#模型配置
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# 聊天历史存储
chat_history_storage_path = "./chat_history"
