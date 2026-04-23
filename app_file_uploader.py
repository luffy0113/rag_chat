import time

import streamlit as st

from knowledge_base import KnowledgeBaseService

st.title("知识库更新服务")

uploader_file = st.file_uploader(
    label = "请上传TXT文件",
    type = ['TXT'],
    accept_multiple_files = False,      #仅接受一个文件上传
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024    #字节转KB

    st.subheader(f"文件名:{file_name}")
    st.write(f"文件类型:{file_type}")
    st.write(f"文件大小:{file_size:.2f} KB")

    # 得到文件内容
    text = uploader_file.getvalue().decode("utf-8")

    with st.spinner("载入知识库中。。。"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)

