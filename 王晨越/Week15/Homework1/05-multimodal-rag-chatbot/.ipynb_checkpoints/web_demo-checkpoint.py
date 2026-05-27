import streamlit as st
from PIL import Image
import io

# --- 模拟后端函数 (需要替换为实际的 RAG 逻辑) ---
def get_rag_response(query, selected_docs):
    response_text = f"基于选定的文档 {selected_docs}，这是对 {query} 的回答。"
    image_flag = "placeholder_image" if "图片" in query or "图" in query else None
    return response_text, image_flag

def get_existing_documents():
    return ["文档A - RAG原理.pdf", "文档B - Streamlit教程.txt", "文档C - 部署指南.docx"]

def handle_file_upload(uploaded_file):
    st.session_state.existing_docs.append(uploaded_file.name)

# --- Streamlit 页面配置 ---
st.set_page_config(
    page_title="RAG 图文问答平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 RAG 图文知识问答平台")

# --- 初始化 Session State ---
if 'existing_docs' not in st.session_state:
    st.session_state.existing_docs = get_existing_documents()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# --- 侧边栏：文档管理 ---
with st.sidebar:
    st.header("📄 文档管理")
    
    st.subheader("上传新文档")
    uploaded_file = st.file_uploader(
        "选择文件（支持PDF/TXT/DOCX等）", 
        type=["pdf", "txt", "docx", "pptx", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=False
    )
    if uploaded_file is not None and uploaded_file not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(uploaded_file)
        handle_file_upload(uploaded_file)

    st.markdown("---")
    
    st.subheader("选择问答范围")
    selected_docs = st.multiselect(
        "选择要问答的文档",
        options=st.session_state.existing_docs,
        default=st.session_state.existing_docs
    )
    
    st.markdown("---")
    
    if st.button("清空聊天记录"):
        st.session_state.chat_history = []
        st.rerun()

# --- 主体区域：问答界面 ---

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["type"] == "mixed":
            st.image(message["image"], caption="检索/生成的图片结果", use_column_width=True)
            
if prompt := st.chat_input("输入您的问题... (可尝试包含 '图片' 或 '图' 进行图片检索)"):
    st.session_state.chat_history.append({"role": "user", "content": prompt, "type": "text_only"})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text, image_flag = get_rag_response(prompt, selected_docs)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
        
        if image_flag == "placeholder_image":
            image_url = "https://via.placeholder.com/600x300.png?text=RAG+Retrieved+Image"
            st.image(image_url, caption="检索到的相关图片", use_column_width=True)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text, 
                "type": "mixed",
                "image": image_url
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text, 
                "type": "text_only"
            })

# --- 附加功能：直接输入文本检索图片 ---
st.markdown("---")
st.subheader("🖼️ 文本检索图片 (独立功能)")

image_search_query = st.text_input("输入关键词以检索相关图片：", key="image_search_input")

if st.button("检索图片"):
    if image_search_query:
        st.info(f"正在使用关键词 '{image_search_query}' 检索...")
        
        image_url = "https://via.placeholder.com/400x200.png?text=Search+Result+for+" + image_search_query.replace(" ", "+")
        st.success("图片检索成功！")
        st.image(image_url, 
                 caption=f"检索结果：{image_search_query}", 
                 use_column_width="always")
    else:
        st.warning("请输入检索关键词。")