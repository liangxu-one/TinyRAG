import os
import json
import shutil
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# LLM与embedding初始化
class LLM_and_embedding:
    def __init__(self, api_key, model_path, model_kwargs, encode_kwargs):
        self.llm = ChatOpenAI(
            api_key = api_key, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", # 填写DashScope base_url
            model = "qwen-long",
            )
        self.embedding = HuggingFaceEmbeddings(model_name = model_path, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)

# 文件读取, 支持处理pdf, txt, docx文件
class FileReader:
    def __init__(self, file_path, chunk_size = 200, chunk_overlap = 50, length_function = len):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = length_function)
        self.loader = None

    def load_text(self):
        if self.file_path.endswith('.pdf'):
            self.loader = PyPDFLoader(self.file_path, extract_images = True)
        elif self.file_path.endswith('.txt'):
            self.loader = TextLoader(self.file_path, encoding = 'utf-8')
        elif self.file_path.endswith('.docx'):
            self.loader = Docx2txtLoader(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .txt, .pdf, or .docx file.")
        return self.loader

    def load_and_split(self):
        self.load_text()
        pages_splitter = self.loader.load_and_split(self.text_splitter)
        return pages_splitter

# 向量数据库
class VectorDb:
    def __init__(self, pages_splitter, embedding, persist_directory):
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.mkdir(persist_directory)

        self.vectorstore = Chroma.from_documents(
            documents = pages_splitter,
            embedding = embedding,
            persist_directory = persist_directory
        )

    def similarity_search(self, query, topK = 4):
        docs = self.vectorstore.similarity_search(query = query, k = topK)
        return docs

def getResponse(llm, prompt):
    messages = [
        {"role":"user","content":prompt}
    ]
    try:
        response = llm.invoke(messages)
        response = json.loads(response.json(ensure_ascii=False))['content']
    except:
        response = ''
    return response

prompt = '''你是一个智能助手, 你的任务是采用以下【文档】的内容, 回答用户咨询的【问题】.
【文档】: ###文档###
【问题】: ###问题###
你的回答必须紧密依托于提供的【文档】资料, 鼓励适当的语言润色来增强回答的可读性, 但必须忠实于【文档】的内容. 如果你认为提供的【文档】无法回答问题, 那请你回答不知道, 不要进行任何杜撰. 你的回答中不要存在"根据【文档】"这类表述.'''

# 将prompt, query, contexts拼接在一起
def data_process(query, contexts):
    rag_content = '\n'
    for i in range(len(contexts)):
        rag_content = rag_content + contexts[i]
        if i < len(contexts) - 1:
            rag_content = rag_content + '\n\n'
    result = prompt.replace('###文档###', rag_content).replace('###问题###', query)
    return result