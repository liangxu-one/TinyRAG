import os
import sys
import argparse
from util import LLM_and_embedding, FileReader, VectorDb, getResponse, data_process
from evaluate import RagEvaluator

os.environ['API_KEY'] = 'your-api-key'

def parse_arguments():
    parser = argparse.ArgumentParser(description = "RAG system.")
    parser.add_argument("--file_path", type = str, required = False, default = "doc",
                        help = "The path to the doc file directory")
    parser.add_argument("--file_name", type = str, required = True,
                        help = "The name to the doc file.")
    parser.add_argument("--model_path", type = str, required = False, default = "embedding_model",
                        help = "The path to the model directory.")
    parser.add_argument("--model_name", type = str, required = True,
                        help = "The name to the model.")
    parser.add_argument("--persist_directory", type = str, required = False, default = "chroma",
                        help = "The directory to persist the vector database.")

    return parser.parse_args()

def main(args):
    # 初始化LLM与embedding
    model_path = os.path.join(sys.path[0], args.model_path, args.model_name)
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    llm_and_embedding = LLM_and_embedding(api_key = os.environ['API_KEY'], model_path = model_path, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)

    # 知识库读取
    file_path = os.path.join(sys.path[0], args.file_path, args.file_name)
    Reader = FileReader(file_path = file_path)
    pages_splitter = Reader.load_and_split()

    # 初始化向量数据库
    persist_directory = os.path.join(sys.path[0], args.persist_directory)
    vectordb = VectorDb(pages_splitter = pages_splitter, embedding = llm_and_embedding.embedding, persist_directory = persist_directory)

    # 根据query召回片段
    query = input("请输入您想要咨询的问题:\n")
    docs = vectordb.similarity_search(query)
    contexts = []
    for i in range(len(docs)):
        contexts.append(docs[i].page_content)

    # prompt处理
    prompt = data_process(query, contexts)
    print("Prompt: ", prompt, "\n")

    # 模型调用
    response = getResponse(llm = llm_and_embedding.llm, prompt = prompt)
    if response == '':
        print("response: 调用失败, 请重新提问\n")
    else:
        print("response: ", response, "\n")

    # 对结果进行评测
    ragEvaluator = RagEvaluator()
    rag_results = {'question': [query], 
                   'answer': [response],
                   'contexts': [contexts]}
    result = ragEvaluator.evaluate(llm = llm_and_embedding.llm, embedding = llm_and_embedding.embedding, rag_results = rag_results)
    print(result)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)