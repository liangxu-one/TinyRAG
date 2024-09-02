# TinyRAG
一个适用中文场景的轻量RAG系统, 基于langchain构建, 并使用RAGAS对检索内容与模型回答进行评价. 使用的评价指标可以查看evaluate.py文件, 对RAGAS中适用于英文场景的指标进行了改写, 以适用于中文场景下的RAG.  

执行命令: python main.py --file_name your_doc_name --model_name your_model_name

注意:  
python环境请使用3.11版本及以上, 本代码中使用了Qwen作为LLM, 若要使用, 请使用自己的API-KEY.