
# 从PDF到智能回答：LLama3和Adaptive RAG在问答系统中的应用
本文深入探讨了如何使用LLama3和Adaptive RAG构建一个高效、智能的问答系统。从设置环境变量开始，到处理用户上传的PDF文件，再到使用大语言模型（LLM）和向量存储技术优化问题回答过程，本文提供了一步步的指南。我们还探讨了如何通过问题重写和文档相关性评分，提高系统的准确性和效率。通过实际案例，我们展示了如何将理论应用到实践，打造出能够理解和回答复杂查询的智能问答机器人。此外，文章还讨论了根据查询确定执行路线的重要性，以及未来可能出现的用于路由确定的LLM和其他模型的趋势。

## 安装

- 步骤1: 安装Ollama https://ollama.ai
- 步骤2: pip install -r requirements.txt
- 启动命令
```shell
$ streamlit run adaptive_rag.py 
```

## 资源
- [Tavily KEY 申请地址](https://app.tavily.com/home)