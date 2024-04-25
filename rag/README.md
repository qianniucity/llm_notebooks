# 检索增强生成（RAG）

**执行代码的步骤:**
1. 创建一个虚拟环境，名称为 `myenv`， python 3.10.13
`conda create --name myenv python=3.10.13`
2. 激活虚拟环境
`conda activate myenv`
3. 安装必要的依赖
`pip install -r requirements.txt`
1. 在 `rag/` 下创建名为 model 的文件夹
2. 从 https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF/tree/main 下载 Llama 模型 `nous-hermes-llama-2-7b.Q4_0.gguf` 并将其添加到 `model/` 中
3. 运行笔记本

## 文件夹结构:
------------

    ├── RAG
    │
    ├──────── base           <- Configuration class
    ├──────── encoder        <- Encoder class
    ├──────── generator      <- Generator class
    ├──────── retriever      <- Retriever class
    │
    │──── config.yaml        <- Config definition
    │──── requirements.txt   <- package version for installing
    │
    └──── rag.ipynb          <- notebook to run the code
--------
