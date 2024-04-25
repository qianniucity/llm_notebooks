# 翻译

**执行步骤:**
1. Install docker
2. 创建一个虚拟环境，名称为 `myenv`， python 3.10.13
`conda create --name myenv python=3.10.13`
3. 激活虚拟环境
`conda activate myenv`
4. 安装必要的依赖
`pip install -r requirements.txt`
5. 在 "translation/src/"下创建名为"/data "的文件夹，并从 https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset 添加审查数据。
6. 在 `translation/src/` 下创建名为 `/env`的文件夹，并添加包含以下内容的文件：
    - postgres.env
    ```
    POSTGRES_DB=postgres
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=root
    ```
    - connection.env
    ```
    DRIVER=psycopg2
    HOST=postgres
    PORT=5432     
    DATABASE=postgres
    USERNAME=admin
    PASSWORD=root
    ```

7. 从 https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF/tree/main 下载 Llama 模型 `nous-hermes-llama-2-7b.Q4_0.gguf` 并将其添加到 `translation/src/model/` 中。
8. 运行命令 `docker-compose up --build`，然后就可以在浏览器中打开 http://localhost:8501/ 并进行聊天了！
9. 或在笔记本中运行 `translation.ipynb` 命令

## Folder Structure:
------------

    ├── translation
    │
    ├──────── src 
    ├────────── base                                          <- Configuration class
    ├────────── classifier                                    <- Language Detector class
    ├────────── encoder                                       <- Encoder class
    ├────────── generator                                     <- Generator class
    ├────────── retriever                                     <- Retriever class
    ├────────── translator                                    <- Translator class
    ├────────── data                                          <- csv file
    ├────────── env                                           <- env files
    ├────────── model                                         <- GGUF Llama
    │
    │────────── config.yaml                                   <- Config definition
    │────────── lang_map.yaml                                 <- language mapping between XLM-RoBERTa and mBART
    │
    │────────── translation.ipynb                             <- notebook
    │────────── populate.py                                   <- python script to populate PGVector
    │────────── app.py                                        <- streamlit application to chat with our LLM
    │
    │──────── requirements.txt                                <- package versions
    │──────── docker-compose.yaml
    └──────── Dockerfile
--------
