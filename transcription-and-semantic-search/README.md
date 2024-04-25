# 多语种转录和语义搜索

**运行代码的步骤:**
1. 安装 docker
2. 创建一个虚拟环境，名称为 `myenv`， python 3.10.13
`conda create --name myenv python=3.10.13`
3. 激活虚拟环境
`conda activate myenv`
4. 安装必要的依赖
`pip install -r requirements.txt`
5. 在 "transcription-and-semantic-search/"下创建名为"/data "的文件夹，然后添加视频
6. 创建名为`/env`的文件夹，并添加 3 个文件，内容如下：
    - connection.env
    ```
    DRIVER=psycopg2
    HOST=localhost
    PORT=5432     
    DATABASE=postgres
    USERNAME=admin
    PASSWORD=root
    ```
    - pgadmin.env
    ```
    PGADMIN_DEFAULT_EMAIL=admin@admin.com
    PGADMIN_DEFAULT_PASSWORD=root
    ```
    - postgres.env
    ```
    POSTGRES_DB=postgres
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=root
    ```
3. Run the command `docker-compose up -d`
4. Run the notebook

## 文件结构:
------------

    ├── transcription-and-semantic-search
    │
    ├──────── base                                          <- Configuration class
    ├──────── encoder                                       <- Encoder class
    ├──────── transcriptor                                  <- WhisperX class
    ├──────── data                                          <- videos and audios
    ├──────── env                                           <- env files
    │
    │──── config.yaml                                       <- Config definition
    │──── requirements.txt                                  <- package version for installing
    │
    └──── multilingual_transcription_semantic_search.ipynb  <- notebook to run the code
--------
