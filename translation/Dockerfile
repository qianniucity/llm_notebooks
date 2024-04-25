FROM python:3.10.13-slim
RUN mkdir /install
WORKDIR /install
COPY ./requirements.txt /requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y gcc-11 g++-11 build-essential clang
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /usr/src/translation
COPY ./src /usr/src/translation

CMD ["python", "populate.py"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
