# 利用 RAG 管道和内存构建实时 AI 语音助手 | Mistral LLM | Ollama

该存储库包含语音助手的代码，该语音助手与 AI 模型交互以实现自然语言理解 (NLU)。该助手旨在记录用户的音频输入，进行转录，然后与人工智能模型交互以提供相关响应。

## 特征

- 分块记录用户的音频输入。
- 使用预先训练的 AI 模型转录录制的音频。
- 与 AI 模型交互，根据用户输入生成响应。
- 利用知识库进行上下文感知响应。

## 先决条件


在运行代码之前，请确保您已安装以下依赖项：

- Python above 3.8
- `pyaudio`
- `numpy`
- `faster_whisper` (Installable via pip)
- `qdrant_client` (Installable via pip)
- Other dependencies specified in `requirements.txt`

## 用法

1. 将此存储库克隆到您的本地计算机。

   ```bash
   git clone https://github.com/mcks2000/llm_notebooks.git
   cd voice_assistant_llm
   ```

2. 使用 pip 安装依赖项。

   ```bash
   pip install -r requirements.txt
   ```

3. 运行主脚本 app.py

   ```bash
   python app.py
   ```

4. 按照提示与语音助手进行交互。出现提示时对着麦克风讲话。

## 配置
- 您可以根据您的要求调整脚本中的默认模型大小和块长度。
- 根据需要修改知识库和AI模型相关的路径和设置。

## Notes
- 确保系统的麦克风已正确配置并且可由脚本访问。
- 确保妥善处理异常和错误，尤其是在录音和转录过程中。


## 致谢
- 本项目使用的AI模型基于faster_whisper。
- 特别感谢 pyaudio、numpy 和 scipy 开发者的贡献。