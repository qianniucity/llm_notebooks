# *RAG，即 检索增强生成（Retrieval Augmented Generation）*
## RAG 简介及其应用
RAG，即 检索增强生成（Retrieval Augmented Generation），是一种通过将信息检索（IR）融入生成过程来提高大型语言模型（LLM）在自然语言生成（NLG）任务中性能的技术。

## 简而言之：

**大型语言模型（LLM）** 功能强大，能够生成文本、翻译语言、创作不同类型的创意内容，并以信息丰富的方式回答您的问题。但是，它们在生成准确和相关的响应方面有时会遇到困难，尤其是在处理复杂或开放式任务时。

**信息检索（IR）** 是从大量数据集合中查找相关文档或信息的过程。IR 系统擅长识别和检索与特定查询或关键字匹配的信息。

## RAG 结合了 LLM 和 IR 的优势，在 NLG 任务中取得了更好的成果：

1. LLM 提供生成创造性和流畅文本的能力。

2. IR 帮助 LLM 访问和利用来自外部来源的相关信息，确保生成的文本准确、一致并得到证据支持。

## 使用 RAG 的优势：

**提高准确性和相关性：** 支持 RAG 的 LLM 可以生成更准确和更相关的响应，尤其是在处理复杂或开放式任务时。

**知识基础：** RAG 允许 LLM 将其响应建立在真实世界的信息之上，使其更加可靠和可信。

**提高创造力：** RAG 可以激发 LLM 生成更具创意和趣味性的文本格式，例如诗歌、代码、脚本、音乐作品、电子邮件、信件等。

## RAG 的应用：

**问答：** RAG 可以帮助 LLM 提供更全面和更具信息性的问题解答，即使这些问题是开放式的、具有挑战性的或需要多个步骤。

**摘要：** RAG 可以帮助 LLM 生成对冗长文档或文本段落的简洁而准确的摘要。

**创意写作：** RAG 可以激发 LLM 创作更具创意和吸引力的文本格式，例如故事、诗歌、脚本和音乐作品。

**对话生成：** RAG 可以促进人与 LLM 之间的更自然和更吸引人的对话。

---

## 案例

### 1、RAG 简单用例
- 描述：我们将探讨如何在不对此类数据进行微调的情况下，利用 RAG 的力量，使 LLM 准确回答 OpenAI 的最新消息
- 代码地址：[ Simple Demo](./simple_demo/)



### 2、使用 LLaMA3 的本地 RAG 代理
- 描述：对所提供的电子邮件进行全面分析，并将其归类为回复。
- 代码地址：[使用 LLaMA3 的本地 RAG 代理](./langgraph_rag_agent_llama3_local.ipynb)
- <a target="_blank" href="https://colab.research.google.com/github/mcks2000/llm_notebooks/blob/main/rag/langgraph_rag_agent_llama3_local.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### 要点：
我们将把 RAG 论文中的观点结合到 RAG 代理中
- **路由:**  自适应 RAG ([paper](https://arxiv.org/abs/2403.14403)). 将问题路由到不同的检索方法
- **回退:** 纠正 RAG ([paper](https://arxiv.org/pdf/2401.15884.pdf)). 如果文档与查询不相关，则回退到网络搜索
- **自我纠正:** Self-RAG ([paper](https://arxiv.org/abs/2310.11511)). 修正有幻觉或不符合问题的答案


### 3、Groq LPU 与 DataStax Enterprise：构建下一代AI驱动网络应用
- 描述：本文将深入探讨 Groq 的创新语言处理单元（LPU）与 DataStax Enterprise（DSE）、Cassio 和 Langchain 的强大数据管理功能之间的完美结合。 
- 代码地址：[Groq LPU 与 DataStax Enterprise：构建下一代AI驱动网络应用](./langchain_rag_groq.ipynb)
- <a target="_blank" href="https://colab.research.google.com/github/mcks2000/llm_notebooks/blob/main/rag/langchain_rag_groq.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 4、使用 Ollama 和 Weaviate 构建用于隐私保护的本地 RAG 系统
- 描述：如何在没有外部依赖的本地环境中，仅使用以下本地组件，用Python实现一个基于RAG的聊天机器人
- 使用Ollama的本地LLM和嵌入模型  
- 通过Docker使用Weaviate的本地向量数据库实例 
- 代码地址：[使用 Ollama 和 Weaviate 构建用于隐私保护的本地 RAG 系统](./Ollama_Weaviate_Local_rag.ipynb)
- <a target="_blank" href="https://colab.research.google.com/github/mcks2000/llm_notebooks/blob/main/rag/Ollama_Weaviate_Local_rag.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>