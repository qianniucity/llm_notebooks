# *知识图谱*
## 什么是知识图谱？
知识图谱，也称为语义网络，表示现实世界实体的网络，即对象、事件、情境或概念，并说明它们之间的关系。这些信息通常存储在图数据库中，并以图形结构可视化，因此称为知识“图谱”。

来源: https://www.ibm.com/topics/knowledge-graph

## 为什么使用图？
一旦建立了知识图谱（KG），我们可以用它进行许多用途。我们可以运行图算法并计算任何节点的中心性，了解一个概念（节点）对这篇作品有多重要。我们可以计算社区，将概念分组在一起以更好地分析文本。我们可以了解看似不相关的概念之间的关系。

最重要的是，我们可以实现图检索增强生成（GRAG），并使用图作为检索器以更深入地与我们的文档交流。这是**检索增强生成（RAG）**的新版本，其中我们使用向量数据库作为检索器，以便与我们的文档进行交互。

---

## 案例

### 1、如何使用Llama3和Hugging Face优化关系提取任务
- 描述：关系提取(RE)是一项任务，旨在从非结构化文本中识别出各种命名实体之间的联系。它与命名实体识别(NER)配合使用，是自然语言处理流程中不可或缺的一步。随着大型语言模型(LLM)的崛起，那些需要标注实体范围并对它们之间的关系进行分类的传统监督方法得到了增强，甚至被基于LLM的方法所取代。
- 代码地址：[如何使用Llama3和Hugging Face优化关系提取任务](./llama3_re/)
- <a target="_blank" href="https://colab.research.google.com/github/mcks2000/llm_notebooks/blob/main/knowlage_graph/llama3_re/Llama3_RE_Inference_SFT.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
