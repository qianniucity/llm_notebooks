# *AI Agent（AI 代理）*
## 什么是AI Agent？
AI Agent，也称为智能代理，是指能够感知周围环境并根据其感知做出自主决策以实现其目标的实体。它通常是一个软件程序，但也可以是机器人或其他类型的物理系统。

AI Agent 通常具有以下几个特征：

- **自主性**: AI Agent 能够独立行动，而无需人工干预。
- **感知能力**: AI Agent 能够感知周围环境，包括其他实体、物体和事件。
- **行动能力**: AI Agent 能够采取行动来改变其周围环境。
- **推理能力**: AI Agent 能够对感知到的信息进行推理，并根据其推理结果做出决策。
- **学习能力**: AI Agent 能够从经验中学习，并随着时间的推移提高其性能。

## AI Agent 可以帮助我们做什么
AI Agent 可以用于各种各样的任务，包括：

- **机器人控制**: AI Agent 可以用于控制机器人，使其能够执行各种任务，例如搬运物体、清洁房间和导航障碍物。
- **游戏**: AI Agent 可以用于玩游戏，例如棋类游戏和电子游戏。
- **自然语言处理**: AI Agent 可以用于处理自然语言，例如机器翻译和文本摘要。
- **计算机视觉**: AI Agent 可以用于从图像和视频中提取信息，例如对象识别和场景理解。
- **个性化推荐**: AI Agent 可以用于向用户推荐产品、服务和内容。
- **欺诈检测**: AI Agent 可以用于检测欺诈行为，例如信用卡欺诈和保险欺诈。
AI Agent 是人工智能研究的一个重要领域，具有广泛的潜在应用。随着 AI Agent 技术的不断发展，我们可以期待它们在未来发挥越来越重要的作用。

以下是一些 AI Agent 在现实世界中的应用示例：

- **Roomba 机器人真空吸尘器** 使用 AI Agent 来导航房间并清理污垢和灰尘。
- **Siri 和 Alexa 等虚拟助手** 使用 AI Agent 来理解自然语言并响应用户的请求。
- **特斯拉自动驾驶汽车** 使用 AI Agent 来感知周围环境并做出驾驶决策。
- **Facebook 的新闻提要** 使用 AI Agent 来个性化用户看到的新闻和广告。
- **阿里巴巴的推荐引擎** 使用 AI Agent 来向用户推荐产品。

---

## 案例

### 1、专业的经济学家 代理
- 描述：llama-3-70b-Instruct 代理一位专业的经济学家，并输出为 Markdown 格式的指令。
- 代码地址：[专业的经济学家](./grop_llama3_agent.ipynb)
- <a target="_blank" href="https://colab.research.google.com/github/mcks2000/llm_notebooks/blob/main/agent/grop_llama3_agent.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

####  要点：
- 🌐 以 API 的方式接入 LlaMa-3-70b-Instruct
- ⚡️ Groq 加成，闪电生成
- ⚙️ Syetem Prompt，提升中文回复的稳定性 & 其他增益
- ⏱️ 实现一个基于字符的简单计速——time 模块
- 🪄 在Notebook IDE 中直接渲染 Markdown——display(Markdown())

### 2、邮件回复 代理
- 描述：对所提供的电子邮件进行全面分析，并将其归类为回复。
- 代码地址：[邮件回复代理](./Email_Reply_Llama3_CrewAI_+_Groq.ipynb)
- <a target="_blank" href="https://colab.research.google.com/github/mcks2000/llm_notebooks/blob/main/agent/Email_Reply_Llama3_CrewAI_+_Groq.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 3、 GenAI 新闻搜索器
- 描述：使用 Llama-3 和 Groq 快速实现 GenAI 新闻摘要代理。可在 6 秒内检索新闻并提供摘要。
- 项目部署： [GenAI 新闻搜索器](./GenAINewsAgent/README.md)
- 代码地址：[GenAI 新闻搜索器](./GenAINewsAgent)

### 4、任务代理
- 描述：我今天花了 5 美元买了一杯咖啡，请记录我的支出。税率为 0.2
- 代码地址：[任务代理](./Agent_OpenAI/)