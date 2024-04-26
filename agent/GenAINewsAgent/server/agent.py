from llms.groq import GroqLLMStream
from configs import GROQ_API_KEY, GROQ_MODEL_NAME
from news import getNews
from prompts import SYSTEM_PROMPT
from brave_search import BraveSearch
from time import time

llm = GroqLLMStream(GROQ_API_KEY)

bs = BraveSearch()


async def newsAgent(query: str):
    # retrieved_news_items = await getNews(query)
    st_time = time()
    retrieved_news_items = await bs(query)
    en_time = time()
    print(f'Search Time: {en_time - st_time}s')
    if not retrieved_news_items:
        yield "\n_Cannot fetch any relevant news related to the search query._"
        return
    messages = [{
        "role":
        "user",
        "content":
        f"Query: {query}\n\nNews Items: {retrieved_news_items}"
    }]
    async for chunk in llm(GROQ_MODEL_NAME,
                           messages,
                           system=SYSTEM_PROMPT,
                           max_tokens=1024,
                           temperature=0.2):
        yield chunk
