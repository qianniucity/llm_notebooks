import os
import httpx
from configs import NEWS_API_KEY, NEWS_BASE_URL


async def getNews(query: str, max_size: int = 8):
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            os.path.join(NEWS_BASE_URL, "news") +
            f"?apiKey={NEWS_API_KEY}&q={query}&size={max_size}")
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(
                f"Error resposne {e.response.status_code} while requesting {e.request.url!r}"
            )
            return None
