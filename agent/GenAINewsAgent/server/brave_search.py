from brave import AsyncBrave
from configs import BRAVE_API_KEY
from typing import List, Dict, Union


class BraveSearch:

    def __init__(self) -> None:
        self.brave_client = AsyncBrave(BRAVE_API_KEY)

    def __parse_results__(self, search_results: Dict):
        # print(type(search_results))
        result_keys = ["web", "news"]
        results = []
        for key in result_keys:
            if hasattr(search_results, key):
                # if key in search_results:
                for result in getattr(getattr(search_results, key), "results"):
                    # for result in search_results[key].get("results"):
                    results += [{
                        "title":
                        result.title,
                        "url":
                        result.url,
                        "description":
                        result.description,
                        "page_age":
                        result.page_age,
                        "age":
                        result.age,
                        "is_breaking_news":
                        getattr(result, "breaking", False)
                    }]
        return results

    async def __call__(self, query: str, **kwargs):
        search_results = await self.brave_client.search(query,
                                                        count=3,
                                                        **kwargs)
        # print(type(search_results.__dict__))
        # print(search_results.__dict__)
        results = self.__parse_results__(search_results)
        return results


if __name__ == "__main__":
    import asyncio
    bs = BraveSearch()
    asyncio.run(bs("lok sabha elections 2024"))
