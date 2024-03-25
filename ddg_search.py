import requests
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

RESULTS_PER_QUESTION = 5   


ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

query_1="is this even working?"

  
print(web_search(query_1))
