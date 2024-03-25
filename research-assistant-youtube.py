# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser #parses the output into a string, used in line 107
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json
# from langchain_community.chat_models import ChatOpenAI ###
from langchain_openai import ChatOpenAI 
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from io import BytesIO
import PyPDF2
import os #
from dotenv import load_dotenv#
load_dotenv()#
os.environ["OPENAI_API_KEY"]= str(os.getenv("OPENAI_API_KEY"))#
LANGCHAIN_TRACING_V2 = 'true'#
os.environ["LANGCHAIN_API_KEY"] =str(os.getenv("LANGCHAIN_API_KEY"))#
from langchain.callbacks.tracers import LangChainTracer#

tracer = LangChainTracer(project_name="RA_tavily_LC")#



RESULTS_PER_QUESTION = 5   

# result_block = 0
# start_point=5*result_block
# RESULTS_PER_QUESTION = start_point+5

ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text} 

-----------

Using the above text, answer in short the following question: 

> {question}

-----------
if the question cannot be answered using the text, simply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)




def scrape_text(url: str):
    # Check if the link is a PDF file
    if url.lower().endswith('.pdf'):
        try:
            # Send a GET request to the PDF link
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Read the PDF content using PyPDF2
                pdf_file = BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pdf_text = ""

                # Extract text from each page in the PDF
                for page_number in range(len(pdf_reader.pages)):
                    pdf_text += pdf_reader.pages[page_number].extract_text()

                # Print or return the extracted text from the PDF
                return pdf_text
            else:
                return f"Failed to retrieve the PDF file: Status code {response.status_code}"
        except Exception as e:
            print(e)
            return f"Failed to retrieve the PDF file: {e}"
    else:
        # If the link is not a PDF file, proceed with HTML scraping
        try:
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                page_text = soup.get_text(separator=" ", strip=True)

                # Print or return the extracted text from the webpage
                return page_text
            else:
                return f"Failed to retrieve the webpage: Status code {response.status_code}"
        except Exception as e:
            print(e)
            return f"Failed to retrieve the webpage: {e}"



scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:15000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

## This is for Arxiv

# from langchain.retrievers import ArxivRetriever
# 
# retriever = ArxivRetriever()
# SUMMARY_TEMPLATE = """{doc} 
# 
# -----------
# 
# Using the above text, answer in short the following question: 
# 
# > {question}
# 
# -----------
# if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
# SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
# 
# 
# scrape_and_summarize_chain = RunnablePassthrough.assign(
#     summary =  SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
# ) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")
# 
# web_search_chain = RunnablePassthrough.assign(
#     docs = lambda x: retriever.get_summaries_as_docs(x["question"])
# )| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()



SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "draft a search query which can "
            "search for and retrieve relevant information asked for in the following question:"
            "question: {question}\n"
            "You must respond with a list of string in the following format: "
            '[" detailed query "].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are a Food Science research assistant. Your sole purpose is to retrieve authentic information from the given text containing URL-Summary Pairs."  # noqa: E501

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
# RESEARCH_REPORT_TEMPLATE = """
# From this question identify the item_name and parameters queried for,  question: "{question}" \
# In the Example Question : What are the ph and titratable acidity of tomato?, the item_name is tomato and the parameter_1 is ph and parameter_2 is titratable acidity.

# Now based on the information stuctured as URL-Summary pairs given below, 
# Information:
# --------
# {research_summary}
# --------

# For each URL-summary pair, extract the item_name and parameter values.
# Always provide the output in the following json format: 

#   (curly brace opening) "results": [
#     (curly brace opening)
#       "source_url": "https://example1.com",
#       "item_name": "tomato",
#       "ph_value": "6.0-6.8",
#       "titratable_acidity": ""
#     (curly brace closing),
#    (curly brace opening)
#       "source_url": "https://example2.com",
#       "item_name": "tomato",
#       "ph_value": "",
#       "titratable_acidity": ""
#     (curly brace closing)
#   ]
# (curly brace closing)

# Let's think step by step:

# """  # noqa: E501

RESEARCH_REPORT_TEMPLATE = """
Based only on the the information stuctured as URL-Summary pairs given below, 
Information:
--------
{research_summary}
--------

answer this question independently for each URL using it's given summary,  question: "{question}" \
Give the answer in the following format.
###
1.source url : www.example1.com
answer : "answer retrieved based on www.example1.com"
2.source url : www.example2.com
answer : "answer retrieved based on www.example2.com"
....
###
"""  

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    # print(content)
    return "\n\n".join(content)

# chain = RunnablePassthrough.assign(
#     research_summary= full_research_chain | collapse_list_of_lists
# ) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106",response_format={ "type": "json_object" }) |StrOutputParser()|json.loads

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") |StrOutputParser()
#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes

# test_text = '''In order to properly answer the original question ("how can langsmith help with testing?"), we need to provide additional context to the LLM. We can do this via retrieval. Retrieval is useful when you have too much data to pass to the LLM directly. You can then use a retriever to fetch only the most relevant pieces and pass those in.

# In this process, we will look up relevant documents from a Retriever and then pass them into the prompt. A Retriever can be backed by anything - a SQL table, the internet, etc - but in this instance we will populate a vector store and use that as a retriever. For more information on vectorstores, see this documentation.

# First, we need to load the data that we want to index. In order to do this, we will use the WebBaseLoader. This requires installing BeautifulSoup.
# '''

# chain.invoke({"text": test_text, "question": "how does langsmith help?"})


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/research-assistant",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

    
