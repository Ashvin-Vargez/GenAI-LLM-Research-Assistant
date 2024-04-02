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

from langchain_community.tools import DuckDuckGoSearchResults

RESULTS_PER_QUESTION = 5   

ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    # print("these are the search results", results)
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
            "draft a detailed search query which can "
            "search for and retrieve relevant information asked for in the following question:"
            "question: {question}\n"
            "Return the  detailed search query  in the following format: "
            '[" detailed search query "].'
            ,
        ),
    ]
)

# search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

# full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()


WRITER_SYSTEM_PROMPT = "You are a Food Science research assistant. Your sole purpose is to retrieve authentic information from the given text containing URL-Summary Pairs."  # noqa: E501

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """
From this question identify the item_name and parameters queried for,  question: "{question}" \
In the Example Question : What are the ph and titratable acidity of tomato?, the item_name is tomato and the parameter_1 is ph and parameter_2 is titratable acidity.

Now based on the information stuctured as URL-Summary pairs given below, 
Information:
--------
{research_summary}
--------

For each URL-summary pair in the above information, extract the item_name and parameter values into a dictionary of the following format: 

  
    (curly brace opening)
      "source_url": (should show the exact url name from which the values are extracted)
      "item_name": (the name of the item regarding which the information is requested)
      "name_of_parameter_1":((replace "name_of_parameter_1" with the actual name of the parameter 1(in the given example, it is "ph"), and the key's value must be the numerical value, if the numerical value is not available, then the value range, and if numerical value or value range is unavailable,  then description  of parameter_1 obtained from the snippet corresponding to the url)
      "name_of_parameter_2": ((replace "name_of_parameter_2" with the actual name of the parameter 2(in the given example, it is "titratable acidity"), and the key's value must be the numerical value, if the numerical value is not available, then the value range, and if numerical value or value range is unavailable,  then description  of parameter_2 obtained from the snippet corresponding to the url)
      "url_text": ( the text snippet associated with this specific url)
    (curly brace closing)

The number of parameters should be as many as identified from the question and the names of the parameters should always match what is asked for in the question. 
You must mandatorily create dictionaries for each source url, even if the parameter values are not present in the text snippet associated with it. Also, each dictionary must mandatorily have keys corresponding to all the parameters identified from the question.
Now, create a json object containing all the dictionaries as values, the keys must be of the format source_1, source_2 etc. The json object must strictly follow this format. Output only the final json object  

Let's think step by step:

# """  # noqa: E501

 # noqa: E501

# RESEARCH_REPORT_TEMPLATE = """
# Based only on the the information stuctured as URL-Summary pairs given below, 
# Information:
# --------
# {research_summary}
# --------

# answer this question independently for each URL using it's given summary,  question: "{question}" \
# Give the answer in the following format.
# ###
# 1.source url : www.example1.com
# answer : "answer retrieved based on www.example1.com"
# 2.source url : www.example2.com
# answer : "answer retrieved based on www.example2.com"
# ....
# ###
# # """  
# RESEARCH_REPORT_TEMPLATE = """
# ###
# Example  1 : What are the ph and titratable acidity of tomato?-> item_name: "tomato", parameter_1: "ph', parameter_2: 'titratable acidity'

# Example  2 : What are the ph, brix and water activity of watermelon?-> item_name: "warermelon", parameter_1: "ph', parameter_2: 'brix', parameter_3: 'water activity'
# ###

# Based on the examples above, from the given question, identify the item_name and parameters queried for,  question: "{question}" \

# Now based on the information given below stuctured as URL-Summary pairs, extract all the source_urls and their summaries.
# Information:
# --------
# {research_summary}
# --------

# Now for each source_url, create separate dictionaries based on the summary text corresponding to that source_url alone.
 
# The dictionaries must find and populate the values for the following keys: source_url, item_name, summary, separate keys for parameters identified from the question given in the beginning. 

# provide output as json. 

# Let's think step by step:

# """ 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    # print("this is the list",list_of_lists[:5])
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    # print("this is the content",content)
    return "\n\n".join(content)
    

# chain = RunnablePassthrough.assign(
#     research_summary= full_research_chain | collapse_list_of_lists
# ) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106",response_format={ "type": "json_object" }) |StrOutputParser()|json.loads

chain = RunnablePassthrough.assign(
    research_summary= web_search_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106",response_format={ "type": "json_object" }) |StrOutputParser()|json.loads

# chain_2=RunnablePassthrough.assign(
#     research_summary= full_research_chain | collapse_list_of_lists
# )
# response=chain_2.invoke({"question":"what is the ph value of tomato" })
# print("this is the response", response)



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


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)

    
