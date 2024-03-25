# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser #parses the output into a string, used in line 107
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json
from langchain_community.chat_models import ChatOpenAI ###
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

WRITER_SYSTEM_PROMPT = "You are a Food Science research assistant. Your sole purpose is to retrieve authentic information from the given text."  # noqa: E501

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """

Information:
--------
{research_summary}
--------
Using the only the above information, answer the following question: "{question}"  \
Always provide the output in json format: 
example question:  what is the ph value and titratable acidity of item_name?
in the example output json oject, there would be key-value pairs for 'item_name', 'ph value', 'titratable acidity' and 'description'. 
The 'item_name' key should always be included mandatorily. For the question: "what is the ph value of mango?", the 'item_name' refers to 'mango'.
For each key, the value must be another json object having two keys , 1. 'key_value', 2.'source_url'.
An additional key 'other_sources' must be included listing other urls as key values.

"""  # noqa: E501

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
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106",response_format={ "type": "json_object" }) |StrOutputParser()|json.loads

#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes

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

    
