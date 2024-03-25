def modified_scrape_and_summarize_chain(data):
    text_to_summarize = ""
    if data.get("local_pdf_path"):
        text_to_summarize += scrape_text(data["local_pdf_path"])
    if data.get("url"):
        text_to_summarize += scrape_text(data["url"])[:15000]  # Limit character count
    return RunnablePassthrough.assign(
        summary=RunnablePassthrough.assign(
            text=lambda x: x
        ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
    )(data | {"text": text_to_summarize})

# Update the chain usage to use the modified version
chain = RunnablePassthrough.assign(
    summary=modified_scrape_and_summarize_chain.map()
) | (lambda x: f"URL: {x.get('url', '')}\n\nSUMMARY: {x['summary']}")
