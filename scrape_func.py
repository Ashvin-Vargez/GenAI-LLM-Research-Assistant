def scrape_text(path_or_url):
    if path_or_url.lower().startswith('http'):
        # Existing logic for web scraping (using requests and BeautifulSoup)
        if path_or_url.lower().endswith('.pdf'):
            try:
            # Send a GET request to the PDF link
                response = requests.get(path_or_url)

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
            response = requests.get(path_or_url)

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
        
        else:
        # Handle local PDF files
            try:
                with open(path_or_url, 'rb') as f:
                    pdf_file = BytesIO(f.read())
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    pdf_text = ""
                    for page_number in range(len(pdf_reader.pages)):
                        pdf_text += pdf_reader.pages[page_number].extract_text()
                    return pdf_text
            except Exception as e:
                print(f"Failed to read PDF file: {e}")
                return f"Failed to read PDF file: {e}"