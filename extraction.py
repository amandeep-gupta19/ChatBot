from langchain_community.document_loaders import WebBaseLoader

# # URL to scrape
url = "https://brainlox.com/courses/category/technical"



# # Load data from the URL
loader = WebBaseLoader(url)
documents = loader.load()

# # Print the extracted data
for doc in documents:
      print(doc.page_content)