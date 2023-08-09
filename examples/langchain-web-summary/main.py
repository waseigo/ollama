from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

llm = Ollama(model="llama2")
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.run(docs)
print(result)
