from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.core.openapi.inference.models import EmbeddingsList
import os
from dotenv import load_dotenv, find_dotenv

class PineconeEmbeddingManager:
    def __init__(self, api_key: str, index_name: str, name_space: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.name_space = name_space
    
    def create_embeddings(self, documents: List[Document], model_name: str = "llama-text-embed-v2") -> EmbeddingsList:
        return self.pc.inference.embed(
            model=model_name,
            inputs=[d.page_content for d in documents],
            parameters={"input_type": "passage", "truncate": "END"}
        )
    
    def store_embeddings(self, embeddings: EmbeddingsList, documents: List[Document]):
        index = self.pc.Index(name=self.index_name)
        records = [
            {"id": f"vector{idx}", "values": e['values'], "metadata": {"text": d.page_content}}
            for idx, (d, e) in enumerate(zip(documents, embeddings))
        ]
        return index.upsert(vectors=records, namespace=self.name_space)
    
    def create_and_store_embeddings(self, documents: List[Document]):
        embeddings = self.create_embeddings(documents)
        result = self.store_embeddings(embeddings, documents)
        print(result)
    
    def search_matching(self, query: str, model: str = "llama-text-embed-v2", top_k: int = 2):
        index = self.pc.Index(name=self.index_name)
        query_embedding = self.pc.inference.embed(
            model=model,
            inputs=[query],
            parameters={"input_type": "query"}
        )


        results =  index.query(
            namespace=self.name_space,
            vector=query_embedding[0].values,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )['matches']

        documents = [result['metadata']['text'] for result in results]

        return documents
        
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index = os.environ.get('INDEX_NAME')
    pinecone_namespace = os.environ.get('NAMESPACE')

    manager = PineconeEmbeddingManager(api_key=api_key, index_name='kifiya', name_space='test')
    
    # markdown_document = '''
    # # Kifiya Company Information

    # Welcome to the Kifiya Company documentation! Below you'll find helpful information about Kifiya.

    # ## Contact Information

    # You can contact us via email at [contact@kifiya.com](mailto:contact@kifiya.com) or call us at +123-456-7890.

    # ## Frequently Asked Questions

    # ### What services does Kifiya provide?
    # Kifiya provides a wide range of services, including but not limited to:
    # - Virtual assistant systems
    # - AI chatbots
    # - Data analytics solutions

    # ### How can I track my order?
    # To track your order, please visit [this page](https://www.kifiya.com/order-status) and enter your order number.

    # ## Company Address

    # Kifiya's headquarters are located at:
    # - 123 Kifiya Street, Addis Ababa, Ethiopia
    # ''' 

    with open("prm_faq.md", "r", encoding="utf-8") as f:
        markdown_document = f.read()

    headers_to_split_on = [("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    documents = [Document(page_content=str(part)) for part in md_header_splits]
    
    manager.create_and_store_embeddings(md_header_splits)
    
    result = manager.search_matching("Where can I contact kifiya?")
    for doc in result:
        print(doc)
