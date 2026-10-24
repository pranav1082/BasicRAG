from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import dotenv
def main():
   
    dotenv.load_dotenv() 

    dirload = DirectoryLoader(path="./Data",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True)

    documents = dirload.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
    )
    
    docs = text_splitter.split_documents(documents)


    embeddings = OpenAIEmbeddings()
    verctorstore = Chroma.from_documents(documents=docs,embedding=embeddings,collection_name="rag_collection",persist_directory="./chromadb")


    llm = ChatOpenAI(model="gpt-6-luna")
    
    query = "What are the 3 types of Machine Learning ?"
    retriever = verctorstore.as_retriever(search_kwarg={"k":3})

    system_prompt = """ You are a helpful assistant. Please answer the query from the retrieved context below. 
    If you do not know the answer, please reply 'I do not Know'. Use 3 sentence maximum and keep the answer precise.

    {context}
    """

    prompt =  ChatPromptTemplate.from_messages([("system",system_prompt),
                                               ("human","{input}")])

    docchain = create_stuff_documents_chain(llm,prompt)
    Ragchain = create_retrieval_chain(retriever=retriever,combine_docs_chain=docchain)
    response=Ragchain.invoke({"input":query})
    print(response)
if __name__ == "__main__":
    main()
