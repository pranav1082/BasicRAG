import dotenv
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def main():
    dotenv.load_dotenv()

    print("Welcome to conversational chat history")
    dirload = DirectoryLoader(path="Data",
                              glob="**/*.txt",
                              loader_cls=TextLoader,
                              loader_kwargs={"encoding":"UTF-8"}
                              )
    documents = dirload.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                            chunk_overlap=20,
                            length_function=len)
    
    chunks = splitter.split_documents(documents)

    embeddings=OpenAIEmbeddings()
    vectorStore = Chroma.from_documents(chunks,embedding=embeddings,collection_name="rag_conversation",persist_directory="chroma")

    
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    model = ChatOpenAI(model="gpt-6-luna")
    output_parser = StrOutputParser()

    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{question}")
    ])

    contextualize_chain = contextualize_q_prompt|model|output_parser

    qa_system_prompt = (
    "You are a helpful assistant. Use the following pieces of retrieved context to answer the question: {context}."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{question}"),
        ])
    

    def get_context(input_dict):
        if input_dict.get("chat_history"):
            question = contextualize_chain.invoke(input_dict)
        else:
            question = input_dict["question"]
        docs = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)
        


   
    conversational_rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | qa_prompt
        | model
        | output_parser
    )
    
    chat_history = []
    response = conversational_rag_chain.invoke({
    "question": "What is Machine Learning ?",
    "chat_history": chat_history
    })
    print(response)
    chat_history.append(HumanMessage("What is Machine Learning ?"))
    chat_history.append(AIMessage(response))
    print(chat_history)
    response = conversational_rag_chain.invoke({
        "question": "What are its type ?",
        "chat_history": chat_history
        })
    print(response)
if __name__ == "__main__":
    main()
