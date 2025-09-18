import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer





# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Initializing RAG Chatbot...")


pdf_path = r"D:\01_Projects\06_agent\01_codes\data\Suraj Meshram_CV.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF.")



# Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks")

# Create Embeddings and Store in Chroma Vectorstore

model_name = "sentence-transformers/all-mpnet-base-v2"

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("In-Progress")

persist_dir = "./chromadb"

try:
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print("Loading existing ChromaDB vectorstore...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("Creating new ChromaDB vectorstore...")
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=persist_dir)
        vectorstore.persist()
except Exception as e:
    print(f"Error with vectorstore: {e}")
    exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define Prompt Templates
condense_prompt_template = """

Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:
"""

qa_prompt_template = """

You are a helpful assistant for Chetu, answering questions based on their website content.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't have enough information, don't try to make up an answer.

Context: {context}

Question: {question}

Conversation history: {chat_history}

Answer:

"""

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="gemma2-9b-it",
    max_tokens=500,
    temperature=0.7
)

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=PromptTemplate.from_template(condense_prompt_template),
    combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(qa_prompt_template)}
)

# Conversational loop
print("\nRAG Chatbot initialized. Type 'exit' to quit.")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    try:
        response = qa_chain({"question": user_query})
        print("\nBot:", response["answer"])
    except Exception as e:
        print(f"Error processing your query: {e}")



    
