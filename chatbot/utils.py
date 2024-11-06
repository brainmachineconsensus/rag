# chatbot/utils.py
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import CSVLoader
from langchain.memory import ConversationBufferWindowMemory
import os

# Chemin vers le fichier .env
from dotenv import load_dotenv
load_dotenv()

def chunk_embedder():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", temperature=0.5)
    return embeddings

# Fonction pour charger et traiter le fichier CSV
def load_and_process_csv(csv_path):
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = chunk_embedder()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_db")
    return vectorstore

# Initialisation du chatbot avec LangChain
def init_chat():
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    csv_path = "vector_db/faq_chatbot.csv"
    
    if not os.path.exists("vector_db/index.faiss"):
        vectorstore = load_and_process_csv(csv_path)
    else:
        embeddings = chunk_embedder()
        vectorstore = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    
    template = """Tu es un assistant juridique au Burkina Faso, chargé de répondre à des questions sur les lois en vigueur dans le pays. Présente-toi clairement en spécifiant ton rôle, en veillant à adopter un ton amical et accessible. Évite de poser des questions et ne fais pas usage de caractères spéciaux tels que les étoiles ou les tirets. Lors de tes réponses, privilégie des phrases fluides et naturelles, et lorsque tu as plusieurs points à aborder, utilise des connecteurs logiques pour assurer une bonne cohérence entre les idées. L'objectif est de fournir des réponses claires et informatives, tout en maintenant une discussion conviviale. 
    context:{context}
    Sois convivial pendant la discussion, aies des reponses naturelles et sans caractères speciaux, fais des phrases correctes en reponses aux questions, pour les élements avec plusieurs points utilisent des connecteurs logiques et non des caractères entre ces points.
    Question :{question}
    faq_chatbot
    """
    message_history = ChatMessageHistory()
    chat_memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                                 faq_chatbot_key="answer",
                                                 chat_memory=message_history,
                                                 k=5,
                                                 return_messages=True)

    prompt = PromptTemplate(template=template, input_variables=["context", "input"])
    chain_type_kwargs = {"prompt": prompt}
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=chat_memory,
        combine_docs_chain_kwargs=chain_type_kwargs,
    )

    return conversation_chain

chatbot = init_chat()  # Initialiser le chatbot au démarrage