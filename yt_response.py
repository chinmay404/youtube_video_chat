from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

class YTResponseGenerator:
    def __init__(self, groq_api_key):
        self.llm_instance = None
        self.init_llm(groq_api_key)
        self.chain = None 
        self.db = None  
        self.retriver = None  
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)

    def init_llm(self, groq_api_key):
        self.llm_instance = ChatGroq(
            model="llama-3.1-70b-versatile",
            api_key=groq_api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def load_qa_chain(self, llm, chain_type="stuff", verbose=False):
        """Loads a QA chain."""
        self.chain = load_qa_chain(llm, chain_type=chain_type, verbose=verbose)

    def get_transcript(self, video_url, translation_language="en"):
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=True,
            language=["en", "id"],
            translation=translation_language,
        )
        text = loader.load()
        docs = self.text_splitter.split_documents(text)
        persistent_dir = "chroma_db"
        self.db = Chroma.from_documents(documents=docs, embedding=self.gen_embeddings(), persist_directory=persistent_dir)
        self.db.persist()
        self.retriver = self.db.as_retriever()
        return self.db.as_retriever()

    def gen_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def init_chroma(self, docs):
        persistent_dir = "chroma_db"
        db = Chroma.from_documents(documents=docs, embedding=self.gen_embeddings(), persist_directory=persistent_dir)
        db.persist()
        self.db = db  
        return self.db.as_retriever()

    def process_query(self, user_input):
        """Processes a query using the QA chain."""
        matching_docs = self.db.similarity_search(user_input)
        print(matching_docs)
        output = self.chain.invoke(input_documents=matching_docs, question=user_input)
        return output

    def handle_user_input(self, user_input):
        """Handles user input by initializing the QA chain and processing queries."""
        if not self.chain:
            self.load_qa_chain(self.llm_instance)  
        return self.process_query(user_input)
