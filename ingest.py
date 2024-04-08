from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from test import QuestionAnswerText


DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450,
                                                   chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    obj1 = QuestionAnswerText()
    Raw_Text = obj1.read_pdf("Policy_Document_FAQ.pdf")
    strings_to_store = obj1.question_answer_list(Raw_Text)
    db = FAISS.from_documents(texts, embeddings)
    for string in strings_to_store:
        embedding = embeddings.encode([string])[0]
        db.add(string, embedding)
    db.save_local(DB_FAISS_PATH)


if __name__=="__main__":
    create_vector_db()