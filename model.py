from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from validate_results import ValidateResultsInteractor
from rich.console import Console


console = Console()
class Solution:

    def set_custom_prompt(self):
        custom_prompt_template = '''

        You are an query resolver for HR Department that provides relevant answers in 
        plain text. Always prioritize factual accuracy.
        If you don't know the answer, state that and recommend contacting HR.
        Focus solely on answering the question using
        relevant sections of the context. Ignore extraneous information.
        strictly avoid repeated text!

        context: {context}
        question: {question}

        answer : 

        '''
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    def load_llm(self):
        # Load model directly
        from transformers import AutoTokenizer, AutoModelForCausalLM


        llm = CTransformers(
            model="TheBloke/zephyr-7B-beta-GGUF",
            model_type="mistral",
            max_new_tokens=20,
            temperature=0.01
        )
        return llm
    def semantic_search(self,text):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        retriever= db.as_retriever(search_kwargs={'k': 1})
        similar_chunk=retriever.get_relevant_documents(text)
        return similar_chunk


    def get_retrievalqa(self, retriever, prompt, llm):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 1})
        chain_type_kwargs = {"prompt": prompt}

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs=chain_type_kwargs,
            verbose=True
        )

    # output function
    def final_result(self, query):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                           model_kwargs={'device': 'cpu'})
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 1})
        qa_retrieval = self.get_retrievalqa(
            retriever=retriever,
            llm=self.load_llm(),
            prompt=self.set_custom_prompt()
        )
        bot_response = qa_retrieval.invoke(query)

        return bot_response
