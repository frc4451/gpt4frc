import os
from pprint import pprint
from typing import List

from langchain.chains import AnalyzeDocumentChain, ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


class GPT4FRC:
    """
    GPT interface wrapper that manages QA Chain, LLM choice, API Key, etc.

    All supported models have static methods that make them more accessible. If you plan to
    utilize the default constructor, you will need to provide your own definition for LLM.

    The following Clients are supported with GPT4FRC (more to come)
        - OpenAI (ChatGPT)


    Planned for inclusion, but not currently supported
        - GPT4All
        - TextGen

    At the time of writing, this only supports ChromaDB for the vectorstore. This means
    that if you plan to use embeddings and allow the user to read notes, you will need to
    provide a PDF or directory of PDFs for the AI to read context.
    """

    def __init__(
        self,
        llm,
        retriever_args: dict,
        chain_type: str = "stuff",
        use_memory: bool = False,
        verbose: bool = False,
    ):
        self._verbose = verbose

        if retriever_args.get("documents_path", None) is None:
            raise ValueError("`documents_path` is required when using a retriever.")

        retriever = self._get_db_retriever(
            documents=self._load_documents_from_pdf(
                documents_path=retriever_args.get("documents_path")
            ),
            chunk_size=retriever_args.get("chunk_size", 250),
            chunk_overlap=retriever_args.get("chunk_overlap", 0),
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self._chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type=chain_type,
        )

    @staticmethod
    def from_chatgpt(
        openai_api_key: str,
        openai_model: str = "gpt-3.5-turbo",
        openai_args: dict = None,
        chain_type: str = "stuff",
        retriever_args: dict = None,
        use_memory: bool = False,
        verbose: bool = False,
    ):
        """
        Static Method that builds GPT4FRC specifically for using OpenAI (ChatGPT)
        """

        if not openai_api_key:
            raise ValueError(
                "When using OpenAI/ChatGPT, you need to provide an OpenAI API key."
            )

        llm = ChatOpenAI(
            model=openai_model, openai_api_key=openai_api_key, **openai_args
        )

        return GPT4FRC(
            llm=llm,
            chain_type=chain_type,
            retriever_args=retriever_args,
            use_memory=use_memory,
            verbose=verbose,
        )

    def ask_question(self, question: str, only_answers: bool = True):
        """
        Ask questions to the QA Chain
        """
        response = self._chain({"question": question})

        if only_answers:
            return response["answer"]
        return response

    def _get_db_retriever(
        self,
        documents: List[Document],
        chunk_size: int = 250,
        chunk_overlap: int = 0,
    ):
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_documents = splitter.split_documents(documents)
        db = Chroma.from_documents(
            documents=split_documents,
            embedding=OpenAIEmbeddings(),
            persist_directory="chromadb",
        )
        return db.as_retriever()

    def _load_documents_from_pdf(
        self,
        documents_path: str,
    ) -> List[Document]:
        if not os.path.exists(documents_path):
            raise ValueError(
                f"You need to provide a valid PDF file path. Received :: {documents_path}"
            )

        documents = (
            PyPDFLoader(documents_path).load()
            if os.path.isfile(documents_path)
            else PyPDFDirectoryLoader(documents_path).load()
        )

        if self._verbose:
            pprint("Successfully able to load documents from :: ", documents)

        return documents
