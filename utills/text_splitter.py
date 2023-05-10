from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def split_text(text):
    """
    Splits the given text into chunks.

    Args:
        text (str): The input text to be split.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_knowledge_base(chunks):
    """
        Creates a knowledge base from the given text chunks.

        Args:
            chunks (list): A list of text chunks.

        Returns:
            knowledge_base: The created knowledge base.
    """
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base
