from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def retrieve_related_documents(knowledge_base, user_question):
    """
    Retrieves related documents from a knowledge base based on a user question.
    Args:
        knowledge_base: The knowledge base to search for related documents.
        user_question: The user's question.
    Returns:
        A list of related documents.
    """
    doc_list = []
    if user_question:
        doc_list = knowledge_base.similarity_search(user_question)
    return doc_list


def generate_response(llm, docs, user_question):
    """
    Generates a response using a language model and a list of documents.
    Args:
        llm: The language model to generate the response.
        docs: The list of documents to provide context.
        user_question: The user's question.
    Returns:
        The generated response and the callback object.
    """
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
    return response, cb


def generate_answer(knowledge_base, user_question):
    """
    Displays the user input and generates a response.
    Args:
        knowledge_base: The knowledge base to search for related documents.
        user_question: The user's question.
    Returns:
        The generated response or an error message if no related documents are found.
    """
    docs = retrieve_related_documents(knowledge_base, user_question)
    if not docs:
        return "Sorry, we couldn't find any related documents for your question."

    llm = OpenAI()
    response, cb = generate_response(llm, docs, user_question)
    print(cb)

    return response
