import streamlit as st
from openai import OpenAI, load_qa_chain, get_openai_callback


# Define a function to display user input and generate responses based on it
def show_user_input(knowledge_base):
    # Display a text input box for the user to enter a question
    user_question = st.text_input("Ask a question about your PDF:")

    # Check if the user has entered a question
    if user_question:
        # Use the similarity_search function to find documents related to the user's question
        docs = knowledge_base.similarity_search(user_question)

        # Create an OpenAI object and load a pre-trained QA chain
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            # Run the QA chain on the retrieved documents and user's question
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        # Display the response generated by the QA chain
        st.write(response)