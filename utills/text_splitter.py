from textsplitter import CharacterTextSplitter


# define a function named split_text that takes a text string as input
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # use the split_text method of the text_splitter object to split the input text into chunks and return the result
    return text_splitter.split_text(text)
