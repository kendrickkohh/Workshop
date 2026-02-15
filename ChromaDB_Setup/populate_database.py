from langchain_community.vectorstores import Chroma
from get_embedding import get_embedding
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDFs into docs variable
def load_documents():
    document_loader = PyPDFDirectoryLoader(
        path = "./Upload_documents",
    )

    docs = document_loader.load()
    return docs

# Split document into chunks so that its easily stored
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )
    return text_splitter.split_documents(documents)

# Add chunks to chromaDB
def add_to_chroma(chunks: list[Document]):
    # Load the existing database
    db = Chroma(
        persist_directory="chroma", embedding_function=get_embedding()
    )

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks] # Adding IDs with the chunks
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

# This function will create IDs like "data/pdfName.pdf:6:2"
# Page Source : Page Number : Chunk Index
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page metadata
        chunk.metadata["id"] = chunk_id

    return chunks

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

main()

# Youtube video says that his experience with RAG on ollama isnt very good, he felt that openAI or AWS are better. Need to use larger ollama model to know
# Good video: https://www.youtube.com/watch?v=2TJxpyO3ei4
# Initial video: https://www.youtube.com/watch?v=E4l91XKQSgw
# Next step will be updating the database, because ids and chunk ids will be fixed, how do we know that a chunk is edited?
