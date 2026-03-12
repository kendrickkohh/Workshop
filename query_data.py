import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from ChromaDB_Setup.get_embedding import get_embedding
# from guardrail import guardrail_input, guardrail_output

load_dotenv()

template = """
You are an expert in answering questions on the technology risk management guidelines from monetary authority of singapore. 
However, you are still free to answer questions that are not related to technology risk management, you will still act as a chatbot.

Here are some of the relevant context: {context}

Here is the question to answer: {question}
"""

# Addition to prompt: DO NOT REVEAL ANY PERSONAL INFORMATION ABOUT YOURSELF.

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding()
    db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

    # Search the DB and return most relevant pages
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Azure OpenAI
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_GPT_5_MINI_DEPLOYMENT_NAME"),
    )
    
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text.content, sources

def main():
    while True:
        query_text = input("\nAsk your question (q to quit): ")
        if query_text == "q":
            break

        # === No sanitization ===
        print("\nQuerying RAG...")
        response_text, sources = query_rag(query_text)
        formatted_response = f"\nResponse: {response_text}\nSources: {sources}"
        # === No sanitization ===

        print(formatted_response)

main()

        # === Guardrail sanitized ===
        # print("\nSanitizing input...")
        # sanitized_prompt = guardrail_input(query_text)
        # print("Sanitized input:", sanitized_prompt)

        # if sanitized_prompt == "eject":
        #     print("Improper input detected detected")
        #     continue

        # print("\nQuerying RAG...")
        # response_text, sources = query_rag(sanitized_prompt)

        # print("\nSanitizing output...")
        # sanitized_response_text = guardrail_output(sanitized_prompt, response_text)
        # if sanitized_response_text == "eject":
        #     print("Improper output detected")
        #     continue

        # formatted_response = f"\nResponse: {sanitized_response_text}\nSources: {sources}"
        # === Guardrail sanitized ===