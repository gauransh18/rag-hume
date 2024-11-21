import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'mps'}
    )
    
    # Initialize the DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Debug: Print total documents in DB
    collection = db._collection
    print(f"Total documents in database: {collection.count()}")

    # Lower the threshold and increase k for testing
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Debug: Print all results with scores
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print("---")

    if len(results) == 0 or results[0][1] < 0.3:  # Lowered threshold significantly
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the language model
    model = HuggingFaceHub(
        repo_id="google/flan-t5-small",  # Using a smaller model for faster responses
        huggingfacehub_api_token=os.environ['HUGGINGFACE_API_TOKEN'],
        model_kwargs={
            "temperature": 0.5,
            "max_length": 512
        }
    )

    response_text = model.predict(prompt)
    print("\nFinal Response:", response_text)

if __name__ == "__main__":
    main()
