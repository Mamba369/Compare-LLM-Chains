import json, os
from PyPDF2 import PdfReader
from langchain_openai import OpenAI, ChatOpenAI
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.indexes import GraphIndexCreator
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import networkx as nx
import pdfplumber

OPENAI_APIKEY_PATH = "debug/openaiapikey.txt"


def print_and_save_n_line_of_json(input_file_path, output_file_path, n):
    with open(input_file_path, "r") as input_file, open(
        output_file_path, "w"
    ) as output_file:
        data = json.load(input_file)

        if isinstance(data, list):
            str_line = json.dumps(data[n], indent=4)
            print(str_line)
            output_file.write(str_line)

        elif isinstance(data, dict):
            for i, (key, value) in enumerate(data.items()):
                if i >= n:
                    pair_str = f'"{key}": {json.dumps(value, indent=4)}'
                    print(pair_str)
                    output_file.write(pair_str + "\n")
                    break

        else:
            print("JSON structure not recognized.")


def get_embeddings(embeddings_model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=embeddings_model_name)


def get_retriever(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    chunk_size: int = 200,
    chunk_overlap: int = 25,
    k_documents: int = 3,
) -> VectorStore:
    print(f"Number of Document objects before recursive splitting: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)
    print(f"Number of Document objects after recursive splitting: {len(documents)}")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k_documents}
    )


def load_questions(file_path: str):
    with open(file_path, "rb") as file:
        questions = [line.decode("utf-8").strip() for line in file.readlines()]
    return questions


def load_pdf_document(file_path: str, start_page: int, end_page: int) -> list[Document]:
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages[start_page:end_page]:
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"page": page.page_number})
                documents.append(doc)
    return documents


def get_connection(local: bool = False, model_name: str = "gpt-3.5-turbo-instruct"):
    with open(OPENAI_APIKEY_PATH, "r") as file:
        apikey = file.read()
    os.environ["OPENAI_API_KEY"] = apikey

    if local:
        return ChatOpenAI(
            model=model_name,
            base_url="http://localhost:1234/v1",
            openai_api_key=apikey,
            temperature=0,
        )
    return OpenAI(model_name=model_name, temperature=0)


def invoke_chain(chain: Chain, input: str):
    return chain.invoke(input=input)[chain._run_output_key]


def validate_answer(embeddings: HuggingFaceEmbeddings, question: str, answer: str):
    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)
    cosine_sim = cosine_similarity([question_embedding], [answer_embedding])[0][0]
    euclidean_dist = euclidean(question_embedding, answer_embedding)
    print(f"Cosine Similarity: {cosine_sim:.03f}")
    print(f"Euclidean Distance: {euclidean_dist:.03f}")


def compare_chains_based_on_question(
    chains: list[Chain],
    labels: list[str],
    embeddings: HuggingFaceEmbeddings,
    question: str,
) -> None:
    print(f"Starting comparison for following question: {question}")
    for chain, label in zip(chains, labels):
        answer = invoke_chain(chain=chain, input=question)
        print(f"For chain {label} was provided following answer:\n{answer}")
        validate_answer(embeddings=embeddings, question=question, answer=answer)
        print()


def create_graph(llm: OpenAI, documents: list[Document]) -> NetworkxEntityGraph:
    index_creator = GraphIndexCreator(llm=llm)
    graphs = [index_creator.from_text(document.page_content) for document in documents]
    graph_nx = graphs[0]._graph
    for g in graphs[1:]:
        graph_nx = nx.compose(graph_nx, g._graph)

    return NetworkxEntityGraph(graph_nx)
