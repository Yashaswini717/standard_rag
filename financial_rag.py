import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from openai import OpenAI

load_dotenv()

DOCS_PATH = "data/raw" 
CHROMA_PATH = "data/financial_chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/llama-3-8b-instruct"

# ============================================================
# 1. INGESTION - Load and chunk documents
# ============================================================
def load_documents(docs_path=DOCS_PATH):
    """Load all 10-K documents from the data folder."""
    print(f"\n[1] Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Path not found: {docs_path}")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding='utf-8')
    )
    
    documents = loader.load()
    print(f"    Loaded {len(documents)} documents")
    
    for doc in documents:
        # Extract company ticker from filename (e.g., "AAPL_10-K_2022.txt" -> "AAPL")
        filename = os.path.basename(doc.metadata['source'])
        parts = filename.replace('.txt', '').split('_')
        doc.metadata['company'] = parts[0] if parts else 'UNKNOWN'
        doc.metadata['year'] = parts[2] if len(parts) > 2 else 'UNKNOWN'
        print(f"    - {doc.metadata['company']} {doc.metadata['year']}: {len(doc.page_content)} chars")
    
    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks with metadata preserved."""
    print(f"\n[2] Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    # Use RecursiveCharacterTextSplitter for better splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"    Created {len(chunks)} chunks")
    
    # Show sample
    if chunks:
        print(f"\n    Sample chunk:")
        print(f"    Company: {chunks[0].metadata.get('company', 'N/A')}")
        print(f"    Preview: {chunks[0].page_content[:150]}...")
    
    return chunks


# ============================================================
# 2. VECTOR STORE - Create embeddings and store
# ============================================================
def create_vector_store(chunks, persist_directory=CHROMA_PATH):
    """Create ChromaDB vector store from chunks."""
    print(f"\n[3] Creating vector store at {persist_directory}...")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"    Vector store created with {len(chunks)} documents")
    return vectorstore


def load_vector_store(persist_directory=CHROMA_PATH):
    """Load existing vector store."""
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore


# ============================================================
# 3. COMPANY DETECTION - Extract company from query
# ============================================================
# Map company names/aliases to ticker symbols
COMPANY_ALIASES = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "facebook": "META",
    "fb": "META",
}

def detect_companies(query: str) -> List[str]:
    """Detect company names/tickers in the query and return list of tickers."""
    query_lower = query.lower()
    detected = set()
    
    for alias, ticker in COMPANY_ALIASES.items():
        # Use word boundary matching to avoid partial matches
        import re
        if re.search(rf'\b{alias}\b', query_lower):
            detected.add(ticker)
    
    return list(detected)


# ============================================================
# 4. HYBRID RETRIEVER - BM25 + Vector Search (Custom Implementation)
# ============================================================
class HybridRetriever:
    """Custom hybrid retriever combining BM25 and vector search with query expansion and metadata filtering."""
    
    def __init__(self, chunks: List[Document], vectorstore, vector_weight=0.7, bm25_weight=0.3, k=5, use_query_expansion=True, company_boost=5.0):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        self.use_query_expansion = use_query_expansion
        self.company_boost = company_boost  # Boost factor for matching company docs
        
        # Build BM25 index
        tokenized_docs = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _search_single_query(self, query: str, target_companies: List[str] = None) -> dict:
        """Perform hybrid search for a single query, return doc_scores dict."""
        doc_scores = {}
        
        # Retrieve more docs initially to ensure coverage across companies
        retrieve_k = max(self.k * 4, 20)  # At least 20 docs for initial retrieval
        
        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=retrieve_k)
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:retrieve_k]
        
        # Add vector results with RRF
        for rank, (doc, score) in enumerate(vector_results):
            # Skip non-matching companies if filtering is active
            if target_companies and doc.metadata.get('company', '') not in target_companies:
                continue
            doc_id = doc.page_content[:100]
            rrf_score = self.vector_weight * (1 / (rank + 60))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0}
            doc_scores[doc_id]['score'] += rrf_score
        
        # Add BM25 results with RRF
        for rank, idx in enumerate(bm25_top_indices):
            doc = self.chunks[idx]
            # Skip non-matching companies if filtering is active
            if target_companies and doc.metadata.get('company', '') not in target_companies:
                continue
            doc_id = doc.page_content[:100]
            rrf_score = self.bm25_weight * (1 / (rank + 60))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0}
            doc_scores[doc_id]['score'] += rrf_score
        
        return doc_scores
    
    def invoke(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search with optional query expansion and company filtering."""
        
        # Detect companies mentioned in query
        target_companies = detect_companies(query)
        if target_companies:
            print(f"    Detected companies: {target_companies}")
        
        # Get query variations (original + expanded)
        if self.use_query_expansion:
            queries = expand_query(query)
            print(f"    Query expansion: {len(queries)} variations")
            for i, q in enumerate(queries):
                print(f"      {i+1}. {q}")
        else:
            queries = [query]
        
        # Aggregate results from all query variations
        combined_scores = {}
        
        for q in queries:
            query_scores = self._search_single_query(q, target_companies)
            
            # Merge into combined scores
            for doc_id, data in query_scores.items():
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {'doc': data['doc'], 'score': 0}
                combined_scores[doc_id]['score'] += data['score']
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return [item[1]['doc'] for item in sorted_results[:self.k]]
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return [item[1]['doc'] for item in sorted_results[:self.k]]


def create_hybrid_retriever(chunks, vectorstore, vector_weight=0.7, bm25_weight=0.3, k=5):
    """Create hybrid retriever combining BM25 and vector search."""
    print(f"\n[4] Creating hybrid retriever (vector={vector_weight}, bm25={bm25_weight})...")
    
    retriever = HybridRetriever(
        chunks=chunks,
        vectorstore=vectorstore,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        k=k
    )
    
    print(f"    Hybrid retriever ready (k={k})")
    return retriever


# ============================================================
# 4. QUERY EXPANSION - Generate multiple query variations
# ============================================================
def expand_query(query, model=LLM_MODEL, num_variations=3):
    """Generate multiple variations of the query for better retrieval."""
    import json
    import re
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    prompt = f"""Generate {num_variations} different variations of this financial query to help retrieve relevant SEC 10-K documents.
Focus on different phrasings, synonyms, and related financial terms.

Original query: "{query}"

Return ONLY a JSON object in this exact format:
{{"queries": ["variation1", "variation2", "variation3"]}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial search query optimizer. Return only valid JSON, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle potential markdown code blocks
        if "```" in content:
            # Extract content between code blocks
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        parsed = json.loads(content)
        variations = parsed.get("queries", [])
        
        if variations:
            return [query] + variations[:num_variations]  # Include original query
        else:
            return [query]
            
    except Exception as e:
        print(f"    Query expansion failed: {e}")
        return [query]  # Fall back to original query only


# ============================================================
# 5. ANSWER GENERATION - OpenRouter LLM
# ============================================================
def generate_answer(query, retrieved_docs, model=LLM_MODEL):
    """Generate answer using OpenRouter LLM."""
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    # Build context from retrieved docs
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('company', 'N/A')} {doc.metadata.get('year', '')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    
    prompt = f"""Based on the following SEC 10-K financial documents, answer the question.

DOCUMENTS:
{context}

QUESTION: {query}

RULES:
- Only use information from the documents above
- Cite the company name when providing financial figures
- If the answer is not in the documents, say "I cannot find this information in the provided documents."

ANSWER:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst assistant that answers questions based on SEC 10-K filings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content.strip()


# ============================================================
# 5. MAIN RAG PIPELINE
# ============================================================
class FinancialRAG:
    """Complete RAG pipeline for financial documents."""
    
    def __init__(self):
        self.chunks = None
        self.vectorstore = None
        self.retriever = None
        self.is_initialized = False
    
    def ingest(self, docs_path=DOCS_PATH):
        """Ingest documents and create retriever."""
        print("=" * 60)
        print("FINANCIAL RAG - INGESTION")
        print("=" * 60)
        
        # Load and chunk
        documents = load_documents(docs_path)
        self.chunks = chunk_documents(documents)
        
        # Create vector store
        self.vectorstore = create_vector_store(self.chunks)
        
        # Create hybrid retriever
        self.retriever = create_hybrid_retriever(self.chunks, self.vectorstore)
        
        self.is_initialized = True
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
    
    def load(self):
        """Load existing vector store."""
        print("Loading existing vector store...")
        
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(f"No vector store found at {CHROMA_PATH}. Run ingest() first.")
        
        self.vectorstore = load_vector_store()
        
        # We need chunks for BM25, so reload documents
        documents = load_documents()
        self.chunks = chunk_documents(documents)
        self.retriever = create_hybrid_retriever(self.chunks, self.vectorstore)
        
        self.is_initialized = True
        print("Loaded successfully!")
    
    def ask(self, query, k=5):
        """Ask a question and get an answer."""
        if not self.is_initialized:
            print("RAG not initialized. Running ingest()...")
            self.ingest()
        
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        
        # Retrieve relevant documents
        print("\n[Retrieving relevant documents...]")
        retrieved_docs = self.retriever.invoke(query)
        
        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs[:5], 1):
            company = doc.metadata.get('company', 'N/A')
            year = doc.metadata.get('year', '')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  [{i}] {company} {year}: {preview}...")
        
        # Generate answer
        print("\n[Generating answer...]")
        answer = generate_answer(query, retrieved_docs)
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {"company": d.metadata.get('company'), "year": d.metadata.get('year')}
                for d in retrieved_docs
            ]
        }


# ============================================================
# CLI INTERFACE
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial RAG Pipeline")
    parser.add_argument("action", choices=["ingest", "ask"], help="Action to perform")
    parser.add_argument("query", nargs="?", help="Question to ask (for 'ask' action)")
    
    args = parser.parse_args()
    
    rag = FinancialRAG()
    
    if args.action == "ingest":
        rag.ingest()
        
    elif args.action == "ask":
        if not args.query:
            print("Please provide a question")
            print("Usage: python financial_rag.py ask \"What was Apple's revenue in 2023?\"")
            return
        
        # Try to load existing, or ingest if not found
        try:
            rag.load()
        except FileNotFoundError:
            print("No existing index found. Running ingestion first...")
            rag.ingest()
        
        rag.ask(args.query)


if __name__ == "__main__":
    main()
