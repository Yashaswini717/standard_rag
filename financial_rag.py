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
# 5. SELF-CORRECTION - Validate retrieval and rewrite query
# ============================================================
def validate_retrieval(query: str, retrieved_docs: List[Document], model=LLM_MODEL) -> dict:
    """
    LLM acts as a JUDGE to validate if retrieved documents are relevant to the query.
    
    Returns:
        dict with keys:
        - is_relevant: bool - True if docs can answer the query
        - confidence: float - 0.0 to 1.0 confidence score
        - reason: str - Why relevant or not
        - missing_info: str - What information is missing (used for rewriting)
    """
    import json
    import re
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    # Build context from retrieved docs
    docs_summary = "\n\n".join([
        f"[Doc {i+1} - {doc.metadata.get('company', 'N/A')} {doc.metadata.get('year', '')}]\n{doc.page_content[:500]}..."
        for i, doc in enumerate(retrieved_docs[:5])
    ])
    
    prompt = f"""You are a retrieval validator for a financial RAG system. Your job is to judge whether the retrieved documents contain relevant information to answer the user's query.

USER QUERY: "{query}"

RETRIEVED DOCUMENTS:
{docs_summary}

Analyze carefully:
1. Does the query ask about a specific company? Do the documents match that company?
2. Does the query ask about a specific year? Do the documents match that year?
3. Do the documents contain the specific information requested (revenue, risk factors, etc.)?

Respond with ONLY a JSON object in this exact format:
{{
    "is_relevant": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation of why relevant or not",
    "missing_info": "what specific information is missing or wrong (e.g., 'documents are from 2022 but query asks for 2023')"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict retrieval validator. Return only valid JSON. Be critical - if there's a year mismatch or company mismatch, mark as not relevant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle potential markdown code blocks
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        result = json.loads(content)
        
        # Ensure all required fields exist
        return {
            "is_relevant": result.get("is_relevant", False),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", "Unknown"),
            "missing_info": result.get("missing_info", "")
        }
        
    except Exception as e:
        print(f"    Validation failed: {e}")
        # Default to relevant on error to avoid blocking
        return {
            "is_relevant": True,
            "confidence": 0.5,
            "reason": f"Validation error: {e}",
            "missing_info": ""
        }


def rewrite_query(original_query: str, validation_result: dict, model=LLM_MODEL) -> str:
    """
    LLM acts as a REWRITER to improve the query based on validation feedback.
    
    Uses the 'reason' and 'missing_info' from validation to create a better query
    that will retrieve more relevant documents.
    
    Returns:
        - Rewritten query string, OR
        - "QUERY_IMPOSSIBLE" if the data simply doesn't exist in our dataset
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    prompt = f"""You are a query rewriter for a financial RAG system. The original query failed to retrieve relevant documents.

AVAILABLE DATA: We only have SEC 10-K filings for these 5 companies: Apple (AAPL), Microsoft (MSFT), Google/Alphabet (GOOGL), Amazon (AMZN), Meta/Facebook (META). Years available: 2022 and 2023 only.

ORIGINAL QUERY: "{original_query}"

PROBLEM: {validation_result['reason']}

MISSING INFORMATION: {validation_result['missing_info']}

RULES:
1. DO NOT change the company or topic the user is asking about
2. If the user asks about Netflix, Tesla, or any company NOT in our dataset → return exactly: QUERY_IMPOSSIBLE
3. If the user asks about years before 2022 or after 2023 → return exactly: QUERY_IMPOSSIBLE  
4. If the query IS about one of our 5 companies (AAPL, MSFT, GOOGL, AMZN, META) for 2022-2023, rewrite it to be more specific:
   - Add the ticker symbol (e.g., AAPL, MSFT)
   - Add the year (2022 or 2023)
   - Add relevant financial terms from SEC 10-K filings

Return ONLY the rewritten query OR "QUERY_IMPOSSIBLE", nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial search query rewriter. Return only the rewritten query or QUERY_IMPOSSIBLE. No explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        rewritten = response.choices[0].message.content.strip()
        
        # Clean up any quotes that might be in the response
        rewritten = rewritten.strip('"\'')
        
        return rewritten
        
    except Exception as e:
        print(f"    Query rewrite failed: {e}")
        return original_query  # Fall back to original


# ============================================================
# 6. ANSWER GENERATION - OpenRouter LLM
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
    
    def ask(self, query, k=5, max_retries=2, use_self_correction=True):
        """
        Ask a question and get an answer with optional self-correction.
        
        Self-Correction Loop:
        1. Retrieve documents
        2. Validate retrieval (LLM judges relevance)
        3. If not relevant and retries left → Rewrite query and retry
        4. Generate answer from final retrieved docs
        
        Args:
            query: The user's question
            k: Number of documents to retrieve
            max_retries: Maximum number of query rewrites (default: 2)
            use_self_correction: Enable/disable self-correction loop (default: True)
        """
        if not self.is_initialized:
            print("RAG not initialized. Running ingest()...")
            self.ingest()
        
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        
        current_query = query
        retrieved_docs = None
        validation_history = []  # Track all validation attempts
        
        # ========== SELF-CORRECTION LOOP ==========
        for attempt in range(max_retries + 1):
            attempt_num = attempt + 1
            
            # Step 1: Retrieve documents
            print(f"\n[Attempt {attempt_num}/{max_retries + 1}] Retrieving documents...")
            if attempt > 0:
                print(f"    Using rewritten query: \"{current_query}\"")
            
            retrieved_docs = self.retriever.invoke(current_query)
            
            print(f"\nRetrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs[:5], 1):
                company = doc.metadata.get('company', 'N/A')
                year = doc.metadata.get('year', '')
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"  [{i}] {company} {year}: {preview}...")
            
            # Step 2: Validate retrieval (skip if self-correction disabled)
            if not use_self_correction:
                print("\n[Self-correction disabled, skipping validation]")
                break
            
            print("\n[Validating retrieval...]")
            validation = validate_retrieval(current_query, retrieved_docs)
            validation_history.append({
                "attempt": attempt_num,
                "query": current_query,
                "validation": validation
            })
            
            print(f"    Relevant: {validation['is_relevant']} (confidence: {validation['confidence']:.2f})")
            print(f"    Reason: {validation['reason']}")
            
            # Step 3: Check if relevant or out of retries
            if validation['is_relevant']:
                print("\n[Validation PASSED - proceeding to answer generation]")
                break
            
            if attempt >= max_retries:
                print(f"\n[Max retries ({max_retries}) reached - proceeding with best available docs]")
                break
            
            # Step 4: Rewrite query using feedback
            print(f"\n[Validation FAILED - rewriting query...]")
            print(f"    Missing info: {validation['missing_info']}")
            
            new_query = rewrite_query(current_query, validation)
            print(f"    Rewritten: \"{new_query}\"")
            
            # Check if query is impossible (data doesn't exist in our dataset)
            if new_query.upper().strip() == "QUERY_IMPOSSIBLE":
                print("\n[QUERY IMPOSSIBLE - requested data is not in our dataset]")
                print("    Available: AAPL, MSFT, GOOGL, AMZN, META (years 2022-2023)")
                # Return early with a clear message
                answer = f"I cannot answer this question. The requested information is not available in our dataset. We only have SEC 10-K filings for: Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), and Meta (META) for fiscal years 2022 and 2023."
                print("\n" + "=" * 60)
                print("ANSWER:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                return {
                    "query": query,
                    "final_query": current_query,
                    "answer": answer,
                    "sources": [],
                    "self_correction": {
                        "enabled": True,
                        "attempts": len(validation_history),
                        "query_impossible": True,
                        "history": validation_history
                    }
                }
            
            # Avoid infinite loop if rewrite returns same query
            if new_query.lower().strip() == current_query.lower().strip():
                print("    [Warning: Rewritten query is identical, stopping retries]")
                break
            
            current_query = new_query
        
        # ========== GENERATE ANSWER ==========
        print("\n[Generating answer...]")
        answer = generate_answer(query, retrieved_docs)  # Use ORIGINAL query for answer
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        
        # Show self-correction summary if it was used
        if use_self_correction and len(validation_history) > 1:
            print("\n" + "-" * 40)
            print("SELF-CORRECTION SUMMARY:")
            for vh in validation_history:
                status = "PASS" if vh['validation']['is_relevant'] else "FAIL"
                print(f"  Attempt {vh['attempt']}: [{status}] {vh['query'][:50]}...")
        
        print("=" * 60)
        
        return {
            "query": query,
            "final_query": current_query,
            "answer": answer,
            "sources": [
                {"company": d.metadata.get('company'), "year": d.metadata.get('year')}
                for d in retrieved_docs
            ],
            "self_correction": {
                "enabled": use_self_correction,
                "attempts": len(validation_history),
                "history": validation_history
            }
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
