"""Pipeline orchestration: Ingests 35 pages, indexes Strategies A/B/C, benchmarks 20 Qs."""
import json
import time
from pathlib import Path
from .chunking import by_paragraph, fixed_size
from .embeddings import FastEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import ChromaVectorStore


def main() -> None:
    print("=== STARTING DATA INGESTION & BENCHMARK PIPELINE ===")
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_pdf = root_dir / "data" / "llm_survey.pdf"
    benchmark_file = root_dir / "benchmark" / "questions.json"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Corpus
    print(f"Loading corpus from: {data_pdf}")
    pages = load_pdf(
        path=data_pdf,
        document_id="llm_survey_2024",
        source="llm_survey.pdf",
        max_pages=35
    )
    print(f"Loaded {len(pages)} substantive content pages (Pages 1 to 35).")
    total_chars = sum(p.metadata["char_count"] for p in pages)
    print(f"Total normalized characters: {total_chars:,}")

    # 2. Load Benchmark
    with open(benchmark_file, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} benchmark questions (15 dev / 5 test).")

    # 3. Setup Embedder and Vector Store
    embedder = FastEmbedder(model_name="BAAI/bge-small-en-v1.5")
    store = ChromaVectorStore(persist_directory=str(results_dir / "chromadb"), dimensions=384)

    # 4. Ingest and Index Strategies
    # Strategy A: Fixed-size 500 chars, overlap 50, no filter
    # Strategy B: Paragraph/Structure aware, max_size 800 chars, no filter
    # Strategy C: Paragraph/Structure aware, max_size 800 chars, WITH metadata filter
    strategies = ["A", "B", "C"]
    strategy_info = {}

    for strat in strategies:
        print(f"\n--- Ingesting Strategy {strat} ---")
        t0 = time.time()
        chunks = []
        if strat == "A":
            for p in pages:
                chunks.extend(fixed_size(p, size=500, overlap=50))
        else: # B and C use paragraph chunking
            for p in pages:
                chunks.extend(by_paragraph(p, max_size=800))
        
        # Build vector store collection
        store.create_schema(strategy=strat)
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)
        store.upsert(strategy=strat, chunks=chunks, vectors=vectors)
        elapsed = time.time() - t0
        
        sizes = [len(c.text) for c in chunks]
        strategy_info[strat] = {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes) if sizes else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "ingest_time_sec": round(elapsed, 2),
            "chunks": chunks
        }
        print(f"Strategy {strat}: {len(chunks)} chunks, time={elapsed:.2f}s, avg_size={strategy_info[strat]['avg_chunk_size']:.1f}")

    # 5. Benchmark Execution
    traces_by_strategy = {}
    summary_by_strategy = {}

    for strat in strategies:
        print(f"\n--- Running Benchmark for Strategy {strat} ---")
        traces = []
        for q in questions:
            q_vec = embedder.embed_query(q["question"])
            # Strategy C applies metadata filter if question specifies filter
            active_filter = q.get("filter") if strat == "C" else None
            
            hits = store.search(
                strategy=strat,
                query_vector=q_vec,
                top_k=3,
                filters=active_filter
            )
            trace = evaluate_query(q, hits)
            traces.append(trace)
            
        traces_by_strategy[strat] = traces
        summary = summarize(traces)
        summary_by_strategy[strat] = summary
        
        # Save traces to JSON
        trace_file = results_dir / f"trace_strategy_{strat}.json"
        with open(trace_file, "w") as f:
            json.dump(traces, f, indent=2)
            
        print(f"Strategy {strat} Dev:  Hit@1={summary['dev']['hit1_str']}, Evidence Hit@3={summary['dev']['hit3_str']}")
        print(f"Strategy {strat} Test: Hit@1={summary['test']['hit1_str']}, Evidence Hit@3={summary['test']['hit3_str']}")

    # Save summary report
    overall_summary = {
        "model": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "vector_store": "ChromaDB (cosine metric, persistent)",
        "corpus_pages": len(pages),
        "total_chars": total_chars,
        "strategy_info": {s: {k: v for k, v in info.items() if k != "chunks"} for s, info in strategy_info.items()},
        "metrics": summary_by_strategy
    }
    with open(results_dir / "summary_metrics.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    print("\n=== BENCHMARK COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
