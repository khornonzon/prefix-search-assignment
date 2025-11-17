#!/usr/bin/env python3
"""Evaluation script for prefix search queries."""
from __future__ import annotations

import argparse
import csv
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Any


def search_query(base_url: str, query: str, top_k: int = 5, use_embeddings: bool = False) -> tuple[List[Dict[str, Any]], float]:
    start_time = time.time()
    try:
        response = requests.get(f"{base_url}/search", params={"q": query, "top_k": top_k, "use_embeddings": use_embeddings}, timeout=10)
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        return data.get("results", []), latency_ms
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        print(f"Error searching '{query}': {e}")
        return [], latency_ms


def tokenize(value: str) -> list[str]:
    if not value:
        return []
    return re.findall(r"[0-9a-zа-яё]+", value.lower())


def is_relevant_result(
    result: Dict[str, Any],
    query_tokens: list[str],
    relevance_threshold: float,
) -> bool:
    if not result:
        return False

    score = result.get("score", 0.0) or 0.0
    if score < relevance_threshold:
        return False

    fields_to_check = [
        tokenize(result.get("name", "")),
        tokenize(result.get("brand", "")),
        tokenize(result.get("category", "")),
        tokenize(result.get("keywords", "")),
    ]
    field_tokens = [token for field in fields_to_check for token in field]

    meaningful_query_tokens = [t for t in query_tokens if len(t) >= 2]
    if not meaningful_query_tokens or not field_tokens:
        return False

    match_count = 0
    for qt in meaningful_query_tokens:
        if any(token.startswith(qt) for token in field_tokens):
            match_count += 1

    required_matches = max(1, int(len(meaningful_query_tokens) * 0.6 + 0.5))
    return match_count >= required_matches


def calculate_precision_scores(
    results: List[Dict[str, Any]],
    query: str,
    relevance_threshold: float,
    max_k: int,
) -> tuple[list[float], list[bool]]:
    if max_k <= 0:
        return [], []

    query_tokens = tokenize(query)
    relevances: list[bool] = []

    for result in results[:max_k]:
        relevances.append(is_relevant_result(result, query_tokens, relevance_threshold))

    precisions: list[float] = []
    for k in range(1, max_k + 1):
        limit = min(k, len(relevances))
        if limit == 0:
            precisions.append(0.0)
            continue
        precisions.append(sum(relevances[:k]) / limit)

    return precisions, relevances


def evaluate_queries(
    queries_path: Path,
    base_url: str,
    output_path: Path,
    query_type: str = "open",
    top_k: int = 5,
    precision_k: int = 3,
    relevance_threshold: float = 0.5,
    use_embeddings: bool = False,
) -> None:
    with queries_path.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        queries = [row for row in reader if row.get("type", "").lower() == query_type.lower()]
    
    if not queries:
        print(f"No '{query_type}' queries found in {queries_path}")
        return

    print(f"Evaluating {len(queries)} {query_type} queries...")
    
    results = []
    precision_fieldnames = list(
        dict.fromkeys(
            [
                "precision_at_1",
                f"precision_at_{precision_k}",
                f"precision_at_{top_k}",
            ]
        )
    )
    total_precision_at_1 = 0.0
    total_precision_at_k = 0.0
    total_precision_at_top = 0.0
    total_coverage_any = 0.0
    total_coverage_relevant = 0.0
    total_latency = 0.0
    total_relevant_results = 0
    
    for query_row in queries:
        query = query_row.get("query", "").strip()
        if not query:
            continue
        
        site = query_row.get("site", "")
        notes = query_row.get("notes", "")
        
        search_results, latency_ms = search_query(base_url, query, top_k=top_k, use_embeddings=use_embeddings)

        precisions, relevances = calculate_precision_scores(
            search_results,
            query,
            relevance_threshold,
            max_k=top_k,
        )

        precision_at_1 = precisions[0] if precisions else 0.0
        target_k_index = min(max(precision_k, 1), top_k) - 1
        precision_at_k = precisions[target_k_index] if precisions else 0.0
        precision_at_top_k = precisions[-1] if precisions else 0.0
        has_results = len(search_results) > 0
        has_relevant = any(relevances)

        total_precision_at_1 += precision_at_1
        total_precision_at_k += precision_at_k
        total_precision_at_top += precision_at_top_k
        total_coverage_any += 1.0 if has_results else 0.0
        total_coverage_relevant += 1.0 if has_relevant else 0.0
        total_latency += latency_ms
        total_relevant_results += sum(1 for rel in relevances if rel)
        
        # Format results
        relevant_in_top_k = sum(1 for rel in relevances if rel)

        result_row: Dict[str, Any] = {
            "query": query,
            "site": site,
            "type": query_type,
            "notes": notes,
            "top_1": search_results[0].get("name", "") if len(search_results) > 0 else "",
            "top_1_score": f"{search_results[0].get('score', 0):.2f}" if len(search_results) > 0 else "",
            "top_2": search_results[1].get("name", "") if len(search_results) > 1 else "",
            "top_2_score": f"{search_results[1].get('score', 0):.2f}" if len(search_results) > 1 else "",
            "top_3": search_results[2].get("name", "") if len(search_results) > 2 else "",
            "top_3_score": f"{search_results[2].get('score', 0):.2f}" if len(search_results) > 2 else "",
            "latency_ms": f"{latency_ms:.2f}",
            "coverage_any": "1" if has_results else "0",
            "coverage_relevant": "1" if has_relevant else "0",
            "relevant_in_top_k": str(relevant_in_top_k),
            "judgement": (
                "strong"
                if precision_at_k >= 0.8
                else "ok"
                if has_relevant
                else "needs_review"
            ),
        }
        precision_values = {
            "precision_at_1": precision_at_1,
            f"precision_at_{precision_k}": precision_at_k,
            f"precision_at_{top_k}": precision_at_top_k,
        }
        for column in precision_fieldnames:
            result_row[column] = f"{precision_values[column]:.2f}"
        results.append(result_row)

        print(
            f"  {query}: {len(search_results)} results, "
            f"P@1={precision_at_1:.2f}, "
            f"P@{precision_k}={precision_at_k:.2f}, "
            f"latency={latency_ms:.1f}ms, "
            f"relevant_in_top_k={relevant_in_top_k}"
        )
    
    query_count = len(queries) or 1
    avg_precision_at_1 = total_precision_at_1 / query_count
    avg_precision_at_k = total_precision_at_k / query_count
    avg_precision_at_top = total_precision_at_top / query_count
    avg_coverage_any = total_coverage_any / query_count
    avg_coverage_relevant = total_coverage_relevant / query_count
    avg_latency = total_latency / len(queries) if queries else 0.0
    
    print(f"\nSummary for {query_type} queries:")
    print(f"  Average Precision@1: {avg_precision_at_1:.2f}")
    print(f"  Average Precision@{precision_k}: {avg_precision_at_k:.2f}")
    print(f"  Average Precision@{top_k}: {avg_precision_at_top:.2f}")
    print(
        f"  Coverage (any): {avg_coverage_any:.2%} "
        f"({int(total_coverage_any)}/{len(queries)})"
    )
    print(
        f"  Coverage (relevant): {avg_coverage_relevant:.2%} "
        f"({int(total_coverage_relevant)}/{len(queries)})"
    )
    print(f"  Average Latency: {avg_latency:.1f}ms")
    print(f"  Total relevant results (top {top_k}): {total_relevant_results}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as dst:
        fieldnames = [
            "query",
            "site",
            "type",
            "notes",
            "top_1",
            "top_1_score",
            "top_2",
            "top_2_score",
            "top_3",
            "top_3_score",
            "latency_ms",
            *precision_fieldnames,
            "coverage_any",
            "coverage_relevant",
            "relevant_in_top_k",
            "judgement",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        summary_precision_values = {
            "precision_at_1": avg_precision_at_1,
            f"precision_at_{precision_k}": avg_precision_at_k,
            f"precision_at_{top_k}": avg_precision_at_top,
        }
        writer.writerow({
            "query": "SUMMARY",
            "site": "",
            "type": "",
            "notes": "",
            "top_1": "",
            "top_1_score": "",
            "top_2": "",
            "top_2_score": "",
            "top_3": "",
            "top_3_score": "",
            "latency_ms": f"{avg_latency:.1f}",
            **{
                field: f"{summary_precision_values[field]:.2f}"
                for field in precision_fieldnames
            },
            "coverage_any": f"{avg_coverage_any:.2%}",
            "coverage_relevant": f"{avg_coverage_relevant:.2%}",
            "relevant_in_top_k": str(total_relevant_results),
            "judgement": (
                f"Any: {avg_coverage_any:.1%}, "
                f"Rel: {avg_coverage_relevant:.1%}, "
                f"P@{precision_k}: {avg_precision_at_k:.2f}"
            ),
        })
    
    print(f"\nResults written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prefix search queries")
    parser.add_argument("--queries", default="data/prefix_queries.csv", help="CSV with queries")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Search service base URL")
    parser.add_argument("--output", default="reports/evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--type", choices=["open", "hidden", "all"], default="open", help="Query type to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="How many results to request from the service")
    parser.add_argument("--precision-k", type=int, default=3, help="Primary precision@k cutoff for reporting")
    parser.add_argument("--relevance-threshold", type=float, default=0.5, help="Minimum score to consider a result relevant")
    parser.add_argument("--use-embeddings", action="store_true", help="Search with embeddings")
    args = parser.parse_args()
    
    queries_path = Path(args.queries)
    if not queries_path.exists():
        raise SystemExit(f"Queries file not found: {queries_path}")
    
    try:
        response = requests.get(f"{args.base_url}/health", timeout=5)
        response.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Search service is not available at {args.base_url}: {e}")
    
    top_k = max(1, args.top_k)
    precision_k = max(1, min(args.precision_k, top_k))
    relevance_threshold = max(0.0, args.relevance_threshold)

    if args.type in ["open", "all"]:
        output_path = Path(args.output)
        if args.type == "all":
            output_path = output_path.parent / f"{output_path.stem}_open{output_path.suffix}"
        evaluate_queries(
            queries_path,
            args.base_url,
            output_path,
            "open",
            top_k=top_k,
            precision_k=precision_k,
            relevance_threshold=relevance_threshold,
            use_embeddings=args.use_embeddings,
        )
    
    if args.type in ["hidden", "all"]:
        output_path = Path(args.output)
        if args.type == "all":
            output_path = output_path.parent / f"{output_path.stem}_hidden{output_path.suffix}"
        evaluate_queries(
            queries_path,
            args.base_url,
            output_path,
            "hidden",
            top_k=top_k,
            precision_k=precision_k,
            relevance_threshold=relevance_threshold,
            use_embeddings=args.use_embeddings,
        )


if __name__ == "__main__":
    main()
