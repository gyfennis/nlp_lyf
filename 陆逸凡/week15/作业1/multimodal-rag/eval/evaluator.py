"""RAG system evaluator for computing metrics."""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Single query evaluation result."""
    query_id: str
    filename_score: float
    page_score: float
    content_score: float
    total_score: float
    answer: str
    sources: List[Dict]


class RAGEvaluator:
    """RAG system evaluator for computing page match, filename match, and content similarity."""

    def __init__(self, test_queries: List[Dict], client=None):
        """
        Initialize evaluator.

        Args:
            test_queries: List of test query dicts with expected_answer and expected_source
            client: Optional FastAPI test client for live testing
        """
        self.test_queries = test_queries
        self.client = client

    def calc_filename_match(self, sources: List[Dict], expected_filename: str) -> float:
        """Calculate filename match score (0-1)."""
        if not expected_filename:
            return 1.0
        if not sources:
            return 0.0

        for source in sources:
            filename = source.get("filename", "")
            if expected_filename in filename:
                return 1.0
        return 0.0

    def calc_page_match(self, sources: List[Dict], expected_page: int) -> float:
        """Calculate page match score (0-1)."""
        if expected_page is None or expected_page <= 0:
            return 1.0
        if not sources:
            return 0.0

        for source in sources:
            if source.get("page") == expected_page:
                return 1.0
        return 0.0

    def calc_content_similarity(self, generated: str, expected: str) -> float:
        """Calculate Jaccard similarity between generated and expected answers."""
        if not expected:
            return 1.0 if not generated else 0.0
        if not generated:
            return 0.0

        def tokenize(text: str) -> set:
            """Extract Chinese characters and English words."""
            chinese = re.findall(r'[一-鿿]+', text)
            english = re.findall(r'[a-zA-Z0-9]+', text)
            return set(chinese + english)

        gen_tokens = tokenize(generated.lower())
        exp_tokens = tokenize(expected.lower())

        intersection = len(gen_tokens & exp_tokens)
        union = len(gen_tokens | exp_tokens)

        return intersection / union if union > 0 else 0.0

    def evaluate_single(self, query: Dict, response: Dict) -> EvaluationResult:
        """Evaluate a single query response."""
        expected = query.get("expected_answer", "")
        expected_source = query.get("expected_source", {})

        sources = response.get("sources", [])
        answer = response.get("answer", "")

        filename_score = self.calc_filename_match(
            sources,
            expected_source.get("filename", "")
        )
        page_score = self.calc_page_match(
            sources,
            expected_source.get("page", 0)
        )
        content_score = self.calc_content_similarity(answer, expected)

        total_score = (
            filename_score * 0.25 +
            page_score * 0.25 +
            content_score * 0.5
        )

        return EvaluationResult(
            query_id=query.get("id", "unknown"),
            filename_score=filename_score,
            page_score=page_score,
            content_score=content_score,
            total_score=total_score,
            answer=answer,
            sources=sources
        )

    def evaluate_all(self) -> Dict:
        """Evaluate all test queries and compute aggregate metrics."""
        results = []

        for query in self.test_queries:
            if self.client:
                # Make live request
                response = self.client.post(
                    "/chat",
                    json={
                        "question": query["question"],
                        "knowledge_base_id": query.get("knowledge_base_id", 1)
                    }
                )
                if response.status_code == 200:
                    result = self.evaluate_single(query, response.json())
                else:
                    result = EvaluationResult(
                        query_id=query.get("id", "unknown"),
                        filename_score=0.0,
                        page_score=0.0,
                        content_score=0.0,
                        total_score=0.0,
                        answer="",
                        sources=[]
                    )
            else:
                # Offline evaluation mode
                result = self.evaluate_single(query, query.get("generated_response", {}))

            results.append(result)

        return self.calculate_metrics(results)

    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict:
        """Calculate aggregate metrics from evaluation results."""
        total = len(results)
        if total == 0:
            return {"error": "No results to calculate"}

        return {
            "overall_score": sum(r.total_score for r in results) / total,
            "avg_filename_score": sum(r.filename_score for r in results) / total,
            "avg_page_score": sum(r.page_score for r in results) / total,
            "avg_content_score": sum(r.content_score for r in results) / total,
            "perfect_scores": sum(1 for r in results if r.total_score == 1.0),
            "failed_scores": sum(1 for r in results if r.total_score < 0.5),
            "detailed_results": [
                {
                    "query_id": r.query_id,
                    "filename_score": r.filename_score,
                    "page_score": r.page_score,
                    "content_score": r.content_score,
                    "total_score": r.total_score,
                    "answer": r.answer,
                    "sources": r.sources
                }
                for r in results
            ]
        }


def run_evaluation(test_set_path: str, api_url: str = None) -> Dict:
    """
    Run evaluation from a test set JSON file.

    Args:
        test_set_path: Path to test data JSON file
        api_url: Optional API URL for live testing

    Returns:
        Dictionary with evaluation metrics
    """
    import json

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    queries = test_data.get("queries", [])

    client = None
    if api_url:
        import requests
        client = requests.Session()
        client.base_url = api_url

    evaluator = RAGEvaluator(queries, client)
    return evaluator.evaluate_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("test_set", help="Path to test data JSON file")
    parser.add_argument("--api-url", help="API base URL for live testing")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    results = run_evaluation(args.test_set, args.api_url)

    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Filename Match: {results['avg_filename_score']:.3f}")
    print(f"Page Match: {results['avg_page_score']:.3f}")
    print(f"Content Similarity: {results['avg_content_score']:.3f}")
    print(f"Perfect Scores: {results['perfect_scores']}/{len(results['detailed_results'])}")
    print(f"Failed Scores: {results['failed_scores']}/{len(results['detailed_results'])}")

    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")