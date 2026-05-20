"""Evaluation script for RAG answers."""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def evaluate_page_match(reference_pages: List[int], cited_pages: List[int]) -> float:
    """Evaluate page matching score."""
    if not reference_pages:
        return 1.0  # No reference pages means no mismatch
    matches = sum(1 for cp in cited_pages if cp in reference_pages)
    return matches / len(cited_pages) if cited_pages else 0.0


def evaluate_filename_match(reference_filenames: List[str], cited_filenames: List[str]) -> float:
    """Evaluate filename matching score."""
    if not reference_filenames:
        return 1.0
    matches = sum(1 for cf in cited_filenames if cf in reference_filenames)
    return matches / len(cited_filenames) if cited_filenames else 0.0


def evaluate_answer_content(reference_answer: str, generated_answer: str) -> float:
    """Evaluate answer content similarity using Jaccard."""
    return calculate_jaccard_similarity(reference_answer, generated_answer)


def evaluate_rag_response(
    reference: Dict,
    generated: Dict,
    weights: Tuple[float, float, float] = (0.25, 0.25, 0.5)
) -> Dict[str, float]:
    """
    Evaluate RAG response with the three metrics.

    Args:
        reference: Reference answer containing expected pages, filenames, and content
        generated: Generated answer containing cited pages, filenames, and content
        weights: Weights for page, filename, and content scores

    Returns:
        Dictionary with individual scores and weighted total
    """
    page_score = evaluate_page_match(
        reference.get("pages", []),
        generated.get("cited_pages", [])
    )

    filename_score = evaluate_filename_match(
        reference.get("filenames", []),
        generated.get("cited_filenames", [])
    )

    content_score = evaluate_answer_content(
        reference.get("answer", ""),
        generated.get("answer", "")
    )

    total_score = (
        weights[0] * page_score +
        weights[1] * filename_score +
        weights[2] * content_score
    )

    return {
        "page_match_score": page_score,
        "filename_match_score": filename_score,
        "content_similarity_score": content_score,
        "total_score": total_score
    }


def run_evaluation(test_set_path: str) -> None:
    """Run evaluation on a test set."""
    import json

    print("Loading test set from:", test_set_path)
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for i, item in enumerate(test_data):
        reference = item["reference"]
        generated = item["generated"]
        scores = evaluate_rag_response(reference, generated)

        print(f"\nTest {i + 1}:")
        print(f"  Page Match: {scores['page_match_score']:.2f}")
        print(f"  Filename Match: {scores['filename_match_score']:.2f}")
        print(f"  Content Similarity: {scores['content_similarity_score']:.2f}")
        print(f"  Total Score: {scores['total_score']:.2f}")

        results.append(scores)

    avg_total = sum(r["total_score"] for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Average Total Score: {avg_total:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_evaluation(sys.argv[1])
    else:
        print("Usage: python evaluate.py <test_set.json>")
        print("\nTest set format:")
        print('{"reference": {"pages": [1,2], "filenames": ["doc.pdf"], "answer": "..."}, "generated": {...}}')