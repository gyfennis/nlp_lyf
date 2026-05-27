"""Offline evaluation helpers aligned with README scoring rules."""


def jaccard_similarity(expected: str, actual: str) -> float:
    a = set(expected.strip())
    b = set(actual.strip())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_answer(
    *,
    expected_answer: str,
    actual_answer: str,
    expected_page: str | None,
    actual_page: str | None,
    expected_filename: str | None,
    actual_filename: str | None,
) -> dict[str, float]:
    page_score = 0.25 if expected_page and actual_page and expected_page in actual_page else 0.0
    file_score = (
        0.25
        if expected_filename and actual_filename and expected_filename in actual_filename
        else 0.0
    )
    content_score = 0.5 * jaccard_similarity(expected_answer, actual_answer)
    total = page_score + file_score + content_score
    return {
        "page_match": page_score,
        "filename_match": file_score,
        "content_similarity": content_score,
        "total": total,
    }
