from app.evaluation import jaccard_similarity, score_answer


def test_jaccard_identical():
    assert jaccard_similarity("abc", "abc") == 1.0


def test_jaccard_disjoint():
    assert jaccard_similarity("abc", "xyz") == 0.0


def test_score_answer_full_match():
    result = score_answer(
        expected_answer="销售额在第三季度下降",
        actual_answer="销售额在第三季度下降",
        expected_page="第3页",
        actual_page="见第3页图表",
        expected_filename="report.pdf",
        actual_filename="来源: report.pdf",
    )
    assert result["page_match"] == 0.25
    assert result["filename_match"] == 0.25
    assert result["content_similarity"] == 0.5
    assert result["total"] == 1.0
