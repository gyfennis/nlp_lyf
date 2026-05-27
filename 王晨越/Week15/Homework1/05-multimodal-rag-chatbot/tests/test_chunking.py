from app.ingestion.chunking import split_text2chunks
from app.ingestion.paths import rewrite_image_paths_in_chunk


def test_split_text2chunks_merges_short_lines():
    lines = ["# Title", "段落一内容较短。", "段落二。"]
    chunks = split_text2chunks(lines, chunk_size=256)
    assert len(chunks) >= 1
    assert "段落一" in chunks[0]


def test_split_skips_references_and_numeric_citations():
    lines = ["正文行。", "# References", "[1] citation", ""]
    chunks = split_text2chunks(lines)
    assert all("# References" not in c for c in chunks)
    assert all(not c.startswith("[1]") for c in chunks)


def test_rewrite_image_paths():
    text = "见下图\n![](images/chart.png)"
    out = rewrite_image_paths_in_chunk(text, "uploads/abc123.pdf")
    assert "./processed/abc123/vlm/images/" in out
