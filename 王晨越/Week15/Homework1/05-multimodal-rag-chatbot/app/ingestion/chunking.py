def split_text2chunks(lines: list[str], chunk_size: int = 256) -> list[str]:
    """Split markdown lines into chunks; each chunk length is bounded by chunk_size."""
    chunks: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "# References":
            continue
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        if not chunks:
            chunks.append(line)
        elif len(chunks[-1]) <= chunk_size:
            chunks[-1] += "\n" + line
        else:
            chunks.append(line)
    return chunks
