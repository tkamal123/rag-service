import pytest

from app.rag import chunk_text


def test_short_paragraphs_are_packed_together():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    assert chunk_text(text, chunk_size=200) == ["First paragraph.\n\nSecond paragraph.\n\nThird paragraph."]


def test_chunks_never_exceed_size_and_are_not_empty():
    text = "\n\n".join(["word " * 60 for _ in range(6)])
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert chunks and all(0 < len(c) <= 120 for c in chunks)


def test_long_paragraph_is_split_with_overlap():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) > 1
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:8])  # shared words at the seam


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=50, overlap=50)
