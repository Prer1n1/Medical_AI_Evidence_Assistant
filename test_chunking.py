"""Free/offline — a fake, fixed-vector embeddings double proves the
semantic-split boundary math without ever loading the real (400MB)
biomedical model or making any network call. Ported test strategy from
enterprise-agentic-rag.
"""

from __future__ import annotations

from ingestion.chunking import chunk_documents, semantic_split
from ingestion.schema import Document, DocumentMetadata


class FakeEmbeddings:
    """Returns a FIXED vector per distinct sentence content — sentences
    about "hypertension" cluster near one vector, sentences about
    "vaccination" cluster near an orthogonal one, so the cosine-distance
    breakpoint logic has a real, predictable jump to detect without
    needing a real model."""

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            if "hypertension" in text.lower():
                vectors.append([1.0, 0.0, 0.0])
            elif "vaccination" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.5, 0.5, 0.0])
        return vectors


def test_semantic_split_detects_topic_shift():
    text = (
        "Hypertension is a major risk factor for cardiovascular disease. "
        "Hypertension affects over a billion adults worldwide. "
        "Vaccination programs have reduced childhood mortality significantly. "
        "Vaccination schedules vary by country and age group."
    )
    pieces = semantic_split(text, FakeEmbeddings())
    assert len(pieces) == 2, f"expected exactly one split at the topic change, got {len(pieces)} pieces"
    assert "hypertension" in pieces[0].lower()
    assert "vaccination" in pieces[1].lower()
    print("OK: semantic_split cuts exactly at the topic change (hypertension -> vaccination)")


def test_chunk_documents_tables_batched_not_semantically_split():
    table_doc = Document(
        content="Drug | Dose\nAmoxicillin | 25 mg/kg",
        metadata=DocumentMetadata(source="a.pdf", doc_type="pdf", section="Table 1", extra={"is_table": True}),
    )
    chunks = chunk_documents([table_doc], embeddings=FakeEmbeddings())
    assert len(chunks) == 1
    assert chunks[0].content == table_doc.content
    print("OK: a single table Document produces exactly one (unsplit) chunk")


def test_chunk_documents_images_pass_through_as_single_chunks():
    image_doc = Document(
        content="A dosage chart showing amoxicillin dosing by weight band.",
        metadata=DocumentMetadata(source="a.pdf", doc_type="pdf", section="Figure 1", extra={"is_image": True, "image_path": "x.png"}),
    )
    chunks = chunk_documents([image_doc], embeddings=FakeEmbeddings())
    assert len(chunks) == 1
    assert chunks[0].metadata.extra["is_image"] is True
    assert chunks[0].metadata.extra["image_path"] == "x.png"
    print("OK: an image Document passes through as a single chunk, image_path preserved")


def test_chunk_index_unique_across_combined_list():
    docs = [
        Document(content="Row A | 1", metadata=DocumentMetadata(source="a.pdf", doc_type="pdf", section="Table 1", extra={"is_table": True})),
        Document(
            content="Hypertension prose about risk factors. Vaccination prose about immunity.",
            metadata=DocumentMetadata(source="a.pdf", doc_type="pdf", section="Body"),
        ),
    ]
    chunks = chunk_documents(docs, embeddings=FakeEmbeddings())
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), f"chunk_index not unique/sequential: {indices}"
    print(f"OK: chunk_index reindexed uniquely across {len(chunks)} combined chunks")


if __name__ == "__main__":
    test_semantic_split_detects_topic_shift()
    test_chunk_documents_tables_batched_not_semantically_split()
    test_chunk_documents_images_pass_through_as_single_chunks()
    test_chunk_index_unique_across_combined_list()
    print("All test_chunking.py checks passed.")
