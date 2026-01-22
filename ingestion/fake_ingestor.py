import uuid
from core.models.block import Block
from core.models.section import Section
from core.models.document import Document
from core.models.chunk import Chunk
import numpy as np


def fake_ingest_text(text: str) -> Document:
    block = Block(block_id=str(uuid.uuid4()), type="text", content=text, metadata={})
    section = Section(
        section_id=str(uuid.uuid4()),
        title="Intro",
        level=1,
        blocks=[block],
        metadata={},
    )
    document = Document(
        document_id=str(uuid.uuid4()), sections=[section], metadata={"source": "fake"}
    )
    return document


def fake_generate_chunks(doc: Document) -> list[Chunk]:
    chunks = []
    for section in doc.sections:
        text = " ".join([blk.content for blk in section.blocks if blk.type == "text"])
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc.document_id,
            section_id=section.section_id,
            text=text,
            metadata=doc.metadata,
            dense_embedding=np.random.rand(768),  # Renamed from 'embedding'
            image_embedding=np.random.rand(768),  # Optional: can be None
        )
        chunks.append(chunk)

    return chunks
