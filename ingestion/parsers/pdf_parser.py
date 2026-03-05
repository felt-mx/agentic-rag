import uuid
import base64
import fitz  # PyMuPDF
from pathlib import Path
from core.models.document import Document
from core.models.section import Section
from core.models.block import Block
from core.models.generator import VLLMClient
from retrieval.answer.prompt_builder import build_image_description_prompt
from .base import BaseParser


class PDFParser(BaseParser):
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    async def parse(self, file_path: Path, next_page_lines: int = 20) -> Document:
        sections = []
        vllm_client = VLLMClient()

        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)

        # Normalize and map keys used across the pipeline
        doc_meta = {
            "source": str(file_path),
            "file_name": file_path.name,
        }

        processed_xrefs = set()
        xref_descriptions = {}

        for page_num in range(total_pages):
            page = pdf_document[page_num]
            text = page.get_text().strip()

            # 1. Extract raster images
            image_descriptions = []
            image_list = page.get_images(full=True)

            # Filter out "junk" images (tiled slices, tiny icons, etc.)
            # We only want images that are likely meaningful (e.g., > 100px)
            valid_images = [img for img in image_list if img[2] > 100 and img[3] > 100]

            print(
                f"Page {page_num + 1}: Found {len(image_list)} raw refs, "
                f"processing {len(valid_images)} filtered image(s)"
            )

            # 2. Process filtered images
            for img_index, img in enumerate(valid_images):
                try:
                    xref = img[0]
                    if xref in processed_xrefs:
                        if xref in xref_descriptions:
                            image_descriptions.append(
                                f"[Image {img_index + 1}]: {xref_descriptions[xref]}"
                            )
                        continue

                    processed_xrefs.add(xref)
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_data = f"data:image/{image_ext};base64,{image_base64}"

                    description = await self._get_vllm_description(
                        vllm_client, image_data
                    )
                    if description:
                        xref_descriptions[xref] = description
                        image_descriptions.append(
                            f"[Image {img_index + 1}]: {description}"
                        )

                except Exception as e:
                    print(f"Error extracting image {xref} on page {page_num + 1}: {e}")

            # 3. Fallback: If no text and no images found, it's likely a vector or scan
            # We render the whole page as one image to be safe.
            if not text and not image_descriptions:
                print(
                    f"Page {page_num + 1}: Low content detected, rendering page as image fallback."
                )
                pix = page.get_pixmap(dpi=600)  # 2x zoom for clarity
                img_data = f"data:image/png;base64,{base64.b64encode(pix.tobytes('png')).decode('utf-8')}"

                page_desc = await self._get_vllm_description(vllm_client, img_data)
                if page_desc:
                    image_descriptions.append(f"[Full Page Visual]: {page_desc}")

            # 4. Contextual Text Building
            combined_text = text
            if page_num < total_pages - 1:
                next_page_text = pdf_document[page_num + 1].get_text()
                context_lines = "\n".join(next_page_text.split("\n")[:next_page_lines])
                combined_text = f"{combined_text}\n{context_lines}"

            if image_descriptions:
                image_section = "\n\n--- Visual Descriptions ---\n" + "\n\n".join(
                    image_descriptions
                )
                combined_text = f"{combined_text}{image_section}"

            # Create Block and Section
            if combined_text.strip():
                block = Block(
                    block_id=str(uuid.uuid4()),
                    type="text",
                    content=combined_text,
                    metadata={"page_number": page_num + 1},
                )
                sections.append(
                    Section(
                        section_id=str(uuid.uuid4()),
                        title=f"Page {page_num + 1}",
                        level=1,
                        blocks=[block],
                        metadata={
                            "page_number": page_num + 1,
                            "image_count": len(valid_images),
                        },
                    )
                )

        pdf_document.close()
        # Ensure the document metadata reflects actual PDF metadata collected
        return Document(
            document_id=str(uuid.uuid4()),
            metadata={**doc_meta, "total_pages": len(sections)},
            sections=sections,
        )

    async def _get_vllm_description(self, client, image_data: str) -> str:
        """Helper to call VLLM API"""
        try:
            image_prompt = build_image_description_prompt(image_data)
            response = await client.generate(image_prompt, tools=None, tool_choice=None, enable_thinking=False)

            print(response)
            return response.get("content", "").strip()
        except Exception:
            return ""
