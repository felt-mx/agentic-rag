import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import random
from typing import Any, List, Optional
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr
from configs.config import VLLM_API_URL, VLLM_GEN_API_PORT, VLLM_GEN_MODEL_NAME


# Helper function to run async code safely in threads
def run_async(coro):
    """Run async coroutine in a thread-safe way"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop in current thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    else:
        # There's already a running loop, we need to run in a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


# Import your local VLLM clients
import sys

sys.path.append("../..")
from core.models.generator import VLLMClient as GeneratorClient
from core.models.embedder import VLLMClient as EmbedderClient


# Custom LangChain-compatible wrapper for your Llama4 Scout generator
class CustomChatModel(BaseChatModel):
    """Custom chat model wrapper for local llama4 scout via VLLMClient"""

    model_name: str = VLLM_GEN_MODEL_NAME
    temperature: float = 0.7
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = GeneratorClient()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using your VLLMClient"""
        # Convert LangChain messages to the format your API expects
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})

        # Call your async generate method synchronously
        response = run_async(
            self._client.generate(
                formatted_messages,
                tools=None,
                tool_choice=None,
                temperature=self.temperature,
            )
        )

        # Extract content from response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = str(response)

        # Return in LangChain format
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "custom-llama4-scout"


# Custom LangChain-compatible wrapper for your Jina embeddings
class CustomEmbeddings(Embeddings):
    """Custom embeddings wrapper for local Jina via VLLMClient with late chunking"""

    def __init__(self):
        self.client = EmbedderClient()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using your Jina late_chunking_embed API"""
        embeddings = []

        for text in texts:
            # Use late_chunking_embed for each document
            chunk_results = run_async(
                self.client.late_chunking_embed(
                    text=text,
                    task="retrieval.passage",
                    late_chunking=True,
                    batch_size=4096,
                )
            )
            # Take the first embedding from the chunked results
            if chunk_results and len(chunk_results) > 0:
                embeddings.append(chunk_results[0]["embedding"])
            else:
                raise Exception(f"No embeddings returned for text: {text[:100]}...")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using your Jina late_chunking_embed API"""
        chunk_results = run_async(
            self.client.late_chunking_embed(
                text=text, task="retrieval.query", late_chunking=True, batch_size=4096
            )
        )
        # Return the first embedding from the chunked results
        if chunk_results and len(chunk_results) > 0:
            return chunk_results[0]["embedding"]
        else:
            raise Exception(f"No embeddings returned for query: {text[:100]}...")


# 1. Load YOUR actual documents
documents_dir = project_root / "documents"
all_documents = []

# Load all PDF files using PyMuPDFLoader
for pdf_file in documents_dir.glob("**/*.pdf"):
    print(f"Loading: {pdf_file.name}")
    loader = PyMuPDFLoader(str(pdf_file))
    documents = loader.load()
    all_documents.extend(documents)

print(
    f"\nLoaded {len(all_documents)} document(s) from {len(list(documents_dir.glob('**/*.pdf')))} PDF file(s)"
)

# 2. Configure the Generator using your local llama4 scout and jina embeddings
generator_llm = CustomChatModel(temperature=0.7)
critic_llm = CustomChatModel(temperature=0.7)
embeddings = CustomEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

sample_docs = random.sample(all_documents, min(len(all_documents), 50))
# 3. Generate the Dataset
# Using default query distribution (single-hop and multi-hop queries)
testset = generator.generate_with_langchain_docs(
    sample_docs,
    testset_size=20,  # Start small to save cost
)

# 4. Save it
test_df = testset.to_pandas()
test_df.to_csv("./dataset/marriott_dataset.csv")
print(test_df.head())
