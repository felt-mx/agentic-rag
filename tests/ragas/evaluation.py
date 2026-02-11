import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set dummy OpenAI key to satisfy client validation
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-usage"

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
import asyncio
import ast

# Import your pipeline components
from config.config import (
    MILVUS_DATABASE,
    VLLM_API_URL,
    VLLM_GEN_API_PORT,
    VLLM_GEN_MODEL_NAME,
    VLLM_EMBED_API_PORT,
    VLLM_EMBED_MODEL_NAME,
)
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.answer.prompt_builder import (
    build_prompt,
    build_reformulation_prompt,
    build_retry_prompt,
    build_relevance_check_prompt,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ==========================================
# 1. DEFINE YOUR RAG INTERFACE
# ==========================================
async def my_rag_pipeline_async(query: str, database: str = None):
    """
    Connect this to your actual RAG application.
    Returns:
        answer (str): The final generated response.
        contexts (list[str]): The list of text chunks retrieved by your Hybrid Search.
    """
    try:
        # Use specified database or fall back to config
        database = database or MILVUS_DATABASE
        retrieval_pipeline = RetrievalPipeline(database=database)

        vllm_client = VLLMClient()
        count = 0
        input_texts = []

        reformulation_prompt = build_reformulation_prompt(query)
        reformulated_response = await vllm_client.generate(
            reformulation_prompt, tools=None, tool_choice=None, temperature=0.1
        )

        reformulated_text = ""
        if isinstance(reformulated_response, dict):
            reformulated_text = reformulated_response.get("content", "").strip()
        else:
            reformulated_text = str(reformulated_response).strip()

        print(f"Reformulated: {reformulated_text}")
        input_texts = [query]

        results = await retrieval_pipeline.retrieve(
            reformulated_text,
            top_k=5,
            retrieval_k=20,
            rerank_method="weighted",
        )

        while True:
            # First check: Are there any results?
            if not results and count < 3:
                count += 1

                input_texts.append(reformulated_text)
                retry_prompt = build_retry_prompt(input_texts)

                reformulated_response = await vllm_client.generate(
                    retry_prompt, tools=None, tool_choice=None
                )
                if isinstance(reformulated_response, dict):
                    reformulated_text = reformulated_response.get("content", "").strip()
                else:
                    reformulated_text = str(reformulated_response).strip()

                print(f"Retry {count}: {reformulated_text}")

                results = await retrieval_pipeline.retrieve(
                    reformulated_text,
                    top_k=5,
                    retrieval_k=20,
                    rerank_method="weighted",
                )
            elif results and count < 3:
                # Second check: Are the results actually relevant and sufficient?
                relevance_prompt = build_relevance_check_prompt(query, results)
                relevance_response = await vllm_client.generate(
                    relevance_prompt, tools=None, tool_choice=None
                )

                relevance_verdict = ""
                if isinstance(relevance_response, dict):
                    relevance_verdict = (
                        relevance_response.get("content", "").strip().upper()
                    )
                else:
                    relevance_verdict = str(relevance_response).strip().upper()

                print(f"Relevance check: {relevance_verdict}")

                if "INSUFFICIENT" in relevance_verdict:
                    # Results exist but are not relevant enough, retry with reformulation
                    count += 1
                    input_texts.append(reformulated_text)
                    retry_prompt = build_retry_prompt(input_texts)

                    reformulated_response = await vllm_client.generate(
                        retry_prompt, tools=None, tool_choice=None
                    )

                    if isinstance(reformulated_response, dict):
                        reformulated_text = reformulated_response.get(
                            "content", ""
                        ).strip()
                    else:
                        reformulated_text = str(reformulated_response).strip()

                    print(f"Retry {count} (insufficient results): {reformulated_text}")

                    results = await retrieval_pipeline.retrieve(
                        reformulated_text,
                        top_k=5,
                        retrieval_k=20,
                        rerank_method="weighted",
                    )
                else:
                    # Results are sufficient, exit the loop
                    break
            else:  # Either max retries reached or results found and relevant
                break

        # Prepare contexts for RAGAS
        contexts = []
        if results and isinstance(results, list):
            contexts = [res.get("answer", "") for res in results]

        if not results:
            results = "No relevant information found."

        prompt = build_prompt(results, query, None)

        # Pure HTTP response - no streaming
        response = await vllm_client.generate(prompt, tools=None, tool_choice=None)

        # Extract answer text
        answer = ""
        if isinstance(response, dict):
            answer = response.get("content", str(response))
        else:
            answer = str(response)

        # Clean up the answer (remove headers if present)
        if "## Answer to User's Question" in answer:
            answer = answer.split("## Answer to User's Question")[-1].strip()

        return answer, contexts

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return f"Error: {str(e)}", []


def my_rag_pipeline(query: str, database: str = None):
    """
    Synchronous wrapper for the async RAG pipeline.
    Returns:
        answer (str): The final generated response.
        contexts (list[str]): The list of text chunks retrieved by your Hybrid Search.
    """
    return asyncio.run(my_rag_pipeline_async(query, database))


# ==========================================
# 2. THE EVALUATION HARNESS
# ==========================================
def run_evaluation(csv_path, output_name, database=None, sample_size=None):
    print(f"--- Starting Evaluation for {csv_path} ---")
    if database:
        print(f"Using database: {database}")

    # Load the Golden Data
    df = pd.read_csv(csv_path)

    # Randomly sample the dataset if sample_size is provided
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size} random questions from {len(df)} total...")
        df = df.sample(n=sample_size, random_state=42)

    # Parse the 'answers' column to extract ground_truth
    # The answers column is a string representation of a dict like:
    # "{'text': array(['answer text'], dtype=object), 'answer_start': array([123], dtype=int32)}"
    def extract_ground_truth(answers_str):
        if not isinstance(answers_str, str):
            return ""

        # Try regex pattern matching for numpy array text representation
        # Matches: 'text': array(['ANSWER'], or "ANSWER", dtype=object)
        import re

        # Pattern 1: Standard single quoted array
        match = re.search(r"'text':\s*array\(\s*\[\s*'([^']*)'\s*\]", answers_str)
        if match:
            return match.group(1)

        # Pattern 2: Double quoted array
        match = re.search(r"'text':\s*array\(\s*\[\s*\"([^\"]*)\"\s*\]", answers_str)
        if match:
            return match.group(1)

        return ""

    df["ground_truth"] = df["answers"].apply(extract_ground_truth)

    # Prepare lists to store your RAG's output
    generated_answers = []
    retrieved_contexts = []

    # Loop through questions and query your RAG
    print(f"Processing {len(df)} queries...")
    for index, row in df.iterrows():
        question = row["question"]

        try:
            # CALL YOUR SYSTEM
            ans, ctx = my_rag_pipeline(question, database=database)

            generated_answers.append(ans)
            retrieved_contexts.append(ctx)

            # Simple progress log
            if index % 10 == 0:
                print(f"Processed {index}/{len(df)}")

        except Exception as e:
            print(f"Error on row {index}: {e}")
            generated_answers.append("Error")
            retrieved_contexts.append([])

    # Add results back to the dataframe
    df["answer"] = generated_answers
    df["contexts"] = retrieved_contexts  # RAGAS needs this column name strictly

    # Ensure we have all required columns: question, answer, contexts, ground_truth
    print(f"Dataset preview:\n{df[['question', 'answer', 'ground_truth']].head(2)}")

    # Convert to HuggingFace Dataset
    ragas_dataset = Dataset.from_pandas(
        df[["question", "answer", "contexts", "ground_truth"]]
    )

    # Setup Local LLM & Embeddings for Ragas
    print("Setting up local VLLM for evaluation...")
    evaluator_llm = ChatOpenAI(
        model=VLLM_GEN_MODEL_NAME,
        openai_api_base=f"http://{VLLM_API_URL}:{VLLM_GEN_API_PORT}/v1",
        openai_api_key="EMPTY",
        temperature=0,
    )
    evaluator_embeddings = OpenAIEmbeddings(
        model=VLLM_EMBED_MODEL_NAME,
        openai_api_base=f"http://{VLLM_API_URL}:{VLLM_EMBED_API_PORT}/v1",
        openai_api_key="EMPTY",
    )

    # Run Metrics
    print("Skipping RAGAS metrics calculation...")
    # results = evaluate(
    #     ragas_dataset,
    #     metrics=[
    #         context_precision,  # Quality of Reranker
    #         context_recall,  # Quality of Hybrid Search
    #         faithfulness,  # Quality of LLM adherence to context
    #         answer_relevancy,  # Quality of Agent/Answer
    #     ],
    #     llm=evaluator_llm,
    #     embeddings=evaluator_embeddings,
    # )

    print(f"Skipping scores display.")

    # Save detailed breakdown (essential for debugging)
    # results_df = results.to_pandas()
    results_df = df

    # Define output path - prefer 'documents' folder so it's accessible outside container if mounted
    output_path = project_root / "documents" / f"{output_name}_detailed_results.csv"

    try:
        results_df.to_csv(output_path, index=False)
        print(f"Saved details to {output_path}")
    except Exception as e:
        print(f"Could not save to {output_path}: {e}")
        # Fallback to current directory as last resort, or /tmp in docker
        fallback_path = f"{output_name}_detailed_results.csv"
        try:
            results_df.to_csv(fallback_path, index=False)
            print(f"Saved details to {fallback_path}")
        except Exception as e_fallback:
            print(f"Could not save to {fallback_path}: {e_fallback}")


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Run evaluation on train.csv dataset (SQuAD format)
    # Ensure your Milvus database is populated before running this!
    # CSV columns: 'id', 'title', 'context', 'question', 'answers'
    # The script will automatically extract ground_truth from the 'answers' column
    dataset_path = Path(__file__).parent / "datasets" / "train.csv"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    SAMPLE_SIZE = 100
    run_evaluation(
        str(dataset_path),
        "train_evaluation_results",
        database=None,
        sample_size=SAMPLE_SIZE,
    )
