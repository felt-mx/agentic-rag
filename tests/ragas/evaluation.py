import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from retrieval.answer.prompt_builder import (
    build_prompt,
    build_critique_prompt,
    build_clarifying_question_prompt,
)
from retrieval.query.corpus_summary import get_corpus_summary
from retrieval.query.dispatcher import dispatch
from retrieval.state import AgentState
from retrieval.pipeline import RetrievalPipeline
from core.models.generator import VLLMClient
from config.config import (
    MILVUS_DATABASE,
    VLLM_API_URL,
    VLLM_GEN_API_PORT,
    VLLM_GEN_MODEL_NAME,
    VLLM_EMBED_API_PORT,
    VLLM_EMBED_MODEL_NAME,
)
import ast
import asyncio
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas import evaluate

try:
    from ragas.run_config import RunConfig
except ImportError:
    # Compatibility fallback for older/newer ragas package layouts.
    from ragas import RunConfig
from datasets import Dataset
import pandas as pd


# Set dummy OpenAI key to satisfy client validation
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-usage"


# Import your pipeline components


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
        corpus_summary = get_corpus_summary()

        state = AgentState(original_query=query)
        results = []

        # ------------------------------------------------------------------
        # Agentic dispatch loop
        # ------------------------------------------------------------------
        while state.retries_remaining >= 0:
            # 1. Dispatch: choose strategy + processed queries
            state = await dispatch(query, vllm_client, state, corpus_summary)
            print(
                f"[dispatch] strategy={state.current_strategy} "
                f"queries={state.processed_queries} "
                f"reason={state.reasoning}"
            )

            # 2. Execute the chosen strategy worker
            if state.current_strategy == "Expansion":
                results = await retrieval_pipeline.retrieve_with_expansion(
                    original_query=query,
                    top_k=5,
                    retrieval_k=20,
                )
            elif state.current_strategy == "Decomposition":
                results = await retrieval_pipeline.retrieve_with_decomposition(
                    original_query=query,
                    queries=state.processed_queries,
                    top_k=5,
                    retrieval_k=20,
                )
            else:
                # Hybrid: decompose then expand each sub-query
                results = await retrieval_pipeline.retrieve_hybrid(
                    original_query=query,
                    sub_queries=state.processed_queries,
                    top_k=5,
                    retrieval_k=20,
                )

            print(f"[retrieval] got {len(results)} results")

            # 3. Sufficiency check
            if results:
                critique_prompt = build_critique_prompt(query, results)
                critique_response = await vllm_client.generate(
                    critique_prompt, tools=None, tool_choice=None
                )
                critique_text = critique_response.get("content", "").strip()
                first_line = critique_text.splitlines()[0].strip().upper()
                print(f"[sufficiency] {critique_text}")

                if "INSUFFICIENT" not in first_line:
                    state.best_results = results
                    break
                else:
                    lines = critique_text.splitlines()
                    critique_sentence = (
                        lines[1].strip() if len(lines) > 1 else critique_text
                    )
                    state.critique_log.append(critique_sentence)
                    if results and not state.best_results:
                        state.best_results = results
            else:
                state.critique_log.append("Retrieval returned no results.")

            state.retries_remaining -= 1

        # ------------------------------------------------------------------
        # Exit strategy
        # ------------------------------------------------------------------
        final_results = results if results else state.best_results

        # Prepare contexts for RAGAS
        contexts = []
        if final_results and isinstance(final_results, list):
            contexts = [res.get("answer", "") for res in final_results]

        if not final_results:
            clarify_prompt = build_clarifying_question_prompt(query, state.critique_log)
            clarify_response = await vllm_client.generate(
                clarify_prompt, tools=None, tool_choice=None
            )
            clarifying_text = clarify_response.get("content", "").strip()
            return clarifying_text, []

        prompt = build_prompt(final_results, query, None)
        response = await vllm_client.generate(
            prompt, tools=None, tool_choice=None, enable_thinking=False
        )

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
def run_evaluation(
    csv_path,
    output_name,
    database=None,
    sample_size=None,
    ragas_timeout_seconds=900,
):
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
    empty_gt_count = int((df["ground_truth"].str.strip() == "").sum())
    if empty_gt_count > 0:
        print(
            f"Warning: {empty_gt_count} rows have empty ground_truth. "
            "Some metrics may be null for these rows regardless of timeout."
        )

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
    # RAGAS needs this column name strictly
    df["contexts"] = retrieved_contexts

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
        max_completion_tokens=8192,
        model_kwargs={
        "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        },
    )

    evaluator_embeddings = OpenAIEmbeddings(
        model=VLLM_EMBED_MODEL_NAME,
        openai_api_base=f"http://{VLLM_API_URL}:{VLLM_EMBED_API_PORT}/v1",
        openai_api_key="EMPTY",
    )

    # Run Metrics
    print("Running RAGAS metrics calculation...")
    run_config = RunConfig(timeout=ragas_timeout_seconds, max_workers=1, max_retries=3)
    print(
        f"Using RAGAS timeout: {ragas_timeout_seconds}s per metric call. "
        "Increase this value for slow local models."
    )
    results = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,  # Quality of Reranker
            context_recall,  # Quality of Hybrid Search
            faithfulness,  # Quality of LLM adherence to context
            answer_relevancy,  # Quality of Agent/Answer
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    print(f"Scores: {results}")

    # Save detailed breakdown (essential for debugging)
    results_df = results.to_pandas()

    # Define output path - prefer 'documents' folder so it's accessible outside container if mounted
    output_path = project_root / "documents" / f"{output_name}_detailed_results.csv"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
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
    ragas_timeout_seconds = 86400
    run_evaluation(
        str(dataset_path),
        "train_evaluation_results",
        database=None,
        sample_size=SAMPLE_SIZE,
        ragas_timeout_seconds=ragas_timeout_seconds,
    )
