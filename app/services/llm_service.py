"""
Ollama LLM generation service with multi-step reasoning.
Uses Chain-of-Thought (CoT) evidence extraction, verification, and grounded synthesis
to actively prevent hallucinations.
"""

import re
import time
from typing import Tuple, Union
import ollama

from app.core.exceptions import LLMGenerationError
from app.core.logging_config import get_logger
from config import (
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_TIMEOUT,
)

logger = get_logger(__name__)

# Initialize client with configured host and timeout
_client = ollama.Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)


def parse_reasoned_response(raw_text: str) -> Tuple[str, str]:
    """
    Extract the clean final answer and internal reasoning steps from the model output.

    Args:
        raw_text: Full output from the LLM containing <reasoning> and <answer> blocks.

    Returns:
        tuple of (clean_answer, reasoning_trace)
    """
    if not raw_text:
        return "", ""

    answer_match = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL | re.IGNORECASE)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw_text, re.DOTALL | re.IGNORECASE)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    if answer_match:
        clean_answer = answer_match.group(1).strip()
    else:
        # Fallback: remove <reasoning> block if present
        clean_answer = re.sub(r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
        # Clean any stray tags
        clean_answer = re.sub(r"</?answer>", "", clean_answer, flags=re.IGNORECASE).strip()
        if not clean_answer:
            clean_answer = raw_text.strip()

    return clean_answer, reasoning


def generate_answer(
    question: str,
    context: str,
    return_reasoning: bool = False,
) -> Union[str, Tuple[str, str]]:
    """
    Generate an answer using multi-step grounded reasoning to eliminate hallucinations.

    Process:
        1. Step 1 (Evidence Extraction): Locates exact quotes from context addressing the question.
        2. Step 2 (Verification & Grounding): Confirms whether quotes directly answer the question without extrapolation.
        3. Step 3 (Final Answer): Formulates clean response using only verified quotes.

    Args:
        question: User query string.
        context: Concatenated context chunks from retrieved documents.
        return_reasoning: Whether to return the (clean_answer, reasoning_trace) tuple.

    Returns:
        clean_answer string (if return_reasoning=False) OR
        (clean_answer, reasoning_trace) tuple (if return_reasoning=True).

    Raises:
        LLMGenerationError: If the LLM call fails or times out.
    """
    truncated_context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""You are a strict, hallucination-resistant document reasoning assistant.

You must follow a strict 3-STEP REASONING PROCESS before answering:

STEP 1 - EVIDENCE EXTRACTION:
Extract all verbatim sentences, phrases, or rules from CONTEXT that relate to the QUESTION. If nothing in the context relates to the question, state 'NO_DIRECT_EVIDENCE'.

STEP 2 - VERIFICATION & GROUNDING:
Verify what the extracted evidence explicitly confirms. Identify the exact facts, requirements, or conditions stated in the text without assuming or guessing unmentioned details.

STEP 3 - FINAL ANSWER:
- If relevant evidence was found: Provide a clear, structured answer using bullet points based ONLY on the verified evidence from Step 1.
- If NO_DIRECT_EVIDENCE: State: 'This information is not available in the provided document.'

---CONTEXT---
{truncated_context}

---QUESTION---
{question}

Respond in the following format:
<reasoning>
[Your Step 1 and Step 2 analysis here]
</reasoning>
<answer>
[Your final verified answer here]
</answer>
"""

    logger.info(
        f"Executing multi-step reasoning using model '{OLLAMA_LLM_MODEL}' for query: '{question[:80]}'..."
    )
    start_time = time.time()

    try:
        response = _client.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_NUM_PREDICT,
            },
        )

        elapsed = time.time() - start_time
        raw_content = response.get("message", {}).get("content", "").strip()
        logger.info(f"Multi-step reasoning completed in {elapsed:.2f}s (raw length: {len(raw_content)} chars)")

        clean_answer, reasoning = parse_reasoned_response(raw_content)

        if not clean_answer:
            logger.warning("LLM generated no answer content after parsing.")
            clean_answer = "This information is not available in the provided document."

        if return_reasoning:
            return clean_answer, reasoning
        return clean_answer

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Multi-step reasoning failed after {elapsed:.2f}s using {OLLAMA_LLM_MODEL}: {e}",
            exc_info=True,
        )
        raise LLMGenerationError(f"Multi-step reasoning failed: {str(e)}") from e
