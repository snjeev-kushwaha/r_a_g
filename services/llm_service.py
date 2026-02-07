import ollama

def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a helpful assistant.

Use ONLY the information from the context below.
You MAY summarize, count, or infer totals if the data is present.
If the context does not contain relevant information, say:
"No data found in the document."

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# def generate_answer(question: str, context: str) -> str:
#     prompt = f"""
# You are an AI assistant answering questions using the provided document content.

# Rules:
# - You MUST only use the information present in the context.
# - You ARE allowed to analyze, count, summarize, and infer answers from the context.
# - If the answer is not explicitly written, derive it from the data.
# - If the information truly does not exist, say: "Information not found in the document."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = ollama.chat(
#         model="llama3.1",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return response["message"]["content"]