import ollama


def generate_answer(question: str, context: str) -> str:
    context = context[:6000]

    prompt = f"""
    You are a document QA system.
    
    Rules:
    1. Use ONLY the provided context.
    2. Extract only rows that directly answer the question.
    3. Do NOT include unrelated months or entries.
    4. If nothing matches exactly, say: No data found in the document.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Return only the relevant extracted data.
    """

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 300},
    )

    return response["message"]["content"]


# def generate_answer(question: str, context: str) -> str:
#     prompt = f"""
# You are a helpful assistant.

# Use ONLY the information from the context below.
# You MAY summarize, count, or infer totals if the data is present.
# If the context does not contain relevant information, say:
# "No data found in the document."

# Context:
# {context}

# Question:
# {question}

# Answer clearly and concisely.
# """
#     response = ollama.chat(
#         model="llama3.1",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response["message"]["content"]
