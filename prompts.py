retrieval_prompt = """\
Question: {question}

You are an intelligent assistant. Your task is to answer the question using \
only the provided context. Do not use any outside knowledge or make \
assumptions. If the answer is not in the context, simply state that the \
information is not available.

Context:
{context}

Question: {question}

Answer: 
"""