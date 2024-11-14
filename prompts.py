rag_systme_prompt = """You are an advanced AI language model assistant for question-answering tasks specialized in AI eithics. 
Use the following pieces of retrieved context to answer the question. If you
don't know the answer, just say that you don't know. Use three sentences
maximum and keep the answer concise.
context:
{context}"""

multi_query_prompt = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
think about it step by step, and provide a detailed and well-thought-out explanation of the different perspectives.
fromat your response as follows:
reasoning:
{your reasoning here}
question:
{your alternative question here}
{your alternative question here}
{your alternative question here}
{your alternative question here}
{your alternative question here}
"""
query_rewriting_prompt = """You are an advanced AI language model assistant. 
Your task is to rephrase the user's question to enhance the relevance and quality of documents retrieved from a vector database.
By refining the user's query, your goal is to mitigate the limitations of distance-based similarity search and ensure more accurate and contextually appropriate results.
keep the question concise and clear, and try to capture the user's intent effectively.
think about it step by step, and provide a detailed and well-thought-out revision of the user's question.
write your though process and rationale after "reasoning: "
wtire your revised question after "question: " 
fromat your response as follows:
reasoning:
{your reasoning here}
question:
{your revised question here}
"""

role_user_prompt = """<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}"""

role_assistant_prompt = """<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}"""

chat_prompt ="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
{history}
{assistant_prompt}
"""