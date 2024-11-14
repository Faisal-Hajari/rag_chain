from llm import LLM, Prompt, VectorDB
import prompts


def gen_history(session_messages:list[dict[str:str]]) -> str:
    history = ""
    assistant= Prompt(prompts.role_assistant_prompt)
    user = Prompt(prompts.role_user_prompt)

    for message in session_messages:
        if message["role"] == "assistant":
            history+= "\n"+assistant(response=message["content"])
        elif message["role"] == "user":
            history+= "\n"+user(query=message["content"])
        else:
            raise ValueError("Unknown role in message")
    return history


def query_rewriter(llm:LLM, session_messages:list[dict[str:str]]) -> str:
    prompt = Prompt(prompts.chat_prompt)(
            system_prompt=prompts.query_rewriting_prompt,
            history=gen_history(session_messages),
            assistant_prompt= Prompt(prompts.role_assistant_prompt)(response="")
    )
    response = llm(prompt)
    response = llm("extract the exact query from the text do not add any other text: {response}".format(response=response))
    return response, prompt


def multi_query(llm:LLM, query:str) -> str:
    prompt = Prompt(prompts.chat_prompt)(
            system_prompt=prompts.multi_query_prompt,
            history=Prompt(prompts.role_user_prompt)(query=query),
            assistant_prompt= Prompt(prompts.role_assistant_prompt)(response="")
    )
    response = llm(prompt)
    response = llm("extract the exact five query from the text do not add any other text: {response}".format(response=response))
    return response, prompt


def concat_docs(docs):
    context = "".join([doc.page_content+"\n\n" for doc in docs])
    return context

def referance_gen(doc):
    source = doc.metadata["source"].split("/")[-1]
    page = doc.metadata["page"]
    referance = f"{source} : {page}"
    return referance

def concat_docs_with_referance(docs):
    context = "\n".join(
        [f"{referance_gen(doc)}\n{doc.page_content}\n\n" 
         for doc in docs ]
    )
    return context

def rag_with_ref(llm:LLM, vdb:VectorDB, messages:list[dict[str:str]]):
    rew_query, _ = query_rewriter(llm, messages)
    multi_query_res, _ = multi_query(llm, rew_query)
    queries = multi_query_res.split("\n")
    docs = [] 
    for query in queries:
        docs.extend(vdb.compress_search(query)) 
    ids = []
    for i, doc in enumerate(docs):
        if doc.metadata["id"] in set(ids):
            docs.pop(i)
        else:
            ids.append(doc.metadata["id"])
    context = concat_docs_with_referance(docs)

    rag_prompt = Prompt(prompts.chat_prompt)(
            system_prompt=Prompt(prompts.rag_systme_prompt)(context=context),
            history=gen_history(messages),
            assistant_prompt= Prompt(prompts.role_assistant_prompt)(response="")
    )
    for chunk in llm.stream(rag_prompt):
        yield chunk, rag_prompt