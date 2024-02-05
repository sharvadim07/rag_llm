import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import LlamaCpp
# To allow CUDA support install llama-cpp-python with following way
# CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

### Chroma imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import Chroma
###
# from langchain_community.vectorstores import Qdrant

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate


from langchain_core.runnables import RunnablePassthrough, RunnablePick

# Start
torch.cuda.empty_cache()

# Load documents
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
loader = TextLoader("/home/vadim/Work/rag_llm/my_document.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


# Init embeddings on GPU
model_name = 'intfloat/multilingual-e5-large'
model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

# Create vectorstore for documents with embeddings
### Qdrant
# vectorstore = Qdrant.from_documents(
#     all_splits[:10],
#     embeddings,
#     location=":memory:",  # Local mode with in-memory storage only
# )
###
### Chroma
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
)
###

# Init question
# question = "Какое превышение вредных веществ в Красноярске?"
question = "Какие новости про Красноярск?"

# Search question in vectorstore
docs = vectorstore.similarity_search(question)

# Init prompt
my_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="Ты администратор новостной ленты, тебя зовут НовостиGPT и ты отвечаешь на вопросы о новостях. У тебя есть контекст с новостями. Не упоминай контекст или его отрывки при ответе, клиент ничего не должен знать о контексе, по которому ты отвечаешь. То чего нет в контексте не сообщается. Не употребляй фразы вида “По нашему контексту“, “в нашем контексте”, “в контексте“.\Вопрос: {question} \Контекст: {context} \nОтвет:"))
from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages = [my_prompt]

# How to format all documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Init local LLM on GPU
llm = LlamaCpp(
    model_path="/home/vadim/Work/rag_llm/mistral-7b-openorca.Q8_0.gguf",
    n_gpu_layers=8,
    n_batch=512,
    n_ctx=10000,
    verbose=True,
    # offload_kqv=True,
    f16_kv=True,
)

# Test LLM
res = llm.invoke("Who are you?")
print(res)

# Init chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run asking LLM using chain
res = chain.invoke({"context": docs, "question": question})
print(res)
