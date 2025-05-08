import json
import os
import re
import time
import torch
import datetime
from accelerate import infer_auto_device_map, init_empty_weights
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv

from functools import partial
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.globals import set_verbose, set_debug
import faiss.contrib.torch_utils
# Disable verbose logging
set_verbose(False)

# Disable debug logging
set_debug(False)

load_dotenv("secrets.env")
token = os.getenv('hf_token')
together_key = os.getenv('TOGETHER_API_KEY')
novati_key = os.getenv('novati_key')
def backoff(delay=2, retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_retry = 0
            current_delay = delay
            while current_retry < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    current_retry += 1
                    if current_retry >= retries:
                        raise e
                    print(f"Failed to execute function '{func.__name__}'. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2
        return wrapper
    return decorator

#print(os.environ['HF_HOME'])


# Load text chunks ------
# Train/val text directory
raw_text_train_val_path = "cpiqa/extracted_paragraphs/"
raw_image_train_path = "results/openai/cpiqa"


# Define embedding model
# Other potential model options for embeddings
#modelPath = "sentence-transformers/all-MiniLM-l6-v2"
#modelPath = "climatebert/distilroberta-base-climate-f"
#modelPath = "sentence-transformers/all-mpnet-base-v2"
modelPath = "NovaSearch/stella_en_1.5B_v5"
#modelPath = "nvidia/NV-Embed-v2"



# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device': 'cuda',
                "trust_remote_code": True,
                #'device_map': device_map,
                }

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

print("Embeddings loading...")


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,  # Provide the pre-trained model's path
    model_kwargs=model_kwargs,  # Pass the model configuration options
    encode_kwargs=encode_kwargs  # Pass the encoding options
)

print("Embeddings loaded")

train_index = False

# FAISS index training/loading
if train_index:

    text_splits = []
    documents_str = {}

    # List of directories
    text_paths = [raw_text_train_val_path]

    for tp in text_paths:
        # Iterate over all input files
        directory = os.fsencode(tp)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            with open(tp + filename, "r", encoding="utf-8") as f:
                text = f.read()
                splits_str = text.split("\n\n")
                splits = [Document(page_content=a) for a in splits_str]
            for split in splits:
                split.metadata = {'source': filename[:-4], 'type': "text"}
            text_splits.extend(splits)

            # Full text
            documents_str[filename[:-4]] = text



    # training images
    for file in os.listdir(os.fsencode(raw_image_train_path)):
        filename = os.fsdecode(file)
        with open(raw_image_train_path + filename, "r", encoding='utf-8') as open_file:
            for line in open_file:
                try:
                    data = json.loads(line)
                    content = data["response"]["body"]["choices"][0]["message"]["content"]
                    metadata = {"figure": data.get("custom_id"),
                                "source": data.get("custom_id").split("-")[0],
                                "type": "figure"}  # Extract metadata
                    doc = Document(page_content=content, metadata=metadata)
                    text_splits.append(doc)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error processing line: {line.strip()}: {e}")

                # Add text to full text
                filename_temp = re.split(r'-(Figure|Table)', data.get("custom_id"))[0]
                documents_str[filename_temp] = documents_str[filename_temp] + "\n\n" + content


    documents = []
    for key, value in documents_str.items():
        documents.append(Document(page_content=value, metadata={
            "source": key,
            "type": "full_text"
        }))

    print(len(text_splits))
    print(len(documents))
    print(documents[0])
    tot = len(text_splits) + len(documents)

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    print("vectorstore setup done")

    # Figures and chunks embedding
    done = 0
    temp = []
    for text in text_splits:
        temp.append(text)
        if len(temp) > 9:
            vectorstore.add_documents(documents=temp)
            done += 10
            print(str(done) + " / " + str(tot))
            temp = []
    # Add leftovers
    vectorstore.add_documents(documents=temp)
    done += 10
    print(str(done) + " / " + str(tot))
    temp = []

    # Full doc embedding
    full_doc_tokenizer = AutoTokenizer.from_pretrained(modelPath, padding=True, truncation=True, token=token)
    for doc in documents:
        chars = len(doc.page_content)
        ts = full_doc_tokenizer.tokenize(doc.page_content)
        # if document is too long
        if len(ts) > 131000:
            # work out splits required and characters per split
            splits = (len(ts) / 100000) + 1
            cps = chars / splits

            full_doc_textSplitter = RecursiveCharacterTextSplitter(
                #separator="\n\n",
                chunk_size=cps,
                # chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )

            # Split text
            ss = full_doc_textSplitter.split_text(doc.page_content)
            ds = [Document(page_content=t, metadata=doc.metadata) for t in ss]

            vectorstore.add_documents(ds)

        else:
            vectorstore.add_documents(documents=[doc])
        done += 1
        if done % 20 == 0:
            print(str(done) + " / " + str(tot))

    vectorstore.save_local("cpiqa_train_val.index")
else:
    print("Loading vectorstore...")
    vectorstore = FAISS.load_local(
        "cpiqa_train_val.index", embeddings, allow_dangerous_deserialization=True
    )
    print("Loaded vectorstore")

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})


def filter_full_text(_, doc):
    return doc['type'] in ['full_text']

def filter_figure_text(a, doc):
    if not doc['type'] in ['figure', 'text']:
        return False
    try:
        sources = a["sources"]
        return doc['source'] in sources
    except:
        return True


template = (
    "You are an assistant for climate research question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "Retrieved information:\n"
    "{context}"
    "\n\n"
    "Question:\n"
    "{question}\n"
    "Answer:\n"
)

prompt = ChatPromptTemplate.from_template(
    template
)


def format_docs(docs):
    print(len("\n\n".join(doc.page_content for doc in docs)))
    return "\n\n".join(doc.page_content for doc in docs)


def get_source_ids(docs):
    l = []
    for d in docs:
        l.append(d.metadata['source'])
    return l

openai = True
if openai:
    from langchain_openai import OpenAI

    from langchain_community.chat_models import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=500,
    )



else:
    # Specify the model name you want to use
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    #model_name = "meta-llama/Llama-3.2-1B"

    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    no_split_modules = ["Qwen2DecoderLayer", "Qwen2MLP"]
    device_map = infer_auto_device_map(model, no_split_module_classes=no_split_modules, dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, torch_dtype=torch.bfloat16, device_map=device_map)

    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, token=token)

    generation_config = model.generation_config
    generation_config.temperature = 0.1
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        #torch_dtype=torch.bfloat16
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs={"temperature": 0.1, "max_length": 512,}# "torch_dtype": torch.bfloat16},

    )

#@backoff(retries=5)
def first_retrieval(query: str):
    initial_docs = retriever.invoke(query, filter=partial(filter_full_text, ""))
    print(len(initial_docs))
    returnv = {
        "initial_context": "\n".join([doc.page_content for doc in initial_docs]),
        "metadata": [doc.metadata for doc in initial_docs]  # Extract metadata for filtering
    }
    return returnv


# Extract metadata and refine filters
def extract_filters(inputs: dict):
    metadata_list = inputs["metadata"]
    sources = list(set(meta.get("source") for meta in metadata_list if meta.get("source")))

    # Construct filter
    filter_dict = {}
    if sources:
        filter_dict["source"] = sources  # Match any of these sources
        filter_dict["type"] = ["text", "figure"]

    try:
        return {"question": inputs["question"], "filters": filter_dict, "concepts": inputs["concepts"]}
    except:
        return {"question": inputs["question"], "filters": filter_dict}


# Second retrieval with filters
@backoff(retries=5)
def second_retrieval(inputs: dict):
    refined_query = inputs["question"]
    filters = inputs["filters"]
    refined_docs = retriever.invoke(refined_query, filter=partial(filter_figure_text, filters))
    return {"question": refined_query,
            "context": "\n".join([doc.page_content for doc in refined_docs]),
            "filters": filters,
            "filters2": [doc.metadata for doc in refined_docs]
            }

def getAnswer(result):
    print("result")
    if openai:
        return result
    #return result.split('Answer:\n')[-1]
    s = re.search(r'(?<=Answer:\n)(.+\n(?!Question:|Answer:))*.*(?=\n|$)', result)
    return s.group(0)

def _dict_to_json(x: dict) -> str:
  return "```\n" + json.dumps(x) + "\n```"



llm_chain = (
    prompt | llm | getAnswer | StrOutputParser()
)

rag_chain = (
        {
            "question": RunnablePassthrough(),
            "initial_results": RunnableLambda(first_retrieval)  # Fetch initial context & metadata
        }
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "initial_context": x["initial_results"]["initial_context"],
            "metadata": x["initial_results"]["metadata"]
        })
        | RunnableLambda(extract_filters)  # Extract relevant filters from metadata
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "filters": x["filters"]
        })
        | RunnableLambda(second_retrieval)
        | RunnableLambda(lambda x: {
            "prompt": prompt.format(**x),
            "output": llm_chain.invoke(x),
            "filters": x["filters"],
            "filters2": x["filters2"]
        })
        | RunnableLambda(lambda x: {
            "prompt": x["prompt"],
            "output": x["output"],
            "filters1": x["filters"],
            "filters2": x["filters2"]
        })
        #| RunnableLambda(_dict_to_json)
        #| output_parser
)

breakdown_template1 = (
    "Given a question, describe in detail 3 contexts or domains in which it can be asked, explain the contexts with a paragraph each. Include titles of academic documents that could be used in the context. "
    "Give the contexts as 3 paragraphs with no headings.\n"
    "Question: {question}\n"
    "Contexts:\n"
)

var_prompt1 = ChatPromptTemplate.from_template(
    breakdown_template1
)

breakdown_template2 = (
    "Given a question and context about the question, decompose the question and context into a set of relevant long-form query sentences for evidence document retrieval (RAG) that can answer the question. Present each sentence on a newline only with no headings. \n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Decomposed phrases:\n"
)

var_prompt2 = ChatPromptTemplate.from_template(
    breakdown_template2
)

def getConcepts(str):
    #print(str)
    if openai:
        concepts = str.content.split("\n\n")
        #concepts = str.split("\n\n")
        return sorted(concepts, key=len)[:3]
    concepts = re.split(r'(?<=\nContexts:\n)', str)[1]
    concepts = concepts.split("\n\n")
    return sorted(concepts, key=len)[:3]

def getPhrases(str):
    if openai:
        str = str.content
    concepts = re.split(r'\n+', str)
    return concepts

breakdown_chain1 = (
    RunnablePassthrough() | var_prompt1 | llm | getConcepts
)

breakdown_chain2 = (
    RunnablePassthrough() | var_prompt2 | llm | getPhrases
)

@backoff(retries=5)
def do_retrieval(query, filters):
    refined_docs = retriever.invoke(query, filter=partial(filter_figure_text, filters))
    return refined_docs

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')

import random
from multiprocessing.dummy import Pool as ThreadPool

def second_retrieval_aug(inputs: dict):
    print("second")
    refined_query = inputs["question"]
    filters = inputs["filters"]

    concepts = inputs["concepts"]
    phrases = []
    for concept in concepts:
        @backoff(retries=5)
        def _f():
            phrases_ = breakdown_chain2.invoke({"question": refined_query, "context": concept})
            phrases.extend(phrases_)
        _f()

    pool = ThreadPool(16)
    results = pool.starmap(do_retrieval, [(query, filters) for query in phrases])
    docs = [
        x
        for xs in results
        for x in xs
    ]

    docs = list({item.page_content: item for item in docs}.values())


    while len(tokenizer.tokenize("\n".join([doc.page_content for doc in docs]))) > 15500: #15500 for qwen, 7500 for gemma
        docs.pop(random.randrange(len(docs)))

    return {"question": refined_query,
            "context": "\n".join([doc.page_content for doc in docs]),
            "filters": filters,
            "filters2": [doc.metadata for doc in docs]
            }

@backoff(retries=5)
def first_retrieval_aug(query: str):
    print("first")
    concepts = breakdown_chain1.invoke({"question": query})

    initial_docs = []

    for concept in concepts:
        initial_docs.extend(retriever.invoke(concept, filter=partial(filter_full_text, "")))

    returnv = {
        "initial_context": "\n".join([doc.page_content for doc in initial_docs]),
        "metadata": [doc.metadata for doc in initial_docs],  # Extract metadata for filtering
        "concepts": concepts
    }
    return returnv

context_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "initial_results": RunnableLambda(first_retrieval_aug)  # Fetch initial context & metadata
        }
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "initial_context": x["initial_results"]["initial_context"],
            "metadata": x["initial_results"]["metadata"],
            "concepts": x["initial_results"]["concepts"]
        })
        | RunnableLambda(extract_filters)  # Extract relevant filters from metadata
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "filters": x["filters"],
            "concepts": x["concepts"]
        })
        | RunnableLambda(second_retrieval_aug) # Second retrieval done here
        | RunnableLambda(lambda x: {
            "prompt": prompt.format(**x),
            "output": llm_chain.invoke(x),
            "filters": x["filters"],
            "filters2": x["filters2"]
        })
        | RunnableLambda(lambda x: {
            "prompt": x["prompt"],
            "output": x["output"],
            "filters1": x["filters"],
            "filters2": x["filters2"]
        })
        #| RunnableLambda(_dict_to_json)
        #| output_parser
)


question = "What are some examples of energy measurements that scientists look at to understand changes in Earth's atmosphere and oceans?"


cpiqa_eval = True

if cpiqa_eval:
    data_json_path = "cpiqa/cpiqa.jsonl"
    test_split_path = "cpiqa/test.txt"
    train_split_path = "cpiqa/train.txt"
    val_split_path = "cpiqa/val.txt"

    qtypes = ["figure", "numerical", "any", "reasoning"]
    promptTypes = ["public", "expert", "sceptic"]

    with open(test_split_path, encoding='utf-8') as f:
        test_ids = [line.rstrip() for line in f if line != ""]

    with open(train_split_path, encoding='utf-8') as f:
        train_ids = [line.rstrip() for line in f if line != ""]

    with open(val_split_path, encoding='utf-8') as f:
        val_ids = [line.rstrip() for line in f if line != ""]

    with open(data_json_path, encoding='utf-8') as f:
        papers = [json.loads(line.rstrip()) for line in f if line != ""]

    # Standard-RAG -------------------------------------------------------------------------
    qa_results = {}
    skip = 0
    train_figs = 0
    test_figs = 0
    val_figs = 0
    fig_path = 'cpiqa/figures/image/'
    for row in papers:

        file_id = list(row.keys())[0]
        v = row[list(row.keys())[0]]
        path = fig_path + file_id.replace('/', '_')
        fig_count = len(os.listdir(path))
        print(fig_count)
        #continue

        # check if in test split
        if file_id not in test_ids:
            continue
        if skip > 0:
            skip -= 1
            continue
        print("Running: " + file_id + "...")
        # get questions section
        questions = row[list(row.keys())[1]]

        qa_results[file_id] = {}
        file_qas = {}

        for promptType in promptTypes:
            qa_results[file_id][promptType] = {}
            file_qas[promptType] = {}
            for qtype in qtypes:
                qa_results[file_id][promptType][qtype] = {}
                file_qas[promptType][qtype] = {}

                question = questions[promptType][qtype]["question"]
                answer = questions[promptType][qtype]["answer"]

                output = rag_chain.invoke(question)
                pred_ans = output['output']

                try:
                    sources1 = output['filters1']
                except:
                    sources1 = []

                try:
                    sources2 = output['filters2']
                except:
                    sources2 = []

                qa_results[file_id][promptType][qtype] = {
                    'question': question,
                    'true_ans': answer,
                    'pred_ans': pred_ans,
                    'sources1': sources1,
                    'sources2': sources2}

                file_qas[promptType][qtype] = {
                    'question': question,
                    'true_ans': answer,
                    'pred_ans': pred_ans,
                    'sources1': sources1,
                    'sources2': sources2}


        with open(f'results/QA/cpiqa/standard_test-qwen.jsonl', 'a') as file:
            json_line = json.dumps(file_qas)
            file.write(json_line + '\n')
        print("Done: " + file_id)
    # Output results
    json.dump(qa_results, open(f'results/QA/cpiqa/standard_test-openai.json', 'w+', encoding='utf-8'))

    # Context-RAG -------------------------------------------------------------------------
    print("CONTEXT-RAG ________________________________________________________________________________________")
    qa_results = {}
    skip = 0
    stop = 250
    i = 0
    for row in papers:
        file_id = list(row.keys())[0]
        v = row[list(row.keys())[0]]

        # check if in test split
        if file_id not in test_ids:
            continue

        if i == stop:
            break
        i += 1

        if skip > 0:
            skip -= 1
            continue

        print("Running: " + file_id + "...")
        print(datetime.datetime.now())
        # get questions section
        questions = row[list(row.keys())[1]]

        qa_results[file_id] = {}
        file_qas = {}

        for promptType in promptTypes:
            qa_results[file_id][promptType] = {}
            file_qas[promptType] = {}
            for qtype in qtypes:
                qa_results[file_id][promptType][qtype] = {}
                file_qas[promptType][qtype] = {}

                question = questions[promptType][qtype]["question"]
                answer = questions[promptType][qtype]["answer"]

                output = context_rag_chain.invoke(question)
                pred_ans = output['output']

                try:
                    sources1 = output['filters1']
                except:
                    sources1 = []

                try:
                    sources2 = output['filters2']
                except:
                    sources2 = []

                file_qas[promptType][qtype] = {
                    'question': question,
                    'true_ans': answer,
                    'pred_ans': pred_ans,
                    'sources1': sources1,
                    'sources2': sources2}

        with open(f'results/QA/cpiqa/crag_test-qwen.jsonl', 'a') as file:
            json_line = json.dumps(file_qas)
            file.write(json_line + '\n')
        print(file_id)

    # Output results
    json.dump(qa_results, open(f'results/QA/cpiqa/crag_test.json', 'w+', encoding='utf-8'))




















