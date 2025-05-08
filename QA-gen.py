import errno
import json
import os
import re
import time

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from filelock import Timeout, FileLock

load_dotenv("secrets.env")

client = OpenAI()

def silent_remove(f):
    try:
        os.remove(f)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


with open('cpiqa/metadata.jsonl', 'r', encoding='utf-8') as json_file:
    paper_list = list(json_file)

figure_list = []

# Get list of figure descriptions
directory = os.fsencode('results/openai/cpiqa/')
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    with open('results/openai/cpiqa/' + filename, 'r', encoding='utf-8') as json_file:
        figure_list1 = list(json_file)
        figure_list2 = [json.loads(a) for a in figure_list1]
        figure_list.extend(figure_list2)

print(len(figure_list))

full_prompt_template = "Context:\n\n{full_text}\n\nInstruction:\n\n{instruction}"

qtype_fig = """The question should be answerable from the figure descriptions only but don't reference the figure or picture."""
qtype_num = """The question should query a useful numerical value without mentioning the document or figure directly."""
qtype_any = """"""
qtype_reason = """The question should require reasoning to answer."""

qtypes = {"figure": qtype_fig, "numerical": qtype_num, "any": qtype_any, "reasoning": qtype_reason}

public_prompt = """As a lay member of the public, generate a single question-answer pair that are answered by the given academic document. {qtype} Use information from the descriptions of figures. Do not reference any part of the document directly. Do not refer to the study or any figure directly. Keep the question simple. Assume the user has never seen the document. Assume the asker knows little about climate science. The question could be written by a child. Answer such that a child will understand. Include a mix of basic factual, analytical and inferential questions. DO NOT MENTION THE CONTEXT DIRECTLY."""
expert_prompt = """As an expert of the topic, and climate science generally, generate one meaningful question and its answer based on the context. {qtype} Use information from the descriptions of figures. Do not reference any part of the document directly. Do not refer to the study directly. The question may be asked with no knowledge of the document content."""
sceptic_prompt = """Generate a single question-answer pair about the context as an extreme climate sceptic. Do not mention that you are a climate sceptic directly. {qtype} Include doubt, previous beliefs. Use information from the descriptions of figures. Do not reference any part of the document directly. Do not refer to the study directly. The question may be asked with no knowledge of the document content. Do not blindly agree with the critic's question. Demonstrate evidence to dispel scepticism. Give examples. Answers should be 1 paragraph or shorter."""

prompts = {"public": public_prompt, "expert": expert_prompt, "sceptic": sceptic_prompt}

skip = 0

with open("cpiqa/cpiqa.jsonl", "r") as file:
    done = [list((json.loads(line)).keys())[0] for line in file]

def task(paper):
    result = json.loads(paper)
    all_figs_data = next(iter(result.values()))['all_figures']

    local_file = list(result.keys())[0]

    if local_file in done:
        return


    figure_descs = []

    for k, v in all_figs_data.items():
        found_fig = [d for d in figure_list if d["custom_id"] == k][0]
        figure_descs.append(found_fig['response']['body']['choices'][0]['message']['content'])


    with open("cpiqa/extracted_paragraphs/" + (
    (next(iter(result.values()))['paper_id'] + ".txt").replace('/', '_')), encoding='utf-8') as f:
        figString = f.read()

    i = 1
    for finfo in figure_descs:
        figString = figString + f"figure {i} description:  \n\n{finfo}\n\n"
        i += 1

    questions = {}

    batch = []
    try:
        # For each prompt
        for target, template in prompts.items():
            for qtype, fill in qtypes.items():
                prompt = template.format(qtype=fill)
                full_prompt = full_prompt_template.format(full_text=figString, instruction=prompt)

                json_line = {"custom_id": local_file + "-" + target + "-" + qtype, "method": "POST", "url": "/v1/chat/completions",
                             "body": {"model": "gpt-4o",
                                      "messages": [{"role": "user",
                                                    "content": [
                                                        {"type": "text",
                                                         "text": f"{full_prompt}"},
                                                    ]}],
                                      "max_tokens": 500}}
                batch.append(json_line)

        batch_path = "batches/cpiqa/QA-gen/" + local_file.replace('/', '_') + ".jsonl"

        silent_remove(batch_path)

        with open(batch_path, "a", encoding='utf-8') as outfile:
            for a in batch:
                outfile.write(json.dumps(a) + "\n")

        # Upload the file
        batch_input_file = client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        )

        # Start the batch
        status = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "qa-gen"
            }
        )

        batch_id = status.id

        while True:
            try:
                batch = client.batches.retrieve(batch_id)
            except:
                time.sleep(5)
                continue

            if batch.status == "completed":
                file_id = batch.output_file_id

                file_response = client.files.content(file_id)
                in_file_name = client.files.retrieve(batch.input_file_id).filename
                with open("results/QA/" + in_file_name[:-6] + ".jsonl", "wb") as out:
                    for chunk in file_response.iter_bytes():  # Iterate through the content in chunks
                        out.write(chunk)


                # Read the file and generate the QA json
                with open("results/QA/" + in_file_name[:-6] + ".jsonl", "r", encoding='utf-8') as br:
                    openai_results = [line.rstrip() for line in br]

                i=0
                for target, template in prompts.items():
                    target_questions = {}
                    for qtype, fill in qtypes.items():
                        openai_question = json.loads(openai_results[i])
                        o = openai_question["response"]["body"]["choices"][0]["message"]["content"]

                        qf = re.match(r'\A(([*]{2})?Question:?([*]{2})? ?)?\n*.+\?', o).group(0)
                        q = re.sub(r'\A([*]{2})?Question:?([*]{2})? ?', '', qf)
                        q = re.sub(r'\A\s*', '', q)
                        a = o.replace(qf, '')
                        a = re.sub(r'\A\s*', '', a)
                        a = re.sub(r'\A([*]{2})?Answer:?([*]{2})? ?', '', a)
                        a = re.sub(r'\A\s*', '', a)

                        target_questions[qtype] = {'question': q, 'answer': a}
                        i += 1
                    questions[target] = target_questions

                result['questions'] = questions

                lock = FileLock("cpiqa/cpiqa.jsonl.lock")
                with lock:
                    with open("cpiqa/cpiqa.jsonl", "a") as outfile:
                        outfile.write(json.dumps(result) + "\n")

                return

            elif batch.status == "failed":
                in_file_name = client.files.retrieve(batch.input_file_id).filename
                print("Failed: " + in_file_name)
                return
            time.sleep(5)


    except:
        print("Error: " + list(result.keys())[0])
        raise Exception





from threading import Thread


threads = []
i=0
for paper in paper_list:
    i += 1
    if skip > 0:
        skip -= 1
        continue

    while len(threads) > 50:
        threads = [t for t in threads if t.is_alive()]

    thread = Thread(target=task, args=(paper,))
    threads.append(thread)
    thread.start()

    if i == 4700:
        break
    break
    continue

