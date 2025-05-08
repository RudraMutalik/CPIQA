import time
from threading import Thread

from dotenv import load_dotenv
import os
load_dotenv("secrets.env")

from openai import OpenAI
client = OpenAI()

def process_batch(path):
    # Upload file
    batch_input_file = client.files.create(
        file=open(path, "rb"),
        purpose="batch"
    )


    status = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "image-labelling"
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
            with open("results/openai/cpiqa/" + in_file_name[:-6] + ".jsonl", "wb") as out:
                for chunk in file_response.iter_bytes():  # Iterate through the content in chunks
                    out.write(chunk)
            break
        elif batch.status == "failed":
            in_file_name = client.files.retrieve(batch.input_file_id).filename
            print("Failed: " + in_file_name)
            break
        time.sleep(5)

threads = []

file_i = 0
while True:

    while len(threads) > 30:
        threads = [t for t in threads if t.is_alive()]

    batch_path = "batches/cpiqa/figure_labelling/cpiqa_image" + str(file_i) + ".jsonl"
    if os.path.isfile(batch_path):
        print("creating")
        thread = Thread(target=process_batch, args=(batch_path,))
        threads.append(thread)
        thread.start()
    else:
        break
    file_i += 1