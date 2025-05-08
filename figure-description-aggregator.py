import json
import os

import pandas as pd

figure_list = []

directory = os.fsencode('results/openai/cpiqa/')
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    with open('results/openai/cpiqa/' + filename, 'r', encoding='utf-8') as json_file:
        figure_list1 = list(json_file)
        figure_list2 = [json.loads(a) for a in figure_list1]
        figure_list.extend(figure_list2)

accepted_figures = []
with open('cpiqa/cpiqa.jsonl', 'r', encoding='utf-8') as json_file:
    entries = [json.loads(a) for a in list(json_file)]
for entry in entries:
    accepted_figures.extend(list(entry[list(entry.keys())[0]]["all_figures"].keys()))

print(len(accepted_figures))



rows = []

for i in range(len(figure_list)):
    if figure_list[i]["custom_id"] not in accepted_figures:
        continue
    rows.append({'filename': figure_list[i]["custom_id"],
                 'figure_desc': figure_list[i]['response']['body']['choices'][0]['message']['content']})

df = pd.DataFrame(rows)
df.to_csv("cpiqa/figure_desc.csv", encoding='utf-8')
print(df)





