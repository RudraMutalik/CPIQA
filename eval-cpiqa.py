import json
import re
import sys
from bert_score import BERTScorer
from numpy import mean


def find_repeated(str):
    str_len = len(str)
    #print("in: " + str)
    try:
        start = str[0]
    except:
        return str
    p = 1

    while p < str_len:
        if str[p] == start:
            try:
                if str[p:p + 20] == str[0:20]:
                    return str[:p]
            except:
                continue
        p += 1
    return str


def remove_repetition(text: str, min_repeat_length: int = 3) -> str:
    """
    Removes repetitive phrases in the text that occur in loops.
    :param text: The input text from the LLM.
    :param min_repeat_length: Minimum length of repeating sequences to be removed.
    :return: The cleaned-up text without excessive repetition.
    """
    # Look for repeated words or phrases using regex
    pattern = re.compile(r'\b(\w{' + str(min_repeat_length) + r',})\b(?:\s+\1\b)+', re.IGNORECASE)

    # Replace repetitive phrases with a single instance
    def deduplicate(match):
        return match.group(1)

    cleaned_text = pattern.sub(deduplicate, text)

    return cleaned_text


def remove_word_repetition(string):
    # Split the string by spaces (or use any other delimiter) to check the repetition
    words = string.split(' ')  # This is assuming each part is space-separated

    final = [words[0]]
    for i in range(1, len(words)):
        max_len = len(final)
        for j in range(0, max_len):
            if final[-j - 1:] == words[i:i + j + 1]:
                return " ".join(final)

        final.append(words[i])

    return " ".join(final)

def remove_think(string):
    return string.split("</think>")[-1]


#scorer = BERTScorer(model_type='bert-base-uncased')
scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
#scorer = BERTScorer(model_type='allenai/scibert_scivocab_uncased')
#bertscore = load("bertscore")

prompts = ['public', 'expert', 'sceptic']
qtypes = ["figure", "numerical", "any", "reasoning"]

results = []
sources = []

data_json_path = "cpiqa/cpiqa.jsonl"

with open(data_json_path, encoding='utf-8') as f:
    papers = [json.loads(line.rstrip()) for line in f if line != ""]


# Open results file
#with open('results/QA/cpiqa/standard_test-openai.jsonl', 'r', encoding='utf-8') as rfile:
with open('results/QA/cpiqa/standard_test-openai.jsonl', 'r', encoding='utf-8') as rfile:
    for line in rfile:
        a = json.loads(line)
        results.append(a)
        for paper in papers:
            if paper[list(paper.keys())[1]]['public']['figure']["answer"] == a['public']['figure']["true_ans"]:
                sources.append((list(paper.keys())[0], paper[list(paper.keys())[1]], paper[list(paper.keys())[0]]))
                break

eva = True
if eva:
    pred = []
    true = []
    ss = []

    pred_breakdown = {}
    true_breakdown = {}
    scores = {}

    correct_source_f1s = []
    incorrect_source_f1s = []


    for prompt in prompts:
        pred_breakdown[prompt] = {}
        true_breakdown[prompt] = {}
        scores[prompt] = {}
        for qtype in qtypes:
            pred_breakdown[prompt][qtype] = []
            true_breakdown[prompt][qtype] = []
            scores[prompt][qtype] = []

    i=0
    for result in results:
        #if i == 9:
        #    break
        for prompt in prompts:
            for qtype in qtypes:
                q = result[prompt][qtype]["question"]

                if re.fullmatch(r'\s+|\s*[\w*:]+\s*', q):
                    continue


                p = result[prompt][qtype]["pred_ans"]
                t = result[prompt][qtype]["true_ans"]

                p = remove_think(p)
                p = remove_word_repetition(p)

                try:
                    _, _, F1 = scorer.score([p], [t])
                except:
                    F1 = 0

                #print(sources[i][0].replace('/', '_'))
                #print(result[prompt][qtype]["sources1"])
                #print(result[prompt][qtype]["sources2"])
                #print("------")

                if result[prompt][qtype]["sources1"] == {}:
                    pass
                elif sources[i][0].replace('/', '_') in result[prompt][qtype]["sources1"]["source"]:
                    correct_source_f1s.append(F1)
                else:
                    incorrect_source_f1s.append(F1)

                if mean(F1.tolist()) < 0.1:
                    print(t)
                    print(p)
                    print("----------------")

                true.append(t)
                pred.append(p)
                ss.append(mean(F1.tolist()))

                scores[prompt][qtype].append(mean(F1.tolist()))
                pred_breakdown[prompt][qtype].append(p)
                true_breakdown[prompt][qtype].append(t)
        i += 1

    print(str(len(correct_source_f1s)) + " : " + str(mean(correct_source_f1s)))
    print(str(len(incorrect_source_f1s)) + " : " + str(mean(incorrect_source_f1s)))


    for prompt in prompts:
        for qtype in qtypes:
            #_, _, F1 = scorer.score(pred_breakdown[prompt][qtype], true_breakdown[prompt][qtype])
            F1 = scores[prompt][qtype]
            if qtype == "any":
                qtype = "general"
            print(f"{prompt}:{qtype} ::: Macro f1: " + str(mean(F1)))

    #_, _, F1 = scorer.score(pred, true)
    F1 = mean(ss)
    print("Overall Macro f1: " + str(mean(F1.tolist())))

sys.exit()

test_split_path = "cpiqa/test.txt"
with open(test_split_path, encoding='utf-8') as f:
    test_ids = [line.rstrip() for line in f if line != ""]

b = list(set(test_ids) - set([a[0] for a in sources]))
print(b)
for i in range(23,41):
    questions_out = []
    answers_out = []

    for prompt in prompts:
        for qtype in qtypes:
            questions_out.append(results[i][prompt][qtype]["question"])
            answers_out.append(remove_think(results[i][prompt][qtype]["pred_ans"]))

    print(sources[i][0].replace('/', '_') + ".pdf")
    for question in questions_out:
        print(question)
    print("-------------------")
    for answers in answers_out:
        print(answers)
    input()
    print("########################\n########################\n")
    print("########################\n########################\n")



#qwen
#0.6478490735927638
#0.6390882063947038

#openai ---------------------------------------------------------------------
#1048 : 0.6919465
#1083 : 0.65903455
#public:figure ::: Macro f1: 0.6689177934004336
#public:numerical ::: Macro f1: 0.7456417369599245
#public:general ::: Macro f1: 0.6365171382351528
#public:reasoning ::: Macro f1: 0.6227477853575711
#expert:figure ::: Macro f1: 0.7112125223729668
#expert:numerical ::: Macro f1: 0.8053549995738498
#expert:general ::: Macro f1: 0.6909260934863994
#expert:reasoning ::: Macro f1: 0.6578448077694314
#sceptic:figure ::: Macro f1: 0.6396598279155711
#sceptic:numerical ::: Macro f1: 0.6465230998346361
#sceptic:general ::: Macro f1: 0.6385647362686809
#sceptic:reasoning ::: Macro f1: 0.6411267834740716
#Overall Macro f1: 0.6752202613624482

#706 : 0.7024565
#1139 : 0.6673573
#public:figure ::: Macro f1: 0.6701628558734465
#public:numerical ::: Macro f1: 0.7613426845454605
#public:general ::: Macro f1: 0.6382212865572034
#public:reasoning ::: Macro f1: 0.6211918342260667
#expert:figure ::: Macro f1: 0.719440077265648
#expert:numerical ::: Macro f1: 0.8195626872360326
#Warning: Empty reference sentence detected; setting raw BERTScores to 0.
#expert:general ::: Macro f1: 0.7013241869933677
#Warning: Empty reference sentence detected; setting raw BERTScores to 0.
#expert:reasoning ::: Macro f1: 0.669212770239215
#sceptic:figure ::: Macro f1: 0.6441616502978047
#sceptic:numerical ::: Macro f1: 0.6444635286050684
#sceptic:general ::: Macro f1: 0.6400756911340967
#sceptic:reasoning ::: Macro f1: 0.6438189232721925
#Warning: Empty reference sentence detected; setting raw BERTScores to 0.
#Warning: Empty reference sentence detected; setting raw BERTScores to 0.
#Overall Macro f1: 0.6807882237886672

#gemini
#0.638507843115474
#0.655240116417408

#gemma:
#0.6336960821350882
#0.5963117847247954

#llama







