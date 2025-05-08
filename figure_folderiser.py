import os
import re

# Used to put extracted pdf images into the relevant paper's folder

img_path = 'cpiqa/figures/image/'

directory = os.fsencode(img_path)
for file in os.listdir(img_path):
    filename = os.fsdecode(file)


    #paper_path = filename.split('-')[0] + '/'

    paper_path = re.split(r'-(Figure|Table)', filename)[0] + '/'


    if not os.path.exists(img_path + paper_path):
        os.mkdir(img_path + paper_path)

    os.rename(img_path + filename, img_path + paper_path + filename)

