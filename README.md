[![CC BY 4.0][cc-by-shield]][cc-by]

# CPIQA: Climate Paper Image Question Answering Dataset for Retrieval-Augmented Generation with Context-based Query Expansion

**Paper**: [CPIQA: Climate Paper Image Question Answering Dataset for Retrieval-Augmented Generation with Context-based Query Expansion](https://aclanthology.org/2025.climatenlp-1.15/)

**Authors**: [Rudra Mutalik](https://rudramutalik.com/), Abiram Panchalingam, [Loitongbam Gyanendro Singh](https://sites.google.com/view/gyanendro), [Timothy J. Osborn](https://crudata.uea.ac.uk/~timo/), [Ed Hawkins](https://edhawkins.org/), and [Stuart E. Middleton](https://www.southampton.ac.uk/~sem03/)

**Dataset (huggingface)**: [Dataset](https://huggingface.co/datasets/RudraMutalik/CPIQA)

**Dataset (Zenodo)**: [Dataset](#)

This repository contains all of the code required to reproduce and utilise the work from the published workshop paper. 

**Abstract:** Misinformation about climate science is a serious challenge for our society. This paper introduces CPIQA (Climate Paper Image Question-Answering), a new question-answer dataset featuring 4,551 full-text open-source academic papers in the area of climate science with 54,612 GPT-4o generated question-answer pairs. CPIQA contains four question types (numeric, figure-based, non-figure-based, reasoning), each generated using three user roles (expert, non-expert, climate sceptic). CPIQA is multimodal, incorporating information from figures and graphs with GPT-4o descriptive annotations. We describe Context-RAG, a novel method for RAG prompt decomposition and augmentation involving extracting distinct contexts for the question. Evaluation results for Context-RAG on the benchmark SPIQA dataset outperforms the previous best state of the art model in two out of three test cases. For our CPIQA dataset, Context-RAG outperforms our standard RAG baseline on all five base LLMs we tested, showing our novel contextual decomposition method can generalize to any LLM architecture. Expert evaluation of our best performing model (GPT-4o with Context-RAG) by climate science experts highlights strengths in precision and provenance tracking, particularly for figure-based and reasoning questions.

**Dataset Summary:** CPIQA is a large scale QA dataset focused on figured extracted from scientific research papers from various peer reviewed venues in the climate science domain. The figures extracted include tables, graphs and diagrams, which inform the generation of questions using large language models (LLMs). Notably this dataset includes questions for 3 audiences: general public, climate skeptic and climate expert. 4 types of questions are generated with various focusses including figures, numerical, text-only and general. This results in 12 questions generated per scientific paper. Alongside figures, descriptions of the figures generated using multimodal LLMs are included and used.

## Citation Information

```bibtex
@inproceedings{mutalik-etal-2025-cpiqa,
    title = "{CPIQA}: Climate Paper Image Question Answering Dataset for Retrieval-Augmented Generation with Context-based Query Expansion",
    author = "Mutalik, Rudra  and
      Panchalingam, Abiram  and
      Singh, Loitongbam Gyanendro  and
      Osborn, Timothy J.  and
      Hawkins, Ed  and
      Middleton, Stuart E.",
    editor = "Dutia, Kalyan  and
      Henderson, Peter  and
      Leippold, Markus  and
      Manning, Christoper  and
      Morio, Gaku  and
      Muccione, Veruska  and
      Ni, Jingwei  and
      Schimanski, Tobias  and
      Stammbach, Dominik  and
      Singh, Alok  and
      Su, Alba (Ruiran)  and
      A. Vaghefi, Saeid",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing Meets Climate Change (ClimateNLP 2025)",
    month = jul,
    year = "2025",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.climatenlp-1.15/",
    pages = "218--232",
    ISBN = "979-8-89176-259-6",
    abstract = "Misinformation about climate science is a serious challenge for our society. This paper introduces CPIQA (Climate Paper Image Question-Answering), a new question-answer dataset featuring 4,551 full-text open-source academic papers in the area of climate science with 54,612 GPT-4o generated question-answer pairs. CPIQA contains four question types (numeric, figure-based, non-figure-based, reasoning), each generated using three user roles (expert, non-expert, climate sceptic). CPIQA is multimodal, incorporating information from figures and graphs with GPT-4o descriptive annotations. We describe Context-RAG, a novel method for RAG prompt decomposition and augmentation involving extracting distinct contexts for the question. Evaluation results for Context-RAG on the benchmark SPIQA dataset outperforms the previous best state of the art model in two out of three test cases. For our CPIQA dataset, Context-RAG outperforms our standard RAG baseline on all five base LLMs we tested, showing our novel contextual decomposition method can generalize to any LLM architecture. Expert evaluation of our best performing model (GPT-4o with Context-RAG) by climate science experts highlights strengths in precision and provenance tracking, particularly for figure-based and reasoning questions."
}
```

## Licence

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
