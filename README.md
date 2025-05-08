[![CC BY 4.0][cc-by-shield]][cc-by]

# CPIQA: Climate Paper Image Question Answering Dataset for Retrieval-Augmented Generation with Context-based Query Expansion

**Paper**: [Link](#)

**Dataset (huggingface)**: [Dataset](https://huggingface.co/datasets/RudraMutalik/CPIQA)

**Dataset (Zenodo)**: [Dataset](#)

This repository contains all of the code required to reproduce and utilise the work from the published workshop paper. 

**Abstract:** Misinformation about climate science is a serious challenge for our society. This paper introduces CPIQA (Climate Paper Image Question-Answering), a new question-answer dataset featuring 4,551 full-text open-source academic papers in the area of climate science with 54,612 GPT-4o generated question-answer pairs. CPIQA contains four question types (numeric, figure-based, non-figure-based, reasoning), each generated using three user roles (expert, non-expert, climate sceptic). CPIQA is multimodal, incorporating information from figures and graphs with GPT-4o descriptive annotations. We describe Context-RAG, a novel method for RAG prompt decomposition and augmentation involving extracting distinct contexts for the question. Evaluation results for Context-RAG on the benchmark SPIQA dataset outperforms the previous best state of the art model in two out of three test cases. For our CPIQA dataset, Context-RAG outperforms our standard RAG baseline on all five base LLMs we tested, showing our novel contextual decomposition method can generalize to any LLM architecture. Expert evaluation of our best performing model (GPT-4o with Context-RAG) by climate science experts highlights strengths in precision and provenance tracking, particularly for figure-based and reasoning questions.

**Dataset Summary:** CPIQA is a large scale QA dataset focused on figured extracted from scientific research papers from various peer reviewed venues in the climate science domain. The figures extracted include tables, graphs and diagrams, which inform the generation of questions using large language models (LLMs). Notably this dataset includes questions for 3 audiences: general public, climate skeptic and climate expert. 4 types of questions are generated with various focusses including figures, numerical, text-only and general. This results in 12 questions generated per scientific paper. Alongside figures, descriptions of the figures generated using multimodal LLMs are included and used.

## Citation Information

```bibtex
@inproceedings{
  mutalik2025cpiqa,
  title={{CPIQA}: Climate Paper Image Question Answering Dataset for Retrieval-Augmented Generation with Context-based Query Expansion},
  author={Rudra Mutalik and Stuart E. Middleton and Abiram Panchalingam and Loitongbam Gyanendro Singh and Timothy J. Osborn and Ed Hawkins},
  booktitle={The 2nd Workshop of Natural Language Processing meets Climate Change},
  year={2025},
  url={https://openreview.net/forum?id=XRwgMjwa2m}
}
```

## Licence

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
