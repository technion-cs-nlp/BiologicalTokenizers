# Effect of Tokenization on Transformers for Biological Sequences
## Abstract:
Deep learning models are transforming biological research. Many bioinformatics and comparative genomics algorithms analyze genomic data, either DNA or protein sequences. Examples include sequence alignments, phylogenetic tree inference and automatic classification of protein functions. Among these deep learning algorithms, models for processing natural languages, developed in the natural language processing (NLP) community, were recently applied to biological sequences.  However, biological sequences are different than natural languages, such as English, and French, in which segmentation of the text to separate words is relatively straightforward. Moreover, biological sequences are characterized by extremely long sentences, which hamper their processing by current machine-learning models, notably the transformer architecture. In NLP, one of the first processing steps is to transform the raw text to a list of tokens. Deep-learning applications to biological sequence data mostly segment proteins and DNA to single characters. In this work, we study the effect of alternative tokenization algorithms on eight different tasks in biology, from predicting the function of proteins and their stability, through nucleotide sequence alignment, to classifying proteins to specific families. We demonstrate that applying alternative tokenization algorithms can increase accuracy and at the same time, substantially reduce the input length compared to the trivial tokenizer in which each character is a token. Furthermore, applying these tokenization algorithms allows interpreting trained models, taking into account dependencies among positions. Finally, we trained these tokenizers on a large dataset of protein sequences containing more than 400 billion amino acids, which resulted in over a three-fold decrease in the number of tokens. We then tested these tokenizers trained on large-scale data on the above specific tasks and showed that for some tasks it is highly beneficial to train database-specific tokenizers. Our study suggests that tokenizers are likely to be a critical component in future deep-network analysis of biological sequence data. 

![image](https://github.com/idotan286/BiologicalTokenizers/assets/58917533/d69893e2-7114-41a8-8d46-9b025b2d2840)

Different tokenization algorithms can be applied to biological sequences, as exemplified for the sequence “AAGTCAAGGATC”. (a) The baseline “words” tokenizer assumes a dictionary consisting of the nucleotides: “A”, “C”, “G” and “T”. The length of the encoded sequence is 12, i.e., the number of nucleotides; (b) The “pairs” tokenizer assumes a dictionary consisting of all possible nucleotide pairs. The length of the encoded sequences is typically halved; (c) A sophisticated dictionary consisting of only three tokens: “AAG”, “TC” and “GA”. The encoded sequence for this dictionary contains only five tokens.

## Data:
The "data" folder contains the train, valid and test data of seven of the eight datasets used in the paper.

## BFD Tokenizers:

We trained BPE, WordPiece and Unigram tokenizers on samples of proteins from the 2.2 billion protein sequences of the BFD dataset (Steinegger and Söding 2018). We evaluate the average sequences length as a function of the vocabulary size and number of sequences in the training data.

![BFD_BPE_table](https://github.com/idotan286/BiologicalTokenizers/assets/58917533/710b7aa7-0dde-46bb-9ddf-39a84b579d71)
![BFD_WPC_table](https://github.com/idotan286/BiologicalTokenizers/assets/58917533/8adfe5a7-25f5-4723-a87a-8598c6a76ff6)
![BFD_UNI_table](https://github.com/idotan286/BiologicalTokenizers/assets/58917533/4462e782-0b21-4377-a5fe-309685141538)

Effect of vocabulary size and number of training samples on the three tokenizers: BPE, WordPiece and Unigram. The darker the color the higher the average number of tokens per protein. Increasing the vocabulary and the training size reduces the number of tokens per protein for all of the tested tokenizers. 

The "BFD_Tokenizers" contains the trained tokenizers on the BFD datasset. The path to the tokenizers is as follows: "/BFD_Tokenizers/\<NUMBER OF TRAINING SAMPLES\>/\<TOKENIZER TYPE\>/\<VOCABULARY SIZE\>"

## APA

```
Dotan, E., Jaschek, G., Pupko, T., & Belinkov, Y. (2023). Effect of Tokenization on Transformers for Biological Sequences. bioRxiv. https://doi.org/10.1101/2023.08.15.553415
```


## BibTeX
```
@article{Dotan_Effect_of_Tokenization_2023,
  author = {Dotan, Edo and Jaschek, Gal and Pupko, Tal and Belinkov, Yonatan},
  doi = {10.1101/2023.08.15.553415},
  journal = {bioRxiv},
  month = aug,
  title = {{Effect of Tokenization on Transformers for Biological Sequences}},
  year = {2023}
}

```
