# Effect of Tokenization on Transformers for Biological Sequences
## Abstract:
Deep-learning models are transforming biological research, including many bioinformatics and comparative genomics algorithms, such as sequence alignments, phylogenetic tree inference, and automatic classification of protein functions. Among these deep-learning algorithms, models for processing natural languages, developed in the natural language processing (NLP) community, were recently applied to biological sequences.  However, biological sequences are different from natural languages, such as English, and French, in which segmentation of the text to separate words is relatively straightforward. Moreover, biological sequences are characterized by extremely long sentences, which hamper their processing by current machine-learning models, notably the transformer architecture. In NLP, one of the first processing steps is to transform the raw text to a list of tokens. Deep-learning applications to biological sequence data mostly segment proteins and DNA to single characters. In this work, we study the effect of alternative tokenization algorithms on eight different tasks in biology, from predicting the function of proteins and their stability, through nucleotide sequence alignment, to classifying proteins to specific families. We demonstrate that applying alternative tokenization algorithms can increase accuracy and at the same time, substantially reduce the input length compared to the trivial tokenizer in which each character is a token. Furthermore, applying these tokenization algorithms allows interpreting trained models, taking into account dependencies among positions. Finally, we trained these tokenizers on a large dataset of protein sequences containing more than 400 billion amino acids, which resulted in over a three-fold decrease in the number of tokens. We then tested these tokenizers trained on large-scale data on the above specific tasks and showed that for some tasks it is highly beneficial to train database-specific tokenizers. Our study suggests that tokenizers are likely to be a critical component in future deep-network analysis of biological sequence data. 

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

## Training Script:

You can use the provided script `train_tokenizer_bert.py` to perform the training and evaluation.

## Usage Flags

The script supports various flags for customization:

+ --tokenizer-type (-t): Choose the type of tokenizer to use. Options include:

  + "BPE" (Byte Pair Encoding, (Sennrich, Haddow, and Birch 2016))
  + "WPC" (WordPiece, (Schuster and Nakajima 2012))
  + "UNI" (Unigram, (Kudo 2018))
  + "WORDS" (each token is a single character)
  + "PAIRS" (each token is a pair of two characters)

+ --vocab-size (-s): Set the vocabulary size for the tokenizer. (Used only when tokenizer type is "BPE", "WPC", or "UNI").

+ --results-path (-r): Specify the path to save the tokenizer, transformer, and results.

+ --layers-num (-l): Define the number of BERT layers.

+ --attention-heads-num (-a): Set the number of BERT attention heads.

+ --hidden-size (-z): Specify the hidden size of BERT layer.

+ --data-path (-d): Provide the path to the folder containing three files: train.csv, valid.csv, and test.csv. For the datasets used in our paper, you may download them from the "data" folder.

+ --epochs (-e): Define the number of training epochs.

+ --print-training-loss (-p): Specify the number of steps to print the loss.

+ --task-type (-y): Choose the task type:

  + "REGRESSION" (for regression datasets, i.e., predicting a score)
  + "CLASSIFICATION" (for classification datasets, i.e., predicting a class).

+ --max-length (-m): Set the maximum tokens per sequence.

+ --learning-rate (-lr): Set the learning rate for the model training.

## Example Usage:

```
# running the SuperFamily classification training with a "BPE" tokenizer of 3,000 tokens
python train_tokenizer_bert.py --tokenizer-type BPE --vocab-size 3000 --results-path ./results_SuperFamily --layers-num 6 --attention-heads-num 8 --hidden-size 256 --data-path ./data/SuperFamily/ --epochs 10 --print-training-loss 1000 --task-type CLASSIFICATION --max-length 128

# running the fluorescence prediction training with a "PAIRS" tokenizer
python train_tokenizer_bert.py --tokenizer-type PAIRS --results-path ./results_fluorescence --layers-num 2 --attention-heads-num 4 --hidden-size 128 --data-path ./data/fluorescence/ --epochs 30 --print-training-loss 100 --task-type REGRESSION --max-length 256 --learning-rate 0.001

# running the stability prediction training with a "WPC" tokenizer of 200 tokens
python train_tokenizer_bert.py --tokenizer-type WPC --vocab-size 200 --results-path ./results_stability --layers-num 6 --attention-heads-num 4 --hidden-size 128 --data-path ./data/stability/ --epochs 15 --print-training-loss 1000 --task-type REGRESSION --max-length 512 --learning-rate 0.000001
```

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

