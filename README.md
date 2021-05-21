# Medical-Question-Answering-System



**System architecture**
The system is consisting of 3 main modules: Knowledge Graph, Keyword Extractor, and BERT, BiLSTM, BiGRU semantic similarity model

![Overall System Architecture](https://user-images.githubusercontent.com/39261594/119176328-41044d00-ba6b-11eb-8b44-5a247845b483.png)

**To run the code on your own machine:**

1. Download Neo4j Graph Database using the following link: https://neo4j.com/download/
2. Create a new folder in Neo4j and import the following DB in Neo4j: https://drive.google.com/file/d/1aodIZ6Dl5qCPJZg7W5ki_UQug-gt5iET/view?usp=sharing
3. Download and install requirements.txt -> pip install -r requirements.txt
4. Download the .h5 file: https://drive.google.com/file/d/1gNGI4nmKXp9g38-rrpraIweXaJhkaQLc/view?usp=sharing
5. Download google/bert_uncased_L-12_H-768_A-12: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
6. Clone the main branch and run GUI.py

**Experiments and Results**

1. Exploration of different WordEmbeddings

Word embeddings which are also known as word representations are needed to map words and phrases into a low-dimensional continuous space. The base model that we are comparing through the experiments  uses BERT. The first embedding that was explored was word2vec and the second one was FastText. The results are shown in the table below. 

| Word Embedding  | Evaluation Accuracy |
| ------------- | ------------- |
| word2vec  | 86.31% |
| fast-text | 82.8% |
| BERT | 86.91% |

 2. Exploration on different feature extraction:
 
Feature extraction is the process of reducing the data into more manageable groups for processing. The first method experimented was a basic CNN layer instead of BiLSTM; however, the results were not promising as the accuracy decreased. The second method was BiGRU and it yielded an accuracy very close to the BiLSTM. The last method experimented was the concatenation of both the BiLSTM and BiGRU layers and together they composed the feature extraction layer of the semantic similarity model.  Below are the results:
 
| Accuracy  | Precision | Recall | F1 |
| ------------- | ------------- | ------------- | ------------- |
| BiLSTM  | 86.3% | 86.2% | **88.7%** | 87.4% |
| CNN | 82.8% | 80.9% | 82.9% | 81.9% |
| BiGRU | 86.9% | 87.1% | 87.2% | 87.1% |
| BBBSM | **87.2%** | **87.2%** | 88% | **87.6%** |

The  examples  below  illustrate  the  retrieved answers for sample questions that we used during evaluating our QA system.

Q1: _What are the symptoms of lung cancer?_

A1: _The   symptoms   of   lung   cancer   include:hemoptysis with chest pain stem in bloodshot couwith weight loss sticky or purulent sputum with, slightly, chest  pain  fever  with  cough  planting spread lung nodules._

Q2: _What is coronavirus?_

A2: _Coronavirus are a type of virus - there aremany kinds, and some of them can cause disease. Examples of coronaviruses include the commoncold,  flu,  to  more  severe  diseases  like  MiddleEast  Respiratory  Syndrome  (MERS-CoV)  and Severe Acute Respiratory Syndrome (SARS-CoV)._




