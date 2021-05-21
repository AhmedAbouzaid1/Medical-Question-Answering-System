# Medical-Question-Answering-System



# System Architecture

The system consists of 3 main modules: Knowledge Graph, Keyword Extractor, and BERT, BiLSTM, BiGRU semantic similarity model.

![Overall System Architecture](https://user-images.githubusercontent.com/39261594/119176328-41044d00-ba6b-11eb-8b44-5a247845b483.png)

#Knowledge Graph

Neo4j-based graph database is further modified to store the medical information. Cypher language and index adjacency are used in target access of data queries, which increases the query speed and eases the subsequent retrievals.

**Answer Extraction
**
1.	Input question is provided.
2.	Key entities are from the input question.
3.	Question intentions are derived.  
4.	Disease & Symptom Extractor is used to extract entities and the intentions.
5.	 Cypher Language is built to query from the knowledge graph by integrating the D&S entities and the user intention extracted from the previous stage.
6.	Returned answers are cleaned and provided to user. 

<p align="center">
<img src="https://user-images.githubusercontent.com/39261594/119187647-ea523f80-ba79-11eb-90ef-316e53b5330c.png" width="300" height="550">
</p>

**Knowledge Graph Question Classification
**

| Question Type  | Question Example |
| ------------- | ------------- |
| Disease_symptom  | Symptoms of lung cancer? |
| Symptom_disease | What causes fever? |
| Disease_Cause | Lung cancer causes? |
| Disease_prevent | How to prevent cold? |
| Disease_cureaway | Medications for cold? |
| Disease_lasttime | Sars Lifetime |



#Keyword Extraction

The keyword extraction section is placed over the similarity model as a preprocessing layer to filters the dataset questions in order to extract the most relevant ones based on the user question, and an optimization layer that significantly speeds up the question answering process. The extraction layer consists of two major sections: a populated SQL database that contains the dataset questions id with the keywords and the synonyms of each question, and an extraction algorithm that retrieves the most relevant questions. As shown in the figure below, the extraction flow starts with receiving a question that is being processed using NLTK to extract the questions' keywords and their corresponding synonyms. A SQL selection query is then used to select all questions that match the extracted keywords. Accordingly, the questions that have the most keywords occurrences are extracted to be passed to the similarity model.

![Keyword_extraction (1)](https://user-images.githubusercontent.com/39261594/119188040-69e00e80-ba7a-11eb-82c3-249b86d544a6.png)


#BERT, BiLSTM, BiGRU Semantic Similarity Model

Our Similarity Model is divided into different layers as mentioned earlier. Our similarity model takes two sentences as an input. Then for each sentence we use BERT as word embedding. Then each word embedding vector get send once to the BiLSTM layer and another time to the BiGRU layer. Then the output of each feature extraction layer gets send to a max pooling and average pooling layers. We concatenate the 2 feature extraction outputs of the max pooling and 2 feature extraction outputs of the average pooling then the output of the concatenation layer gets send to a dense layer as a final step.    

![simModel](https://user-images.githubusercontent.com/39261594/119187728-048c1d80-ba7a-11eb-9427-cca9b24ccf5e.png)

#Environment setup

1. Download Neo4j Graph Database using the following link: https://neo4j.com/download/
2. Create a new folder in Neo4j and import the following DB in Neo4j: https://drive.google.com/file/d/1aodIZ6Dl5qCPJZg7W5ki_UQug-gt5iET/view?usp=sharing
3. Download and install requirements.txt -> pip install -r requirements.txt
4. Download the .h5 file: https://drive.google.com/file/d/1gNGI4nmKXp9g38-rrpraIweXaJhkaQLc/view?usp=sharing
5. Download google/bert_uncased_L-12_H-768_A-12: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
6. Clone the main branch and run GUI.py

#Experiments and Results

**1. Exploration of different WordEmbeddings
**

Word embeddings which are also known as word representations are needed to map words and phrases into a low-dimensional continuous space. The base model that we are comparing through the experiments  uses BERT. The first embedding that was explored was word2vec and the second one was FastText. The results are shown in the table below. 

| Word Embedding  | Evaluation Accuracy |
| ------------- | ------------- |
| word2vec  | 86.31% |
| fast-text | 82.8% |
| BERT | 86.91% |

** 2. Exploration on different feature extraction:
**

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

#System Accuracy 

| Overall System  |  Accuracy |
| ------------- | ------------- |
| KG+BBBSM | 86.48% |
