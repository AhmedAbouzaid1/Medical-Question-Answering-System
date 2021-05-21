# Medical-Question-Answering-System



System architecture
![Overall System Architecture](https://user-images.githubusercontent.com/39261594/119176328-41044d00-ba6b-11eb-8b44-5a247845b483.png)

**To run the code on your own machine:**

1. Download Neo4j Graph Database using the following link: https://neo4j.com/download/
2. Create a new folder in Neo4j and import the following DB in Neo4j: https://drive.google.com/file/d/1aodIZ6Dl5qCPJZg7W5ki_UQug-gt5iET/view?usp=sharing
3. Download the .h5 file: https://drive.google.com/file/d/1gNGI4nmKXp9g38-rrpraIweXaJhkaQLc/view?usp=sharing
4. Download google/bert_uncased_L-12_H-768_A-12: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
5. Clone the main branch and run GUI.py

**Experiments and Results**

1. Exploration of different WordEmbeddings

| Word Embedding  | Evaluation Accuracy |
| ------------- | ------------- |
| word2vec  | 86.31% |
| fast-text | 82.8% |
| BERT | 86.91% |

 2. Exploration on different feature extraction:
| Accuracy  | Precision | Recall | F1 |
| ------------- | ------------- | ------------- | ------------- |
| BiLSTM  | 86.3% | 86.2% | **88.7%** | 87.4% |
| CNN | 82.8% | 80.9% | 82.9% | 81.9% |
| BiGRU | 86.9% | 87.1% | 87.2% | 87.1% |
| BBBSM | **87.2%** | **87.2%** | 88% | **87.6%** |






