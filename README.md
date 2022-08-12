# A tool for identifying and visualizing key topics 
An unsupervised, semantic-similarity based topic analyzing tool detects and visualizes the topics in a collection of publication abstracts.

By utilizing the pre-trained BERT model, the abstracts are transformed into context vector representation (or embedding). 
The HDBSCAN clustering algorithm and UMAP dimension reduction technique are used to find the clusters of abstracts based on semantic similarity.
Keywords are extracted from each abstract and diversified with MMR ranking algorithm.
Topics are labeled with groups of keywords in each cluster.
The clustering results are evaluated using Silhouette Coefficient.
A case study was conducted to detect key research topics from the abstracts at the intersection of Urban Study and Machine Learning fields.
A web-based interactive visualization is presented to help examine and investigate the topics.

Our system operate in an unsupervised manner and does not need to be trained, neither it depends on dictionaries, external-corpus, size of the text or domain.


# Usage
To run our visualisation, use Google Chrome and open the below files. 
- Abstract cluster visualization: `frontend\abstract_cluster.html`
- Keyword group visualization: `frontend\keyword_group.html`



## Folder Structures
This repository contains the source code that is published for the purpose of giving the details of our paper. 

**Frontend**
1. `abstract_cluster.html`: abstract cluster visualisation
2. `keyword_group.html`: keyword group visualisation

**Backend**
1. `BERTArticleCluster.py`: convert abstracts to context vectors and find the clusters based on cosine similarities.
2. `ArticleClusterTermTFIDF.py`: pick up distinct TFIDF terms of each abstract cluster
3. `KeywordExtraction.py`: extract keywords from each abstract with MMR ranking algorith.
4. `KeywordGroup.py`: group keywords based on semantic similarity to represent the topic
5. `TermKeywordGroup.py`: combine keyword groups and abstract clusters together
6. `Evaluation.py`: a script runs the evaluations (such topic coherence score)

**Required Libraries**
1. `Sentence Transformer`: https://www.sbert.net/index.html
2. `UMAP`: https://umap-learn.readthedocs.io/en/latest/
3. `HDBSCAN`: https://hdbscan.readthedocs.io/en/latest/index.html
4. `scikit-learn`: https://scikit-learn.org/stable/
5. `nltk`: https://www.nltk.org/
6. `plotly`: https://plotly.com/python/
7. `stanza`: https://stanfordnlp.github.io/stanza/

