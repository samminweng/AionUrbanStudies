# A tool for identifying and visualizing key topics 
This paper describes a topic analyzing tool to detect and visualize the topics in a collection of publications.
By utilizing the pre-trained BERT model, the abstracts are transformed into contextualized vector representation (or embedding). 
The HDBSCAN clustering algorithm and keyword extraction method are used to find the clusters of abstracts based on semantic similarity and label the topics with groups of keywords in each cluster.
The clustering results are evaluated using Silhouette Coefficient.
A case study was conducted to detect key research topics from the publications at the intersection of Urban Study and Machine Learning fields.
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
1. `BERTArticleCluster.py`: convert article texts to context vectors and cluster them.
2. `ArticleClusterTermTFIDF.py`: pick up distinct terms of each article cluster using TFIDF
3. `KeywordExtraction.py`: extract keywords from article abstract
4. `KeywordCluster.py`: cluster keywords
5. `TopicKeywordCluster.py`: pick up topic words for a keyword cluster



