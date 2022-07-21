# A tool for identifying and visualizing key topics 

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



