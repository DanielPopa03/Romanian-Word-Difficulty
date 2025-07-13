As part of our Machine Learning course, this was a project and competition. We ranked first with a score of 0.51 on the public test data and 0.6 on the private test data, which was used for the final evaluation.
The dataset was created by our professor and annotated by us. The report was originally written in Romanian, and you can find the translated version below.

# Word Complexity Evaluation  
**Popa Robert-Daniel, Georgian Sofronea**

## Abstract  
Using a dataset containing a token and a context, we attempt to determine the complexity of that token (ideally, a word).  
Using various models and sets of features, we concluded several things, including the necessity of data augmentation.  
Also, we were unable to extract significant information from the sentence (context), but rather from the token itself.  
The trained model is an ensemble of Gradient Boosting techniques (CatBoost + LightGBM), where some of the most important inputs are the word frequency in a large text corpus and various features from the word embedding space.  
Next, we present the solution we found, as well as the intermediate steps we took to reach it.

---

## 1 Features

### 1.1 Immediate Features  
Initially, we began with a data analysis process: we tried to plot each feature to observe any correlations.  
Among the first features verified were: word length, number of syllables, number of vowels, vowel-to-consonant ratio.

---

### 1.2 spaCy  
We extracted the following using `spaCy Romanian News Large`:  
- OOV (is the word out-of-vocabulary?),  
- lemma (dictionary form),  
- part of speech,  
- vector length from the embedding space,  
- 10 vectors obtained through Principal Component Analysis (PCA).  

Principal Component Analysis (PCA) is a technique that uses covariance and eigenvalues to reduce dimensionality.  
In our case, we reduced the space from several hundred dimensions (300) to 10, which helped in the training process.  
We noticed that if the part of speech is unknown or a proper noun, it correlates with a higher score. Its final importance is minor but still contributes.  
OOV, although initially correlated, turned out to have no importance in the final analysis.  

We also tried extracting information about word relationships using the syntactic tree from spaCy, but it did not yield positive results.  

**Figure 1: Syntactic Analysis**  
<img width="607" height="601" alt="image" src="https://github.com/user-attachments/assets/719fd706-4ad7-4ed6-8f65-dddfd207e3a7" />

---

### 1.3 Corolla  
From the Corolla corpus, we extracted the frequency of each lemma of each token and normalized it.

Initially, we removed outliers (those tokens whose lemmas were not found in the Corolla corpus).

The correlation was clearly linear. Pearson correlation had a value of 0.56.  
We centered the data and assigned outliers a mean value.

**Figure 2: Correlation with Frequency in Corolla**  
**Figure 3: Corolla Correlation Without Outliers**  
**Figure 4: Corolla Correlation with Outliers Replaced by Mean**  
<img width="362" height="729" alt="image" src="https://github.com/user-attachments/assets/b38d2425-03e9-41d3-9962-7ef00f8336fc" />


---

### 1.4 Extracted Embeddings  
We extracted embeddings using `word2vec`, `fasttext`, and a pretrained BERT model, generically referred to as "transformer".  
These embeddings were also reduced using PCA to 10 dimensions, except for BERT, which was reduced to 20 dimensions.

---

### 1.5 Wikipedia Scrape  
Using the `wikiextractor` API, we extracted the entire Romanian-language Wikipedia corpus and applied the same procedure used for the Corolla corpus.

---

### 1.6 Geometric Semantics  
Knowing that certain operations can be performed in embedding spaces, such as:
E(King) − E(Queen) = E(Man) − E(Woman)
**Figure 5: Semantic Relationships in Embedding Space**  
**Figure 6: Semantic Relationships in Embedding Space**  
<img width="912" height="443" alt="image" src="https://github.com/user-attachments/assets/4f04cc99-43bf-4901-9ef4-8770d319e3ad" />


We deduced that by subtracting two synonymous words that differ in complexity, we could extract a **semantic direction** in the embedding space of complexity.

For example, `E(achizitiona)` − `E(cumpara)` determines a direction in space.  
Since the words are synonyms, they are semantically similar, ideally differing only in complexity.  
We used a set of 1000 such word pairs (initially 100) to hypothetically determine a direction of complexity in the BERT embedding space.  

We then used this (normalized) direction to perform a dot product with the token’s embedding, hoping to obtain a complexity metric — which worked.

---

## 2 Architectures and Progress  
We tried several machine learning models.  
We started with linear regression (which initially worked well, surpassing the baseline using only spaCy features), then moved to an MLP which didn’t improve results much (only ~1% better than linear regression).  

We tried a KNN Regressor to detect non-linear relationships but it was unsuccessful.  

Later, with the addition of embeddings, we tried Random Forest Regressor and Support Vector Regressor, which had similar results (~35%).

Finally, we used Gradient Boosting, where **CatBoost** (for categorical data) performed best, followed closely by **LightGBM**, and then **XGBoost**, so we averaged results from the first two. This yielded results around **40%**.  

After discovering **geometric semantics**, we found that XGBoost underperformed compared to the other two methods in our case, so we discarded it.

The geometric semantic feature, when used alone in an SVR model, scored approximately **17%**.  
When integrated into the best ensemble model, we achieved a score of **49%**, which increased to **51.66%** after expanding the simple-complex word pair list from 100 to 1000.

---

## 3 Feature Importance Analysis  
In **Figures 7, 8, and 9**, we plotted the importance of each feature in the final model, in ascending order.

**Figure 7: Feature Importance in CatBoost**  
<img width="645" height="702" alt="image" src="https://github.com/user-attachments/assets/6100d5ac-dde2-4c5a-b963-98d8b30ac0d2" />


**Figure 8: Feature Importance in LightGBM**  
<img width="465" height="577" alt="image" src="https://github.com/user-attachments/assets/33f17292-2026-4ad0-bedb-1a6fab56e678" />

**Figure 9: Feature Importance in CatBoost + LightGBM Ensemble**  
<img width="467" height="536" alt="image" src="https://github.com/user-attachments/assets/9d04b001-b36e-453c-a37e-b6227219d715" />



