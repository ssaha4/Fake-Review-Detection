# Fake Review Detection on Yelp

This project explores whether supervised learning can identify Yelp reviews flagged by Yelp's filter. It combines review text with reviewer activity and business-level patterns, then compares models on how many flagged reviews they find and how often those flags are correct.

In the saved holdout run, the **class-weighted neural network reached 73% overall accuracy, 66% recall and 23% precision for flagged reviews**. The class imbalance is central to the result. Predicting every review as genuine would yield **89.73% accuracy** with zero recall for flagged reviews.

## Dataset and labels

The experiments use **YelpNYC**, from the dataset published with Shebuti Rayana and Leman Akoglu's *Collective Opinion Spam Detection: Bridging Review Networks and Metadata* (KDD 2015). The source files pair review text with reviewer ID, business ID, rating, filter label, and posting date. The notebooks map source label `+1` to genuine (`0`) and filtered reviews to fake (`1`). These are proxy labels derived from Yelp's filter. The experiments measure agreement with that filter.

| Model label | Reviews | Share |
| --- | ---: | ---: |
| Genuine (`0`) | 322,167 | 89.73% |
| Flagged / fake (`1`) | 36,885 | 10.27% |
| Total | 359,052 | 100% |

Counts come from the archived `labels.csv`. Both classes appear in the ANN holdout, with 64,434 genuine reviews and 7,377 flagged reviews.

## How the pipeline works

1. **Parse reviews and metadata.** Read the tab-delimited text and metadata, clean the text with regular expressions, and derive sentiment and text-length signals. The notebooks also explore stop-word removal, stemming, lemmatization, and bigrams.
2. **Build behavioral and business features.** Group reviews by reviewer to capture posting frequency and rating behavior. Group reviews by business to calculate rating and volume statistics and compare wording across reviews for the same business.
3. **Assemble one feature row per review.** Attach reviewer and business aggregates to each review, then add similarity and cluster features. The archived model inputs are `featuresAll.csv` with 17 columns and a separately stored `labels.csv` in matching row order.
4. **Explore and train.** Inspect feature distributions, outliers, and correlations. Compare probabilistic, tree-based, and neural classifiers using each notebook's documented split and preprocessing.
5. **Evaluate by class.** Read the confusion matrix and report precision, recall, and F1 for each class. Use ELI5 to explore which features the trained models rely on.

## The 17 model inputs

| Scope | Feature | What it captures |
| --- | --- | --- |
| Review | `polarity` | TextBlob sentiment direction, from negative to positive |
| Review | `subjectivity` | TextBlob score from objective to subjective |
| Review | `wordCount` | Text-length signal. The review extraction notebook uses the character length of the cleaned, stemmed string. |
| Business | `stdrating` | Standard deviation of ratings for the business |
| Review and business | `clusterid` | K-means cluster assignment from numeric reviewer ID, business ID, and posting time |
| Review and business | `meanSimilarity` | Mean cosine similarity between a review and reviews for the same business |
| Review and business | `varOfSimilarity` | Variance of those similarity scores |
| Business | `meanRating` | Mean rating for the business |
| Business | `numberOfReviews` | Review count for the business |
| Reviewer | `avgNoOfWords` | Average number of processed word tokens in the reviewer's reviews |
| Reviewer | `totalReviews` | Number of reviews by the reviewer |
| Reviewer | `isSingleton` | Whether the reviewer has exactly one review |
| Reviewer | `posReviewRatio` | Fraction of the reviewer's ratings at or above 3 |
| Reviewer | `negReviewRatio` | Fraction of the reviewer's ratings below 3 |
| Reviewer | `avgNoOfReviews` | Mean review count per day on days the reviewer posted |
| Reviewer | `maxNoOfReviews` | Largest review count on any one day for that reviewer |
| Reviewer | `dateVariance` | Variance of the gaps in days between consecutive reviews after sorting by date |

### Text similarity and clustering

The similarity notebook fits TF-IDF separately within each business. TF-IDF gives more weight to terms that occur in a review and less weight to terms that appear across many reviews in that business. The notebook takes the dot product of normalized TF-IDF vectors to obtain a pairwise cosine similarity matrix, then summarizes each row by its mean and variance and max-normalizes each summary within the business. The calculation includes self-similarity on the diagonal. For example, two reviews that reuse several distinctive phrases can be closer in this representation than two reviews that share only common words.

The clustering notebook explores K-means inertia over different cluster counts and exports assignments from a 1,000-cluster experiment. The clustering input includes time expressed in seconds relative to a fixed reference date. Because reviewer and business IDs are numeric identifiers, their numeric proximity is an experimental signal whose value depends on the ID encoding.

## Models and evaluation

| Model | Experiment |
| --- | --- |
| Random Forest | Compared an 80/20 random split with five-fold stratified cross-validation. The stratified experiment uses 50 trees and standardizes features within each fold. |
| Gaussian Naive Bayes | Fits a Gaussian feature distribution per class and uses five-fold stratified cross-validation. The scaler is fit on the training rows and applied to the validation rows. |
| Decision Tree | Uses ten stratified folds, with PCA reducing 17 features to seven components. PCA is fit separately on each training fold. |
| XGBoost | Uses the same ten-fold PCA setup. The dedicated notebook uses the default classifier settings. |
| ANN | Uses four hidden dense layers with 64, 128, 128, and 32 units, each followed by batch normalization and ReLU. A single sigmoid unit outputs the flagged-class score. The active run uses Adam, binary cross-entropy, 10 epochs, batch size 128, and class weights `{0: 1, 1: 7}`. |

Stratified folds keep the class proportions close to the original dataset. Class weighting changes the training objective by penalizing mistakes on flagged reviews more heavily. In the active ANN run, an approximately 80/20 holdout is created by slicing each class separately, shuffling the combined training rows, and rounding sigmoid scores to binary predictions. The saved output comes from that weighted holdout experiment. The archive also includes exploratory ANN cross-validation output and an SVM experiment.

### Saved ANN holdout results

| Class | Precision | Recall | F1 | Test examples |
| --- | ---: | ---: | ---: | ---: |
| Genuine | 0.95 | 0.74 | 0.83 | 64,434 |
| Flagged / fake | 0.23 | 0.66 | 0.34 | 7,377 |

Overall accuracy was **0.73**. Fake-class recall of **0.66** means the ANN found about two-thirds of reviews with the filtered label. Precision of **0.23** means about one in four reviews it flagged matched that label. The saved XGBoost experiment illustrates the opposite tradeoff. It achieved around 90% accuracy while fake-class recall stayed below 1.1% in every fold.

The project report's separate comparison summarizes fake-class precision, recall, and F1 at **0.23 / 0.61 / 0.34** for Gaussian Naive Bayes, **0.21 / 0.26 / 0.24** for the decision tree, and **0.29 / 0.13 / 0.18** for Random Forest. These are historical experiment results. The ANN holdout and the cross-validation results use different evaluation setups and should be compared with that distinction in mind.

### Feature analysis

The visualization notebook plots histograms, density curves, boxplots, and feature correlations. The ELI5 notebook permutes a feature column, then measures how much the trained model's score changes when that feature's relationship to the labels is disrupted. The saved Naive Bayes experiment applies permutation importance to the final fold's training partition. The report lists positive and negative rating ratios among the leading signals for that model. Those two ratios are complementary, so their rankings should be interpreted together. The experiment provides an exploratory view of model behavior and uses training-set scores.

## Notebooks

| Notebook | Contents |
| --- | --- |
| [Feature Extraction](notebooks/Feature_Extraction.ipynb) | Review parsing, sentiment, cleaned-text length, rating spread, and exploration of posting data |
| [Reviewer Features and Model Training](notebooks/SWM_UserFeatureExtraction_ModelTraining.ipynb) | Reviewer activity, date gaps, K-means, feature assembly, Random Forest and Gaussian Naive Bayes |
| [Business Features and Content Similarity](notebooks/Product_features_and_content_similarity_and_model_train.ipynb) | Business rating and review counts, TF-IDF, cosine similarity, and classifier experiments |
| [Visualization](notebooks/Visualization.ipynb) | Distributions, correlations, outliers, and SVM exploration |
| [ANN](notebooks/ANN.ipynb) | Class-weighted neural network and holdout metrics |
| [XGBoost and Decision Tree](notebooks/XGB%26DecisionTree.ipynb) | PCA within stratified folds and class-level evaluation |
| [Feature Importance](notebooks/Feature_Importance.ipynb) | Gaussian Naive Bayes permutation importance with ELI5 |

## Run locally

```bash
git clone https://github.com/ssaha4/Fake-Review-Detection.git
cd Fake-Review-Detection
python -m pip install -r requirements.txt
jupyter notebook
```

- For extraction, obtain the YelpNYC `reviewContent`, `metadata`, and `reviewGraph` files. Install the NLTK resources requested by the tokenization, stop-word removal, and lemmatization cells.
- For model notebooks, supply aligned `featuresAll.csv` and `labels.csv` files. Use the feature table above to check the 17 inputs, keep `label` as a separate target, and verify that feature and label rows refer to the same reviews. The visualization notebook also uses `dataAll.csv`.
- Update notebook paths to match your working directory. The original experiments contain local and Colab paths and refer to intermediate CSVs, including `review_features.csv`, `user.csv`, `product.csv`, and `datacluster_seconds.csv`. Check CSV headers and join keys when wiring the separate stages together.
- The notebooks are preserved as experiment code, with saved outputs stripped for browsing and raw data kept separate. Dependencies are unpinned, and some notebook cells use older pandas or scikit-learn APIs. The numbers above come from archived runs. A fresh end-to-end execution has yet to be verified for this publication.

## Scope and limitations

Reviewer and business aggregates can be computed across the full dataset before splitting. A review-level split can therefore share reviewer or business history across training and validation rows, including activity that would be unavailable when scoring a new review. A time-based evaluation with historical-only aggregates would measure performance on future reviews. A reviewer- or business-grouped split would test generalization to unseen entities. The pairwise similarity calculation also materializes a dense matrix for each business, so its memory grows quadratically with that business's review count.
