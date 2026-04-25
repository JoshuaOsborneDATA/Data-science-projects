# EDA
## outliers
1. looking at the histogram of popularity there are many zeros
	1. made the decision to drop any with 0 upon closer inspection. Many are Christmas songs likely from very small artists that simply had little to no outreach, but also likely songs that may have been taken off of spotify.

## a first glance at feature gridplot
1. at first glance, it seems one of the most influential features will be instrumentalness. Makes sense since songs with more vocals are more likely to be hits (like pop I'm guessing).
## boxplots
1. looking at the boxplots, what stands out the most is that in acousticness, the outliers of class 1 (hit songs) is outside the distribution of class 0. Class 1 also has a much tighter distribution although it overlaps with class 0.

## Top 10 genres based on counts alone

| genre   | count |
| ------- | ----- |
| pop     | 282   |
| dance   | 241   |
| electro | 222   |
| house   | 206   |
| rock    | 197   |
| k-pop   | 196   |
| metal   | 186   |
| indie   | 172   |
| edm     | 167   |
| latino  | 161   |


## surprising findings
1. the apparent bimodality in the popularity
2. the duration distribution is much tighter for hit songs (at least visually. Better to actually calculate it)

# Preprocessing
__audio_features__
"danceability","energy","loudness","speechiness","acousticness","instrumentalness", "liveness","valence", "tempo","duration_ms" 

categorical_features = "explicit", "key","mode","time_signature","track_genre"


# Results
The best performing model seems to be the random forest based on the PR-AUC score. I seemed to have chosen the optimal n_estimators by luck because gridsearch did not find a better parameter combination. I chose the best model using the PR-AUC because the dataset in this case is inbalanced (so ROC would not be useful). Naive bayes performed poorly in this case because the underlying assumption of NB is that the features are conditionally independent, which in this case there are features that violate this assumption. There are other models which may be a better fit, but I am using this to test specific models I have looked at recently. Additionally, KNN may do better with different k values. To find the optimal value of k we would need to do cross validation.
