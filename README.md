# StockMarketPrediction_NLP
## Objective
Project consisted on developing an NLP model to predict the daily closing values of a stock market index based on news. The jupyter notenook presents all the experiments with all the technique used and to achieave the model with higher performance.
## DataSet
The data was divided in a file for training and another file for testing. After applying different NLP techniques, a ready-to-run final code with the model which achieves the highest classification performance was created with the evaluation of the model for the test dataset. 
## Data Exploration 
### Analyze the data and provide some conclusions and visual information that contextualize the data.
In the data exploration, two histograms were plotted to infer the count of the binary variable, in this case, the Closing Status and to identify what were the most frequent words in the headlines for each Id.

It can be observed that the most frequent words appear to be stop words or punctuation, emphasizing the importance of preprocessing the data to reveal what will be the most relevant words in each headline.
## Data Preprocessing 
### Implementation and experiment different preprocessing techniques
For splitting our dataset, the function train_test_split from sklearn was used to obtain training and test sets, selecting the size of each, and fixing the random generator. 

` C_train, C_test, y_train, y_test = train_test_split(corpora, corpora['Closing Status'], test_size=0.20, random_state=4)`

For pre-processing the data, two approaches were used: noise removal and normalization. In normalization, to better uniformize the data, was decided to use lemmatization to normalize the words of the headlines in the data set, since it considers vocabulary and the morphological analysis of each word (returns the lemma of a word, dictionary form) and can increase the recall in our predictions – relevant words are considered correctly. 

Removing the noise from the headlines was considered important to implement a successful text analysis: lowercasing all the words was helpful to make the text simpler to apply the tasks ahead; removing punctuation to each sentence was important to better train our models, since it can add noise and ambiguity when performing the training step; 

Stop Words were removed so the model only considers the key features in the headlines - these words don’t offer much important information.

It was taken in consideration that in News headlines the text used is already simpler than when dealing with extensive documents or news articles, so this step was done with moderation, only using techniques that would provide the best results for this analysis.

## Feature Engineering
### Implementation of two feature engineering techniques
After implementing the preprocessing methods to our text data, its needed to extract the features that will be used in the classification models, from the text data.

In this project two methods were used: Bag of Words and TF – IDF.

The Bag of Words representation transforms the text in fixed length vectors using Count Vectorizer. This method does not consider the structure/order of words, it only checks the word occurrences. Although this model is very simple to implement, it does not consider the meaning of words like other feature engineering techniques.

TF – IDF is helpful in considering the importance of each word to understand our data set. The Term Frequency indicates the frequency of each word in the data set, while the Inverse Document Frequency indicates the importance of the word in the data set (term frequency alone gives equal importance to every word). For example, if a word is repeated in a dataset very frequently, then it may not be carrying important information compared to less frequent words.

## Classification Models
### Test of four classification algorithms 
The K-Nearest Neighbors assumes that similar things exist near each other. This model can be very useful in classification problems since it can easily identify the class of a particular data point. In the KNeighborsClassifier the number of neighbors chose was the default and the weigh function used in prediction was the distance, where closer neighbors of a query point will have a greater influence than neighbors which are further.

Logistic Regression is a supervised learning method used to predict the probability of a binary event.

Naïve Bayes is a classification algorithm for binary (two-class) and multi-class classification problems, that assumes inputs as independent given the target value, calculating the probability of each class in the train dataset, then calculate the distance to neighbors and calculate the respective probabilities.

The Random Forest is an ensemble model that fits a number of decision tree classifiers on various sub-samples of the dataset and is used to improve the predictive accuracy and control over-fitting. The number of trees considered was 200 and since the goal of the random forest classifier is to try to predict classes accurately, the criteria chosen was entropy to maximize information gained with the split.

## Evaluation 
### Evaluate the models resorting to Recall, Precision and Accuracy
Precision answers the question of what proportion of positive identifications was correct. Recall, on the other hand, tells us what proportion of actual positives were identified correctly.

For example, the KNN model (using BoW method) has a precision of 0,27, meaning that when it predicts if the index closing value will be higher or equal it is correct 27% of the time. With a recall result of 0,53, the model identifies 53% of index’s that closed with a higher or equal value.

The accuracy showed poor results from the train to the test data set in every model, not going above 55%, with KNN model using the BoW feature engineering method being the model with the highest accuracy and the Random Forest using the TF-IDF method being the seconds best accuracy wise.

The overall result for each classification model translates overfitting problems, meaning that the models cannot generalize well in the test data set.
In almost every model presented in this project, with exception of the KNN model, the TF – IDF method provided the best results for the models considered, when compared to the use of Bag of Words in each classification model.

## Results and Final Conclusions
There can be different approaches to choose the preferred model for predicting the daily closing values of a tock market index.

The emphasis should rely on general performance measures, such as the precision, recall and accuracy. Following this approach, the KNN model was the one that retrieved higher values when compared to the other models: accuracy of 55% (although still low for the main goal of the present project), a precision of 67% and a recall of 55%.

Furthermore, taking into consideration the objective of the project and the results that all the models presented, it would be necessary to apply more data preprocessing methods and extra feature engineering methods to enhance the training and the learning for this data set.
