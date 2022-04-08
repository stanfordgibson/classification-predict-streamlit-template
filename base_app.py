"""

	Simple Streamlit webserver application for serving developed classification
	models.

	Author: Explore Data Science Academy.

	Note:
	---------------------------------------------------------------------
	Please follow the instructions provided within the README.md file
	located within this directory for guidance on how to use this script
	correctly.
	---------------------------------------------------------------------

	Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from operator import index
import streamlit as st
import joblib 
import os
from PIL import Image

# Data dependencies
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# NLP packages
import string # For punctuation removal
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk import ngrams
import collections



st.set_option('deprecation.showPyplotGlobalUse', False)
# Vectorizer
news_vectorizer = open("resources/tfidfvect1.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#----------------------------------------#
# Functions for analysis of twitter data
#----------------------------------------#
# 1) N-grams count
# 2) Word count
# 3) Length of tweet
# 4) Average word length
# 5) Table for length metrics
# 6) Most common hashtags


# 1) N-GRAMS COUNT
raw_analysis = raw.copy()
lem = WordNetLemmatizer()

# Normalization
def normalizer(tweet):
	"""
	Normalises a tweet string by removing URLs, punctuation, converting to
	lowercase, tokenisation and lemmatization.
	
	parameters:
			tweet: (string) A tweet that will be normalised
	Returns:
			lemmas: A list of the preprocessed strings
	"""
	tweet_no_url = re.sub(r'http[^ ]+', '', tweet) # Remove URLs beginning with http
	tweet_no_url1 = re.sub(r'www.[^ ]+', '', tweet_no_url) # Remove URLs beginning with http
	only_letters = re.sub("[^a-zA-Z]", " ",tweet_no_url1)  # Remove punctuation
	tokens = nltk.word_tokenize(only_letters) # Tokenization
	lower_case = [l.lower() for l in tokens] # Lowercase
	filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
	lemmas = [lem.lemmatize(t) for t in filtered_result] 
	return lemmas
raw_analysis['normalized'] = raw_analysis['message'].apply(normalizer)



# Return bigrams and trigrams
def ngrams(input_list):
	"""
	Creates a list of 2 and 3 consecutive words within the input list.
	
	Parameters:
			input_list: A list of strings that come from a normalized tweet
	Returns:
			bigrams+trigrams: A list of the bigrams and trigrams for the input list
	"""
	bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
	trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
	return bigrams+trigrams
raw_analysis['grams'] = raw_analysis['normalized'].apply(ngrams)

# Count bigrams and trigrams
def count_words(input):
	"""
	Counts the number of occurences of strings within the input list.
	
	Parameters:
		input_list: A list of strings
		
	Return:
		A list of tuples containing n-grams and a count of occurences for the n-gram
	"""
	cnt = collections.Counter()
	for row in input:
		for word in row:
			cnt[word] += 1
	return cnt

dictionary = {}

def tuples_to_dict(tup, di): 
	"""
	Convert a list of tuples into a dictionary
	"""
	di = dict(tup) 
	return di 

def show_ngrams(category, amount):
		
		"""
	Finds a specified amount of top ngrams for a category
	Parameters:
		category: (int) training data label (-1, 0, 1, 2)
		amount: (int) number of ngrams to return
	Output:
		ngrams_df: A dataframe containing a specified amount of ngrams for
		a category.
	"""
		ngrams_tup = raw_analysis[(raw_analysis.sentiment == category)][['grams']].apply(count_words)['grams'].most_common(amount+1)

		ngrams_dict = tuples_to_dict(ngrams_tup, dictionary)
		ngrams_df = pd.DataFrame(ngrams_dict.items(), columns = ['Ngram', 'Count'])
		return ngrams_df

def show_words(category, amount):
		
	
		"""
	Finds a specified amount of top words for a category
	Parameters:
		category: (int) training data label (-1, 0, 1, 2)
		amount: (int) number of words to return
	Output:
		ngrams_df: A dataframe containing a specified amount of words for
		a category.
	"""
		words_tup = raw_analysis[(raw_analysis.sentiment == category)][['normalized']].apply(count_words)['normalized'].most_common(amount+1)

		words_dict = tuples_to_dict(words_tup, dictionary)
		words_df = pd.DataFrame(words_dict.items(), columns = ['Ngram', 'Count'])
		return words_df


# 2) WORD COUNT
def word_count(tweet):
	"""
	Returns the number of words in a string.
  
	Parameters:
			A pandas series (str)
	Returns:
			An length of the tweet string (int).
	"""
	return len(tweet.split())



raw_analysis['word_count'] = raw_analysis['message'].apply(word_count)
word_count_believers = raw_analysis[raw_analysis['sentiment'] == 1]['word_count']
avg_word_count_believers = word_count_believers.mean()

word_count_deniers = raw_analysis[raw_analysis['sentiment'] == -1]['word_count']
avg_word_count_deniers = word_count_deniers.mean()

word_count_neutrals = raw_analysis[raw_analysis['sentiment'] == 0]['word_count']
avg_word_count_neutrals = word_count_neutrals.mean()

word_count_factuals = raw_analysis[raw_analysis['sentiment'] == 2]['word_count']
avg_word_count_factuals = word_count_factuals.mean()


# 3) LENGTH OF TWEET
def length_of_tweet(tweet):
	"""
	Returns the number of characters in each tweet.
	
	parameters: 
			A pandas series (str)
	Returns:
			The number of characters in each tweet (int).
	"""
	return len(tweet)

raw_analysis['tweet_length'] = raw_analysis['message'].apply(length_of_tweet)

t_length_believers = raw_analysis[raw_analysis['sentiment'] == 1]['tweet_length']
avg_t_length_believers = t_length_believers.mean()

t_length_deniers = raw_analysis[raw_analysis['sentiment'] == -1]['tweet_length']
avg_t_length_deniers = t_length_deniers.mean()

t_length_neutrals = raw_analysis[raw_analysis['sentiment'] == 0]['tweet_length']
avg_t_length_neutrals = t_length_neutrals.mean()

t_length_factuals = raw_analysis[raw_analysis['sentiment'] == 2]['tweet_length']
avg_t_length_factuals = t_length_factuals.mean()


# 4) AVERAGE WORD LENGTH
def average_word_length(tweet):
	"""
	Returns the avarage length of words withing each tweet.
	
	parameters: 
			A pandas series(str)
	Returns:
			The average length of words within each tweet (float).
	"""
	words = tweet.split()
	average = sum(len(word) for word in words) / len(words)
	return round(average, 2)
raw_analysis['avg_word_length'] = raw_analysis['message'].apply(average_word_length)

w_length_believers = raw_analysis[raw_analysis['sentiment'] == 1]['avg_word_length']
avg_w_length_believers = w_length_believers.mean()

w_length_deniers = raw_analysis[raw_analysis['sentiment'] == -1]['avg_word_length']
avg_w_length_deniers = w_length_deniers.mean()

w_length_neutrals = raw_analysis[raw_analysis['sentiment'] == 0]['avg_word_length']
avg_w_length_neutrals = w_length_neutrals.mean()

w_length_factuals = raw_analysis[raw_analysis['sentiment'] == 2]['avg_word_length']
avg_w_length_factuals = w_length_factuals.mean()


# 5) TABLE FOR LENGTH METRICS
tweet_metrics = {'Average word count': [avg_word_count_deniers,
										avg_word_count_neutrals,
										avg_word_count_believers,
										avg_word_count_factuals],
				 'Average tweet length': [avg_t_length_deniers,
				 						  avg_t_length_neutrals,
										  avg_t_length_believers,
										  avg_t_length_factuals],
				 'Average word length' : [avg_w_length_deniers,
				 						  avg_w_length_neutrals,
										  avg_w_length_believers,
										  avg_w_length_factuals]}


# Convert dictionary to dataframe
tweet_metrics = pd.DataFrame.from_dict(tweet_metrics, orient='index',
									   columns=['Deniers', 'Neutrals',
												'Believers', 'Factuals'])

# Divide "Average tweet length" by 10 so that it visualises nicely
tweet_metrics.iloc[1,:] = tweet_metrics.iloc[1,:].apply(lambda x: x / 10)

# Scale down "average tweet length" for visualisation
tweet_metrics = tweet_metrics.reset_index()
tweet_metrics_melted = pd.melt(tweet_metrics, id_vars=['index'],
							   value_vars=['Deniers', 'Neutrals',
							   'Believers', 'Factuals'])



# 6) MOST COMMON HASHTAGS
def find_hashtags(tweet):
	"""
	Create a list of all the hashtags in a string
	Parameters:
	tweet: String 
	Outputs:
	hashtags: List of strings containing hashtags in input string
	"""
	hashtags = []         
	for word in tweet.lower().split(' '): 
		#Appending the hashtag into the list hashtags
		if word.startswith('#'):        
			hashtags.append(word)	
	return hashtags

# Create new column for hashtags
raw_analysis['hashtags'] = raw_analysis['message'].apply(find_hashtags)

def show_hashtags(category, amount):
	"""
	Finds a specified amount of top hashtags for a category
	Parameters:
	category: (int) training data label (-1, 0, 1, 2)
	amount: (int) number of hashtags to return
	"""
	hashtags_tup = raw_analysis[(raw_analysis.sentiment == category)][['hashtags']].apply(count_words)['hashtags'].most_common(amount+1)

	hashtags_dict = tuples_to_dict(hashtags_tup, dictionary)
	hashtags_df = pd.DataFrame(hashtags_dict.items(), columns=['Ngram', 'Count'])
	return hashtags_df






	

header =st.container()


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	

	with header:
		st.title('Skyscraper Solutions Tweet Classifier')
		logo = Image.open('teamlogo.png')
		st.image(logo)

	# Creates a main title and subheader on your page -
	# these are static across all pages
		st.subheader("Tweet Classifer")
		st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Information", "Prediction", "Category Explanation", 'Category Analysis']
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "About Us":
		st.info("Meet the great minds!")
		# You can read a markdown file from supporting resources folder
		
		st.markdown('About the Team')
		image = Image.open('team.png')
		st.image(image, caption= 'Team Skyscrapers Solutions')

		
	
	
	
	if selection == "Information":
		

		st.write("Identifying your audience's stance on climate change" +
				 " may reveal important insights about them, such as their " +
				 "personal values, their political inclination, and web behaviour.")
		st.write("This tool allows you to imput sample text from your target audience "+
				 " and select a machine learning model to predict whether the author of "+
				 " that text")
		st.write("* Believes in climate change")
		st.write("* Denies climate change")
		st.write("* Is neutral about climate change")
		st.write("* Provided a factual link to a news site")
		st.write("You can also view an exploratory analysis about each category to gain deeper insights "+
				 "about each category.")
		st.write("Select Prediction in the side bar to get started.")
		train_df = pd.read_csv('train.csv')
		sentiment_labels = pd.DataFrame(train_df.sentiment.unique())

		st.markdown('The sentiment labels are:')
		st.write(sentiment_labels)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page


		#dataset =st.container()
		
		
		#with dataset:
			#st.header('Climate Change Tweeter Dataset')
			#st.text('Data The collection of this data was funded by a Canada Foundation for Innovation JELF Grant\nto Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change\ncollected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected.\nEach tweet is labelled as one of the following classes')
			#st.title('A closer look into the data')
			
			#st.write(train_df.head(5))
			
			
			#st.markdown('We will now create a barplot that will help us view and understand the distribution of the Classes (sentiments).')
			

	
	if selection == "Category Explanation":
			
			

		
				
				
			st.write("## Comparison of categories")
			st.write("The model predicts the text to be classed into one of four categories:")
			st.write("* Denies climate change (-1)")
			st.write("* Is neutral about climate change (0)")
			st.write("* Believes in climate change (1)")
			st.write("* Provided a factual link to a news site (2)")
			st.write("View the raw data used to train the models at the bottom of the page.")
			st.write("More information relating to the most commonly used words for each category can"+
				 " be found in the 'Analysis of each category' page in the sidebar.")

		# Count of each category
			st.write("### Count of each category")
			st.write("Number of tweets available in each category of the training data.")
			if st.checkbox('Show count of each category'):
					
					
				
					fig1 = sns.countplot(x='sentiment', data=raw, palette='rainbow')
					st.pyplot()
					st.write(raw['sentiment'].value_counts())
					st.write("The training data contained more tweets from climate change believers"+
					" than any other group which implies that there may be more information"+
					" available about this group than others. When building the models, the categories"+
					" were resampled so that they each were of the same size.")
		
		# Length metrics
			st.write("### Length metrics")
			st.write("Comparison of categories' average tweet length, word count, and average word length.")
			if st.checkbox('Show length metrics'):
					
				st.write(tweet_metrics)
				fig2 = sns.barplot(x='variable', y='value', hue='index', data=tweet_metrics_melted,
							   palette='rainbow')
				st.pyplot()
				st.write("The 'Average tweet length' metric was divided by 10 so that it could "+
					 "be visualised on the graph.")
				



			

			#st.info('About Models')
			#image1 = Image.open('confusion_mrtx_grdbst.png')
			#st.image(image1, caption= 'Confusion Matrix for Gradient Boost')
			#image2 = Image.open('confusion_mrtx_knn.png')
			#st.image(image2, caption= 'Confusion Matrix for K Nearest Neighbor')
			#image3 = Image.open('confusion_mrtx_lr.png')
			#st.image(image3, caption= 'Confusion Matrix for Logistic Regression')
			#image4 = Image.open('confusion_mrtx_nbm.png')
			#st.image(image4, caption= 'Confusion Matrix for Naive Bayes (Multinomial)')
			#image5 = Image.open('confusion_mrtx_rndmfst.png')
			#st.image(image5, caption= 'Confusion Matrix for Random Forest')





			
	
	
			
	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		st.info("1. Enter a sample text in the box below\n " +
				"2. Select the model used to classify your text\n"+
				"To learn more about each group, please explore the options in the sidebar.")
		
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		def cleaner(tweet):
					
				lowercase = tweet.lower()
				without_handles = re.sub(r'@', r'', lowercase)
				without_hashtags = re.sub(r'#', '', without_handles)
				without_URL = re.sub(r'http[^ ]+', '', without_hashtags)
				without_URL1 = re.sub(r'www.[^ ]+', '', without_URL)    
				return without_URL1

		# Function to remove punctuation
		def remove_punctuation(message):
			return ''.join([l for l in message if l not in string.punctuation])

		def remove_stop_words(tokens):
			return [t for t in tokens if t not in stopwords.words('english')]    
			
		
			

		# Allow user to select model
		model = st.selectbox("Select a model to make the prediction",
							['Support Vector', 'Random Forest',
							'Logistic Regression'])

		# Classify using SVC
		if model =='Support Vector':
			
			if st.button("Classify using Support Vector"):
					# Apply preprocessing functions to user input:
					clean_txt = cleaner(tweet_text)
					# Remove punctuation:
					clean_txt = remove_punctuation(clean_txt)
					# Tokenize:
					clean_txt = TreebankWordTokenizer().tokenize(clean_txt)
					# Remove Stop Words:
					clean_txt = remove_stop_words(clean_txt)
					# Lemmatize:
					clean_txt = [WordNetLemmatizer().lemmatize(word) for word in clean_txt]
					# Restore to sentence:
					clean_txt = " ".join(clean_txt)
		
				
					

				# Load your .pkl file with the model of your choice + make predictions
					vect_text = tweet_cv.transform([clean_txt]).toarray()
					predictor = joblib.load(open(os.path.join("resources/svm.pkl"),"rb"))
					prediction = predictor.predict(vect_text)

					if prediction == 0:
						st.success('Neutral!*** Select "Comparison Analysis" in the sidebar for more information about this category')
					if prediction == -1:
						st.success('Climate change denier!*** Select "Comparison Analysis" in the sidebar for more information about this category')
					if prediction == 2:
						st.success('Provides link to factual news source!*** Select "Comparison Analysis" in the sidebar for more information about this category')
					if prediction == 1:
						st.success('Climate change believer!*** Select "Comparison Analysis" in the sidebar for more information about this category')

		# Classify using Random Forest
		if model =='Random Forest':
			
			if st.button("Classify using Random Forest"):
					
					# Apply preprocessing functions to user input:
					clean_txt = cleaner(tweet_text)
					# Remove punctuation:
					clean_txt = remove_punctuation(clean_txt)
					# Tokenize:
					clean_txt = TreebankWordTokenizer().tokenize(clean_txt)
					# Remove Stop Words:
					clean_txt = remove_stop_words(clean_txt)
					# Lemmatize:
					clean_txt = [WordNetLemmatizer().lemmatize(word) for word in clean_txt]
					# Restore to sentence:
					clean_txt = " ".join(clean_txt)
				# Load your .pkl file with the model of your choice + make predictions
					vect_text = tweet_cv.transform([clean_txt]).toarray()
					predictor = joblib.load(open(os.path.join("resources/rf.pkl"),"rb"))
				#tweet_text = [tweet_text]
					prediction = predictor.predict(vect_text)

					if prediction == 0:
						st.success('Neutral!*** Select "Comparison Analysis" in the sidebar for more information about this category')
					if prediction == -1:
						st.success('Climate change denier!*** Select "Comparison Analysis" in the sidebar for more information about this category')
					if prediction == 2:
						st.success('Provides link to factual news source!*** Select "Comparison Analysis" in the sidebar for more information about this category')
					if prediction == 1:
						st.success('Climate change believer!*** Select "Comparison Analysis" in the sidebar for more information about this category')
		# Classify using Logistic Regression
		if model=='Logistic Regression':

				if st.button("Predict using Logistic Regression"):
						
						# Apply preprocessing functions to user input:
						clean_txt = cleaner(tweet_text)
					# Remove punctuation:
						clean_txt = remove_punctuation(clean_txt)
					# Tokenize:
						clean_txt = TreebankWordTokenizer().tokenize(clean_txt)
					# Remove Stop Words:
						clean_txt = remove_stop_words(clean_txt)
					# Lemmatize:
						clean_txt = [WordNetLemmatizer().lemmatize(word) for word in clean_txt]
					# Restore to sentence:
						clean_txt = " ".join(clean_txt)
					
						
						
						
						predictor = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
						vect_text = tweet_cv.transform([clean_txt]).toarray()
						prediction = predictor.predict(vect_text)
						

						if prediction == 0:
							st.success('Neutral!*** Select "Comparison Analysis" in the sidebar for more information about this category')
						if prediction == -1:
							st.success('Climate change denier!*** Select "Comparison Analysis" in the sidebar for more information about this category')
						if prediction == 2:
							st.success('Provides link to factual news source!*** Select "Comparison Analysis" in the sidebar for more information about this category')
						if prediction == 1:
							st.success('Climate change believer!*** Select "Comparison Analysis" in the sidebar for more information about this category')	
	# Building out the Analysis of each category page
	if selection == 'Category Analysis':
		st.write("## Analysis of Individual Categories")
		st.info("1. Select a category from the dropdown menu for a more detailed analysis.\n\n"+
				"2. Select which analysis you would like to see")

		# Select category of which user would like to view data
		category = st.selectbox("Select category",
							['Deniers', 'Neutrals',
							'Believers', 'Factuals'])

		# Most frequent individual words
		st.write("### Most frequently used words")
		st.write("Most commonly-used words for this category.")
		if st.checkbox('Show most frequent words'):
			st.write("### Most frequent individual words (lemmas)")
			st.write("These are the' lemmas. One of the text preprocessing "+
					"steps involved lemmatization, which is simplifying a word to its most basic "+
					"form. It groups related words together, e.g. 'walked', 'walks', 'walking' "+
					"would be simplified to just 'walk.'")
			if category == 'Deniers':
				cat_slider_w = -1
			if category == 'Neutrals':
				cat_slider_w = 0
			if category == 'Believers':
				cat_slider_w = 1
			if category == 'Factuals':
				cat_slider_w = 2
			number_to_show_w = st.slider('Amount of entries to show', 1, 50, 10)
			st.write(show_words(cat_slider_w, number_to_show_w))

		# N-grams analysis
		#show_ngrams(category, amount)
		st.write("### Most frequently used phrases (n-grams)")
		st.write("Most commonly-used phrases for this category.")
		if st.checkbox('Show most frequent phrases'):
			st.write("n-grams refer to a sequence of n consecutive items. In this case, it"+
					" refers to a n consecutive words in a text. The most common bigrams (two words) and"+
					" trigrams (three words) were counted in the training data. These show the most frequently"+
					" used sets of two and three words by each category.")
			if category == 'Deniers':
				cat_slider_ngram = -1
			if category == 'Neutrals':
				cat_slider_ngram = 0
			if category == 'Believers':
				cat_slider_ngram = 1
			if category == 'Factuals':
				cat_slider_ngram = 2
			number_to_show_ngram = st.slider('Amount of entries to show', 1, 50, 10)
			st.write(show_ngrams(cat_slider_ngram, number_to_show_ngram))

		
		# Length-related metrics checkbox
		st.write("### Length-related metrics")
		st.write("Average word count per tweet, average tweet length, average word length.")
		if st.checkbox('Show length-related metrics'):
			if category == 'Deniers':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_deniers, 2))
				st.write("Average tweet length: ", round(avg_t_length_deniers, 2))
				st.write("Average word length: ", round(avg_w_length_deniers, 2))

			if category == 'Neutrals':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_neutrals, 2))
				st.write("Average tweet length: ", round(avg_t_length_neutrals, 2))
				st.write("Average word length: ", round(avg_w_length_neutrals, 2))

			if category == 'Believers':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_believers, 2))
				st.write("Average tweet length: ", round(avg_t_length_believers, 2))
				st.write("Average word length: ", round(avg_w_length_believers, 2))

			if category == 'Factuals':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_factuals, 2))
				st.write("Average tweet length: ", round(avg_t_length_factuals, 2))
				st.write("Average word length: ", round(avg_w_length_factuals, 2))
		
		# Top hashtags
		st.write("### Top hashtags")
		st.write("Most frequently-used hashtags for this category.")
		if st.checkbox("Show top hashtags"):
			if category == 'Deniers':
				cat_slider = -1
			if category == 'Neutrals':
				cat_slider = 0
			if category == 'Believers':
				cat_slider = 1
			if category == 'Factuals':
				cat_slider = 2
			number_to_show = st.slider('Amount of entries to show', 1, 50, 10)
			st.write(show_hashtags(cat_slider, number_to_show))
		
		# Analysis
		if category == "Deniers":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")
			st.write("In this dataset, climate change deniers seem to tend to retweet Donald Trump and Twitter user"+
			 		 " @SteveSGoddard, who has since changed his username to [@Tony__Heller](https://twitter.com/Tony__Heller).")

		if category == "Neutrals":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")

		if category == "Believers":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")
			st.write("Tweets in this category frequently mention the idea of dying as a result of climate change."+
					 " The tweets that are frequently mentioning Twitter user @StephenSchlegel are retweets that"
					 " are responding to a tweet by Melania Trump. Melania posted a picture of a sea creature with "
					 "the caption, 'What is she thinking?' and many people responded with, 'She's thinking about"
					 " how she's going to die because your husband doesn't believe in climate change.' This may "
					 "indicate that those who believe in climate change tend to not follow Donald Trump.")
					 
		if category == "Factuals":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")
			st.write("These tweets seem to be centered around issues relating to policy and Donald Trump and "
					 "Scott Pruitt's (former Administrator of the U.S. Environmental Protection Agency) views on climate change.")
			st.write("There is also mention of the "
					 "[Paris Agreement](https://unfccc.int/process-and-meetings/the-paris-agreement/the-paris-agreement), which"
					 " is an agreement with the United Nations Framework Convention on Climate Change which deals with"
					 " the reduction of the impact of climate change. In 2017, President Donald Trump chose to withdraw"
					 " the U.S.'s participation from this agreement.")

	

		

		#if st.button("Classify"):
			#Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/lr_model.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
