"""Spam filter using Naive Bayes classifier"""


import email.parser 
import os, sys, stat
from tqdm import tqdm
import re, cgi
import math, pickle
from decimal import Decimal
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np

def extract_content(filename):
	''' Extract the subject and payload from the .eml file.'''
	with open(filename, 'rb') as fp:
		msg = email.message_from_bytes(fp.read())
	sub = msg.get('subject')
	#If it is a multipart message, get_payload returns a list of parts.
	if msg.is_multipart():
		payload = msg.get_payload()[0]	
		payload = payload.as_bytes() #We will consider the body as bytes so it is easier to decode into a unicode string.
	else:
		payload =  msg.get_payload()
	return "{}\n{}" . format(sub, payload)

def get_text_from_email(mail):
	""" Removes html tags and punctuations."""
	tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')

	# Remove well-formed tags, fixing mistakes by legitimate users
	mail = tag_re.sub('', mail)

	# Clean up anything else by escaping
	mail = cgi.escape(mail)
	
	mail = re.sub(r'([\\][n|t|x])', ' ', mail)                           #Removes \n\t\b strings
	mail = re.sub(r'[=*/&;.,/\" ?:<>\[\]\(\)\{\}\|%#`~\\]', ' ', mail)   #Removes punctuations
	mail = re.sub(r'[- _=+]{2,}|(?=\s)[-_]|[-_](?=\s)', ' ', mail)       #Removes unnecessary hiphens and underscores
	mail = re.sub(r'[\d]', ' ', mail)                                    #Revoves all digits
	mail = re.sub(r'[\'!=+]', '', mail)                                  #Replaces these punctuations with null string
	return mail.lower()


def preprocess(mail):
	"""Preprocess data"""
	# Currently just one preprocessing step.
	mail = get_text_from_email(mail)
	return mail


def add_words_to_dict(word_set, word_dict, ham):
	"""Checks if the word is presnt or not and increments its respective value"""
	for word in word_set:
		if word not in word_dict:
			word_dict[word] = {'spam_count': 0, 'ham_count': 0}
		if ham:
			word_dict[word]['ham_count'] = word_dict[word]['ham_count'] + 1
		else:
			word_dict[word]['spam_count'] = word_dict[word]['spam_count'] + 1 

def calculate_spaminess(word, word_dict, total_ham, total_spam):
	""" Calculate the probability of a message being spam provided that the word is present."""

	pr_s, pr_h = 0.5, 0.5  #Assumming equal probability for both ham and spam
	threshold = 2   #Strength factor to handle rare words
	total_occurance = word_dict[word]['spam_count'] + word_dict[word]['ham_count']  #Total number of times the word has occured in both ham and spam
	freq_s = word_dict[word]['spam_count'] / total_spam 
	freq_h = word_dict[word]['ham_count'] / total_ham
	spamminess = (freq_s * pr_s) / (freq_s * pr_s + freq_h * pr_h)  #The probability that a given mail is spam, provided that this word is present.
	corrected_spaminess = (0.3 * threshold + total_occurance * spamminess) / (threshold + total_occurance)  #Considering the strength factor.
	word_dict[word]['spaminess'] = corrected_spaminess   

def generate_dictionary(files, labels):
	"""Generates a dictionary of all the words in both ham and spam mails"""
	#Initializing variables
	iterator = 0
	word_dict = {}
	total_spam = 0
	total_ham = 0

	for file in tqdm(files):
		#Read and extract mail contents
		try:
			mail = extract_content(file)
		except:
			print("Corrupted File {}" . format(file))
		# Prepare data
		mail = preprocess(mail)
		word_list = [s for s in mail.split()]
		word_set = set(word_list)

		# Incrementing HAM/SPAM count
		ham = (True if int(labels[iterator].split()[0]) == 1 else False)
		if ham:
			total_ham += 1
		else:
			total_spam += 1

		add_words_to_dict(word_set, word_dict, ham)
		iterator += 1
	for word in word_dict:
		calculate_spaminess(word, word_dict, total_ham, total_spam)
	with open('word_dict.pickle', 'wb') as f:
		pickle.dump(word_dict, f)
	return word_dict

def get_scores(expected, predicted):
	""" Compares predicted and expected values and returns various metrics."""
	scores = {}
	# _ implies we do not care about that metric.
	_, scores['False Positives'], scores['False Negatives'], _= confusion_matrix(expected, predicted).ravel()
	scores['Precision'], scores['Recall'], scores['F_score'], _= precision_recall_fscore_support(expected, predicted, average='macro')
	return scores

def training(files, labels):
	"""Trains the model and returns a word dictionary"""
	try:
		with open('word_dict.pickle', 'rb') as f:
			print("Found pickle file. Skipping training")
			word_dict = pickle.load(f)
	except:
		# Generate Dictionary
		word_dict = generate_dictionary(files, labels)

	return word_dict

def predict(files, word_dict):
	"""Predicts values using the word dictionary and returns a list of predictions"""
	predictions = []
	for file in tqdm(files):
		#Read and extract mail contents
		try:
			mail = extract_content(file)
		except:
			print("Corrupted File {}" . format(file))
		
		# Prepare data
		mail = preprocess(mail)
		word_list = [s for s in mail.split()]
		word_set = set(word_list)

		n = 0
		spaminess_list = []
		for word in word_set:
			if word not in word_dict:
				continue              						# Ignore new words (for now)
				spaminess = 0.6       						# Or... assume it is slightly spam ( Gives better FP, but lower f-score)
			else:
				spaminess = word_dict[word]['spaminess']
				if spaminess < 0.6 and spaminess > 0.4:
					continue                                #ignore the word if spaminess is neutral
			spaminess_list.append(spaminess)

		# Adding up all the word probabilities
		for spaminess in spaminess_list:
			n +=  (math.log(1-spaminess) - math.log(spaminess))
		probability = 1 / (1 + Decimal(math.e) ** Decimal(n))
		
		# Predicting 
		if probability > 0.8:
			prediction = '0'
		else:
			prediction = '1'
		predictions.append(prediction)
	return predictions


def main():

	# Default paths for all the inputs. Overrided if script not in the same locations as them.
	train = './TRAINING'
	test = './TESTING'
	spam = './SPAM.label'

	# Getting user input if defaults are not valid
	print("Please make sure the script is in the same directory as the Training and testing folders.")
	if not (os.path.isdir(train) and os.path.isdir(test) and os.path.exists(spam)):
		print("Testing and training datasets not found: ")
		train = input("Enter training dataset path: ")
		test = input("Enter testing dataset path: ")
		spam = input("Enter labels file path: ")
	
	# Getting training and testing files
	train_files = sorted([os.path.join(train, file) for file in os.listdir(train)])[:3000]
	test_files = sorted([os.path.join(test, file) for file in os.listdir(test)])
	files = train_files + test_files
	print("Found the datasets.")
	
	# Spam labels
	with open(spam, 'r') as f:
		labels = [line.split()[0] for line in f.readlines()]
	train_labels = labels[:3000]
	test_labels = labels[3000:]

	# Training our model
	print("Training the model...")
	word_dict = training(train_files, train_labels)
	
	# Predicting labels for both training and testing data.
	print("Testing on both training and testing datasets...")
	predictions = predict(files, word_dict )
	train_predictions = predictions[:3000]
	test_predictions = predictions[3000:]

	# Get respective scores
	test_scores = get_scores(test_labels, test_predictions)
	train_scores = get_scores(train_labels, train_predictions)
	combined_scores = get_scores(labels, predictions)

	# Output results onto the console
	print("\nTraining Scores:")
	for key, value in sorted(train_scores.items()):
		print("{:15} : {:.5}" .format(key, float(value)))
	print("\nTesting Scores: ")
	for key, value in sorted(test_scores.items()):
		print("{:15} : {:.5}" .format(key, float(value)))
	print("\nCombined Scores: ")
	for key, value in sorted(combined_scores.items()):
		print("{:15} : {:.5}" .format(key, float(value)))

	# Creating a results file. Pandas object is used to help format our output.
	mails = {}
	mails['files'] = [os.path.split(file)[1] for file in files]
	mails['labels'] = labels
	mails['predictions'] = predictions
	df = pd.DataFrame(mails)
	df['result'] = np.where(df['predictions'] == df['labels'], "CORRECT", "WRONG")
	df.set_index('files', inplace=True)
	with open('NBresults.txt', 'w') as f:
		f.write(df.to_string())
	print("Results file created: {}" . format(os.path.abspath('NBresults.txt')))

main()
