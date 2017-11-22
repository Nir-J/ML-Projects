""" Spam filter using linear SVM and bag of words """

from sklearn import feature_extraction, svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os, sys
import email, re, cgi
from tqdm import tqdm

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
	
	return mail.lower()

def get_scores(expected, predicted):
	""" Compares predicted and expected values and returns various metrics."""
	scores = {}
	# _ implies we do not care about that metric.
	_, scores['False Positives'], scores['False Negatives'], _= confusion_matrix(expected, predicted).ravel()
	scores['Precision'], scores['Recall'], scores['F_score'], _= precision_recall_fscore_support(expected, predicted, average='macro')
	return scores

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
	files = sorted([os.path.join(train, file) for file in os.listdir(train)])[:3000]
	test_files = sorted([os.path.join(test, file) for file in os.listdir(test)])
	files.extend(test_files)
	print("Found the datasets.")
	
	# Spam labels
	with open(spam, 'r') as f:
		labels = [line.split()[0] for line in f.readlines()]
	
	# Extracting text from email
	bodies = []
	vectors = []
	words = {}
	for file in tqdm(files):
		bodies.append(get_text_from_email(extract_content(file)))
		
	# Creating a count vector for each email.
	# All the stop words are removed.
	# max_df = 0.5 means that the word should not be present in more that 50% of the emails
	# min_df = 30 means word should occur atleast 30 times in all emails combined.
	cv = feature_extraction.text.CountVectorizer(stop_words='english', max_df=0.5, min_df=30)
	# As we are not providing vocabulary, we use the fit_transform function where vocab is automatically generated.
	vectors = cv.fit_transform(bodies).toarray()

	#Creating a dictionary of features and labels
	mails = {}
	mails['files'] = [os.path.split(file)[1] for file in files]
	mails['vectors'] =  list(vectors)
	mails['labels'] = labels

	# Creating training and testing features and labels
	x_train = list(vectors)[:3000]
	y_train = labels[:3000]
	x_test = list(vectors)[3000:]
	y_test = labels[3000:]

	# Initializing classifier
	classifier = svm.SVC(kernel='linear')
	print("Training dataset...")
	classifier.fit(x_train, y_train)

	# Predicting labels of both training and testing
	print("\nPredicting values in testing and training dataset...")
	to_predict = vectors.reshape(len(vectors), -1) #Reshaping array so that it is of valid input format
	predictions = classifier.predict(to_predict)

	# Adding predictins to the dictionary
	mails['predictions'] = predictions
	train_predictions = predictions[:3000]
	test_predictions = predictions[3000:]

	# Get respective scores
	test_scores = get_scores(y_test, test_predictions)
	train_scores = get_scores(y_train, train_predictions)
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
	df = pd.DataFrame(mails)
	df['result'] = np.where(df['predictions'] == df['labels'], "CORRECT", "WRONG")
	df.set_index('files', inplace=True)
	with open('SVMresults.txt', 'w') as f:
		f.write(df.drop('vectors', 1).to_string())
	print("Results file created: {}" . format(os.path.abspath('SVMresults.txt')))
	
main()

