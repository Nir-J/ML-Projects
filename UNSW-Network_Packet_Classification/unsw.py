import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf 
import os

# Function to encode string features
def encode_features_and_labels(training, testing):

	# To encode string  labels into numbers
	le = LabelEncoder()

	# Creates new dummy columns from each unique string in a particulat feature
	training = pd.get_dummies(data=training, columns=['proto', 'service', 'state'])
	testing = pd.get_dummies(data=testing, columns=['proto', 'service', 'state'])

	# Making sure that the training features are same as testing features.
	# The training dataset has more unique protocols and states, therefore number \
	# of dummy columns will be different in both. We make it the same.
	traincols = list(training.columns.values)
	testcols = list(testing.columns.values)

	# For those in training but not in testing
	for col in traincols:
		# If a column is missing in the testing dataset, we add it
		if col not in testcols:
			testing[col] = 0
			testcols.append(col)
	# For those in testing but not in training
	for col in testcols:
		if col not in traincols:
			training[col] = 0
			traincols.append(col)


	# Moving the labels and categories to the end and making sure features are in the same order
	traincols.pop(traincols.index('attack_cat'))
	traincols.pop(traincols.index('label'))
	training = training[traincols+['attack_cat', 'label']]
	testing = testing[traincols+['attack_cat', 'label']]

	# Encoding the category names into numbers so that they can be one hot encoded later.
	training['attack_cat'] = le.fit_transform(training['attack_cat'])
	testing['attack_cat'] = le.fit_transform(testing['attack_cat'])

	# Returning modified dataframes and the vocabulary of labels for inverse transform
	return (training, testing, le)

# Parameters
training_epochs = 20
batch_size = 9
start_rate = 0.0002

# Network Parameters
n_hidden_1 = 100 # 1st layer number of neurons
n_hidden_2 = 50 # 2nd layer number of neurons
n_features = 196 # There are 194 different features for each packet.
n_classes = 10 # There are 9 different types of malicious packets + Normal

########### Defining tensorflow computational graph ###########

# tf Graph input
# Features
X = tf.placeholder(tf.float32, [None, n_features])
# Labels
Y = tf.placeholder(tf.int32, [None,])
# decay step for learning rate decay
decay_step = tf.placeholder(tf.int32)


# Create model
def deep_neural_network(x):

    # Hidden fully connected layer with 100 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu)
    # Hidden fully connected layer with 50 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, n_classes)
    return out_layer

# Construct model
logits = deep_neural_network(X)

# Define loss and optimizer
# Converting categories into one hot labels
labels = tf.one_hot(indices=tf.cast(Y, tf.int32), depth=n_classes)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    					logits=logits, labels=labels))
global_step = tf.Variable(0, trainable=False)

# Using a learning rate which has polynomial decay
starter_learning_rate = start_rate
end_learning_rate = 0.00005 # we will use a polynomial decay to reach learning this learning rate.29
decay_steps = decay_step
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)
# Using adam optimizer to reduce loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

# Model for testing
pred = tf.nn.softmax(logits)  # Apply softmax to logits

# Model for prediction: Used to just return predicted values
prediction=tf.argmax(pred,1)

########## END of model ############

########## Reading and processing input datasets #########

# Default values. 
train_set = 'UNSW_NB15_training-set.csv'
test_set = 'UNSW_NB15_testing-set.csv'

# Comment if you need to hardcode path
# train_set = input("Enter training dataset: ")
# test_set = input("Enter testing dataset: ")
# if not os.path.exists(train_set) or not os.path.exists(test_set):
# 	print("Files not found")
# 	exit()
# Read data using pandas
training = pd.read_csv(train_set, index_col='id')
testing = pd.read_csv(test_set, index_col='id')

# Encoding string columns
training, testing, le = encode_features_and_labels(training, testing)

# Normalising all numerical features:
cols_to_normalise = list(training.columns.values)[:39]
training[cols_to_normalise] = training[cols_to_normalise].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
testing[cols_to_normalise] = testing[cols_to_normalise].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

######## End of preprocessing #######

######## Training and testing #########

def get_accuracy(df):

	# Calculate accuracy for label classification
	categories = prediction.eval(feed_dict={X: df.iloc[:, 0:-2]}) # Getting back the predictions

	# Function to convert categories back into binary labels
	f = lambda x: 0 if le.inverse_transform(x) == "Normal" else 1

	# Prepating the necessary predictions and labels for comparision; converting categories to normal/malicious
	binary_prediction = np.fromiter((f(xi) for xi in categories), categories.dtype, count=len(categories))
	binary_labels = df.iloc[:, -1].values
	
	# Compating predictions and labels to calculate accuracy
	correct_labels = tf.equal(binary_prediction, binary_labels)
	label_accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))
	result = label_accuracy.eval()
	print("Label accuracy: {:.2f}%".format(result*100))

	# Calculate accuracy for category classification
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = accuracy.eval({X: df.iloc[:, 0:-2], Y: df.iloc[:,-2]})
	print("Category accuracy: {:.2f}%".format(result*100))

def train_and_test_model(training, testing):
	with tf.Session() as sess:
		sess.run(init)

		# Training cycle
		for epoch in range(training_epochs):
			# Shuffling dataset before training
			df = training.sample(frac=1)
			avg_cost = 0.
			total_data = df.index.shape[0] 
			num_batches = total_data // batch_size + 1
			i = 0
			# Loop over all batches
			while i < total_data:
				batch_x = df.iloc[i:i+batch_size, 0:-2].values
				batch_y = df.iloc[i:i+batch_size, -2].values # Last two columns are categories and labels
				i += batch_size
				# Run optimization op and cost op (to get loss value)
				_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
				                                                Y: batch_y,
				                                                decay_step: num_batches * training_epochs})
				# Compute average loss
				avg_cost += c / num_batches
			# Display logs per epoch step
			print("Epoch: {:04} | Cost={:.9f}".format(epoch+1, avg_cost))
			get_accuracy(testing)
			print()
		print("Training complete")

		print("Training results: ")
		get_accuracy(training)
		print("Testing results: ")
		get_accuracy(testing)


# Training the model after shuffling the data.
train_and_test_model(training, testing)




