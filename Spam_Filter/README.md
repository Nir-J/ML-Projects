# Spam-Filter
Basic spam filter using two methods: **Naive Bayes** and **Linear SVM**.

### Dataset:

- Training and testing data are taken from the CSDMC2010 SPAM corpus, which is one of the datasets for the data mining competition associated with ICONIP 2010.
- It is slightly modified as the testing dataset used here was originally a part of the training dataset. The original testing dataset is not used as they do not have labels.
- Read `main_readme.txt` for more information.

### Requirements:

- Python version 3.5+ and all the package requirements listed in `requirements.txt`
- Run `pip install -r requirements.txt` to install them all.

### Executing:

The script has default settings which enables it to run without user input. You can get more information in `NBreadme.txt` and `SVMreadme.txt`.

### NB.py

You can learn more about the algorithm implemented here: http://www.linuxjournal.com/article/6467

### SVM.py

I would suggest looking up count vectorizer as it is the only feature being used: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
