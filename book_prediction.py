# Predict book genre

# Importing the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# load the dataset
df = pd.read_csv("E:/Users/EDriveAkshatRaj/NLTK/Predict Book Genre/data/data.csv")

df =df.drop(columns = 'index')


# Clean the dataset
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation

corpus = []
stop_words = stopwords.words('english')
text = re.sub("[^A-Za-z]", " ", df.iloc[0, 2])
text = text.lower()
text = text.split()
text = [word for word in text if not word in stop_words]
ps = PorterStemmer()
text = [ps.stem(word) for word in text]
text = " ".join(text)


original_0 = df.iloc[0, 2]


# Clean the text by removing stopwords, punctuation, lower
def preprocess_text(data: pd.DataFrame, columns: str=None):
    if columns not in data.columns:
        raise Exception('This column does not exist in the dataframe')
    corpus = []
    stop_words = stopwords.words('english')
    for i in range(0, len(data)):
        summary = re.sub("[^A-Za-z]", " ", df[columns][i])
        summary = summary.lower()
        summary = summary.split()
        summary = [word for word in summary if not word in stop_words]
        ps = PorterStemmer()
        summary = [ps.stem(word) for word in summary]
        summary = " ".join(summary)
        corpus.append(summary)
    return corpus


corpus = preprocess_text(data=df, columns='summary')


# Converting the text to Bag of Words using TfidfVectorizer
tf_vectorizer =  TfidfVectorizer(max_features=38000)
X_tf = tf_vectorizer.fit_transform(corpus).toarray()


# Target variable
genre = df['genre'].value_counts()

encoder = LabelEncoder()
y_tf = encoder.fit_transform(df['genre'])



class_labels  = {'crime':0, 'fantasy':1, 'history':2, 'horror':3, 'psychology':4, 'romance':5,
       'science':7, 'sports':7, 'thriller':9, 'travel':9}
class_labels
# Seprating the dataset into training and test set
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tf, y_tf, test_size=0.25, random_state=0)


# Training the model using different models

models = {"Decision Tree" : DecisionTreeClassifier(),
            "Xgboost": XGBClassifier(),
            "RandomForest": RandomForestClassifier(),
            "Gaussian": GaussianNB(),
            "Multinomial": MultinomialNB(),
            "Bernoulli": BernoulliNB(),
            'Support Vector Machine': SVC()}

accuracy = {}
recall = {}
confusion = {}
precision = {}
for name, model in models.items():
    classifier = model.fit(X_train_tf, y_train_tf)
    y_pred = classifier.predict(X_test_tf)
    ac = accuracy_score(y_test_tf, y_pred)
    cm = confusion_matrix(y_test_tf, y_pred)
    re = recall_score(y_test_tf, y_pred, average='macro')
    pre = precision_score(y_test_tf, y_pred, average='macro')
    accuracy[name] = ac
    recall[name] = re
    confusion[name] = cm
    precision[name] = pre


        

    
# Use XgboostClassifier
xg = XGBClassifier()
xg.fit(X_train_tf, y_train_tf)

y_pred = xg.predict(X_test_tf)

ac_xg = accuracy_score(y_test_tf, y_pred)
print(xg)


# Use BernoulliNB
bn = BernoulliNB()
bn.fit(X_train_tf, y_train_tf)

y_pred = bn.predict(X_test_tf)

ac_bn = accuracy_score(y_test_tf, y_pred)
ac_bn
print(ac_bn)



encoder.inverse_transform()
