import os, re, time, gensim, numpy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# Preprocessing

path1 = "./dataset/train/neg"
path2 = "./dataset/train/pos"
path3 = "./dataset/test/neg"
path4 = "./dataset/test/pos"
files1 = os.listdir(path1)
files2 = os.listdir(path2)
files3 = os.listdir(path3)
files4 = os.listdir(path4)

files0 = open("./dataset/glove/glove.6B.100d.txt", 'r') 
data1 ,data2 = [], []
maxfeatures = 4000
stop_words = set(stopwords.words('english'))

def preprocess_(files, path):
    data_ = []
    for f in files:
	input_ = open(os.path.join(path,f),'r').read()
	output_ = os.path.splitext(f)[0]
	temp_, j = [], 0
	for i in range(len(output_)):
	    if output_[i] == '_':
		output_ = output_[i+1:] 
		break
	input_ = re.sub(r'<br /><br />',' ', input_)
	input_ = re.sub(r'[^a-zA-Z]+',' ', input_)
	data_.append([input_, output_])
    return data_

#####
data1 = preprocess_(files1, path1)
data2 = preprocess_(files2, path2)
data3 = preprocess_(files3, path3)
data4 = preprocess_(files4, path4)

#####

#WordRepresentations

def glove(files):
    model = {}
    for line in files:
        splitLine = line.split()
        word = splitLine[0]
        embedding = numpy.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model

def doc_vec(data_, data__, data___, data____):
    data_.extend(data__)
    data_.extend(data___)
    data_.extend(data____)
    vector = [x[0] for x in data_]
    return vector

def vectors(vector, files, index):
    dictionary = []

    if index == 1:
        vectorizer = CountVectorizer(stop_words = 'english', min_df = 10, max_df = 0.5, max_features = maxfeatures, binary = True)
        print 'Binary Bag Of Words'

    elif index == 2:
	vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 10, max_df = 0.5, max_features = maxfeatures, use_idf = False)
        print 'Normalized Tf'

    elif index == 3:
	vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 10, max_df = 0.5, max_features = maxfeatures)
	print 'TfIdf'

    elif index == 4:
        vector_token = [word_tokenize(x) for x in vector]
	wvec = Word2Vec(vector_token)
        tfidf_vec = TfidfVectorizer(stop_words = 'english', min_df = 10, max_df = 0.5,  max_features = maxfeatures)
        tfidf = tfidf_vec.fit_transform(vector)
        vocab1 = wvec.wv.vocab
        vocab2 = tfidf_vec.get_feature_names()
        dictionary = numpy.zeros((len(vocab2), 100))
        for x in range(len(vocab2)):
            y = vocab2[x]
            if y in vocab1:
                dictionary[x] = wvec[y]
        dictionary = tfidf.dot(dictionary)
	print 'Averaged Word2Vec'	

    elif index == 5:
	vector_token = [word_tokenize(x) for x in vector]
        glove_vec = glove(files)
        tfidf_vec = TfidfVectorizer(stop_words = 'english', min_df = 10, max_df = 0.5,  max_features = maxfeatures)
        tfidf = tfidf_vec.fit_transform(vector)
        vocab2 = tfidf_vec.get_feature_names()
        dictionary = numpy.zeros((len(vocab2), 100))
        for x in range(len(vocab2)):
            y = vocab2[x]
            if y in glove_vec:
                dictionary[x] = glove_vec[y]
        dictionary = tfidf.dot(dictionary)
        print 'Averaged GloveVec'

    elif index == 6:
        vector_ = []
        for count,x in enumerate(vector):
	    stops = [x for x in word_tokenize(x) if x not in stop_words]
            word = TaggedDocument(words = stops, tags = [count])
            vector_.append(word)

        temp = Doc2Vec(vector_, max_vocab_size =  maxfeatures, dm = 0)
        print 'Doc2Vec (Paragraph Vector)'
        for i in range(0, 50000):
            dictionary.append(temp.docvecs[i])

    else: 
	return False
    if index <= 3:
	    dictionary = vectorizer.fit_transform(vector).toarray()	
    return dictionary	


#Classification models

def naive_bayes(input_, input__, output_):
	gnb =  GaussianNB()
	y_pred = gnb.fit(input_, output_).predict(input__)
	return numpy.linalg.norm(numpy.subtract(output_,y_pred),1)
	
def log_regression(input_, input__, output_):
        lr =  LogisticRegression()
        y_pred = lr.fit(input_, output_).predict(input__)
        return numpy.linalg.norm(numpy.subtract(output_,y_pred),1)

def svm_(input_, input__, output_):
        clf =  svm.LinearSVC()
        y_pred = clf.fit(input_, output_).predict(input__)
        return numpy.linalg.norm(numpy.subtract(output_,y_pred),1)

def mlp_classifier(input_, input__, output_):
        nn =  MLPClassifier(tol = 0.001)
        y_pred = nn.fit(input_, output_).predict(input__)
        return numpy.linalg.norm(numpy.subtract(output_,y_pred),1)


def rnn_classifier(input_, input__, output_):
        trainX = numpy.reshape(input_, (input_.shape[0], 2, maxfeatures/2))
        testX = numpy.reshape(input__, (input__.shape[0], 2, maxfeatures/2))
        model = Sequential()
        model.add(SimpleRNN(4, input_shape=(2, maxfeatures/2)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, output_, epochs=100, batch_size=32, verbose=2)    
        y_pred = model.predict(testX)
        return numpy.linalg.norm(numpy.subtract(output_, y_pred),1)


##### Execution

def call(data1, data2, data3, data4, files):
	data_ = doc_vec(data1, data2, data3, data4)
        y_out = numpy.concatenate([numpy.zeros(12500),numpy.ones(12500)])
	for index in range(1,2):
		words = vectors(data_, files, index)	
		
		print 'Classifiers:'
                """
		nb1 = naive_bayes(words[:25000], words[25000:], y_out)
		print 'NB = ', "%.2f" %(100 - nb1/250.0)

		lr1 = log_regression(words[:25000], words[25000:], y_out)
		print 'LR = ', "%.2f" %(100 - lr1/250.0)

		svm1 = svm_(words[:25000], words[25000:], y_out)
		print 'SVM = ', "%.2f" %(100 - svm1/250.0)

		nn1 = mlp_classifier(words[:25000], words[25000:], y_out)
		print 'NN = ', "%.2f" %(100 - nn1/250.0)
                """
                rnn1 = rnn_classifier(words[:25000], words[25000:], y_out)
                print 'RNN = ', "%.2f" %(100 - rnn1/250.0)
                print '___ ___'


#####
print 'Execution:'
call(data1, data2, data3, data4, files0)

#####
