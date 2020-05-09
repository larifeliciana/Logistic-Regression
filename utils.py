from sklearn import feature_extraction
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import linear_model
from sklearn import neighbors
import os
import pickle



def ler(endereco):
    arq = open(endereco, 'rb')
    lista = pickle.load(arq)
    return lista

def carregar(pasta):
    caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
    lista = []
    for i in caminhos:
        review = open(i, 'r')
        lista.append(review.read())
    return lista

def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)



def bag(tipo, treino_doc, teste_doc):

    if tipo is 'tfidf':
        tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english')
        x = tfidf.fit_transform(treino_doc)
        teste = tfidf.transform(teste_doc)


    elif tipo is 'tf':
        tf = feature_extraction.text.CountVectorizer(stop_words='english')
        x = tf.fit_transform(treino_doc)
        teste = tf.transform(teste_doc)

    elif tipo is 'bin':
        tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', binary=True)
        x = tfidf.fit_transform(treino_doc)
        teste = tfidf.transform(teste_doc)


    return x, teste



def modelo(tipo, metrica, treino_doc, teste_doc, treino_classe, teste_classe):
    acerto = 0
    if tipo is 'naiveM': #Naive Bayes Multinominal
        naive = naive_bayes.MultinomialNB()
        naive.fit(treino_doc, treino_classe)
        labels = naive.predict(teste_doc)

    elif tipo is 'logistic': #Regressão Logística
        logistic = linear_model.LogisticRegression()
        logistic.fit(treino_doc, treino_classe)
        labels = logistic.predict(teste_doc)


    elif tipo is 'knn': #K-NN
        knn = neighbors.KNeighborsClassifier(n_neighbors=70)
        knn.fit(treino_doc, treino_classe)
        labels = knn.predict(teste_doc)


    if metrica is 'acuracia':
            acerto = metrics.accuracy_score(teste_classe, labels)
    return acerto


#pq multinominal com entrada binária é melhor que bernuille
"""x = ler("doc_classes")
doc = x[0]
classes = x[1]
treino_doc, treino_classes, teste_doc, teste_classes = doc[0:800], classes[0:800], doc[800:1000], classes[800:1000]
x 
x = modelo('logistic', 'acuracia', x,y, treino_classes, teste_classes)
print(x)
"""