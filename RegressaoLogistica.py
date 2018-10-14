from sklearn import feature_extraction
import datetime
import SklearnEstudando as l
import  math
import preprocessamento as pre
from sklearn import metrics
import random
import nltk
#0.2 10 74
#0.3 10 75
#0.4 10 75
def recalcular(coeficiente,data, classe, prediction):
    return coeficiente + 0.3 * (classe - prediction) * prediction * (1 - prediction) * data

def coeficientes(coeficiente, data, previsao, classe):
    coeficiente[0] = recalcular(coeficiente[0], 1, classe, previsao)
    for i in range(0, len(coeficiente)):
        coeficiente[i] = recalcular(coeficiente[i], data[i - 1], classe, previsao)
    return coeficiente

def prever(coeficiente, data):
     out = coeficiente[0]
     for i in range (0, (len(data)-1)):
         out = out + coeficiente[i+1]*data[i]
     return  1 / (1 + math.exp(-(out)))

def acharCoeficiente(data, classes, rodadas):
 coeficiente = [0 for i in range(0, len(data[0])+1)]
 for n in range(0,rodadas):
    for i in range(0, len(data)):
     classe = classes[i]
     previsao = prever(coeficiente, data[i])
     coeficiente = coeficientes(coeficiente, data[i], previsao, classe)
 return  coeficiente

def treino(treino_doc, treino_classe, tipo, rodadas):
    #data = tipo.fit_transform(treino_doc)
    #data = data.toarray()

    data = pre.bag(treino_doc, tipo)
    print("Procurando coeficiente...")
    print(datetime.datetime.now())
    coeficiente = (acharCoeficiente(data, treino_classe, rodadas))
    print("Achou coeficiente:")

    return coeficiente

def teste(coeficientes, teste_doc, tipo):
    classes = []
    #tipo =  feature_extraction.text.CountVectorizer(stop_words='english')
    #teste = tipo.transform(teste_doc)
    #teste = teste.toarray()

    teste = pre.bag(teste_doc, tipo)
    acerto = 0
    total = len(teste)
    for i in range(0, len(teste)):
        x = (prever(coeficientes, teste[i]))
        if x < 0.5:
            classes.append(0)
        else:
            classes.append(1)
    return classes

def metrica(classe_teste, x, metrica):
    total = len(classe_teste)
    acerto = 0
    if metrica is 'acuracia':
        x = metrica + ": " + str(metrics.accuracy_score(y_true=classe_teste,y_pred=x))
    elif metrica is 'f-mesure':
        x = metrica + ": " + str(metrics.f1_score(y_true=classe_teste,y_pred=x))
    elif metrica is 'recall':
        x = metrica + ": " + str(metrics.recall_score(y_true=classe_teste,y_pred=x))
    elif metrica is "precisão":
        x = metrica + ": " + str(metrics.precision_score(y_true=classe_teste,y_pred=x))

    print(x)
    return x
def main(rodadas, tipo):
    #tipo = feature_extraction.text.CountVectorizer(stop_words='english')
    x = l.ler("doc_classes")
    doc = x[0]
    classes = x[1]
    print("treino")
    treino_doc, treino_classes, teste_doc, teste_classes = doc[0:800], classes[0:800], doc[800:1000], classes[800:1000]
    x = treino(treino_doc, treino_classes, tipo,rodadas)
    print("teste")
    labels = teste(x,teste_doc, tipo)
    return metrica(teste_classes, labels, 'acuracia')



#225
#binario 79
#main()


main(5  , 'tf')