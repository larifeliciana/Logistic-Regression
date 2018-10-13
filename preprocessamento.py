import math
import spacy

def preprocessamento(lista): ##Lê uma pasta com os arquivos de texto e retorna uma lista dos tokens com a classificação na última posição
    #lista = carregar(pasta) #Carrega todos os arquivos de uma pasta para uma lista onde cada elemento é um texto
    nlp = spacy.load('en')
    processada = []
    for n in range(0,len(lista)):
        if n % 10 is 0:
            print(n)
        processada.append(preprocessamentoS(lista[n], nlp))
    return processada

def preprocessamentoS(string, nlp):
    string = nlp(string)

    string = [token.lemma_ for token in string if (not token.is_stop) and ((not token.is_punct or token.text == '!') and  token.is_alpha)]
    return string


def dicionario(docs):
    lista = []
    for i in docs:
        lista = lista + i
    return set(lista)

#CONTA EM QUANTOS DOCUMENTOS DA CLASSE A PALAVRA APARECE
def conta(palavra, documentos, classe):
    soma = 0
    b= False
    if classe is None:
        b = True
    for i in documentos:
        if b or i[-1] is classe:
            if palavra in i:
                soma = soma + 1

    return soma

def idf(docs):
    vetor = {}
    features = dicionario(docs)
    t = len(docs)
    for i in features:
        x =  conta(i, docs, None)
        idf = (t)/x
        vetor.update({i:idf})
    return vetor
def bag(docs, tipo):
    bag = []
    features = dicionario(docs)
    idf1 = idf(docs)
    print("bag fazendo")
    for i in docs:
        bag.append([])
        for j in features:
            if tipo is 'tf':
                  bag[-1].append(i.count(j))

            elif tipo is 'tfidf':
                if j in i:
                    bag[-1].append(idf1[j])
                else: bag[-1].append(0)
            elif tipo is 'bin':
                if j in i:
                    bag[-1].append(1)
                else: bag[-1].append(0)

    return bag

