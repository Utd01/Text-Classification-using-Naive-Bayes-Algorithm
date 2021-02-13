import numpy as np
import sys
import json
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)


def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens)


def readinputs(path):
    x = {}
    y = {}
    i = 0 
    for load in open(path, mode = "r"):
        review = json.loads(load)
        x[i]=review["text"].lower()
        y[i]=int(review["stars"])
        i = i + 1
    return (x,y)

def draw_confusion(confusionMatrix):
    plt.imshow(confusionMatrix)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.set_cmap("Reds")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    plt.show()

def partA(tr,te):
    trX, trY = readinputs(tr)
    teX, teY = readinputs(te)

    # then make the dictionary
    store = {}
    numOccurences = [0, 0, 0, 0, 0]
    starViseWord = [0, 0, 0, 0, 0]
    for i in range(len(trX)):
        starForYi = trY[i]
        splitted_string = trX[i].split()
        starViseWord[starForYi-1] += len(splitted_string)
        numOccurences[starForYi-1] += 1
        for word in splitted_string:
            if word in store:
                store[word][starForYi-1] = store[word][starForYi-1] + 1
            else:
                store[word] = [1, 1, 1, 1, 1]
                store[word][starForYi-1] = store[word][starForYi-1]  + 1
    # the dictionary is present in store
    #  and all the other important details are avaiable in other variables.
    total_train = len(trX)
    total_test = len(teX)
    total_words = len(store)

    for word in store:
        for i in range(5):
            temp1 =float(store[word][i])
            temp2 = float(total_words+starViseWord[i])
            store[word][i] = math.log(temp1/(temp2))

    # the make prediction on test data
    sPredict = [0.0,0.0,0.0,0.0,0.0]
    prediction = [0]* total_test
    for i in range(5):
        t1= float(numOccurences[i])
        t2= float(total_train)
        sPredict[i]= math.log((t1)/(t2))  

    for i in range(total_test):
        p = [0.0,0.0,0.0,0.0,0.0]
        for j in range(5):
            p[j]= p[j]+ sPredict[j]
            aftersplit = teX[i].split()
            for word in aftersplit:
                if word in store:
                    p[j]= p[j]+ store[word][j]
                else:
                    tempo = float(total_words+ starViseWord[j])
                    p[j] = p[j] + math.log(1.0/(tempo))
        ans = 0
        maxval =  p[0]
        for i2 in range(5):
            if(p[i2]>maxval):
                maxval = p[i2]
                ans = i2
        prediction[i] = 1 + ans
    

    # calculating accuracy
    
    print("Finding Accuracy:\n")
    match = 0
    for i in range(total_test):
        if(teY[i] == prediction[i]):
            match = match + 1
    
    matchPercent = (float(match))/(float(total_test))
    print(matchPercent*100)

    # upon running the program:
    # accuracy over test data = 61.68877787582824 %
    # accuracy over training data = 68.6790484452355 %

    # making confusion matrix and finding f1 score

    array_testY = [0]*total_test
    for i in range(total_test):
        array_testY[i] = teY[i]
    
    # confusionMatrix = confusion_matrix(array_testY, prediction)
    # f1_sc = f1_score(array_testY, prediction, average=None)
    # print("F1 Score")
    # print(f1_sc)
    # print("Confusion Matrix")
    # print(confusionMatrix)
    # macro_f1Score = f1_score(array_testY, prediction, average='macro')
    # print("Macro F1 Score")
    # print(macro_f1Score)
    # draw_confusion(confusionMatrix)

def partD(tr,te):
    trX, trY = readinputs(tr)
    teX, teY = readinputs(te)
    store = {}
    numOccurences = [0, 0, 0, 0, 0]
    starViseWord = [0, 0, 0, 0, 0]
    for i in range(len(trX)):
        starForYi = trY[i]
        splitted_string = getStemmedDocuments(trX[i],True)
        starViseWord[starForYi-1] += len(splitted_string)
        numOccurences[starForYi-1] += 1
        for word in splitted_string:
            if word in store:
                store[word][starForYi-1] = store[word][starForYi-1] + 1
            else:
                store[word] = [1, 1, 1, 1, 1]
                store[word][starForYi-1] = store[word][starForYi-1]  + 1

    total_train = len(trX)
    total_test = len(teX)
    total_words = len(store)

    for word in store:
        for i in range(5):
            temp1 =float(store[word][i])
            temp2 = float(total_words+starViseWord[i])
            store[word][i] = math.log(temp1/(temp2))

    # the make prediction on test data
    sPredict = [0.0,0.0,0.0,0.0,0.0]
    prediction = [0]* total_test
    for i in range(5):
        t1= float(numOccurences[i])
        t2= float(total_train)
        sPredict[i]= math.log((t1)/(t2))  

    for i in range(total_test):
        p = [0.0,0.0,0.0,0.0,0.0]
        for j in range(5):
            p[j]= p[j]+ sPredict[j]
            aftersplit = getStemmedDocuments(teX[i],True)
            for word in aftersplit:
                if word in store:
                    p[j]= p[j]+ store[word][j]
                else:
                    tempo = float(total_words+ starViseWord[j])
                    p[j] = p[j] + math.log(1.0/(tempo))
        ans = 0
        maxval =  p[0]
        for i2 in range(5):
            if(p[i2]>maxval):
                maxval = p[i2]
                ans = i2
        prediction[i] = 1 + ans
    print("Finding Accuracy:\n")
    match = 0
    for i in range(total_test):
        if(teY[i] == prediction[i]):
            match = match + 1
    
    matchPercent = (float(match))/(float(total_test))
    print(matchPercent*100)

    # upon running the program:
    # accuracy over test data = 61.68877787582824 %
    # accuracy over training data = 68.6790484452355 %

    # making confusion matrix and finding f1 score

    array_testY = [0]*total_test
    for i in range(total_test):
        array_testY[i] = teY[i]
    
    # confusionMatrix = confusion_matrix(array_testY, prediction)
    # f1_sc = f1_score(array_testY, prediction, average=None)
    # print("F1 Score")
    # print(f1_sc)
    # print("Confusion Matrix")
    # print(confusionMatrix)
    # macro_f1Score = f1_score(array_testY, prediction, average='macro')
    # print("Macro F1 Score")
    # print(macro_f1Score)
    # draw_confusion(confusionMatrix)

def make_double(splitted_string):
    ret =[]
    s = len(splitted_string)
    word =""
    i =0
    while(i<=s-1):
        word = word + splitted_string[i]
        i = i+1
        if(i<=s-1):
            word = word +" "
            word = word + splitted_string[i]
            ret.append(word)
            i = i+1
            word=""
    if(word!=""):
        ret.append(word)
    return ret




def partBigrams(tr,te,output):
    trX, trY = readinputs(tr)
    teX, teY = readinputs(te)

    # then make the dictionary
    store = {}
    numOccurences = [0, 0, 0, 0, 0]
    starViseWord = [0, 0, 0, 0, 0]
    for i in range(len(trX)):
        starForYi = trY[i]
        splitted_string = trX[i].split()
        splitted_string = make_double(splitted_string)
        starViseWord[starForYi-1] += len(splitted_string)
        numOccurences[starForYi-1] += 1
        for word in splitted_string:
            if word in store:
                store[word][starForYi-1] = store[word][starForYi-1] + 1
            else:
                store[word] = [1, 1, 1, 1, 1]
                store[word][starForYi-1] = store[word][starForYi-1]  + 1
    # the dictionary is present in store
    #  and all the other important details are avaiable in other variables.
    total_train = len(trX)
    total_test = len(teX)
    total_words = len(store)

    for word in store:
        for i in range(5):
            temp1 =float(store[word][i])
            temp2 = float(total_words+starViseWord[i])
            store[word][i] = math.log(temp1/(temp2))

    # the make prediction on test data
    sPredict = [0.0,0.0,0.0,0.0,0.0]
    prediction = [0]* total_test
    for i in range(5):
        t1= float(numOccurences[i])
        t2= float(total_train)
        sPredict[i]= math.log((t1)/(t2))  

    for i in range(total_test):
        p = [0.0,0.0,0.0,0.0,0.0]
        for j in range(5):
            p[j]= p[j]+ sPredict[j]
            aftersplit = teX[i].split()
            aftersplit = make_double(aftersplit)
            for word in aftersplit:
                if word in store:
                    p[j]= p[j]+ store[word][j]
                else:
                    tempo = float(total_words+ starViseWord[j])
                    p[j] = p[j] + math.log(1.0/(tempo))
        ans = 0
        maxval =  p[0]
        for i2 in range(5):
            if(p[i2]>maxval):
                maxval = p[i2]
                ans = i2
        prediction[i] = 1 + ans
    
    np.savetxt(output, prediction, fmt="%d", delimiter="\n")
    # calculating accuracy
    
    # print("Finding Accuracy:\n")
    # match = 0
    # for i in range(total_test):
    #     if(teY[i] == prediction[i]):
    #         match = match + 1
    
    # matchPercent = (float(match))/(float(total_test))
    # print(matchPercent*100)

    # upon running the program:
    # accuracy over test data = 61.68877787582824 %
    # accuracy over training data = 68.6790484452355 %

    # making confusion matrix and finding f1 score

    # array_testY = [0]*total_test
    # for i in range(total_test):
    #     array_testY[i] = teY[i]
    
    # confusionMatrix = confusion_matrix(array_testY, prediction)
    # f1_sc = f1_score(array_testY, prediction, average=None)
    # print("F1 Score")
    # print(f1_sc)
    # print("Confusion Matrix")
    # print(confusionMatrix)
    # macro_f1Score = f1_score(array_testY, prediction, average='macro')
    # print("Macro F1 Score")
    # print(macro_f1Score)
    # draw_confusion(confusionMatrix)


def make_triple(splitted_string):
    ret =[]
    s = len(splitted_string)
    word =""
    i =0
    while(i<=s-1):
        word = word + splitted_string[i]
        i = i+1
        if(i<=s-1):
            word = word +" "
            word = word + splitted_string[i]
            i = i+1
            if(i<=s-1):
                word = word +" "
                word = word + splitted_string[i]
                ret.append(word)
                i = i+1
                word=""
    if(word!=""):
        ret.append(word)
    return ret


def partTrigrams(tr,te):
    trX, trY = readinputs(tr)
    teX, teY = readinputs(te)

    # then make the dictionary
    store = {}
    numOccurences = [0, 0, 0, 0, 0]
    starViseWord = [0, 0, 0, 0, 0]
    for i in range(len(trX)):
        starForYi = trY[i]
        splitted_string = trX[i].split()
        splitted_string = make_triple(splitted_string)
        starViseWord[starForYi-1] += len(splitted_string)
        numOccurences[starForYi-1] += 1
        for word in splitted_string:
            if word in store:
                store[word][starForYi-1] = store[word][starForYi-1] + 1
            else:
                store[word] = [1, 1, 1, 1, 1]
                store[word][starForYi-1] = store[word][starForYi-1]  + 1
    # the dictionary is present in store
    #  and all the other important details are avaiable in other variables.
    total_train = len(trX)
    total_test = len(teX)
    total_words = len(store)

    for word in store:
        for i in range(5):
            temp1 =float(store[word][i])
            temp2 = float(total_words+starViseWord[i])
            store[word][i] = math.log(temp1/(temp2))

    # the make prediction on test data
    sPredict = [0.0,0.0,0.0,0.0,0.0]
    prediction = [0]* total_test
    for i in range(5):
        t1= float(numOccurences[i])
        t2= float(total_train)
        sPredict[i]= math.log((t1)/(t2))  

    for i in range(total_test):
        p = [0.0,0.0,0.0,0.0,0.0]
        for j in range(5):
            p[j]= p[j]+ sPredict[j]
            aftersplit = teX[i].split()
            aftersplit = make_triple(aftersplit)
            for word in aftersplit:
                if word in store:
                    p[j]= p[j]+ store[word][j]
                else:
                    tempo = float(total_words+ starViseWord[j])
                    p[j] = p[j] + math.log(1.0/(tempo))
        ans = 0
        maxval =  p[0]
        for i2 in range(5):
            if(p[i2]>maxval):
                maxval = p[i2]
                ans = i2
        prediction[i] = 1 + ans
    

    # calculating accuracy
    
    print("Finding Accuracy:\n")
    match = 0
    for i in range(total_test):
        if(teY[i] == prediction[i]):
            match = match + 1
    
    matchPercent = (float(match))/(float(total_test))
    print(matchPercent*100)

    # upon running the program:
    # accuracy over test data = 61.68877787582824 %
    # accuracy over training data = 68.6790484452355 %

    # making confusion matrix and finding f1 score

    array_testY = [0]*total_test
    for i in range(total_test):
        array_testY[i] = teY[i]
    
    # confusionMatrix = confusion_matrix(array_testY, prediction)
    # f1_sc = f1_score(array_testY, prediction, average=None)
    # print("F1 Score")
    # print(f1_sc)
    # print("Confusion Matrix")
    # print(confusionMatrix)
    # macro_f1Score = f1_score(array_testY, prediction, average='macro')
    # print("Macro F1 Score")
    # print(macro_f1Score)
    # draw_confusion(confusionMatrix)



def make_four(splitted_string):
    ret =[]
    s = len(splitted_string)
    word =""
    i =0
    while(i<=s-1):
        word = word + splitted_string[i]
        i = i+1
        if(i<=s-1):
            word = word +" "
            word = word + splitted_string[i]
            i = i+1
            if(i<=s-1):
                word = word +" "
                word = word + splitted_string[i]
                i= i+1
                if(i<=s-1):
                    word = word +" "
                    word = word + splitted_string[i]
                    ret.append(word)
                    i = i+1
                    word=""
    if(word!=""):
        ret.append(word)
    return ret


def partQuaternarygrams(tr,te):
    trX, trY = readinputs(tr)
    teX, teY = readinputs(te)

    # then make the dictionary
    store = {}
    numOccurences = [0, 0, 0, 0, 0]
    starViseWord = [0, 0, 0, 0, 0]
    for i in range(len(trX)):
        starForYi = trY[i]
        splitted_string = trX[i].split()
        splitted_string = make_four(splitted_string)
        starViseWord[starForYi-1] += len(splitted_string)
        numOccurences[starForYi-1] += 1
        for word in splitted_string:
            if word in store:
                store[word][starForYi-1] = store[word][starForYi-1] + 1
            else:
                store[word] = [1, 1, 1, 1, 1]
                store[word][starForYi-1] = store[word][starForYi-1]  + 1
    # the dictionary is present in store
    #  and all the other important details are avaiable in other variables.
    total_train = len(trX)
    total_test = len(teX)
    total_words = len(store)

    for word in store:
        for i in range(5):
            temp1 =float(store[word][i])
            temp2 = float(total_words+starViseWord[i])
            store[word][i] = math.log(temp1/(temp2))

    # the make prediction on test data
    sPredict = [0.0,0.0,0.0,0.0,0.0]
    prediction = [0]* total_test
    for i in range(5):
        t1= float(numOccurences[i])
        t2= float(total_train)
        sPredict[i]= math.log((t1)/(t2))  

    for i in range(total_test):
        p = [0.0,0.0,0.0,0.0,0.0]
        for j in range(5):
            p[j]= p[j]+ sPredict[j]
            aftersplit = teX[i].split()
            aftersplit = make_four(aftersplit)
            for word in aftersplit:
                if word in store:
                    p[j]= p[j]+ store[word][j]
                else:
                    tempo = float(total_words+ starViseWord[j])
                    p[j] = p[j] + math.log(1.0/(tempo))
        ans = 0
        maxval =  p[0]
        for i2 in range(5):
            if(p[i2]>maxval):
                maxval = p[i2]
                ans = i2
        prediction[i] = 1 + ans
    

    # calculating accuracy
    
    print("Finding Accuracy:\n")
    match = 0
    for i in range(total_test):
        if(teY[i] == prediction[i]):
            match = match + 1
    
    matchPercent = (float(match))/(float(total_test))
    print(matchPercent*100)

    # upon running the program:
    # accuracy over test data = 61.68877787582824 %
    # accuracy over training data = 68.6790484452355 %

    # making confusion matrix and finding f1 score

    array_testY = [0]*total_test
    for i in range(total_test):
        array_testY[i] = teY[i]
    
    # confusionMatrix = confusion_matrix(array_testY, prediction)
    # f1_sc = f1_score(array_testY, prediction, average=None)
    # print("F1 Score")
    # print(f1_sc)
    # print("Confusion Matrix")
    # print(confusionMatrix)
    # macro_f1Score = f1_score(array_testY, prediction, average='macro')
    # print("Macro F1 Score")
    # print(macro_f1Score)
    # draw_confusion(confusionMatrix)


def partB(tr,te):
    trX, trY = readinputs(tr)
    teX, teY = readinputs(te)
    total_test = len(teX)
    array_testY = [0]*total_test
    for i in range(total_test):
        array_testY[i] = teY[i]
    
    rand_prediction = np.random.random_integers(1, 5, (total_test, 1))
    

    # max occurences 
    numOccurences = [0, 0, 0, 0, 0]
    for i in range(total_test):
        starForYi = trY[i]
        numOccurences[starForYi-1] += 1
    maxStars = 1 +np.argmax(numOccurences)
    max_prediction = [maxStars]*len(teX)

    
    matchRandom = 0
    matchMax = 0
    for i in range(total_test):
        if(teY[i] == rand_prediction[i]):
            matchRandom = matchRandom + 1
        if(teY[i] == max_prediction[i]):
            matchMax = matchMax + 1
    
    matchPercent = (float(matchRandom))/(float(total_test))
    print("Accuracy in Random pickup:")
    print(matchPercent*100)
    matchPercent = (float(matchMax))/(float(total_test))
    print("Accuracy in Max pickup:")
    print(matchPercent*100)

    # confusionMatrix = confusion_matrix(array_testY, rand_prediction)
    # print("Confusion Matrix for random prediction")
    # print(confusionMatrix)
    # draw_confusion(confusionMatrix)
    # confusionMatrix = confusion_matrix(array_testY, max_prediction)
    # print("Confusion Matrix for majority prediction")
    # print(confusionMatrix)
    # draw_confusion(confusionMatrix)  

def main():
    tr = sys.argv[1]
    te = sys.argv[2]
    output = sys.argv[3]
    # partA(tr,te)
    # partB(tr,te)
    # partD(tr,te)
    partBigrams(tr,te,output)
    # partTrigrams(tr,te)
    # partQuaternarygrams(tr,te)

        



if __name__ == "__main__":
    main()
