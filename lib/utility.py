import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from lib.args import Args
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def getSentenceForUnigram(ngram:str)->str:
    firstType = ngram.split('type:')[1].split(',')[0].strip()
    firstName = ngram.split('name:')[1].split(',')[0].strip()
    nextNode = ngram.split('[NN]')[1].strip()
    newFormat = firstType + ': ' + firstName + ' [NN] ' + nextNode + '\n'
    newFormat = newFormat.replace(': -,', '.').replace(': -', '.')
    return newFormat

def getSentenceForBigram(ngram:str)->str:
    firstType = ngram.split('type:')[1].split(',')[0].strip()
    firstName = ngram.split('name:')[1].split(',')[0].strip()
    secondType = ngram.split('type:')[2].split(',')[0].strip()
    secondName = ngram.split('name:')[2].split(',')[0].strip()
    nextNode = ngram.split('[NN]')[1].strip()
    newFormat = firstType + ': ' + firstName + ', ' + secondType + ': ' + secondName  + ' [NN] ' + nextNode + '\n'
    newFormat = newFormat.replace(': -,', '.').replace(': -', '.')
    return newFormat

def getSentenceForTrigram(ngram:str)->str:
    firstType = ngram.split('type:')[1].split(',')[0].strip()
    firstName = ngram.split('name:')[1].split(',')[0].strip()
    secondType = ngram.split('type:')[2].split(',')[0].strip()
    secondName = ngram.split('name:')[2].split(',')[0].strip()
    thirdType = ngram.split('type:')[3].split(',')[0].strip()
    thirdName = ngram.split('name:')[3].split(',')[0].strip()
    nextNode = ngram.split('[NN]')[1].strip()
    # newFormat = firstType + ': ' + firstName + ', ' + secondType + ': ' + secondName + ', ' + thirdType + ': ' + thirdName + ' [NN] ' + nextNode + '\n'
    newFormat = firstType + ': ' + firstName + ', ' + secondType + ': ' + secondName + ', ' + thirdType + ': ' + thirdName + ' [next node label] ' + nextNode.split(':')[1] + '\n'
    newFormat = newFormat.replace(': -,', '.').replace(': -', '.')
    return newFormat


def getSentenceForNgram(ngram:str)->str:
    if ngram.count('[AN]')==1: #it's a unigram        
        return getSentenceForUnigram(ngram)
    elif ngram.count('[AN]')==2: #it's a bigram
        return getSentenceForBigram(ngram)
    elif ngram.count('[AN]')==3: #it's a trigram
        return getSentenceForTrigram(ngram)
    else:
        return 'wrong ngram format!'


def getEmbeddingForSentence(sentence:str)->str:
    return embed([sentence])[0]

def getEmbeddingForTrigrm(trigram:str)->str:
    sentence = getSentenceForNgram(trigram)
    embedding = getEmbeddingForSentence(sentence)
    return embedding

def generateSentenceFilesFromTrigramFiles(args:Args):
    inPath_inputModelFeaturs = args.dataRootForNgramFormat + 'inputModelFeatures.txt'
    outPath_inputModelFeaturs = args.dataRootForSentences + 'inputModelSentences.txt'
    inPath_dataset = args.dataRootForNgramFormat + 'dataset.txt'    
    outPath_dataset = args.dataRootForSentences + 'datasetSentences.txt'
    with open(outPath_inputModelFeaturs, 'w') as w_file:
        with open(inPath_inputModelFeaturs, 'r') as r_file:
            for line in r_file:
                if line.startswith('---'):
                    w_file.write('---\n')
                    continue
                s = getSentenceForNgram(line)
                w_file.write(s)
    with open(outPath_dataset, 'w') as w_file:
        with open(inPath_dataset, 'r') as r_file:
            for line in r_file:
                s = getSentenceForNgram(line)
                w_file.write(s)

def compareSentences(args, firstSent:str, secondSent:str)->float:
    emb1 = getEmbeddingForSentence(firstSent)
    emb2 = getEmbeddingForSentence(secondSent)
    simScore = float("{:.2f}".format(cosine_similarity([emb1], [emb2])[0][0]))
    return simScore