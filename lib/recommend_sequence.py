from lib.args import Args
from lib import dataStructure as ds
from tqdm import tqdm

# function iterates through trigrams in the input model and writes recommendations to a single file
def recommedBasedOnInputModelWithSentences(args:Args):
    inputModelSentences_path = args.dataRootForSentences + 'inputModelSentences.txt'
    targetNodeNgramsObjs = [] # lis of type TargetNodeNgram
    ngramsPerNode = []
    with open(inputModelSentences_path, 'r') as file:
        testModelNgrams = file.read()
        ngramsPerNode = testModelNgrams.split('---')
        if '\n' in ngramsPerNode: ngramsPerNode.remove('\n')
        # iterate all ngram lists per target node
        for ngrams in ngramsPerNode:            
            # create a TargetNodeNgram object per ngram set
            n = ds.TargetNodeNgram(args, ngrams, dbPath = args.dataRootForSentences + 'datasetSentences.txt')
            targetNodeNgramsObjs.append(n)


    
    writePath = args.dataRootForSentences + 'recommendations.txt'
    # compare the input trigram with each trigram in each line in the list
    # if the similarity is > n% then output it
    with open(writePath, 'w') as recFile:
        for targetNodeNgram in targetNodeNgramsObjs:
            recStr = targetNodeNgram.returnFinalRecommendationList(args)
            for line in recStr:
                recFile.write(line + '\n')
            recFile.write('-------------------------------\n')
        recFile.write('=========================================================================\n')
 
    
    print('check recommendations.txt')   