import lib.utility as u
from tqdm import tqdm
import heapq

class TargetNodeNgram:
    def __init__(self, args, ngramLines, dbPath):
        self.ComparedSequenceList = []
        self.inputSequenceList = []
        # we build a single rec list out of all recommendation lists or just return the list if there is one ngram in the input.
        self.Recommendation = None
        # given the ngram set, add each ngram to the dictionary
        self.inputSequenceList = ngramLines.split('\n')
        if '' in self.inputSequenceList: 
            self.inputSequenceList = [i for i in self.inputSequenceList if i != '']
        for seq in self.inputSequenceList:      
            #remove NN from the ngram
            seq_groundTruth = str(seq).split('[next node label]')[1].strip() #hack: replaced [NN] with [next node label] 
            seq = str(seq).split('[next node label]')[0].strip() #hack: replaced [NN] with [next node label]               
            comperedSequence = ComparedSequence(seq, seq_groundTruth, dbPath)
            comperedSequence.buildRecommendationList(args)
            comperedSequence.sortElements()
            self.ComparedSequenceList.append(comperedSequence)

    # returns a string list of recommendation. each element is a recommendation
    def returnFinalRecommendationList(self, args)->list:
        # make a single list out of all the recommendation lists
        if len(self.ComparedSequenceList) == 1: 
            # return recommendation string
            recList = []
            recList.append('input: ' + self.ComparedSequenceList[0].input_sentence + '\n')
            recList.append('ground truth: ' + self.ComparedSequenceList[0].seq_groundTruth + '\n')
            hitRate = 0
            for rec in self.ComparedSequenceList[0].recommendationList:                
                recList.append(rec.db_seq + '\n' + str(rec.simScore) + '\n' + rec.next_nodes + '\n')
                # also count up hitRate if there's a hit
                groundtruth = self.ComparedSequenceList[0].seq_groundTruth # hack: groundtruth = self.ComparedSequenceList[0].seq_groundTruth.split(':')[1]
                recommendation =  rec.next_nodes # hack: recommendation =  rec.next_nodes.split(':')[1]
                if groundtruth == recommendation:
                    hitRate += 1
            recList.append('HIT : ' + str(hitRate) + '\n')
            return recList
        # otherwise mix all the rec lists
        else:
            # Combine all the lists into a single list
            combined_list = []
            nextNodesDict = {}  # To keep track of the highest score for each name
            for item in self.ComparedSequenceList:
                for obj in item.recommendationList:
                    if obj.next_nodes not in nextNodesDict or obj.simScore > nextNodesDict[obj.next_nodes].simScore:
                        nextNodesDict[obj.next_nodes] = obj

            # Add the objects with the highest scores to the combined list
            combined_list = list(nextNodesDict.values())               
            # Sort the combined list by score in descending order
            combined_list.sort(key=lambda x: x.simScore, reverse=True)
            N = 10
            top_N = combined_list[:N]    

            recList = []
            recList.append('multiple input sequence' + '\n')
            recList.append('ground truth: ' + self.ComparedSequenceList[0].seq_groundTruth + '\n')
            hitRate = 0
            for rec in top_N:                
                recList.append(rec.db_seq + '\n' + str(rec.simScore) + '\n' + rec.next_nodes + '\n')
                # also count up hitRate if there's a hit
                groundtruth = self.ComparedSequenceList[0].seq_groundTruth.split(':')[1]
                recommendation = rec.next_nodes.split(':')[1]
                if groundtruth == recommendation:
                    hitRate += 1
            recList.append('HIT : ' + str(hitRate) + '\n')
            return recList


class ComparedSequence:
    def __init__(self, input_sentence, sequence_groundTruth, dbPath):
        self.input_sentence = input_sentence
        self.seq_groundTruth = sequence_groundTruth
        self.recommendationList = []
        self.dbPath = dbPath

    def buildRecommendationList(self, args):
        with open(self.dbPath) as file:
            datasetSequence = file.read().splitlines()
            for line in tqdm(datasetSequence):
                db_sequence = line.split('[next node label]')[0].strip() #hack: replaces [NN] with [next node label] 
                simScore = u.compareSentences(args, self.input_sentence, db_sequence) # both ngrams in str format
                if simScore > 0.0:
                    nextNode = line.split('[next node label]')[1].strip() #hack: replaces [NN] with [next node label] 
                    self.updateTopRecommendationList(db_sequence, simScore, nextNode)
    
    # sort the list of recommendations in descending order
    def sortElements(self):        
        self.recommendationList.sort(key=lambda x: x.simScore, reverse=True)

    # helper functions ==========================
    def updateTopRecommendationList(self, db_ngram, simScore, nextNodes):
        # if the list is not filled yet
        if len(self.recommendationList) < 10:
            newRec = Recommendation(db_ngram, simScore, nextNodes)
            heapq.heappush(self.recommendationList, newRec)
        else:
            min_score_obj = self.recommendationList[0]
            if simScore > min_score_obj.simScore:
                newRec = Recommendation(db_ngram, simScore, nextNodes)
                heapq.heapreplace(self.recommendationList, newRec)


class Recommendation:
    def __init__(self, db_seq, simScore, next_node):
        self.db_seq = db_seq
        self.simScore = simScore
        self.next_nodes = next_node
    def __lt__(self, other):
        return self.simScore < other.simScore
    def __le__(self, other):
        return self.simScore <= other.simScore





