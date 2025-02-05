import lib.utility as embedLib
import lib.recommend_sequence as rs
from lib.args import Args


args = Args()
# embedLib.generateSentenceFilesFromTrigramFiles(args)
print('ngram to sentence is done!')
rs.recommedBasedOnInputModelWithSentences(args)
