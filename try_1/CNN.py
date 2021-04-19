from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from statistics import mode

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	# lines = file.readlines()
	# result = []
	# for x in lines:
	# 	type = (x.split(','))
	# 	result.append([type[1], type[2]])
	#
	# print(result[0])
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]

	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)

	# clean doc
	tokens = clean_doc(doc)

	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(filename, vocab):
	# walk through all files in the folder
	# for filename in listdir(directory):
	# 	# skip any reviews in the test set
	# 	if is_trian and filename.startswith('cv9'):
	# 		continue
	# 	if not is_trian and not filename.startswith('cv9'):
	# 		continue
	# 	# create the full path of the file to open
	# 	path = directory + '/' + filename
	# 	# add doc to vocab
	# 	add_doc_to_vocab(path, vocab)

	add_doc_to_vocab(filename, vocab)

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()


# define vocab
vocab = Counter()
# add all docs to vocab
process_docs("snli_1.0_train.csv", vocab)

# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


#filtering the vocab
min_occurane = 2
tokens = [k for k, c in vocab.items() if (c >= min_occurane and k not in ["contradictionA", "entailmentA", "neutralA"])]
print(len(tokens))
print(tokens.most_common(50))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
