import codecs
import pickle

fp = codecs.open('./renmin3.txt', 'r', encoding='utf-8')
data = [[item[0] for item in line.split()] for line in fp.readlines() if len(line.strip())>0]
word = set()
for line in data:
    for item in line:
        word.add(item)
word2id = {value:key+1 for key, value in enumerate(word)}
word2id['<UNK>'] = len(word2id)+1
word2id['<PAD>'] = 0
print(word2id)
with codecs.open('./word2id.pkl', 'wb') as fp:
    pickle.dump(word2id, fp)
