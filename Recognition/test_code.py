vocab = open('vocab_tmp.txt', 'w')
file = open('/home1/vudinh/NomNaOCR/Text-Generator-main/Train.txt').read().splitlines()
new_vocab = []
for f in file:
    _, label = f.split('\t')
    new_vocab += list(label)
new_vocab = set(new_vocab)
for i in new_vocab:
    vocab.write(i+'\n')
