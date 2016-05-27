import numpy as np, random as rand

class LDAModel:
    def __init__(self, Kin, vocabIn={}, alIn=.05, betIn =.05):
        self.vocab = vocabIn
        self.vocabInv = {v:k for k,v in self.vocab.iteritems()}
        self.K = Kin
        self.W = len(self.vocab.keys())
        self.alpha = alIn
        self.beta = betIn

    def build_vocab(self, corpus):
        newVocab = {}
        for doc in corpus:
            for word in doc:
                if word in newVocab.keys():
                    pass
                else:
                    newVocab[word] = len(newVocab.keys())
        self.vocab = newVocab
        self.vocabInv = {v:k for k,v in self.vocab.iteritems()}
        self.W = len(self.vocab.keys())

    def gibbs_init(self, corpus):
        z_assign = []
        ndk = np.zeros((len(corpus), self.K))
        nkw = np.zeros((self.K, self.W))
        nk = np.zeros(self.K)
        for i in range(len(corpus)):
            newZ = []
            convDoc = [self.vocab[w] for w in corpus[i]]
            for word in convDoc:
                label = rand.randint(0,self.K-1)
                ndk[i,label] += 1
                nkw[label, word] += 1
                nk[label] += 1
                newZ.append(label)
            z_assign.append(newZ)
        return z_assign, ndk, nkw, nk

    def gibbs_sample(self, corpus, iters):
        zA, ndk, nkw, nk = self.gibbs_init(corpus)     
        for ITERATION in range(iters):
            for i in range(len(corpus)):
                convDoc = [self.vocab[w] for w in corpus[i]]
                for m in range(len(convDoc)):
                    word = convDoc[m]
                    tp = zA[i][m]
                    ndk[i, tp] -= 1
                    nkw[tp, word] -= 1
                    nk[tp] -= 1
                    pval = np.zeros(self.K)
                    for k in range(self.K):
                        pval[k] = (ndk[i, k] + self.alpha) * (nkw[k, word] + self.beta) / (nk[k] + (self.W*self.beta) )
                    pval = pval / pval.sum()
                    newT = np.random.multinomial(1, pval).argmax()
                    zA[i][m] = newT
                    ndk[i, newT] += 1
                    nkw[newT, word] += 1
                    nk[newT] += 1
        self.ndk = ndk
        self.nkw = nkw
        self.nk = nk
        theta = self.alpha*np.ones(ndk.shape)
        self.theta = ((theta + ndk).T / (theta+ndk).sum(axis=1)).T
        phi = self.beta*np.ones(nkw.shape)
        self.phi = (phi + nkw) / (phi+nkw).sum(axis=0)
        self.zA = zA

    def get_topic_words(self, topic, numwords):
        vals = self.phi[topic,:]
        return sorted(self.vocab.keys(), key = lambda x: -1.0*vals[self.vocab[x]])[:numwords]

    def print_GVIZ_model(self, filename=None):
        newstr = "digraph g1 {\n\tgraph [rankdir=LR];\n"
        for k,v in self.vocab.iteritems():
            newstr += '\tw%d [label="%s"];\n' % (v, k)
        for i in range(self.K):
            newstr += '\tt%d [label="%s"];\n' % (i, ", ".join(self.get_topic_words(i, 3)))
        for i in range(self.W):
            for j in range(self.K):
                if self.phi[j,i] > 0.05:
                    newstr += "\tw%s -> t%s [penwidth=%f];\n" % (str(i), str(j), int(10*self.phi[j,i]))
        newstr += "}"
        if filename:
            f = open(filename, 'w')
            f.write(newstr)
            f.close()
        else:
            print newstr

test = LDAModel(3)
f = open('test_corp')
corp = []
line = f.readline()
while line:
    corp.append(line.strip().split(" "))
    line=f.readline()
test.build_vocab(corp)
print test.gibbs_sample(corp, 100)
test.print_GVIZ_model('test.dot')
