from sklearn import svm
import sys
import pickle
import os
import numpy as np

dir = sys.argv[1]
dest = sys.argv[2]

embedding_files = [file for file in os.scandir(dir) if (file.is_file() and file.name[0] != '.')]
label_dict = {}
embeddings = []
labels = []
i = 0
for file in embedding_files :
    new_embeddings = pickle.load(open(file.path, "rb"))
    label_dict[i] = new_embeddings[0][1]
    for embedding in new_embeddings :
        print(np.shape(embedding))
        embeddings.append(np.array(embedding[0]).tolist())
    labels+=[i]*len(new_embeddings) 
    i+=1
embeddings = np.reshape(embeddings, (len(embeddings), 128))
print(embeddings)
print(np.shape(embeddings))
print(labels)
model = svm.SVC()
model.fit(embeddings, labels)
pickle.dump((model, label_dict), open(dest, "wb"))

