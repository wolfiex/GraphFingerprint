from algorithms import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as Descriptors

graph = pd.read_json('data/fingerprint.json').T

fngroups = pd.read_csv('data/functionalgroups_regexmatched.csv')
fngroups.set_index('name',inplace=True)

#inorganics dont have smiles strings, lets remove these
fngroups = fngroups.loc[map(lambda x: type(x)==str,fngroups.smiles),:]
species = fngroups.index.values
species.sort()

graph = graph.loc[map(lambda x: x in species,graph.index),:]
fngroups = fngroups.loc[map(lambda x: x in species,fngroups.index),:]
smiles = fngroups.smiles.values

fngroups = fngroups[fngroups.columns[1:-2]]


#char to vector tokens
vectorizer = TfidfVectorizer(analyzer='char_wb')
vec_smiles = vectorizer.fit_transform(smiles).toarray()

vectorizer = TfidfVectorizer(analyzer='char_wb')
vec_spec = vectorizer.fit_transform(species).toarray()


#structure
embed_fn=np.nan_to_num(fngroups.values)
embed_graph=graph.values


#molecular fingerprint
#https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf
finger_mqn =[]
finger_morgan = []
finger_maccs = []
finger_ap = []

for i in smiles:
        mol = AllChem.MolFromSmiles(i)

        finger_mqn.append(np.array(Descriptors.MQNs_(mol)))
        finger_maccs.append(np.array(Descriptors.GetMACCSKeysFingerprint((mol))))
        #finger_morgan.append(np.array(Descriptors.GetMorganFingerprint((mol))))
        finger_ap.append(np.array(Descriptors.GetAtomPairFingerprint((mol))))

###
names = 'vec_spec,vec_smiles,embed_fn,finger_mqn,finger_maccs,finger_ap,embed_graph'.split(',')
data = [vec_spec,vec_smiles,embed_fn,finger_mqn,finger_maccs,finger_ap,embed_graph ]
counter = 0
for i in data:
    try:
        res = do_pca(i)
        plt.scatter(res[0],res[1],label = counter,alpha = .4)
        plt.savefig('figs/pca_%s.pdf'%names[counter])
        plt.clf()
    except Exception as e:
        print 'err',counter,e

    try:
        res = do_tsne(i)
        plt.scatter(res[0],res[1],label = counter,alpha = .4)
        plt.savefig('figs/tsne_%s.pdf'%names[counter])
        plt.clf()
    except Exception as e:
        print 'err',counter,e


    try:
        res = do_umap(i)
        plt.scatter(res[0],res[1],label = counter,alpha = .4)
        plt.savefig('figs/umap_%s.pdf'%names[counter])
        plt.clf()

    except Exception as e:
        print 'err',counter,e
    counter+=1
