import math
import itertools
from scipy import linalg
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

data = pd.read_json('./data/jobs.json', orient='records')
data.job_description = data.job_description.apply(lambda t: t[0])
data



# full_skills_doc = data[['job_description', 'skills']]
# full_skills_doc


# # Term-Document Matrix


all_skills = data.skills.tolist()
all_skills = np.asarray(all_skills)

all_skills = list(itertools.chain(*all_skills))
all_skills[:5]


td_matrix = pd.DataFrame()
td_matrix['skills'] = all_skills
td_matrix.drop_duplicates(inplace=True)


def term_frequency(t, d):
    return d.count(t)

idf_values = {}
all_skills = td_matrix['skills'].tolist()
num_of_docs = len(data.index)

for skill in all_skills:
    _skill = skill
    contains_token = map(lambda doc: _skill in doc, data.skills.tolist())
    idf_values[skill] = math.log(float(num_of_docs) / (1 + sum(contains_token)))


idf_values.get('SSRS')

print (len(td_matrix))
print (len(data))

# td_matrix = td_matrix
# data = data

def calc_td_matrix(i, row):
    for ix, tdrow in td_matrix.iterrows():
        doc = 'd' + str(i)
        td_matrix.loc[ix, doc] = idf_values.get(tdrow['skills'] ,0) * term_frequency(tdrow['skills'], row['job_description'])

for i, row in data.iterrows():
    calc_td_matrix(i, row)

# Export
td_matrix.to_csv('tmp/td_matrix.csv', index=False, encoding='utf-8')

td_matrix
_td_matrix = td_matrix
_td_matrix = _td_matrix.set_index('skills')

skills_sparse = sparse.csr_matrix(_td_matrix)
similarities = cosine_similarity(skills_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

distance_matrix = pairwise_distances(skills_sparse, metric="cosine")
distance_matrix

x = pd.DataFrame(similarities)

x.to_csv('tmp/x.csv', index=False)

x.columns = _td_matrix.index
x.set_index(_td_matrix.index, inplace=True)

x
x[(x >= 0.9).any(axis=1)].to_csv('./tmp/related_test.csv',encoding='utf8')

a = np.random.randn(9, 6) + 1.j*np.random.randn(9, 6)
a.shape

U, s, Vh = linalg.svd(_td_matrix, full_matrices=False)

new_d = np.dot(U, np.dot(np.diag(s), Vh))

skills_sparse = sparse.csr_matrix(new_d)
similarities = cosine_similarity(skills_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

x = pd.DataFrame(similarities)
x.columns = _td_matrix.index
x.set_index(_td_matrix.index, inplace=True)

x

x[(x >= 0.9).any(axis=1)].to_csv('./tmp/related_test_svd_lite.csv',encoding='utf8')
