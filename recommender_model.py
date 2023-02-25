import pandas as pd
import numpy as np
import warnings
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from surprise import Reader,Dataset
from surprise import SVD
import surprise
import pickle
import networkx as nx
from networkx.algorithms import bipartite
warnings.filterwarnings('ignore')
path_ = 'C:/Users/ANKUR/Desktop/Project/'

#2. Exploratory Data Analysis

# 2.1 Loading the data

#https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
movies = pd.read_csv(path_+'movies_metadata.csv')
ratings = pd.read_csv(path_+'ratings_small.csv')
keywords = pd.read_csv(path_+'keywords.csv')
links = pd.read_csv(path_+'links_small.csv')

# 2.2 Data Cleaning

# Removing duplicate rows
movies.drop_duplicates(inplace=True)
keywords.drop_duplicates(inplace=True)

# Removing NaN rows
movies.drop(movies[movies.title.isnull()].index, inplace=True)

# Merging the dataframes
movies = movies[['id','title','genres']]
movies.id = movies.id.astype(float)                                 #object --> float
movies = movies.merge(links, left_on='id', right_on='tmdbId')
movies.drop(['imdbId','id'], axis=1, inplace=True)
movies.drop_duplicates(inplace=True)
movies = movies.merge(keywords, left_on='tmdbId', right_on='id')
movies.drop(['id'], axis=1, inplace=True)

def cleans(row):
    if len(row)>0:
        return str([(i['name'].lower().strip()) for i in row])
    return str(row)

#genres and keywords coulumn are of type str and json format.. we get the genre names in a list and store it in str format again
movies.keywords = movies.keywords.apply(lambda x: cleans(ast.literal_eval(x)))
movies.genres = movies.genres.apply(lambda x: cleans(ast.literal_eval(x)))

if not os.path.isfile(path_+'movie_ratings.csv'):
    movie_ratings = movies.merge(ratings, on='movieId')
    movie_ratings.sort_values('timestamp', inplace=True)
    movie_ratings.to_csv(path_+'movie_ratings.csv', index=False)
movie_ratings = pd.read_csv(path_+'movie_ratings.csv')

# 2.4 Train Test Temporal Split

#X_train, X_test = train_test_split(movie_ratings, test_size=0.05, shuffle=False)
if not (os.path.isfile(path_+'xtrain.csv') and os.path.isfile(path_+'xtest.csv')):
    movie_ratings[:int(len(movie_ratings)*0.8)].to_csv(path_+'xtrain.csv', index=False)
    movie_ratings[int(len(movie_ratings)*0.8):].to_csv(path_+'xtest.csv', index=False)

X_train = pd.read_csv(path_+'xtrain.csv')
X_test = pd.read_csv(path_+'xtest.csv')

train_movies = set(X_train.movieId.values)
test_movies = set(X_test.movieId.values)

train_movies = movies[movies.movieId.isin(train_movies)]
test_movies = movies[movies.movieId.isin(test_movies)]


# 2.6 Genre feature analysis

genres = set()
for i in train_movies.genres:
    genres.update(ast.literal_eval(i))
genres=list(genres)

dist = []
den = len(train_movies)
for i in genres:
    v = len(train_movies[train_movies.genres.str.match('.*'+i+'.*')==True])/den*100
    dist.append((v,i))
dist = pd.DataFrame(dist).sort_values(0)

dist[0] = dist[0].apply(lambda x: x**(-1))
weights_genre = dict(zip(list(dist[1].values),list(dist[0].values)))


# 2.7 Keywords feature analysis

keywords = set()
for i in train_movies.keywords:
    keywords.update(ast.literal_eval(i))
keywords=list(keywords)

dist = []
den = len(train_movies)
for i in keywords:
    if i=='':
        continue
    v = len(train_movies[train_movies.keywords.str.match('.*'+i+'.*')==True])/den*100
    dist.append((v,i))
dist = pd.DataFrame(dist).sort_values(0)

dist[0] = dist[0].apply(lambda x: (x+1)**(-1))
weights_keywords = dict(zip(list(dist[1].values),list(dist[0].values)))

# 4. Collaborative Filtering

# 4.1 Trainset and Testset
train_ratings = X_train[['userId','movieId','rating']]
test_ratings = X_test[['userId','movieId','rating']]

#Trainset
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(train_ratings,reader)
trainset = data.build_full_trainset()

#Testset
#list of list (or tuples)==> (user, id, rating)
testset = test_ratings.values

# 4.2 Training SVD Model

if not os.path.isfile(path_+'svdpp_algo.pkl'):
    svdpp_algo = SVDpp(random_state=24)
    svdpp_algo.fit(trainset)
    surprise.dump.dump(path_+'svdpp_algo.pkl',algo=svdpp_algo)
    print('Done Dumping...!')

# 5. Graph based Recommendations

# 5.1 Creating nodes and edges

edges = []
for row in train_movies.iterrows():
    mid = row[1].movieId
    edges.extend([(mid, k, weights_keywords.get(k,0)*10) for k in ast.literal_eval(row[1].keywords)])
    edges.extend([(mid, g, weights_genre[g]*200) for g in ast.literal_eval(row[1].genres)])

data = pd.DataFrame(edges, columns=['movieId','keywords','weights'])

# 5.2 Creating Graph
if not os.path.isfile(path_+'graph.pkl'):
    B = nx.Graph()
    B.add_nodes_from(data.movieId.unique(), bipartite=0, label='movie')
    B.add_nodes_from(data.keywords.unique(), bipartite=1, label='keygen')
    B.add_weighted_edges_from(edges)
    pickle.dump(B,open(path_+'graph.pkl','wb'))

print('Finished!!')
