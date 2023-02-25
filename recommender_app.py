import streamlit as st
import numpy as np
import pandas as pd
import surprise
import pickle
import networkx as nx
from networkx.algorithms import bipartite
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from sklearn.feature_extraction.text import CountVectorizer
import requests

st.set_page_config(
    page_title="Movie App",
    page_icon=":film_projector:",
    layout="centered",
    initial_sidebar_state="expanded")

def fetch_details(tmdbId):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=d8ee6f66bcbb558bffdc1a35af3f725d&language=en-US'.format(tmdbId))
    return response.json()

path_ = 'C:/Users/ANKUR/Desktop/Project/'
base_url = 'https://image.tmdb.org/t/p/w500'
X_train = pd.read_csv(path_+'xtrain.csv')
train_movies = pd.read_csv(path_+'train_movies.csv')

user_no = np.random.choice(range(0,671))             #159,265
#user_no = 265

#Populartiy============================================================================================

top_bool = X_train.groupby('movieId').count()['rating']
top_ind = top_bool[top_bool>100].index
top_movies = X_train[X_train.movieId.isin(top_ind)]
top20_ids = top_movies.groupby('movieId').rating.mean().sort_values(ascending=False)[:20].index        #average rating

#Collaborative Filtering Recommendations================================================================
svdpp_algo = surprise.dump.load(path_+'svdpp_algo.pkl')[1]       #tuple (prediction,algo)
train_ratings = X_train[['userId','movieId', 'rating']]

def recommend_collab(user_id):
    movies_watched = set(train_ratings[train_ratings.userId==user_id].movieId.values)

    if len(movies_watched)==0:
        return [],[]

    movies_unwatched = set(train_movies.movieId) - movies_watched

    results = []
    for mid in movies_unwatched:
        results.append(svdpp_algo.predict(user_id, mid))

    df =  pd.DataFrame([(i.iid,i.est) for i in results], columns=['movieId', 'rating']).sort_values('rating', ascending=False)
    top10_ids = df.movieId[:10]

    liked_ids = train_ratings[train_ratings.userId==user_id].sort_values('rating', ascending=False).movieId.values[:5]
    return (top10_ids, liked_ids)

top10_ids, liked_ids = recommend_collab(user_no)

# Graph based Recommendations================================================================================
B = pickle.load(open(path_+'graph.pkl','rb'))

def graph_recommend(q):

    rw = BiasedRandomWalk(StellarGraph(B))
    walk = rw.run(nodes=[q], n=1, length=10000, p=0.01, q=100, weighted=True, seed=42)

    #with 1/p prob, it returns to the source node
    #with 1/q prob, it moves away from the source node
    #Shape of walk: (1,10000)

    walk = list(filter(lambda x:type(x)==int, walk[0]))
    walk = list(map(str, walk))
    walk = ' '.join(walk)

    vocab = {str(mov):ind for ind,mov in enumerate(train_movies.movieId.sort_values().unique())}   #movieId:index
    vec = CountVectorizer(vocabulary=vocab)
    embed = vec.fit_transform([walk])

    reverse_vocab = {v:int(k) for k,v in vocab.items()}         #index:movieId
    embed = np.array(embed.todense())[0]

    top5_ids=[]
    for ind in embed.argsort()[::-1]:
        if len(top5_ids)==5: break
        movid = reverse_vocab[ind]
        if movid!=q: top5_ids.append(movid)

    return top5_ids

#Recommendations===========================================================================================
def show_blocks(id_lst,steps):

    lst = []
    for ind in id_lst:
        tid = train_movies[train_movies.movieId==ind].tmdbId.values[0]
        details = fetch_details(tid)
        name = details['title']
        img = base_url+details['poster_path']#+train_movies[train_movies.movieId==ind].poster_path.values[0]
        lst.append((name, img))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        for i in lst[:steps[0]]:
            st.image(i[1])
            st.text(i[0])
    with col2:
        for i in lst[steps[0]:steps[1]]:
            st.image(i[1])
            st.text(i[0])
    with col3:
        for i in lst[steps[1]:steps[2]]:
            st.image(i[1])
            st.text(i[0])
    with col4:
        for i in lst[steps[2]:steps[3]]:
            st.image(i[1])
            st.text(i[0])
    with col5:
        for i in lst[steps[3]:steps[4]]:
            st.image(i[1])
            st.text(i[0])

def show_bar():
    mov_name = st.selectbox('What are you looking for...?', train_movies.title.values)
    mid, tid = train_movies[train_movies.title==mov_name][['movieId','tmdbId']].values[0]
    details = fetch_details(tid)

    if st.button('Search'):

        col1,col2 = st.columns(2)
        with col1:
            st.image(base_url+details['poster_path'])
        with col2:
            st.header(details['title'])
            st.caption(details['tagline'])
            st.write(details['overview'])
            st.markdown("**Released in {}**".format(details['release_date']))
            st.write('Runtime: {} mins'.format(details['runtime']))
            st.write('Avg. Rating: {} :star:            Votes: {} :thumbsup:'.format(details['vote_average'], details['vote_count']))

        st.header("More like this...")
        top5_ids = graph_recommend(mid)
        show_blocks(top5_ids,[1,2,3,4,5])

if len(top10_ids)==0:
    st.header('Hello stranger!!')

    show_bar()
#'https://image.tmdb.org/t/p/w500/uexxR7Kw1qYbZk0RYaF9Rx5ykbj.jpg'
    st.title('Most Popular movies on the platform')
    show_blocks(top20_ids,[4,8,12,16,20])

else:
    st.header('Welcome user '+str(user_no))
    show_bar()
    st.title('Your Favourites...')
    show_blocks(liked_ids,[1,2,3,4,5])
    st.title('Based on your taste...')
    show_blocks(top10_ids,[2,4,6,8,10])
