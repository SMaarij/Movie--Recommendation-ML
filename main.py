import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt



movies = pd.read_csv('Movie_Id_Titles.csv')
ratings = pd.read_csv('Dataset.csv')

print(movies.head())
print(ratings.head())

final_dataset = ratings.pivot(index='item_id',columns='user_id',values='rating')
# print(final_dataset.head())

# replacing NaN by 0
final_dataset.fillna(0,inplace=True)
print(final_dataset.head())


no_user_voted = ratings.groupby('item_id')['rating'].agg('count')
no_movie_voted = ratings.groupby('user_id')['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 1].index,:]

# f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
# plt.axhline(y=1,color='r')
# plt.xlabel('Movie Id')
# plt.ylabel('No of users voted')
# plt.show()

# f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(no_movie_voted.index,no_movie_voted,color='mediumseagreen')
# plt.axhline(y=1,color='r')
# plt.xlabel('UserId')
# plt.ylabel('No. of votes by user')
# plt.show()


csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['item_id']
        movie_idx = final_dataset[final_dataset['item_id'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['item_id']
            idx = movies[movies['item_id'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"
    
print(get_movie_recommendation('Toy Story'))