# %%
import numpy as np
import pandas as pd
import warnings
import sys

warnings.filterwarnings('ignore')
# %%
# name=sys.argv[1]
credits_df = pd.read_csv("Data/credits.csv")
movies_df=pd.read_csv("Data/movies.csv")

# %%
movies_df

# %%
credits_df

# %%
movies_df=pd.merge(movies_df,credits_df, on = 'title',how='inner') #merges credits_df and movies_df
movies_df = movies_df.loc[:, ~movies_df.columns.duplicated()]
movies_df

# %%
# movies_df.info() #shows columns

# %%
movies_df=movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]
movies_df

# %%
movies_df.isnull().sum()

# %%
movies_df.dropna(inplace=True) #drops null value rows
movies_df.isnull().sum()

# %%
movies_df.drop_duplicates()

# %%
# movies_df.iloc[0].crew #retrives any index from dataframe


# %%
import ast

# %%
def convert(object):
    l = []
    for i in ast.literal_eval(str(object)):
        l.append(i['name'])
    return l
    
    
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df["genres"] = movies_df["genres"].apply(convert)

movies_df.head()

# %%
def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(['name'])
            counter+=1
        else:
            break
        return l

# %%
movies_df['cast']=movies_df['cast'].apply(convert)

# %%
movies_df.head()

# %%
def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
    return l;

# %%
movies_df['crew']=movies_df['crew'].apply(fetch_director)

# %%
movies_df.head()

# %%
movies_df['overview'][0]

# %%
# seperte all the words so that we can use that for recommendation
movies_df['overview']=movies_df['overview'].apply(lambda x:x.split())


# %%
movies_df.head()

# %%
#now we will write code for removing spaces between 2 words ... for example Science Fiction will become ScinceFiction
movies_df["genres"]=movies_df["genres"].apply(lambda x :[i.replace(" ","") for i in x])
movies_df["keywords"]=movies_df["keywords"].apply(lambda x :[i.replace(" ","") for i in x])
movies_df["cast"]=movies_df["cast"].apply(lambda x :[i.replace(" ","") for i in x])

# %%
movies_df.head()

# %%
#merge all the the things in to a single column called tags
movies_df["tags"]=movies_df["overview"]+movies_df["genres"]+movies_df["cast"]+movies_df["keywords"]+movies_df["crew"]
movies_df

# %%
new_df=movies_df[["movie_id","title","tags"]]
new_df

# %%
#remove brackets in tags
new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))
new_df

# %%
#make everything in lower case

new_df['tags']=new_df['tags'].apply(lambda x: x.lower())

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000 , stop_words='english')

# %%
cv.fit_transform(new_df['tags']).toarray().shape


# %%
vectors=cv.fit_transform(new_df['tags']).toarray()
vectors[0]

# %%
import nltk

# %%
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# %%
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# %%
new_df['tags']=new_df['tags'].apply(stem)

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
similarity = cosine_similarity(vectors)


# %%
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

# %%


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    movies_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# %%
while(1):
    name=input()
    print("\n")
    recommend(name)

