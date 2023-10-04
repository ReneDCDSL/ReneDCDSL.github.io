---
layout: post
title: "Twitter Sentiment Analysis"
subtitle: "Sentiment Analysis classification task performed over a million tweets"
#date:
background: '/img/posts/twitter_SA/twitter_SA.jpg'
---

# [S403011] [Machine Learning Project](https://github.com/ReneDCDSL/Twitter_Sentiment_Analysis)
##### by [De CHAMPS René](https://www.linkedin.com/in/rené-de-champs-2679bb269/) & [MAULET Grégory](https://www.linkedin.com/in/gregory-maulet-4a1879140/)

<br>

# I. Introduction: Description of the data set, imports, notebook structure

<br>

### Description of the dataset

In this analysis, we're looking at a dataset containing tweets : short message posted by users on www.twitter.com. We're aiming at modelling and predicting the sentiment, whether positive or negative, of each tweet. The sentiment of the tweet was based on whether each tweet contained a happy ":)" or sad ":(" smiley. These smileys have been removed from the tweets beforehand.

<br>

### Notebook structure

We'll proceed in this analysis by making a first exploratory data analysis in which we'll take an overall look at our data to get a first intuition on how to approach the modelling. Then, we'll apply different modelling approach and try to compare them using predictive scoring. Finally, after tuning up our best model to get the best possible training fit, we'll apply our model on the test dataset which will serve as our final prediction result. Furthermore, we added an appendix at the end containing a long and perious attempt on using BERT model.

<br>

### Libraries

Let's first load the various libraries needed for this analysis.

```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import keras.backend as K

from numpy import array, asarray, zeros
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GlobalMaxPooling1D, Conv1D, LSTM, Flatten, Dense, Embedding, MaxPooling1D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.utils import np_utils

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re, string

import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from pprint import pprint
from time import time
import logging
```

<br>

### Import

We then import the training dataset under the name "emote".


```python
emote = pd.read_csv("MLUnige2021_train.csv",index_col=0)
print("Dataset shape:", emote.shape)
```

    C:\Users\rened\Anaconda3\lib\site-packages\numpy\lib\arraysetops.py:580: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)
    

    Dataset shape: (1280000, 6)
  
<br> 

# II. Exploratory Data Analysis & Feature Engineering

### First look at the data

Now that we've imported our training dataset, let's take a first look into it.


```python
emote.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2063391019</td>
      <td>Sun Jun 07 02:28:13 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>BerryGurus</td>
      <td>@BreeMe more time to play with you BlackBerry ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2000525676</td>
      <td>Mon Jun 01 22:18:53 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>peterlanoie</td>
      <td>Failed attempt at booting to a flash drive. Th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2218180611</td>
      <td>Wed Jun 17 22:01:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>will_tooker</td>
      <td>@msproductions Well ain't that the truth. Wher...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2190269101</td>
      <td>Tue Jun 16 02:14:47 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>sammutimer</td>
      <td>@Meaghery cheers Craig - that was really sweet...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2069249490</td>
      <td>Sun Jun 07 15:31:58 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>ohaijustin</td>
      <td>I was reading the tweets that got send to me w...</td>
    </tr>
  </tbody>
</table>
</div>



This dataset contains not only the tweets and its corresponding emotions, but also the username of the sender, the date at which it was sent and a last column which indicates if a specific query was used in processing the data. 


```python
print("Missing values in our data :", emote.isna().sum().sum())
# No missing values

found = emote['lyx_query'].str.contains('NO_QUERY')
print("Instances of NO_QUERY in column 'lyx_query':", found.count())
# Full of "NO_QUERY"
```

    Missing values in our data : 0
    Instances of NO_QUERY in column 'lyx_query': 1280000
    

Our dataset doesn't contain any missing values. Morevover, we observe that the column 'lyx_query' is full of the same statement 'NO_QUERY'. Thus, this variable is of no use in the predictive aim of our model since it doesn't make any discrimination between any tweet.


```python
sns.catplot(x="emotion", data=emote, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
plt.show();
print("Number of positive tweets :", sum(emote["emotion"] == 1))
print("Number of negative tweets :", sum(emote["emotion"] == 0))
# About 50/50 positive and negative tweets
```


    
![png](output_12_0.png)
    


    Number of positive tweets : 640118
    Number of negative tweets : 639882
    

We are training on a pretty balanced dataset with as much positive and negative tweets. This will let us perform train/test split without the need of stratifying.


```python
print("Number of different tweet id :", emote["tweet_id"].nunique()) # 1069 tweets have had same id ??
print("Number of different users :", emote["user"].nunique()) # 574.114 different users
print("Number of users that tweeted only once :", sum(emote["user"].value_counts() == 1)) # 365.446 users tweeted once
print("Various users and their number of posted tweets :")
print(emote["user"].value_counts()) # some of them commented a lot
```

    Number of different tweet id : 1278931
    Number of different users : 574114
    Number of users that tweeted only once : 365446
    Various users and their number of posted tweets :
    lost_dog          446
    webwoke           292
    tweetpet          239
    VioletsCRUK       234
    mcraddictal       226
                     ... 
    anne_ccj            1
    JocelynG42          1
    wkdpstr             1
    Strawberry_Sal      1
    MichalMM            1
    Name: user, Length: 574114, dtype: int64
    

About a quarter of the twitter users in our training dataset only tweeted once during that period, while some of them went as far as tweeting several hundred times.

 

### 5 Most talkative users data


```python
emote[emote["user"] == "lost_dog"].head() # SPAM : all the 446 same message "@random_user I am lost. Please help me find a good home."
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8229</th>
      <td>0</td>
      <td>2209419659</td>
      <td>Wed Jun 17 10:22:06 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>lost_dog</td>
      <td>@JamieDrokan I am lost. Please help me find a ...</td>
    </tr>
    <tr>
      <th>9527</th>
      <td>0</td>
      <td>2328965183</td>
      <td>Thu Jun 25 10:11:34 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>lost_dog</td>
      <td>@W_Hancock I am lost. Please help me find a go...</td>
    </tr>
    <tr>
      <th>10645</th>
      <td>0</td>
      <td>2072079020</td>
      <td>Sun Jun 07 20:21:54 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>lost_dog</td>
      <td>@miznatch I am lost. Please help me find a goo...</td>
    </tr>
    <tr>
      <th>14863</th>
      <td>0</td>
      <td>2214285766</td>
      <td>Wed Jun 17 16:31:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>lost_dog</td>
      <td>@kgustafson I am lost. Please help me find a g...</td>
    </tr>
    <tr>
      <th>16723</th>
      <td>0</td>
      <td>1696136174</td>
      <td>Mon May 04 07:41:03 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>lost_dog</td>
      <td>@kneeon I am lost. Please help me find a good ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
emote[emote["user"] == "webwoke"].head() # SPAM : making request to visit some random website (commercial bot ?)
#emote[emote["user"] == "webwoke"].sum() # 68/292 positive messages

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19553</th>
      <td>0</td>
      <td>2067697514</td>
      <td>Sun Jun 07 12:48:05 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>webwoke</td>
      <td>come on... drop by 1  44. blogtoplist.com</td>
    </tr>
    <tr>
      <th>24144</th>
      <td>0</td>
      <td>2072285184</td>
      <td>Sun Jun 07 20:44:08 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>webwoke</td>
      <td>owww god, drop by 18  57. blogspot.com</td>
    </tr>
    <tr>
      <th>25988</th>
      <td>0</td>
      <td>2055206809</td>
      <td>Sat Jun 06 08:54:04 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>webwoke</td>
      <td>F**K! drop by 1  97. zimbio.com</td>
    </tr>
    <tr>
      <th>28219</th>
      <td>1</td>
      <td>2053451192</td>
      <td>Sat Jun 06 04:36:04 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>webwoke</td>
      <td>uhuiii... move up by 1  69. hubpages.com</td>
    </tr>
    <tr>
      <th>28597</th>
      <td>1</td>
      <td>2066463084</td>
      <td>Sun Jun 07 10:34:05 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>webwoke</td>
      <td>GoGoGo... move up by 1  13. slideshare.net</td>
    </tr>
  </tbody>
</table>
</div>




```python
emote[emote["user"] == "tweetpet"].head() # SPAM : 239 messages asking to "@someone_else Clean me"
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11130</th>
      <td>0</td>
      <td>1676425868</td>
      <td>Fri May 01 22:00:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>tweetpet</td>
      <td>@CeladonNewTown  Clean Me!</td>
    </tr>
    <tr>
      <th>13494</th>
      <td>0</td>
      <td>1573611322</td>
      <td>Tue Apr 21 02:00:03 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>tweetpet</td>
      <td>@chromachris  Clean Me!</td>
    </tr>
    <tr>
      <th>17443</th>
      <td>0</td>
      <td>1676426980</td>
      <td>Fri May 01 22:00:49 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>tweetpet</td>
      <td>@Kamryn6179  Clean Me!</td>
    </tr>
    <tr>
      <th>23973</th>
      <td>0</td>
      <td>1677423044</td>
      <td>Sat May 02 02:00:12 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>tweetpet</td>
      <td>@greenbizdaily  Clean Me!</td>
    </tr>
    <tr>
      <th>33463</th>
      <td>0</td>
      <td>1676426375</td>
      <td>Fri May 01 22:00:43 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>tweetpet</td>
      <td>@ANALOVESTITO  Clean Me!</td>
    </tr>
  </tbody>
</table>
</div>




```python
emote[emote["user"] == "VioletsCRUK"].sum() # 180/234 positive messages
emote[emote["user"] == "VioletsCRUK"].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8319</th>
      <td>0</td>
      <td>2057611341</td>
      <td>Sat Jun 06 13:19:41 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>VioletsCRUK</td>
      <td>@marginatasnaily lol i was chucked of 4 times ...</td>
    </tr>
    <tr>
      <th>9102</th>
      <td>1</td>
      <td>1573700635</td>
      <td>Tue Apr 21 02:26:06 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>VioletsCRUK</td>
      <td>@highdigi Nothing worse! Rain has just started...</td>
    </tr>
    <tr>
      <th>16570</th>
      <td>1</td>
      <td>1980137710</td>
      <td>Sun May 31 05:49:01 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>VioletsCRUK</td>
      <td>Will catch up with yas later..goin for a solid...</td>
    </tr>
    <tr>
      <th>37711</th>
      <td>1</td>
      <td>1881181047</td>
      <td>Fri May 22 03:52:11 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>VioletsCRUK</td>
      <td>@Glasgowlassy lol oh that's a big buffet of ha...</td>
    </tr>
    <tr>
      <th>37909</th>
      <td>0</td>
      <td>2067636547</td>
      <td>Sun Jun 07 12:41:40 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>VioletsCRUK</td>
      <td>@jimkerr09 That was a really lovely tribute to...</td>
    </tr>
  </tbody>
</table>
</div>




```python
emote[emote["user"] == "mcraddictal"].sum() # 54/226 positive messages
emote[emote["user"] == "mcraddictal"].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2337</th>
      <td>0</td>
      <td>2059074446</td>
      <td>Sat Jun 06 16:11:42 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>mcraddictal</td>
      <td>@MyCheMicALmuse pleaseeee tell me? -bites nail...</td>
    </tr>
    <tr>
      <th>2815</th>
      <td>0</td>
      <td>1968268387</td>
      <td>Fri May 29 21:05:43 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>mcraddictal</td>
      <td>@MCRmuffin</td>
    </tr>
    <tr>
      <th>7448</th>
      <td>0</td>
      <td>2052420061</td>
      <td>Sat Jun 06 00:40:11 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>mcraddictal</td>
      <td>@chemicalzombie dont make me say it  you know.</td>
    </tr>
    <tr>
      <th>10092</th>
      <td>0</td>
      <td>2061250826</td>
      <td>Sat Jun 06 20:29:01 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>mcraddictal</td>
      <td>@NoRaptors noooooo begging  i hate that. I'm s...</td>
    </tr>
    <tr>
      <th>13533</th>
      <td>0</td>
      <td>1981070459</td>
      <td>Sun May 31 08:20:52 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>mcraddictal</td>
      <td>@Boy_Kill_Boy  so was haunting in ct. That mov...</td>
    </tr>
  </tbody>
</table>
</div>



Out of the 5 users that tweeted the most, it seems like 3 of them are some kind of bot or spam bot. The 4th and 5th ones seem to be random users from which we got a lot tweets in the database.
All these users show pattern in their sent tweets. Indeed, they tend to send messages that are not balanced towards their emotion. 'Lost_dog' and 'tweetpet' both sent only negative tweets out of hundreds of them. 'webwoke' and 'mcraddictal' also sent largely negative tweets while 'VioletsCRUK' sent mostly positive tweets. We'll take this information into account when trying to classify further tweets.


```python
emote = emote[['emotion', 'user', 'text']]
emote.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>user</th>
      <th>text</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>BerryGurus</td>
      <td>@BreeMe more time to play with you BlackBerry ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>peterlanoie</td>
      <td>Failed attempt at booting to a flash drive. Th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>will_tooker</td>
      <td>@msproductions Well ain't that the truth. Wher...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>sammutimer</td>
      <td>@Meaghery cheers Craig - that was really sweet...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>ohaijustin</td>
      <td>I was reading the tweets that got send to me w...</td>
    </tr>
  </tbody>
</table>
</div>



We add a column with the number of words per tweet:


```python
emote['length'] = emote['text'].apply(lambda x: len(x.split(' ')))
emote.head()
```


```python
max_tweet = max(emote["length"])
print('Largest tweet length:', max_tweet)
```

    Largest tweet length: 110
    

 

### Symbols
Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers:


```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text
```


```python
emote['text_clean'] = emote['text'].apply(clean_text) #maybe remove the name with the @
emote.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
      <th>text_clean</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2063391019</td>
      <td>Sun Jun 07 02:28:13 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>BerryGurus</td>
      <td>@BreeMe more time to play with you BlackBerry ...</td>
      <td>breeme more time to play with you blackberry t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2000525676</td>
      <td>Mon Jun 01 22:18:53 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>peterlanoie</td>
      <td>Failed attempt at booting to a flash drive. Th...</td>
      <td>failed attempt at booting to a flash drive the...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2218180611</td>
      <td>Wed Jun 17 22:01:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>will_tooker</td>
      <td>@msproductions Well ain't that the truth. Wher...</td>
      <td>msproductions well aint that the truth whered ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2190269101</td>
      <td>Tue Jun 16 02:14:47 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>sammutimer</td>
      <td>@Meaghery cheers Craig - that was really sweet...</td>
      <td>meaghery cheers craig  that was really sweet o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2069249490</td>
      <td>Sun Jun 07 15:31:58 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>ohaijustin</td>
      <td>I was reading the tweets that got send to me w...</td>
      <td>i was reading the tweets that got send to me w...</td>
    </tr>
  </tbody>
</table>
</div>



 

### Stopwords
Remove stopwords (a list of not useful english words like 'the', 'at', etc.). It permits to reduce dimension of the data when tokenizing.


```python
stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text
    
emote['text_clean'] = emote['text_clean'].apply(remove_stopwords)
emote.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
      <th>text_clean</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2063391019</td>
      <td>Sun Jun 07 02:28:13 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>BerryGurus</td>
      <td>@BreeMe more time to play with you BlackBerry ...</td>
      <td>breeme time play blackberry</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2000525676</td>
      <td>Mon Jun 01 22:18:53 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>peterlanoie</td>
      <td>Failed attempt at booting to a flash drive. Th...</td>
      <td>failed attempt booting flash drive failed atte...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2218180611</td>
      <td>Wed Jun 17 22:01:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>will_tooker</td>
      <td>@msproductions Well ain't that the truth. Wher...</td>
      <td>msproductions well aint truth whered damn auto...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2190269101</td>
      <td>Tue Jun 16 02:14:47 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>sammutimer</td>
      <td>@Meaghery cheers Craig - that was really sweet...</td>
      <td>meaghery cheers craig  really sweet reply pumped</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2069249490</td>
      <td>Sun Jun 07 15:31:58 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>ohaijustin</td>
      <td>I was reading the tweets that got send to me w...</td>
      <td>reading tweets got send lying phone face dropp...</td>
    </tr>
  </tbody>
</table>
</div>



 

### Stemming/ Lematization

Stemming cuts off prefixes and suffixes (ex: laziness -> lazi). Lemma converts words (ex: writing, writes) into its radical (ex: write).


```python
stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text
```


```python
emote['text_clean'] = emote['text_clean'].apply(stemm_text)
emote.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
      <th>text_clean</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2063391019</td>
      <td>Sun Jun 07 02:28:13 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>BerryGurus</td>
      <td>@BreeMe more time to play with you BlackBerry ...</td>
      <td>breem time play blackberri</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2000525676</td>
      <td>Mon Jun 01 22:18:53 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>peterlanoie</td>
      <td>Failed attempt at booting to a flash drive. Th...</td>
      <td>fail attempt boot flash drive fail attempt swi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2218180611</td>
      <td>Wed Jun 17 22:01:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>will_tooker</td>
      <td>@msproductions Well ain't that the truth. Wher...</td>
      <td>msproduct well aint truth where damn autolock ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2190269101</td>
      <td>Tue Jun 16 02:14:47 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>sammutimer</td>
      <td>@Meaghery cheers Craig - that was really sweet...</td>
      <td>meagheri cheer craig  realli sweet repli pump</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2069249490</td>
      <td>Sun Jun 07 15:31:58 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>ohaijustin</td>
      <td>I was reading the tweets that got send to me w...</td>
      <td>read tweet got send lie phone face drop ampit ...</td>
    </tr>
  </tbody>
</table>
</div>



 

### Wordclouds


```python
#pip install wordcloud
from wordcloud import WordCloud
from PIL import Image
```


```python
twitter_mask = np.array(Image.open('user.jpeg'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=twitter_mask,
)
wc.generate(' '.join(text for text in emote.loc[emote['emotion'] == 1, 'text_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for positive messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()
```


    
![png](output_42_0.png)
    



```python
twitter_mask = np.array(Image.open('user.jpeg'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=twitter_mask,
)
wc.generate(' '.join(text for text in emote.loc[emote['emotion'] == 0, 'text_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for negative messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()
```


    
![png](output_43_0.png)
    


A nice representation of the most used words in each sentiment-type of messages

Now that we've taken a good first look at our data. We'll try to compute some models. Since we're trying to predict a binary outcome that is the sentiment of a given tweet, we'll proceed with classification modelling. In order to do so, we first need to transform our text data into digits so that we can apply our models. We do so by using techniques that transform our data into numbers then work on the words frequency. As we were led to believe from the 5 most talkative (+ we compared models with and without) that usernames are good source of information towards predicting further sentiment (in this dataset), we combine the text and username as our predictors for further modelling. 


```python
emote_cut = emote.drop(emote.index[640000:1280000])
print("Cropped dataset shape:", emote_cut.shape)
X_train, X_test, y_train, y_test = train_test_split((emote.text + emote.user), emote.emotion, test_size=0.1, random_state=37)
print("First 5 entries of training dataset :", X_train.head())
```

 

### Feature Engineering : Bag of words

##### The Vectorization and TF-IDF method

We will extract the numerical features of our text content using a first tool that will vectorize our corpus then a second one that will take into account the frequency of appearance of our words tokens. 

First, we make use of CountVectorizer. This method tokenizes strings of words by transforming them into tokens (using white spaces as token separators)


Using CountVectorizer to tokenize and count the occurences of words in our text dataset.


```python
count_vect = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape
```

Now we reweight the words counts through TF-IDF so that they can be used by classifier methods.


```python
tfidftransformer = TfidfTransformer()
X_train_final = tfidftransformer.fit_transform(X_train_counts)
X_train_final.shape
```

 

# III. Model Selection

The aim of this project is to learn from our tweet training dataset in order to being able to classify new tweets as being of positive or negative emotion. This is a classification task with binary outcome. There are several models that we've seen in class that can be of help here. We decided to present you our 3 best classification regression models. This is followed by attempts at building a Neural Network model that could predict better the tweet sentiment.

 

## Preprocessing Effectiveness

We will compare the effectiveness of preprocessing the data in the aim of increasing our prediction accuracy by comparing 2 models respectively including pre-processed data and unprocessed data.

Here's a first logistic model fit on 50 000 observations, with preprocessing :


```python
pipeline_log = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('log', LogisticRegression()),
])
emote_cut = emote.drop(emote.index[50000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)

grid_search_log = GridSearchCV(pipeline_log, parameters_log, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_log.steps])
print("parameters:")
pprint(parameters_log)
t0 = time()
grid_search_log.fit(emote_cut.text_clean, emote_cut.emotion)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_log.best_score_)
print("Best parameters set:")
best_parameters = grid_search_log.best_estimator_.get_params()
for param_name in sorted(parameters_log.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'log']
    parameters:
    {'log__C': (0.5, 0.75, 1.0),
     'log__penalty': ('l2',),
     'vect__ngram_range': ((1, 2), (1, 3))}
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    done in 45.296s
    
    Best score: 0.760
    Best parameters set:
    	log__C: 1.0
    	log__penalty: 'l2'
    	vect__ngram_range: (1, 2)
    

We achieved a prediction score of 76% using logistic regression on 50 000 observations. Let's now compare this score with the one using the unprocessed data :


```python
pipeline_log = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('log', LogisticRegression()),
])
emote_cut = emote.drop(emote.index[50000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)

grid_search_log = GridSearchCV(pipeline_log, parameters_log, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_log.steps])
print("parameters:")
pprint(parameters_log)
t0 = time()
grid_search_log.fit(emote_cut.text, emote_cut.emotion)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_log.best_score_)
print("Best parameters set:")
best_parameters = grid_search_log.best_estimator_.get_params()
for param_name in sorted(parameters_log.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'log']
    parameters:
    {'log__C': (0.5, 0.75, 1.0),
     'log__penalty': ('l2',),
     'vect__ngram_range': ((1, 2), (1, 3))}
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    done in 68.993s
    
    Best score: 0.777
    Best parameters set:
    	log__C: 1.0
    	log__penalty: 'l2'
    	vect__ngram_range: (1, 2)
    

We achieved a prediction score of 77.7% with 50 000 observations using logistic regression on unprocessed data. We observe that the preprocess is actually hurting our prediction accuracy.

##### In this case, we fitted the same models once without any kind of preprocess and a second time using various preprocess methods. Selecting each of these methods separately (not shown here) guided us in the same direction. We found no prepocess techniques worth adding in the aim of better prediction accuracy for this dataset.

 

## Support Vector Machine (SVM)

One of the most efficient prediction tool we saw in class was Support Vector Machine. Here we fit it trying different set of tuning parameters.
First let's define a pipeline which includes the tokenizer, the term weighting scheme and our SVM model.


```python
pipeline_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('svm', LinearSVC()),
])
```

Then, we make a small dictionnary containing all the parameters we want to test during our Cross-Validation. (We have already fitted dozens of models with even larger sets. In order not to make it run several hours, we've decided to crop this parameter set to its few main components).


```python
parameters_svm = {
    # 'vect__max_df': (0.4, 0.5),
    # 'vect__max_features': (None, 50000, 200000, 400000),
    'vect__ngram_range': ((1, 2), (1, 3),),
    #'svm__penalty': ('l2', 'elasticnet'),
    # 'svm__loss': ('squared_hinge',),
    'svm__C': (0.6, 0.7, 0.8),
}
```

Then comes the Cross-Validation part. Here, all possible combination of parameters are tested on our training data in the aim of finding the parameter set that yields the best possible prediction score on various K-Folds. Here is the output for 320 000 observations :


```python
'''emote_cut = emote.drop(emote.index[3200000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)'''

grid_search_svm = GridSearchCV(pipeline_svm, parameters_svm, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_svm.steps])
print("parameters:")
pprint(parameters_svm)
t0 = time()
grid_search_svm.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_svm.best_score_)
print("Best parameters set:")
best_parameters = grid_search_svm.best_estimator_.get_params()
for param_name in sorted(parameters_svm.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'svm']
    parameters:
    {'svm__C': (0.7,),
     'svm__penalty': ('l2',),
     'vect__max_features': (None, 50000, 200000, 400000),
     'vect__ngram_range': ((1, 3),)}
    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    done in 875.074s
    
    Best score: 0.815
    Best parameters set:
    	svm__C: 0.7
    	svm__penalty: 'l2'
    	vect__max_features: None
    	vect__ngram_range: (1, 3)
    

And here the one for computing on 640 000 observations


```python
'''emote_cut = emote.drop(emote.index[6400000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)'''

grid_search_svm = GridSearchCV(pipeline_svm, parameters_svm, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_svm.steps])
print("parameters:")
pprint(parameters_svm)
t0 = time()
grid_search_svm.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_svm.best_score_)
print("Best parameters set:")
best_parameters = grid_search_svm.best_estimator_.get_params()
for param_name in sorted(parameters_svm.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'svm']
    parameters:
    {'svm__C': (0.6, 0.7, 0.8), 'vect__ngram_range': ((1, 2), (1, 3))}
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    done in 3555.632s
    
    Best score: 0.830
    Best parameters set:
    	svm__C: 0.8
    	vect__ngram_range: (1, 3)
    

##### Using half of our training dataset to fit some Cross validation models, we obtain a score of 83% on this SVM model with the above parameters. This is one of the best predictions we were able to make. 

 

In order to compare with the following methods, we fitted this additional SVM on 100 000 observations :


```python
emote_cut = emote.drop(emote.index[100000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)

grid_search_svm = GridSearchCV(pipeline_svm, parameters_svm, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_svm.steps])
print("parameters:")
pprint(parameters_svm)
t0 = time()
grid_search_svm.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_svm.best_score_)
print("Best parameters set:")
best_parameters = grid_search_svm.best_estimator_.get_params()
for param_name in sorted(parameters_svm.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'svm']
    parameters:
    {'svm__C': (0.6, 0.7, 0.8), 'vect__ngram_range': ((1, 2), (1, 3))}
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    done in 238.555s
    
    Best score: 0.794
    Best parameters set:
    	svm__C: 0.6
    	vect__ngram_range: (1, 2)
    

We achieve a score of 79.4% using less than 10% of our training data. This score will be compared with next models' ones.

 

## Logistic Classification

Another strong classifier is the logistic regression. Here we do the same steps as for the SVM part in order to compare final prediction scores.


```python
pipeline_log = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('log', LogisticRegression()),
])
```


```python
parameters_log = {
    # 'vect__max_df': (0.5,),
    'vect__ngram_range': ((1, 2), (1, 3)),
    'log__C': (0.5, 0.75, 1.0),
    'log__penalty': ('l2',),
}
```


```python
emote_cut = emote.drop(emote.index[100000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)

grid_search_log = GridSearchCV(pipeline_log, parameters_log, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_log.steps])
print("parameters:")
pprint(parameters_log)
t0 = time()
grid_search_log.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_log.best_score_)
print("Best parameters set:")
best_parameters = grid_search_log.best_estimator_.get_params()
for param_name in sorted(parameters_log.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'log']
    parameters:
    {'log__C': (0.5, 0.75, 1.0),
     'log__penalty': ('l2',),
     'vect__ngram_range': ((1, 2), (1, 3))}
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    done in 292.206s
    
    Best score: 0.786
    Best parameters set:
    	log__C: 1.0
    	log__penalty: 'l2'
    	vect__ngram_range: (1, 2)
    

    C:\Users\rened\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

Using the logistic regression model, we achieve a score of 78.6%, almost a 1% less than SVM on the sample size. Logistic classification is a quite accurate method.

 

## Multinomial Naive Bayes (MNB)

And a third very efficient classifier could be the Multinomial Naive Bayes one. Again, we apply the same methodology as before in order to find the best set of parameters for our model.


```python
pipeline_mnb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mnb', MultinomialNB()),
])
```


```python
parameters_mnb = {
    'vect__max_df': (0.5,),
    'vect__ngram_range': ((1, 2), (1, 3)),  
    'mnb__alpha': (0.75, 1),
    # 'mnb__penalty': ('l2','elasticnet'),
    # 'mnb__max_iter': (10, 50, 80),
}
```

Here, we are training on 100 000 observations.


```python
emote_cut = emote.drop(emote.index[100000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.1, random_state=37)

grid_search_mnb = GridSearchCV(pipeline_mnb, parameters_mnb, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline_mnb.steps])
print("parameters:")
pprint(parameters_mnb)
t0 = time()
grid_search_mnb.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search_mnb.best_score_)
print("Best parameters set:")
best_parameters = grid_search_mnb.best_estimator_.get_params()
for param_name in sorted(parameters_mnb.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'tfidf', 'mnb']
    parameters:
    {'mnb__alpha': (0.75, 1),
     'vect__max_df': (0.5,),
     'vect__ngram_range': ((1, 2), (1, 3))}
    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    done in 50.199s
    
    Best score: 0.778
    Best parameters set:
    	mnb__alpha: 1
    	vect__max_df: 0.5
    	vect__ngram_range: (1, 2)
    

Using only 100 000 observations, we were able here to achieve a score with 77.8% prediction using MNB, a bit less than using logistic regression and even lesser than SVM. These examples were meant to show the difference between each models. We are aware there is some arbitrary choices here in the choice of the parameters for the several Cross-Validation. However, we chose these parameters based on many attempts of finding the best accuracy for each type of model. Overall, SVM performed better than the 2 other shown models here.

 

## Long Short Term Memory (LSTM) Neural Network

This time, we try to apply a Neural Network method. After trying several RNN and CNN, we came across this method that yielded better accuracy results for us. To be in concordance with the chosen method, we vectorize our text sample using the Tokenizer function from 'Keras' package.


```python
emote_cut = emote.drop(emote.index[320000:1280000])
X_train, X_test, y_train, y_test = train_test_split((emote_cut.text + emote_cut.user), emote_cut.emotion, test_size=0.2, random_state=37)

max_features = 20000
nb_classes = 2
maxlen = 80
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
```

We implement an LSTM layer followed by a Dense one, ending up with a softmax activation as we're working on a binary outcome.


```python
batch_size = 32
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2)) 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=3,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
```

    Build model...
    Train...
    Epoch 1/3
    8000/8000 [==============================] - 665s 83ms/step - loss: 0.1807 - accuracy: 0.7862 - val_loss: 0.1699 - val_accuracy: 0.8058
    Epoch 2/3
    8000/8000 [==============================] - 664s 83ms/step - loss: 0.1516 - accuracy: 0.8295 - val_loss: 0.1667 - val_accuracy: 0.8099
    Epoch 3/3
    8000/8000 [==============================] - 712s 89ms/step - loss: 0.1353 - accuracy: 0.8509 - val_loss: 0.1721 - val_accuracy: 0.8092
    2000/2000 [==============================] - 44s 22ms/step - loss: 0.1721 - accuracy: 0.8092
    Test score: 0.17206811904907227
    Test accuracy: 0.8091718554496765
    Generating test predictions...
    WARNING:tensorflow:From <ipython-input-37-75d41c68389e>:22: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
    

This 80.92% prediction score reflects the accuracy taking 320 000 observations into fitting. Another one using the whole dataset went to 83% on kaggle.

 

# IV. Best Model Analysis & Kaggle Submission

Looking at all previous attempts, we ended up with Support Vector Machine as having the best predictive performance. The best pipeline is therefore constructed out of the Cross-Validated parameters.


```python
pipeline_SVM_BEST = Pipeline([
    ('vect', CountVectorizer(max_df = 0.5, ngram_range = (1, 3))),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(C=0.8 , penalty = 'l2', loss = 'squared_hinge')),
])
```

Thanks to this pipeline, we expect to have a prediction score of around 83.5%. Score that can fluctuate whether our model overfits or not our training sample. The following predictions are posted on Kaggle as our main results.


```python
pipeline_SVM_BEST.fit((emote.text + emote.user), emote.emotion)
```




    Pipeline(steps=[('vect', CountVectorizer(max_df=0.5, ngram_range=(1, 3))),
                    ('tfidf', TfidfTransformer()), ('clf', LinearSVC(C=0.8))])




```python
emote_test = pd.read_csv("MLUnige2021_test.csv")
```


```python
predictions_SVM = pipeline_SVM_BEST.predict((emote_test.text + emote_test.user))
```


```python
output=pd.DataFrame(data={"Id":emote_test["Id"],"emotion":predictions_SVM}) 
output.to_csv(path_or_buf=r"C:\Users\rened\Desktop\____Master in Statistics\__Machine Learning\Project\results_SVM_0.8.csv", index=False)
```

 

# V. Conclusion

This project represents our final class work in this Machine Learning course. We applied most of the methods and models seen in class in order to get the best predictive performance we could. Various data preprocessing methods were tried but none of them ended up increasing our predictive performance. A high number of models and optimization led us towards this final score of about 83% of sentiment prediction using the Support Vector Machine model. Many attempts at using RNN or CNN or alternative models such as BERT were unsuccessful in this case, with prediction scores a bit lower than our conventional model. 

 

# VI. Appendix

## BERT 
This is an attempt at implementing a sophisticated model trained by Google, BERT. 
Unfortunately, BERT model was computationaly too costly and so too difficult for us to implement looking at our time and computation power restrictions. Here's how it would have gone with more resources :

## EDA and Preprocessing


```python
#pip install torch
```


```python
import torch 
from tqdm.notebook import tqdm
```


```python
df = pd.read_csv("MLUnige2021_train.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2063391019</td>
      <td>Sun Jun 07 02:28:13 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>BerryGurus</td>
      <td>@BreeMe more time to play with you BlackBerry ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2000525676</td>
      <td>Mon Jun 01 22:18:53 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>peterlanoie</td>
      <td>Failed attempt at booting to a flash drive. Th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2218180611</td>
      <td>Wed Jun 17 22:01:38 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>will_tooker</td>
      <td>@msproductions Well ain't that the truth. Wher...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>2190269101</td>
      <td>Tue Jun 16 02:14:47 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>sammutimer</td>
      <td>@Meaghery cheers Craig - that was really sweet...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>2069249490</td>
      <td>Sun Jun 07 15:31:58 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>ohaijustin</td>
      <td>I was reading the tweets that got send to me w...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df50 = df.sample(1000)
df50['id'] = range(1, len(df50) + 1)
df50.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>emotion</th>
      <th>tweet_id</th>
      <th>date</th>
      <th>lyx_query</th>
      <th>user</th>
      <th>text</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>726359</th>
      <td>726359</td>
      <td>1</td>
      <td>1548913184</td>
      <td>Fri Apr 17 22:22:39 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>justmikeyhrc</td>
      <td>Sleep mode initiated...long day ahead. Hopeful...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22139</th>
      <td>22139</td>
      <td>1</td>
      <td>1978778261</td>
      <td>Sun May 31 00:26:11 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>babybyndi</td>
      <td>I love it when he wears Express clothes. Yumm,...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1047379</th>
      <td>1047379</td>
      <td>1</td>
      <td>2179011339</td>
      <td>Mon Jun 15 08:31:13 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>eyulo</td>
      <td>beautiful day in the city  it pays to live in ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>716599</th>
      <td>716599</td>
      <td>0</td>
      <td>2186036482</td>
      <td>Mon Jun 15 18:15:53 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>Beejangles</td>
      <td>Grocery shopping. Alone  no one ever comes wit...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>332822</th>
      <td>332822</td>
      <td>0</td>
      <td>2065610910</td>
      <td>Sun Jun 07 08:56:54 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>LucyMarie85</td>
      <td>@AnnaSaccone i know but its just rude  ahhh we...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df50.set_index('id', inplace=True)
df50 = df50[['emotion', 'text']]
df50.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emotion</th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Sleep mode initiated...long day ahead. Hopeful...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>I love it when he wears Express clothes. Yumm,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>beautiful day in the city  it pays to live in ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Grocery shopping. Alone  no one ever comes wit...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>@AnnaSaccone i know but its just rude  ahhh we...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df50.text.iloc[0]
```




    'Sleep mode initiated...long day ahead. Hopefully new things to share tomorrow. Anyone want to see anything at the MK, let me know. '



## Train test split


```python
X_train, X_test, y_train, y_test = train_test_split(
    df50.index.values,
    df50.emotion.values,
    test_size = 0.15, random_state = 42) # no stratification since balanced
```


```python
# Create a column in df50 saying whether data is in training or test set.
df50['data_type'] = ['not_set']*df50.shape[0]
```


```python
df50.loc[X_train, 'data_type'] = 'train'
df50.loc[X_test, 'data_type'] = 'test'
```


```python
df50.groupby(['emotion', 'data_type']).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>emotion</th>
      <th>data_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>test</th>
      <td>75</td>
    </tr>
    <tr>
      <th>train</th>
      <td>434</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>test</th>
      <td>75</td>
    </tr>
    <tr>
      <th>train</th>
      <td>416</td>
    </tr>
  </tbody>
</table>
</div>



## Loading Tokenizer and Encoding the Data


```python
#pip install transformers
```


```python
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
```


```python
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
) #uncased for all lowercase data
```


    Downloading:   0%|          | 0.00/232k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/466k [00:00<?, ?B/s]



```python
encoded_data_train = tokenizer.batch_encode_plus(
    df50[df50.data_type=='train'].text.values,
    add_special_tokens=True,
    return_attention_mask = True,
    pad_to_max_length=True,
    max_length=110,
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    df50[df50.data_type=='test'].text.values,
    add_special_tokens=True,
    return_attention_mask = True,
    pad_to_max_length=True,
    max_length=110,
    return_tensors='pt'
)

# input for BERT to train
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
emotion_train = torch.tensor(df50[df50.data_type=='train'].emotion.values)

# input for BERT to test
input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
emotion_test = torch.tensor(df50[df50.data_type=='test'].emotion.values)
```

    Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
    C:\Users\rened\Anaconda3\lib\site-packages\transformers\tokenization_utils_base.py:2110: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      FutureWarning,
    


```python
# BERT's datatsets 
dataset_train = TensorDataset(input_ids_train, attention_masks_train, emotion_train)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, emotion_test)
```

## Setting up BERT pretrained model


```python
from transformers import BertForSequenceClassification
```


```python
# each tweet is a sequence that will be classified positive or negative emotion
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(df50.emotion.unique()), #how many output it can have 
    output_attentions=False, 
    output_hidden_states=False #doesn't show output 
)
```


    Downloading:   0%|          | 0.00/570 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/440M [00:00<?, ?B/s]


    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.weight']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

## Creating Data Loaders


```python
# Data loaders offer a nice way to iterate through our dataset in batches
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
```


```python
batch_size = 4 #32 previously but we have limited memory on these machines

dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size
)

dataloader_test = DataLoader(
    dataset_test,
    sampler=RandomSampler(dataset_test),
    batch_size=32 #back to 32 because we don't have many computations on the test set 
)
#our dataset is now in a dataloader
```

## Setting up Optimizer and Scheduler


```python
from transformers import AdamW, get_linear_schedule_with_warmup
```


```python
#optimizer (ADAM) is a way to optimize our weights 
optimizer = AdamW(
    model.parameters(),
    lr=1e-5, #recommended by the original paper to be between 2e-5 and 5e-5, can cross-validate this hyper-parameter
    eps=1e-8
)
```


```python
epochs = 5 #can be cross validated

#scheduler defines our learning rate and how it changes through each epoch 

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0, #default
    num_training_steps= len(dataloader_train)*epochs #defines how many times learning rate changes
)
```

## Defining our Performance Metrics


```python
from sklearn.metrics import f1_score


def f1_score_func(preds, emotion):
    preds_flat = np.argmax(preds, axis=1).flatten()
    emotion_flat = emotion.flatten()
    return f1_score(emotion_flat, preds_flat, average = 'weighted') #can put average=macro

def accuracy_per_class(preds, emotion):
    emotion_dict_inverse = {1, 0}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    emotion_flat = emotion.flatten()
    
    for emotion in np.unique(emotion_flat):
        y_preds = preds_flat[emotion_flat==emotion]
        y_true = emotion_flat[emotion_flat==emotion]
        print(f'Class:{emotion_dict_inverse[emotion]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
```

## Creating our Training Loop


```python
import random 

seed_test=42 #can try with seed 17 also (modify upper too)
random.seed(seed_test)
np.random.seed(seed_test)
torch.manual_seed(seed_test)
torch.cuda.manual_seed_all(seed_test) #useful if we use a GPU
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device) #best when cuda
```

    cpu
    


```python
def evaluate(dataloader_test):
    
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_test):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'inputs_ids': batch[0],
              'attention_mask': batch[1],
                     'labels': batch[2]}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        
        logits = logits.detach().cpu().numpy()
        emotion_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(emotion_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_test)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return loss_val_avg, predictions, true_vals
        
```


```python
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train, 
                        desc='Epoch {:1d}'.format(epoch), 
                        leave=False, disable=False) #to see how many batched have been trained and how many remain
    
    for batch in progress_bar:
        
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
             'input_ids': batch[0],
             'attention_mask': batch[1],
             'labels': batch[2]
         }
            
        outputs = model(**inputs)
         
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training loss': '{:.3f}'.format(loss.item()/len(batch))})
    
    torch.save(model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model')
    
    tqdm.write('\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total / len(dataloader)
    tqdm.write(f'Training loss:{loss_train_avg}')
    
    test_loss, predictions, true_vals = evaluate(dataloader_test)
    test_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Test loss:{test_loss}')
    tqdm.write(f'F1 score (weighted): {test_f1}')
```

## Loading and Evaluating model


```python
emotion_dict = {0, 1}
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                     num_labels = len(emotion_dict),
                                                     output_attentions = False, 
                                                     output_hidden_states = False)
```


```python
model.to(device)
pass
```


```python
model.load_state_dict(torch.load('Models/finetuned_bert_epoch_1_gpu_trained.model',
                                map_location=torch.device('cpu')))
```


```python
_, predictions, true_vals = evaluate(dataloader_test)
```


```python
accuracy_per_class(prediction, true_vals)
```
