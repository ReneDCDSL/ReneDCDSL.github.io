# Cuisine and Ingredients clustering

I have always been passionated about food and its polarity. It can make you strong and healthy or weak and sick. It can represent pure fuel for some people, or a form of artistic expression for others. It can be as simple as inserting a slice of cheese between 2 pieces of bread and as complicated as making a Michelin-star plate involving dozens of ingredients into very long and intricate processes. 
Having a meal is a whole experience we get so often yet can be so diverse. From getting a quick but heartwarming panini  sandwich for lunch in a busy work day, to sharing a relaxed Sunday roast with one's family.

As humans, food is a vital element of our lives. Although there are lots of food that can now be eaten raw thanks to our agricultural heritage, it usually involves some thorough choices: plants to breed in specific conditions, processes to preserve on long period of time, processes to make edible, techniques to increase productivity and food supply... The geographical, meteorological, social and cultural differences across our planet is the root of our wildlife diversity. Humans have cultivated plants for millenniums. It is a heritage that is visible through cuisine.

We have a rich and diverse culture around the world, but with people sharing common physiological roots, I was wondering if we could witness similarity in cuisines. By clustering cuisines through ingredients choices, I hope to see common traits arise between cuisine from people with different ethnicity, but who share either similar geographical conditions or whose ancestors have shared a past connection. Eventually, this approach could reveal unexpected similarity or differences between some cuisines.

To achieve this clustering effort, I have found a collection of pairs of recipe ingredients and cuisine origin. The data comes from [Yummly](https://www.yummly.com) , a recipe recommender website. The dataset contains about 40000 recipes from 20 regions. The aim is to use text processing techniques to then apply unsupervised Machine Learning techniques to find clusters of cuisines and ingredients. In a first part I will preprocess the text data, remove some stop words and get it accessible to our following algorithms. In the second part I model the ingredients lists using 3 methods: K-Means, Principal Component Analysis (PCA) then a Latent Dirichlet Allocation (LDA) model in order to find cluster of ingredients and see if it is possible to group regional cuisines together or find regular pattern in the ingredients choice.

## Imports


```python
# Basic Tools
import pandas as pd
import numpy as np
import mglearn
import itertools
import json
from IPython.display import display, HTML

# Visuals
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
from PIL import Image
%matplotlib inline

# Machine Learning
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA, KernelPCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
```


```python
# Dataset of ingredients list associated with a region from Yummly. 
df = pd.read_json('./data/Yummly/train.json')


# Load previous Within Cluster Sum of Squares results (K-Means)
with open("wcss_kmeans_res_1-24.json", "r") as fp:
    wcss_1_24 = json.load(fp)
```

###### Graphs helper


```python
# Labels prep
keys = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 
        'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole', 'brazilian', 
        'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']
color_vals = ['red', 'sienna', 'tan', 'gold', 'darkkhaki', 'chartreuse', 'darkgreen', 'darkcyan',
              'deepskyblue', 'royalblue', 'navy', 'darkorchid', 'mediumvioletred', 'grey', 'maroon',
              'coral', 'darkorange', 'olive', 'lightgreen','g']
colors = dict(zip(keys,color_vals))
markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8'] * 2
```

###### Functions


```python
def add_top_column(df, top_col, inplace=False):
    if not inplace:
        df = df.copy()
    
    df.columns = pd.MultiIndex.from_product([[top_col], df.columns])
    return df
```

## Exploratory Data Analysis


```python
print('Dataset shape: ', df.shape)
df.head()
```

    Dataset shape:  (39774, 3)
    




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
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39774 entries, 0 to 39773
    Data columns (total 4 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   id                  39774 non-null  int64 
     1   cuisine             39774 non-null  object
     2   ingredients         39774 non-null  object
     3   ingredients_string  39774 non-null  object
    dtypes: int64(1), object(3)
    memory usage: 1.2+ MB
    


```python
# Unique cuisines
print('Number of different cuisines: ', len(df['cuisine'].unique()))

# Unique meals
print('Number of unique meals: ', sum(df['id'].value_counts()))

# Missing ingredient list
print('Missing set of ingredients: ', sum(df['ingredients'].isna()))

# Missing cuisine
print('Missing meal origin: ', sum(df['cuisine'].isna()))
```

    Number of different cuisines:  20
    Number of unique meals:  39774
    Missing set of ingredients:  0
    Missing meal origin:  0
    


```python
# Count of recipes per cuisine
df['cuisine'].value_counts().to_frame().transpose()
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
      <th>italian</th>
      <th>mexican</th>
      <th>southern_us</th>
      <th>indian</th>
      <th>chinese</th>
      <th>french</th>
      <th>cajun_creole</th>
      <th>thai</th>
      <th>japanese</th>
      <th>greek</th>
      <th>spanish</th>
      <th>korean</th>
      <th>vietnamese</th>
      <th>moroccan</th>
      <th>british</th>
      <th>filipino</th>
      <th>irish</th>
      <th>jamaican</th>
      <th>russian</th>
      <th>brazilian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cuisine</th>
      <td>7838</td>
      <td>6438</td>
      <td>4320</td>
      <td>3003</td>
      <td>2673</td>
      <td>2646</td>
      <td>1546</td>
      <td>1539</td>
      <td>1423</td>
      <td>1175</td>
      <td>989</td>
      <td>830</td>
      <td>825</td>
      <td>821</td>
      <td>804</td>
      <td>755</td>
      <td>667</td>
      <td>526</td>
      <td>489</td>
      <td>467</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['cuisine'].value_counts().index
```




    Index(['italian', 'mexican', 'southern_us', 'indian', 'chinese', 'french',
           'cajun_creole', 'thai', 'japanese', 'greek', 'spanish', 'korean',
           'vietnamese', 'moroccan', 'british', 'filipino', 'irish', 'jamaican',
           'russian', 'brazilian'],
          dtype='object')




```python
# values to plot
val = df['cuisine'].value_counts().values
ind = df['cuisine'].value_counts().index

#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:5]

#create pie chart
plt.pie(val, labels = ind, colors = colors, autopct='%.0f%%')
plt.title('Distribution of Cuisines')
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Cuisine wordcloud

# Import bowl shape
bowl_mask = np.array(Image.open('./img/bowl.png'))

# Wordcloud
plt.subplots(figsize = (8,8))
wordcloud = WordCloud (
                    background_color = 'white',
                    width = 712,
                    height = 384,
                    mask=bowl_mask,
                    colormap = 'Set1').generate(' '.join(df['cuisine'].values))
plt.imshow(wordcloud) 
plt.axis('off') # remove axis
#plt.savefig('./img/cuisines.png')
plt.show()
```


    
![png](output_16_0.png)
    


The data contains mostly Italian, Mexican, Southern US, Indian, Chinese & French cuisine. It is not hard to guess that the data comes from an American website with such food influences. <br> 
The recipe are not equally distributed among all regions but there still are at least 467 lists of ingredients for each region. 


```python
# Ingredients wordcloud

# Import basket shape
basket_mask = np.array(Image.open('./img/basket.png'))

plt.subplots(figsize = (10,8))
wordcloud = WordCloud (
                    background_color = 'white',
                    width = 712,
                    height = 384,
                    mask=basket_mask,
                    colormap = 'BrBG').generate(df['ingredients'].str.join(' ').to_string())
plt.imshow(wordcloud) 
plt.axis('off') # remove axis
#plt.savefig('./img/ingredients.png')
plt.show()
```


    
![png](output_18_0.png)
    



```python
pd.Series([x for item in df['ingredients'] for x in item]).value_counts()[0:20].to_frame(name='Count')
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
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>salt</th>
      <td>18049</td>
    </tr>
    <tr>
      <th>olive oil</th>
      <td>7972</td>
    </tr>
    <tr>
      <th>onions</th>
      <td>7972</td>
    </tr>
    <tr>
      <th>water</th>
      <td>7457</td>
    </tr>
    <tr>
      <th>garlic</th>
      <td>7380</td>
    </tr>
    <tr>
      <th>sugar</th>
      <td>6434</td>
    </tr>
    <tr>
      <th>garlic cloves</th>
      <td>6237</td>
    </tr>
    <tr>
      <th>butter</th>
      <td>4848</td>
    </tr>
    <tr>
      <th>ground black pepper</th>
      <td>4785</td>
    </tr>
    <tr>
      <th>all-purpose flour</th>
      <td>4632</td>
    </tr>
    <tr>
      <th>pepper</th>
      <td>4438</td>
    </tr>
    <tr>
      <th>vegetable oil</th>
      <td>4385</td>
    </tr>
    <tr>
      <th>eggs</th>
      <td>3388</td>
    </tr>
    <tr>
      <th>soy sauce</th>
      <td>3296</td>
    </tr>
    <tr>
      <th>kosher salt</th>
      <td>3113</td>
    </tr>
    <tr>
      <th>green onions</th>
      <td>3078</td>
    </tr>
    <tr>
      <th>tomatoes</th>
      <td>3058</td>
    </tr>
    <tr>
      <th>large eggs</th>
      <td>2948</td>
    </tr>
    <tr>
      <th>carrots</th>
      <td>2814</td>
    </tr>
    <tr>
      <th>unsalted butter</th>
      <td>2782</td>
    </tr>
  </tbody>
</table>
</div>



We can see that among ingredients, spices and condiments appear very often, it is expected as they are fundamental ingredients in most recipes. We can also observe some ingredients that carry a lot of weight in some specific cuisines: lime juice, parmesan cheese, soy sauce, jalapeno chilie, green onion... These ingredients are specific to some regions and so I expect them to be marker of cuisines. 

## Text processing

In this part, we will apply some text vectorizing tools so that further algorithms can use the ingredients as inputs.

### Text preprocessing

We start by changing the lists of ingredients to a set of strings.


```python
# Ingredients as string to apply TfidVectorizer
df['ingredients_string'] = df['ingredients'].str.join(' ')
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
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
      <th>ingredients_string</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce black olives grape tomatoes ga...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>plain flour ground pepper salt tomatoes ground...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>eggs pepper salt mayonaise cooking oil green c...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>water vegetable oil wheat salt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>black pepper shallots cornflour cayenne pepper...</td>
    </tr>
  </tbody>
</table>
</div>



We can now regroup all the ingredients present in the recipes and apply the TF-IDF algorithm to get a measure of the importance of each ingredients in the whole corpus


```python
# Corpus of ingredients lists
list_corpus = df['ingredients_string'].tolist()

# Initiate TF-IDF
vectorizer = TfidfVectorizer()

# Fit TF-IDF
vectorizer.fit(list_corpus)

# Matrix of TF-IDF features
vector = vectorizer.transform(df['ingredients_string'])
```

We can rank the words by both IDF and TF-IDF scores


```python
# Column-wise max values
max_value = vector.max(axis=0).toarray().ravel()

# Ordered list of TF-IDF scores
sorted_by_tfidf = max_value.argsort()

# List of all features
feature_names = np.array(vectorizer.get_feature_names_out())

# Ordered list of IDF scores
sorted_by_idf = np.argsort(vectorizer.idf_)
```

### Observation of ingredients frequency ranking 

Here we observe the previously computed IDF and TF-IDF rankings of our corpus


```python
# Features with highest TF-IDF scores
print("Features with the highest tfidf:\n{}".format(feature_names[sorted_by_tfidf[-50:]]))
```

    Features with the highest tfidf:
    ['piri' 'meyer' 'coconut' 'bhaji' 'peanut' 'chiles' 'spaghettini' 'bacon'
     'cho' 'nopales' 'walnuts' 'gram' 'vegetable' 'cachaca' 'seaweed' 'vodka'
     'hollandaise' 'watermelon' 'yucca' 'yuca' 'plantains' 'okra' 'cherry'
     'half' 'crab' 'almonds' 'jasmine' 'manioc' 'artichokes' 'sushi' 'fried'
     'wafer' 'duck' 'umeboshi' 'pozole' 'polenta' 'coffee' 'sticky' 'jam'
     'raspberries' 'pappadams' 'espresso' 'barley' 'peanuts' 'breadfruit'
     'butter' 'udon' 'grained' 'phyllo' 'water']
    


```python
# Features with highest TF-IDF scores
print("Features with the highest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:50]]))
```

    Features with the highest tfidf:
    ['multi' 'garland' 'teff' 'psyllium' 'cotto' 'slim' 'blueberri'
     'fruitcake' 'patties' 'romanesco' 'knoflook' 'olie' 'wok' 'gember'
     'woksaus' 'specials' 'harvest' 'hurst' 'parslei' 'moss'
     'chocolatecovered' 'vineyard' 'burgundi' 'premium' 'collect' 'pat'
     'ocean' 'sheet' 'true' 'souchong' 'ginkgo' 'serving' 'lb' 'to'
     'nonhydrogenated' 'better' 'than' 'creations' 'dijonnaise' 'loose'
     'poured' 'fondant' 'gel' 'hillshire' 'layer' 'legumes' 'sections' 'tube'
     'america' 'chilcostle']
    


```python
# Features with lowest IDF scores
print("Features with the lowest idf:\n{}".format(feature_names[sorted_by_idf[:100]]))
```

    Features with the lowest idf:
    ['salt' 'oil' 'pepper' 'garlic' 'ground' 'fresh' 'onions' 'sugar' 'olive'
     'sauce' 'black' 'water' 'chicken' 'cheese' 'butter' 'tomatoes' 'flour'
     'red' 'green' 'cloves' 'powder' 'onion' 'juice' 'chopped' 'eggs' 'white'
     'cilantro' 'milk' 'rice' 'vegetable' 'cream' 'ginger' 'lemon' 'corn'
     'large' 'leaves' 'vinegar' 'all' 'purpose' 'soy' 'cumin' 'broth' 'dried'
     'lime' 'wine' 'chili' 'parsley' 'bell' 'beans' 'kosher' 'carrots'
     'grated' 'extra' 'dry' 'basil' 'brown' 'unsalted' 'parmesan' 'sesame'
     'virgin' 'chilies' 'beef' 'paste' 'oregano' 'boneless' 'seeds' 'cinnamon'
     'potatoes' 'cooking' 'shredded' 'tomato' 'baking' 'thyme' 'pork' 'egg'
     'shrimp' 'fat' 'bread' 'skinless' 'yellow' 'tortillas' 'seasoning' 'low'
     'chile' 'diced' 'sodium' 'cayenne' 'breasts' 'vanilla' 'celery' 'bay'
     'coriander' 'whole' 'spray' 'leaf' 'minced' 'mushrooms' 'sour' 'crushed'
     'flakes']
    

### Stopwords sets

Some words are present very often in recipes and so they do not carry much meaning and specificity towards cuisines. In this part we try a couple of stop words sets to be removed. 

#### Common stopwords


```python
# Vectorizer
count_vect = CountVectorizer(stop_words = ENGLISH_STOP_WORDS)

# Feature matrix
counts = count_vect.fit_transform(df["ingredients_string"])
```

#### Common stopwords + low IDF features 


```python
# Group common stopwords and low IDF features
custom_stop_words = list(ENGLISH_STOP_WORDS) + feature_names[sorted_by_idf[:30]].tolist()
print('Length of stopwords list: ', len(custom_stop_words))

# Vectorizer
count_vect2 = CountVectorizer(stop_words=custom_stop_words)

# Feature matrix
counts2 = count_vect2.fit_transform(df["ingredients_string"])
print('Shape of Vectorized feature matrix: ', counts2.shape)

# Remaining features list
words = count_vect2.get_feature_names_out()
print('Number of remaining features: ', len(words))
```

    Length of stopwords list:  348
    Shape of Vectorized feature matrix:  (39774, 2940)
    Number of remaining features:  2940
    

## Clustering

Now that the data is ready to be processed, we can group ingredients by some similarity measures and frequency distribution. I will apply 3 algorithms: K-Means, PCA and LDA; to make clusters of ingredients commonly found together.

### K-means

First, I will try using K-Means to cluster the ingredients. I set the number of clusters as 25, hoping to recover the 20 different cuisine types with a few added clusters for flexibility. K-Means attempts at creating clusters through elements distance measures.


```python
# 1st Model
# Computation: 25 clusters(~6/7min), 2 clusters(30s)

number_of_clusters=25
km = KMeans(n_clusters = number_of_clusters)
km.fit(counts)
```




    KMeans(n_clusters=25)




```python
# Clusters observation

# Find centroids
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

# List of features
terms = count_vect.get_feature_names_out()

print("Top terms per cluster:")
for i in range(number_of_clusters):
    top_15_words = [terms[ind] for ind in order_centroids[i, :15]]
    print("Cluster {}: {}".format(i, ' '.join(top_15_words)))
```

    Top terms per cluster:
    Cluster 0: chicken broth salt pepper oil garlic fresh olive sodium chopped black white ground cloves fat
    Cluster 1: flour salt purpose water oil eggs sugar butter milk yeast dry vegetable large pepper olive
    Cluster 2: oil olive pepper salt garlic extra virgin fresh cloves tomatoes red ground black cheese wine
    Cluster 3: dried pepper garlic salt oil oregano tomatoes olive ground black cheese basil red fresh onion
    Cluster 4: onions salt pepper garlic oil green water tomatoes cloves olive vegetable red chicken fresh sauce
    Cluster 5: sugar water milk cream juice salt butter lemon fresh white vanilla brown orange cinnamon lime
    Cluster 6: ground salt oil powder cumin garlic coriander ginger seeds chili onions leaves masala green tomatoes
    Cluster 7: fresh cilantro lime salt chopped juice onion garlic oil pepper tomatoes ground chilies jalapeno cumin
    Cluster 8: oil sesame sauce soy garlic rice onions ginger pepper sugar seeds salt green vinegar fresh
    Cluster 9: pepper green onions chicken garlic bell salt celery sauce oil chopped rice tomatoes ground black
    Cluster 10: sauce oil soy pepper chicken garlic corn starch rice ginger sugar sesame water onions salt
    Cluster 11: salt cheese oil butter water fresh pepper milk garlic sauce cream juice tomatoes eggs onion
    Cluster 12: fresh oil pepper salt olive lemon garlic juice chopped ground parsley black cloves cheese tomatoes
    Cluster 13: chicken boneless skinless oil pepper salt garlic breasts onions breast broth fresh sauce halves cheese
    Cluster 14: cheese cream shredded cheddar green tortillas onions sour beans ground chicken corn tomatoes sauce salsa
    Cluster 15: pepper bell red oil salt garlic green fresh ground olive black onions tomatoes onion chopped
    Cluster 16: sauce soy garlic oil sugar onions pepper salt water rice ginger vinegar chicken fresh pork
    Cluster 17: fresh sauce lime fish sugar garlic oil cilantro juice rice leaves red pepper chicken coconut
    Cluster 18: baking sugar flour powder salt butter purpose eggs soda large milk unsalted vanilla buttermilk extract
    Cluster 19: cheese mozzarella parmesan pepper sauce garlic fresh grated salt ground ricotta shredded oil eggs pasta
    Cluster 20: ground pepper salt garlic fresh oil cumin black cloves ginger cinnamon chicken coriander onions chopped
    Cluster 21: cheese parmesan grated pepper salt oil garlic olive fresh butter ground black chicken cream onions
    Cluster 22: powder pepper ground garlic salt cumin chili black oil onion cheese chicken corn cilantro onions
    Cluster 23: pepper ground salt black oil garlic butter olive onions fresh white onion sauce flour water
    Cluster 24: sugar vanilla butter large extract cream milk salt flour eggs egg purpose unsalted yolks ground
    

Some of these clusters seem to recover usual ingredients found in ethnic cuisines (Cluster 6: Indian, Cluster 7: Mexican, Cluster 19: Italian), sometimes several cluster seems to be linked with similar cuisine but different dishes (Clusters 7/14/22 looks more or less Mexican), and some clusters represent common meal ingredients (Cluster 24: cake ingredients, Cluster 1: condiments).

The number of clusters was set arbitrarily to get a first sense of the data. Looking at the obtained results, it may not be the appropriate choice. There is a method called the Elbow method which can help us decide what would be an approximately good amount of clusters. By checking the cumulative sum of squared distances from each cluster's centroids, we can find a good number of clusters to fit our data.


```python
# Checking for optimal number of clusters
# K=range(1,5) :  3min20
# K=range(1,10):  17min
# K=range(10,25): 1hr10

#wcss = [] # Within-cluster sum of squares

K = range(1,25)
for k in K:
    km_ = KMeans(n_clusters=k)
    km_ = km_.fit(counts)
    wcss.append(km_.inertia_)

# Save results
#with open("wcss_kmeans_res_CHANGEHERE.json", "w") as fp:
#    json.dump(wcss, fp)
```


```python
# Plot the elbow curve

plt.plot(K, wcss, marker='o', linestyle='--') # can load previous results of wcss
plt.xlabel('K')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method to choose optimal K')
plt.show()
```


    
![png](output_49_0.png)
    


The Elbow method advise us to pick a number of clusters which is not too big but still encompasses enough dissimilarity between them. With this method, we should choose a number that sits in the angle of the curve. Here, the curve does not have a very sharp angle, I would take a value between 5 and 10 clusters. 


```python
# K-Means with 5 clusters

km_5 = KMeans(n_clusters=5)
km_5 = km_5.fit(counts) # Computation: 1'40
```


```python
# Clusters observation (K=5)

# Find centroids
order_centroids_5 = km_5.cluster_centers_.argsort()[:, ::-1]

print("Top terms per cluster:")
for i in range(5):
    top_15_words = [terms[ind] for ind in order_centroids_5[i, :15]]
    print("Cluster {}: {}".format(i, ' '.join(top_15_words)))
```

    Top terms per cluster:
    Cluster 0: sugar flour butter salt eggs purpose baking milk large vanilla powder cream extract water unsalted
    Cluster 1: salt cheese pepper oil onions garlic chicken water sauce fresh cream butter ground green sugar
    Cluster 2: sauce oil soy garlic pepper sugar sesame rice onions fresh ginger chicken salt green water
    Cluster 3: pepper ground salt garlic oil black onions powder chicken cumin red fresh green tomatoes onion
    Cluster 4: fresh oil pepper olive salt garlic cheese tomatoes cloves black ground red chopped parsley juice
    

There is the again the cake recipe cluster, and what I would attribute in order: a recipe for creamy chicken sauce, Asian cuisine ingredients, Mexican or Indian (southern) style meal, and Italian/Mediterranean ingredients


```python
# K-Means with 10 clusters

km_10 = KMeans(n_clusters=10)
km_10 = km_10.fit(counts) # Computation: 3min
```


```python
# Clusters observation (K=10)

# Find centroids
order_centroids_10 = km_10.cluster_centers_.argsort()[:, ::-1]

print("Top terms per cluster:")
for i in range(10):
    top_15_words = [terms[ind] for ind in order_centroids_10[i, :15]]
    print("Cluster {}: {}".format(i, ' '.join(top_15_words)))
```

    Top terms per cluster:
    Cluster 0: pepper bell red garlic salt green oil onions ground black chicken fresh tomatoes olive sauce
    Cluster 1: pepper salt black ground garlic oil onions chicken butter flour water fresh onion white sauce
    Cluster 2: powder salt garlic oil ground cumin chili onions cilantro tomatoes green leaves ginger pepper coriander
    Cluster 3: sugar flour butter salt eggs purpose baking large vanilla milk powder extract cream unsalted water
    Cluster 4: sugar salt water oil sauce fresh butter milk garlic juice onions cream chicken flour white
    Cluster 5: ground pepper salt garlic oil cumin fresh black cloves chicken onions olive chopped powder ginger
    Cluster 6: oil olive pepper salt garlic fresh tomatoes cheese cloves extra virgin black ground red wine
    Cluster 7: fresh juice chopped oil salt pepper garlic lime cilantro lemon olive ground cloves onion chicken
    Cluster 8: sauce oil soy garlic sesame sugar pepper rice onions ginger chicken salt fresh water green
    Cluster 9: cheese shredded pepper cream sauce ground salt garlic chicken onions cheddar parmesan tomatoes green oil
    

With 10 clusters, we can see similar structure to the 5 ones. With some ingredients for bakery (cluster 3), Asian cuisine (cluster 8) Italian/Mediterranean (clusters 0,6,9) Southern/Mexican/Indian/Sunny&Spicy cuisine (clusters 2,5,7) and casual white sauce/creamy chicken dish (clusters 1,2,4). These clusters seem to be follow the distribution of cuisine regions in the dataset, with a lot of Italian recipes (either Mediterranean ingredients like tomato, olive oil, garlic, or with the American fascination of chicken Alfredo with a creamy/buttery chicken sauce), followed by recipes from countries that are sunny and hot, where they have a lot of fresh ingredients and spices (cilantro, lime, lemon, tomatoes, garlic with various spices: coriander, cumin, chili, pepper, cloves..). <br>
We can clearly see how these recipes come from an American community from the initial cuisine distribution and clusters that emerges from the ingredients lists. These clusters are still fairly general, having at least the 25 clusters allowed for more nuances between the big categories (difference between Mexican and Indian or Japanese and Chinese for example)

------------------


```python
# 2nd model
# Computation: 25 clusters(~5min30)

number_of_clusters=25
km2 = KMeans(n_clusters = number_of_clusters)
km2.fit(counts2)
```




    KMeans(n_clusters=25)




```python
# Clusters observation

# Find centroids
order_centroids2 = km2.cluster_centers_.argsort()[:, ::-1]

# List of features
terms2 = count_vect2.get_feature_names_out()

for i in range(number_of_clusters):
    top_15_words = [terms2[ind] for ind in order_centroids2[i, :15]]
    print("Cluster {}: {}".format(i, ' '.join(top_15_words)))
```

    Cluster 0: baking purpose soda buttermilk large unsalted cream vanilla corn cinnamon extract brown granulated yellow shortening
    Cluster 1: corn tortillas beans starch cumin chili frozen cream yellow kernels chilies broth purpose jack chile
    Cluster 2: parsley leaf flat wine lemon extra virgin dry bay thyme large broth carrots leaves celery
    Cluster 3: vinegar wine cider leaves balsamic mustard soy brown kosher bay pork cucumber purple basil lemon
    Cluster 4: lime chilies cumin jalapeno avocado corn purple tortillas chili chile leaves beans wedges kosher oregano
    Cluster 5: large egg yolks whites cream purpose unsalted vanilla lemon extract chocolate spray heavy cooking fat
    Cluster 6: soy corn starch ginger sesame vinegar wine boneless scallions sodium broth skinless pork chili breasts
    Cluster 7: boneless skinless breasts breast halves broth cumin ginger cream lime seasoning bell lemon dried chili
    Cluster 8: sesame seeds soy toasted ginger vinegar scallions carrots chili brown flakes honey paste noodles sodium
    Cluster 9: beef broth carrots tomato potatoes wine stock lean dried purpose thyme paste parsley bay leaves
    Cluster 10: dried oregano basil thyme tomato parsley leaves wine bay crushed diced broth bell celery parmesan
    Cluster 11: vanilla extract cream purpose large cinnamon unsalted brown baking chocolate egg heavy granulated corn light
    Cluster 12: cumin coriander ginger seeds chili leaves masala garam seed turmeric cinnamon paste chilies curry mustard
    Cluster 13: extra virgin leaves vinegar lemon wine basil kosher large parsley sea bread olives purple parmesan
    Cluster 14: cream sour tortillas corn shredded salsa cumin beans cheddar chili chilies lime jack fat avocado
    Cluster 15: shredded cheddar beans tortillas beef cream salsa taco seasoning corn chili cumin sharp jack sour
    Cluster 16: lemon parsley grated zest leaves large cream shrimp unsalted orange cinnamon wine dry purpose ginger
    Cluster 17: fat free broth sodium cooking spray low reduced cream purpose dried dry parmesan wine cumin
    Cluster 18: soy ginger sesame vinegar pork scallions carrots chili wine chinese noodles brown sodium dark cabbage
    Cluster 19: yeast dry purpose warm active bread large unsalted spray cooking unbleached cornmeal extra virgin kosher
    Cluster 20: bell celery yellow broth seasoning sausage shrimp leaves diced parsley thyme bay cayenne beans tomato
    Cluster 21: purpose potatoes cream large leaves broth unsalted pork kosher parsley bread coconut beans cinnamon carrots
    Cluster 22: lime fish leaves coconut thai paste ginger curry chili basil shrimp brown chile peanuts shallots
    Cluster 23: reggiano parmigiano extra virgin unsalted large leaves basil wine purpose dry kosher parsley cream broth
    Cluster 24: parmesan grated basil mozzarella parsley pasta italian cream ricotta broth large dried wine dry leaves
    

Could remove bit more stopwords (fat, sodium, free, low, large, dried, meal...) but at the same time, some words are more common in specific region. For example, maybe most regions say salt but american could be used to say sodium instead, or 'free' as in free-range chicken/eggs may be something more present in western culture so it still carries some meaning and distinction.

--------------

### PCA

Principal Component Analysis is a popular technique to derive a set of low dimensional features from a much larger set while still preserving as much variance as possible. It is often used to do variable selection or to visualize high-dimensional data.
Here we will use it for the latter purpose. We start with 3010 features which represent the total number of ingredients in our recipes and reduce it down to the 2 principal components which we'll be able to plot on a graph.

####  Overview of cuisines

Here we apply PCA on our whole dataset, keeping only the 2 principal components


```python
# Model
pca = PCA(n_components=2)

# Fit & Transform
counts_array = counts.toarray()
pca.fit(counts_array)
counts_pca = pca.transform(counts_array)
```


```python
print("Original shape: {}".format(str(counts.shape)))
print("Reduced shape: {}".format(str(counts_pca.shape)))
```

    Original shape: (39774, 2970)
    Reduced shape: (39774, 2)
    

We can then plot all the recipes in function of their principal components


```python
# Graph
plt.figure(figsize = (10,8))
mglearn.discrete_scatter(counts_pca[:,0], 
                         counts_pca[:,1], 
                         df.cuisine, 
                         alpha=.3)

plt.legend(['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican',
            'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole', 'brazilian', 'french', 'japanese',
            'irish', 'korean', 'moroccan', 'russian'],bbox_to_anchor=(1.02, 1), loc=2)
plt.title("Cuisines of the world",fontsize=16)
plt.xlabel("First principal component",fontsize=14)
plt.ylabel("Second principal component",fontsize=14);

#plt.savefig('./img/recipes_pca.png',dpi=300,bbox_inches='tight')
```


    
![png](output_69_0.png)
    


The recipes are globally well packed up, although we can see some distinctive regions (pink squares at the bottom, turquoise hexagons at the top). One great thing is that, from the few clusters we can see, they don't have too large intra-cluster variance, we can see clear grouping of them. We could try to reduce the number of cuisines to see clearer separations.

#### Focus on few cuisines

Here I arbitrarily chose to  focus on Japanese, Greek, Russian & Cajun/Creole cuisines.


```python
# Selection
selected_cuisines = ['cajun_creole','japanese','greek','russian']

# Graph
fig, ax = plt.subplots(figsize = (10,8))
for j, g in enumerate(np.unique(selected_cuisines)):
    ix = np.where(df['cuisine'] == g)[0]
    ax.scatter(counts_pca[ix[:200], 0], 
               counts_pca[ix[:200], 1], 
               c=colors[g], 
               label=g, 
               alpha=.7, 
               marker=markers[j], 
               s=50)

ax.legend(bbox_to_anchor=(1.02, 1), loc=2)
ax.set_xlabel("First principal component",fontsize=14)
ax.set_ylabel("Second principal component",fontsize=14)
ax.set_title("4 Cuisines' cluster",fontsize=16);

#fig.savefig('./img/recipes_4_selected_pca.png',dpi=300,bbox_inches='tight')
```


    
![png](output_73_0.png)
    


From this second graph, we can observe more distinction between the cuisines. There are still some heavy overlapping in the middle-left part of the graph but they all seem to diverge in a different direction. Japanese cuisine tend to have a higher degree of the 2nd principal component, while Greek and Cajun/Creole have on average a higher amount of the 1st principal component. 
<br>
To reduce the noise in these graphs and get another view of the resulting PCA model, we can focus on cuisine centroids, the average points of all recipes.

#### Centroids


```python
# Graph
fig, ax = plt.subplots(figsize = (10,8))
for j, g in enumerate(np.unique(keys)):
    ix = np.where(df['cuisine'] == g)[0]
    ax.scatter(np.mean(counts_pca[ix, 0]), 
               np.mean(counts_pca[ix, 1]), 
               c=colors[g], 
               label=g, 
               alpha=.7, 
               marker=markers[j], 
               s=100)
    
ax.legend(bbox_to_anchor=(1.02, 1), loc=2)
ax.set_xlabel("First principal component",fontsize=14)
ax.set_ylabel("Second principal component",fontsize=14)
ax.set_title("Plot of centroids for each cuisine",fontsize=16);

#fig.savefig('./img/recipes_centroids_pca.png',dpi=300,bbox_inches='tight')
```


    
![png](output_76_0.png)
    


This centroid plot is way easier to read. We can see some clusters of cuisines emerging. For example, the 4 points at the very top are Chinese, Korean, Thai & Vietnamese cuisines, which are regionally very close. Japanese and Filipino cuisines are not too far from the first 4, but with each their own offset on the graph. On the bottom left we can see British, Irish, Russian, Southern_us and French cuisine making another cluster of European meals. On the bottom middle right part, another cluster is formed with Italian, Mexican, Indian, Spanish, Greek, Jamaican, Cajun/Creole and a bit further, Moroccan cuisines. These are a mix of countries with cuisine full of spices and Mediterranean countries. A common ingredient shared by these countries could be the tomato. <br>
Although reducing such high dimensional data on only 2 dimensions may seem extreme and lack depth in the specificity between cuisines, we can already see very logical patterns that underlines a working methodology. 

#### Ingredients association 

Another way to look at our PCA results is to look at the way ingredients were classified along the 2 principal components. We can get these 2 values for each ingredient, plot them, then check what are the similarity between them, and the logic the model has been trying to extract.


```python
# Ingredients principal components table
words = count_vect.get_feature_names_out()
pca_df = pd.DataFrame(pca.components_,
                      columns=words,
                      index=['first', 'second'])
pca_df
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
      <th>00</th>
      <th>10</th>
      <th>100</th>
      <th>14</th>
      <th>15</th>
      <th>25</th>
      <th>33</th>
      <th>40</th>
      <th>43</th>
      <th>95</th>
      <th>...</th>
      <th>za</th>
      <th>zatarain</th>
      <th>zatarains</th>
      <th>zero</th>
      <th>zest</th>
      <th>zesty</th>
      <th>zinfandel</th>
      <th>ziti</th>
      <th>zucchini</th>
      <th>épices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first</th>
      <td>-0.000020</td>
      <td>-0.000011</td>
      <td>-0.000029</td>
      <td>0.000020</td>
      <td>-0.00001</td>
      <td>0.000015</td>
      <td>0.000041</td>
      <td>0.000035</td>
      <td>-2.806522e-05</td>
      <td>0.000087</td>
      <td>...</td>
      <td>0.000047</td>
      <td>-0.000016</td>
      <td>-0.000008</td>
      <td>-0.000005</td>
      <td>-0.001009</td>
      <td>-0.000055</td>
      <td>-0.000032</td>
      <td>0.000320</td>
      <td>0.012891</td>
      <td>-0.000048</td>
    </tr>
    <tr>
      <th>second</th>
      <td>-0.000003</td>
      <td>-0.000036</td>
      <td>-0.000001</td>
      <td>-0.000144</td>
      <td>-0.00007</td>
      <td>-0.000032</td>
      <td>-0.000035</td>
      <td>-0.000118</td>
      <td>1.741740e-07</td>
      <td>-0.000089</td>
      <td>...</td>
      <td>-0.000034</td>
      <td>0.000006</td>
      <td>0.000009</td>
      <td>0.000026</td>
      <td>-0.006452</td>
      <td>-0.000226</td>
      <td>-0.000018</td>
      <td>-0.000639</td>
      <td>-0.000763</td>
      <td>-0.000047</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 2970 columns</p>
</div>



Since there are a lot of ingredients in our corpus, we won't be able to plot them all. To get ingredients with various polarity, I will just extract several elements with the highest and lowest values for each of the 2 principal components.


```python
# Ingredients selection
select_ingredients = np.empty([20,], dtype=object)
select_ingredients[:5] = pca_df.iloc[:, np.argsort(pca_df.loc['first'])[-5:]].columns
select_ingredients[5:10] = pca_df.iloc[:, np.argsort(pca_df.loc['first'])[:5]].columns
select_ingredients[10:15] = pca_df.iloc[:, np.argsort(pca_df.loc['second'])[-5:]].columns
select_ingredients[15:] = pca_df.iloc[:, np.argsort(pca_df.loc['second'])[:5]].columns

# Graph
fig, ax = plt.subplots(figsize = (10,8))
for i in select_ingredients:
    x = [0, pca_df.loc['first', i]]
    y = [0, pca_df.loc['second', i]]
    ax.plot(x, y, label=i)
    ax.text(x[1]+.005, y[1], i, fontsize=12)
    
ax.set_xlabel("First principal component", fontsize=14)
ax.set_ylabel("Second principal component", fontsize=14)
ax.set_title("Ingredients most associated with first and second components", fontsize=16)
ax.set_xlim([-.37,.43]);

#fig.savefig('./img/ingredients_pca.png',dpi=300)
```


    
![png](output_82_0.png)
    


This graph shows the vectorized position of the ingredients along the 2 principal components of the model. Again, we can see some groups of ingredients that share really similar attributes. Soy, sesame, rice, sauce, oil, garlic, fresh on the top right corner are all ingredients commonly found in Asian cuisine. While on the bottom left side: flour, butter, eggs, vanilla & sugar are very common pastry ingredients. <br>
It is interesting to see how just 2 variables can capture so much information about a very large number of various ingredients.

Our simple 2 dimensional PCA model has been working great. However, we don't know how much it really captured overall nuances. We can check this with the explained variance ratio which is a ratio of the variance a model with K components can keep compared to one with all of its components.

#### Cumulative explained variance

Here I plot the cumulative explained variance for all k combination of principal components.


```python
# 1min to fit
# Model
pca2 = PCA()

# Fit
pca2.fit(counts_array)

# Check explained variance ratio
print(pca2.explained_variance_ratio_)

# Plot the Cumulative Explained Variance
plt.figure(figsize=(10,8))
plt.plot(range(1,2971), pca2.explained_variance_ratio_.cumsum(), marker='o', linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative Explained Variance');
```

    [7.07604993e-02 4.00388345e-02 3.49737761e-02 ... 4.18149272e-36
     3.45463899e-36 3.56824283e-37]
    


    
![png](output_87_1.png)
    


This curve represent the amount of variance a PCA with k components would contain. We can see a very sharp curve, increasing very fast on the first few several hundreds components, before curving horizontally. This means that a model that would have 500 principal components would contain almost 95% of the variance coming from all ingredients. Unfortunately, we can see that a model with just 2 principal components does not carry a majority of the total variance.


```python
print('Variance ratio of the first 2 principal components: {}%'.format(round(sum(pca2.explained_variance_ratio_[0:2])*100,2)))
```

    Variance ratio of the first 2 principal components: 11.08%
    

Our model with 2 principal components captures only 11% of total variance, yet it still manages to understand a lot of similarity between ingredients and cuisines. <br>
Here are the number of components that would be needed, with our current dataset, to carry more overall variance:


```python
# Choice of number of principle components
print('Number of Principle Components to keep 50% of data variance: ', np.where(pca2.explained_variance_ratio_.cumsum()>0.5)[0][0])
print('Number of Principle Components to keep 80% of data variance: ', np.where(pca2.explained_variance_ratio_.cumsum()>0.8)[0][0])
print('Number of Principle Components to keep 90% of data variance: ', np.where(pca2.explained_variance_ratio_.cumsum()>0.9)[0][0])
print('Number of Principle Components to keep 95% of data variance: ', np.where(pca2.explained_variance_ratio_.cumsum()>0.95)[0][0])
print('Number of Principle Components to keep 99% of data variance: ', np.where(pca2.explained_variance_ratio_.cumsum()>0.99)[0][0])
print('Initial number of features (ingredients): ', len(feature_names))
```

    Number of Principle Components to keep 50% of data variance:  29
    Number of Principle Components to keep 80% of data variance:  143
    Number of Principle Components to keep 90% of data variance:  297
    Number of Principle Components to keep 95% of data variance:  502
    Number of Principle Components to keep 99% of data variance:  1112
    Initial number of features (ingredients):  3010
    

### LDA

LDA is used to discover latent (hidden) topics within data thanks to Dirichlet distributions. There are 2 Dirichlet distributions used in LDA, one over the topics (here, ideally the cuisine) in each document and another over the words (ingredients) in each topic. Contrarily to PCA and other algorithms that use distance measures to determine similarity, LDA is based on the frequency counts of words within topics.


```python
# Model
# Computation: 25_components(5'30)

lda = LatentDirichletAllocation(n_components=25, 
                                learning_method="batch", 
                                max_iter=25, 
                                random_state=0)

# Fit
recipe_topics = lda.fit_transform(counts)
print("lda.components_.shape: {}".format(lda.components_.shape))
```

    lda.components_.shape: (25, 2970)
    


```python
sorting = np.argsort(lda.components_, axis=1)[:,::-1]
feature_names_2 = np.array(count_vect.get_feature_names_out())
```


```python
mglearn.tools.print_topics(topics=range(25),
                           feature_names=feature_names_2,
                           sorting=sorting,
                           topics_per_chunk=9,
                           n_words=10)
```

    topic 0       topic 1       topic 2       topic 3       topic 4       topic 5       topic 6       topic 7       topic 8       
    --------      --------      --------      --------      --------      --------      --------      --------      --------      
    pork          pepper        dried         ground        cheese        sesame        cheese        sugar         chicken       
    pepper        red           oregano       pepper        cream         oil           fresh         egg           boneless      
    ground        vinegar       tomato        salt          sour          sauce         parmesan      cream         skinless      
    salt          oil           garlic        cumin         shredded      soy           oil           vanilla       breasts       
    garlic        olive         pepper        garlic        tortillas     rice          olive         milk          breast        
    sauce         salt          tomatoes      fresh         cheddar       seeds         pepper        extract       halves        
    vinegar       wine          paste         oil           green         sugar         grated        large         broth         
    black         onion         ground        paprika       corn          onions        garlic        butter        pepper        
    beef          fresh         salt          cayenne       beans         vinegar       basil         yolks         oil           
    sugar         black         thyme         black         onions        garlic        salt          chocolate     thighs        
    
    
    topic 9       topic 10      topic 11      topic 12      topic 13      topic 14      topic 15      topic 16      topic 17      
    --------      --------      --------      --------      --------      --------      --------      --------      --------      
    water         olive         fresh         fresh         seeds         mustard       sauce         fat           lemon         
    flour         extra         lime          wine          salt          butter        oil           cooking       juice         
    dry           oil           juice         parsley       powder        meat          soy           low           fresh         
    salt          virgin        mint          white         oil           bread         garlic        sodium        orange        
    bread         salt          cilantro      leaf          coriander     dijon         ginger        spray         zest          
    oil           garlic        leaves        dry           cumin         sea           pepper        broth         grated        
    yeast         cloves        fish          salt          ginger        mayonaise     starch        chicken       peel          
    purpose       pepper        sugar         oil           onions        apples        corn          reduced       apple         
    crumbs        tomatoes      sauce         chopped       chili         fine          onions        free          sugar         
    warm          fresh         chopped       butter        seed          eggs          sesame        chopped       dill          
    
    
    topic 18      topic 19      topic 20      topic 21      topic 22      topic 23      topic 24      
    --------      --------      --------      --------      --------      --------      --------      
    salt          yogurt        flour         cheese        cilantro      pepper        coconut       
    butter        plain         purpose       mozzarella    salt          bell          milk          
    pepper        greek         baking        sauce         lime          green         sauce         
    cheese        nonfat        salt          italian       fresh         onions        red           
    ground        style         eggs          shredded      garlic        garlic        fish          
    eggs          mixed         sugar         parmesan      onion         salt          lime          
    potatoes      vegetables    butter        ricotta       chilies       red           paste         
    cream         mango         powder        pasta         oil           sauce         curry         
    milk          milk          milk          seasoning     pepper        oil           rice          
    black         free          large         eggs          chopped       rice          oil           
    
    
    

## Conclusion

Thanks to the various techniques employed here, we managed to extract simple similarity measures between various cuisines and ingredients association. The dataset I used is not optimal for this task as it feels very influenced by American cooking and habits. I see several big biases that should be accounted for in a further analysis, to extract clearer insights:
- Get a more equal distribution of recipes
- Have the recipes written by locals or taken from locals/chefs with more precise ingredients choices (From experience and looking at the data, I know people often cut corners when doing international dishes (using lemon instead of limes, curry powder instead of specific set of spices, white wine instead of mirin etc.))
- keep n-gram of words as some ingredients share same component (ground black pepper VS bell pepper)

## Additional material

### PCA and K-Means

It is possible to combine PCA and K-Means algorithm for more efficient modelling. Indeed, we can reduce noisy information by preselecting the principal components through PCA before applying a K-Means algorithm for clear clustering.


```python
# Models
pca80 = PCA(n_components=143)
pca90 = PCA(n_components=297)
pca95 = PCA(n_components=502)

# Fit
pca80.fit(counts_array)

# Keep calculated resulting components scores for the elements in our dataset
scores_pca80 = pca80.transform(counts_array)
```


```python
# Applying K-means on 
# Computation time: K=range(1,5):7s, K=range(1,10):23s, K=range(1,30):3min25

wcss_pca80 = [] # Within-cluster sum of squares
K = range(1,30)
for k in K:
    km_pca80 = KMeans(n_clusters=k)
    km_pca80 = km_pca80.fit(scores_pca80)
    wcss_pca80.append(km_pca80.inertia_)
```


```python
# Plot the elbow curve

plt.plot(K, wcss_pca80, marker='o', linestyle='--')
plt.xlabel('K')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method to choose optimal K')
plt.show()
```


    
![png](output_104_0.png)
    


From this graph, we can determine the number of clusters we want to keep. Using the Elbow method, I decide to keep 10 clusters.
We can now implement the K-means clustering algorithm with the chosen number of clusters


```python
# K-means model
kmeans_pca80 = KMeans(n_clusters = 10,
                      init = 'k-means++',
                      random_state = 0)
# Model fit with principal components scores
kmeans_pca80.fit(scores_pca80)
```




    KMeans(n_clusters=10, random_state=0)




```python
# Results

df_pca80 = df.copy()
df_pca80['Cluster'] = kmeans_pca80.labels_
df_pca80.head(3)
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
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
      <th>ingredients_string</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce black olives grape tomatoes ga...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>plain flour ground pepper salt tomatoes ground...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>eggs pepper salt mayonaise cooking oil green c...</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(scores_pca80)[:2]
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>133</th>
      <th>134</th>
      <th>135</th>
      <th>136</th>
      <th>137</th>
      <th>138</th>
      <th>139</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.066841</td>
      <td>-0.510578</td>
      <td>0.430887</td>
      <td>1.154517</td>
      <td>0.620896</td>
      <td>-0.232621</td>
      <td>-0.331172</td>
      <td>-0.622670</td>
      <td>-0.157093</td>
      <td>-0.674858</td>
      <td>...</td>
      <td>-0.053482</td>
      <td>0.307238</td>
      <td>-0.148087</td>
      <td>-0.294309</td>
      <td>0.109994</td>
      <td>-0.238635</td>
      <td>0.128690</td>
      <td>-0.162350</td>
      <td>0.106591</td>
      <td>-0.162417</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.199452</td>
      <td>-1.316914</td>
      <td>-1.224489</td>
      <td>0.069642</td>
      <td>-0.309043</td>
      <td>-0.334106</td>
      <td>-0.108580</td>
      <td>-0.241645</td>
      <td>-0.232802</td>
      <td>0.287942</td>
      <td>...</td>
      <td>0.143459</td>
      <td>0.155415</td>
      <td>-0.012209</td>
      <td>-0.015597</td>
      <td>0.042306</td>
      <td>0.165428</td>
      <td>0.035422</td>
      <td>0.100817</td>
      <td>-0.018154</td>
      <td>-0.069029</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 143 columns</p>
</div>




```python
df_pcaXX = pd.concat([df_pca80, pd.DataFrame(scores_pca80)], axis=1)
df_pcaXX.head(3)
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
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
      <th>ingredients_string</th>
      <th>Cluster</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>...</th>
      <th>133</th>
      <th>134</th>
      <th>135</th>
      <th>136</th>
      <th>137</th>
      <th>138</th>
      <th>139</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce black olives grape tomatoes ga...</td>
      <td>9</td>
      <td>-0.066841</td>
      <td>-0.510578</td>
      <td>0.430887</td>
      <td>1.154517</td>
      <td>0.620896</td>
      <td>...</td>
      <td>-0.053482</td>
      <td>0.307238</td>
      <td>-0.148087</td>
      <td>-0.294309</td>
      <td>0.109994</td>
      <td>-0.238635</td>
      <td>0.128690</td>
      <td>-0.162350</td>
      <td>0.106591</td>
      <td>-0.162417</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>plain flour ground pepper salt tomatoes ground...</td>
      <td>1</td>
      <td>1.199452</td>
      <td>-1.316914</td>
      <td>-1.224489</td>
      <td>0.069642</td>
      <td>-0.309043</td>
      <td>...</td>
      <td>0.143459</td>
      <td>0.155415</td>
      <td>-0.012209</td>
      <td>-0.015597</td>
      <td>0.042306</td>
      <td>0.165428</td>
      <td>0.035422</td>
      <td>0.100817</td>
      <td>-0.018154</td>
      <td>-0.069029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>eggs pepper salt mayonaise cooking oil green c...</td>
      <td>6</td>
      <td>0.312870</td>
      <td>0.769161</td>
      <td>-0.839508</td>
      <td>0.648466</td>
      <td>0.097039</td>
      <td>...</td>
      <td>-0.016164</td>
      <td>-0.055193</td>
      <td>-0.173232</td>
      <td>0.008825</td>
      <td>0.034810</td>
      <td>0.147212</td>
      <td>0.132248</td>
      <td>0.128975</td>
      <td>-0.146260</td>
      <td>-0.044708</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 148 columns</p>
</div>




```python
# Plot data by PCA components
x_axis = df_pcaXX[0]
y_axis = df_pcaXX[1]
plt.figure(figsize=(10,8))
sns.scatterplot(x = x_axis, 
                y = y_axis, 
                hue=df_pcaXX['Cluster'], 
                palette=color_vals[0:10])
plt.title('Clusters by PCA Components')
plt.show()
```


    
![png](output_110_0.png)
    



```python
pca80.components_.shape
```




    (143, 2970)




```python
order_centroids_pca80[0]
```




    array([  5,   6,   7,  11,  12,  10,   9,  44,  55,  61,  48,  23,  43,
            59,  50,  25,  16,  64,  52,  22,  60,  62,  34,  66,  31,  67,
            30,  33,  95,  19,  35,  63,  72, 116,  28,  56, 126,  38,  46,
           142, 115,  70,  90, 110, 100, 119,  15, 140, 112, 134,  92,  86,
            77,  97, 137,  82,  76, 138, 132, 131,  83,  21,  89,  29, 129,
            88,  74, 125, 128,  57,  49, 113, 106, 135,  68, 107, 127, 114,
            94,  96,  85, 109,  73, 139, 108, 133, 130,  98, 103,  99, 120,
           104,  24, 101, 102, 111, 121, 122, 141, 136,  75, 105,  78,  87,
            91, 124,  93,  69, 118, 117,  71,  84,  79,  40,  80,  39,  26,
            81,  47, 123,  65,  58,  41,  54,  36,  27,  32,  45,  53,  37,
            20,  42,  14,  17,  51,  18,  13,   8,   3,   2,   1,   4,   0],
          dtype=int64)




```python
pca80.components_[5].argmax()
```




    1030



### Interpretation

It is hard to extract clear interpretation from principal components but we can attempt it nonetheless. <br>
Since there were initially 3010 ingredients, from which we derived 143 principal components, we have created 143 combinations of 3010 variables.
Each cluster are a distinct combination of these 143 vectors.
It is possible to observe the vectors that represent best a cluster to get an idea of what it represents. Then observe what each of these vectors try to represent from the set of ingredients. <br>
Let's take a look at an example:


```python
# Focus on Principal Component number 5
print('The 5th principal component is a vectorial combination of initial features: ', pca80.components_[5])
print()

# Positions of the elements with largest score
PC5_top10_index = np.argsort(-pca80.components_[5])[:10]
print('Indices of the 10 elements with largest score: ', PC5_top10_index)
print('Scores at the 10 elements: ', [pca80.components_[5][x] for x in PC5_top10_index])
print('List of the specified elements: ', terms[PC5_top10_index])
```

    The 5th principal component is a vectorial combination of initial features:  [-5.14241991e-05 -4.67684294e-06  5.24665521e-07 ... -3.10662492e-05
     -2.53138824e-03  5.09900614e-05]
    
    Indices of the 10 elements with largest score:  [1030 1970 2618  529  369  570 1001  700 2346  515]
    Scores at the 10 elements:  [0.36335562672424143, 0.23305849121649114, 0.21503822237530185, 0.2104099335392782, 0.19537071914665008, 0.1845772773593622, 0.18228863936232156, 0.1742451416025657, 0.1699852299317262, 0.1508701887631323]
    List of the specified elements:  ['fresh' 'pepper' 'sugar' 'chicken' 'butter' 'chopped' 'flour' 'cream'
     'sauce' 'cheese']
    


```python
pd.DataFrame([pca80.components_[5][x] for x in PC5_top10_index],terms[PC5_top10_index], columns=['Score'])
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
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fresh</th>
      <td>0.363356</td>
    </tr>
    <tr>
      <th>pepper</th>
      <td>0.233058</td>
    </tr>
    <tr>
      <th>sugar</th>
      <td>0.215038</td>
    </tr>
    <tr>
      <th>chicken</th>
      <td>0.210410</td>
    </tr>
    <tr>
      <th>butter</th>
      <td>0.195371</td>
    </tr>
    <tr>
      <th>chopped</th>
      <td>0.184577</td>
    </tr>
    <tr>
      <th>flour</th>
      <td>0.182289</td>
    </tr>
    <tr>
      <th>cream</th>
      <td>0.174245</td>
    </tr>
    <tr>
      <th>sauce</th>
      <td>0.169985</td>
    </tr>
    <tr>
      <th>cheese</th>
      <td>0.150870</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.argsort(-pca80.components_[5])[:10]
```




    array([1030, 1970, 2618,  529,  369,  570, 1001,  700, 2346,  515],
          dtype=int64)




```python
pd.set_option('display.max_rows', 500)
number_of_PC = 10
number_of_components=100

def PC_content(pca_model = pca80, number_of_PC=10, number_of_components=10):
    df_total = pd.DataFrame()
    for PC in range(number_of_PC):
        ind = np.argsort(-pca_model.components_[PC])[:number_of_components]
        df_ = pd.DataFrame({'Ingredient':terms[ind], 'Score':[pca_model.components_[PC][x] for x in ind]})
        df_ = add_top_column(df_, "PC {}".format(PC))
        df_total = pd.concat([df_total,df_],axis=1)
    return df_total

PC_res = PC_content(pca80, number_of_PC, number_of_components)
PC_res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">PC 0</th>
      <th colspan="2" halign="left">PC 1</th>
      <th colspan="2" halign="left">PC 2</th>
      <th colspan="2" halign="left">PC 3</th>
      <th colspan="2" halign="left">PC 4</th>
      <th colspan="2" halign="left">PC 5</th>
      <th colspan="2" halign="left">PC 6</th>
      <th colspan="2" halign="left">PC 7</th>
      <th colspan="2" halign="left">PC 8</th>
      <th colspan="2" halign="left">PC 9</th>
    </tr>
    <tr>
      <th></th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pepper</td>
      <td>0.523607</td>
      <td>sauce</td>
      <td>0.474086</td>
      <td>fresh</td>
      <td>0.558289</td>
      <td>cheese</td>
      <td>0.359403</td>
      <td>cheese</td>
      <td>0.290647</td>
      <td>fresh</td>
      <td>0.363356</td>
      <td>cheese</td>
      <td>0.526328</td>
      <td>chicken</td>
      <td>0.538314</td>
      <td>chicken</td>
      <td>0.479741</td>
      <td>onions</td>
      <td>0.594707</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ground</td>
      <td>0.361922</td>
      <td>soy</td>
      <td>0.261672</td>
      <td>olive</td>
      <td>0.221516</td>
      <td>pepper</td>
      <td>0.318516</td>
      <td>ground</td>
      <td>0.280666</td>
      <td>pepper</td>
      <td>0.233058</td>
      <td>sauce</td>
      <td>0.342029</td>
      <td>oil</td>
      <td>0.303247</td>
      <td>ground</td>
      <td>0.202891</td>
      <td>green</td>
      <td>0.395903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>garlic</td>
      <td>0.258455</td>
      <td>oil</td>
      <td>0.255458</td>
      <td>cheese</td>
      <td>0.194892</td>
      <td>bell</td>
      <td>0.131593</td>
      <td>chicken</td>
      <td>0.173870</td>
      <td>sugar</td>
      <td>0.215038</td>
      <td>ground</td>
      <td>0.261870</td>
      <td>flour</td>
      <td>0.237056</td>
      <td>broth</td>
      <td>0.167949</td>
      <td>fresh</td>
      <td>0.144382</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fresh</td>
      <td>0.247066</td>
      <td>sesame</td>
      <td>0.193111</td>
      <td>juice</td>
      <td>0.144890</td>
      <td>parmesan</td>
      <td>0.095975</td>
      <td>cumin</td>
      <td>0.152478</td>
      <td>chicken</td>
      <td>0.210410</td>
      <td>oil</td>
      <td>0.238259</td>
      <td>powder</td>
      <td>0.201450</td>
      <td>white</td>
      <td>0.124195</td>
      <td>water</td>
      <td>0.140545</td>
    </tr>
    <tr>
      <th>4</th>
      <td>oil</td>
      <td>0.225165</td>
      <td>rice</td>
      <td>0.182921</td>
      <td>chopped</td>
      <td>0.133097</td>
      <td>red</td>
      <td>0.092441</td>
      <td>cilantro</td>
      <td>0.139846</td>
      <td>butter</td>
      <td>0.195371</td>
      <td>parmesan</td>
      <td>0.173779</td>
      <td>salt</td>
      <td>0.201422</td>
      <td>boneless</td>
      <td>0.114295</td>
      <td>butter</td>
      <td>0.116424</td>
    </tr>
    <tr>
      <th>5</th>
      <td>black</td>
      <td>0.200506</td>
      <td>ginger</td>
      <td>0.135998</td>
      <td>lemon</td>
      <td>0.114092</td>
      <td>green</td>
      <td>0.088739</td>
      <td>onions</td>
      <td>0.133142</td>
      <td>chopped</td>
      <td>0.184577</td>
      <td>eggs</td>
      <td>0.159981</td>
      <td>broth</td>
      <td>0.196694</td>
      <td>skinless</td>
      <td>0.104954</td>
      <td>parsley</td>
      <td>0.095025</td>
    </tr>
    <tr>
      <th>6</th>
      <td>red</td>
      <td>0.173708</td>
      <td>garlic</td>
      <td>0.129364</td>
      <td>tomatoes</td>
      <td>0.099469</td>
      <td>onions</td>
      <td>0.085838</td>
      <td>shredded</td>
      <td>0.126227</td>
      <td>flour</td>
      <td>0.182289</td>
      <td>soy</td>
      <td>0.153966</td>
      <td>purpose</td>
      <td>0.145914</td>
      <td>wine</td>
      <td>0.099634</td>
      <td>flour</td>
      <td>0.080278</td>
    </tr>
    <tr>
      <th>7</th>
      <td>olive</td>
      <td>0.154176</td>
      <td>onions</td>
      <td>0.121855</td>
      <td>parsley</td>
      <td>0.094211</td>
      <td>shredded</td>
      <td>0.079754</td>
      <td>cream</td>
      <td>0.111774</td>
      <td>cream</td>
      <td>0.174245</td>
      <td>sugar</td>
      <td>0.153048</td>
      <td>olive</td>
      <td>0.145514</td>
      <td>sodium</td>
      <td>0.093258</td>
      <td>tomatoes</td>
      <td>0.074155</td>
    </tr>
    <tr>
      <th>8</th>
      <td>salt</td>
      <td>0.142398</td>
      <td>chicken</td>
      <td>0.114661</td>
      <td>basil</td>
      <td>0.090868</td>
      <td>chicken</td>
      <td>0.079438</td>
      <td>sauce</td>
      <td>0.105296</td>
      <td>sauce</td>
      <td>0.169985</td>
      <td>grated</td>
      <td>0.151882</td>
      <td>garlic</td>
      <td>0.140513</td>
      <td>lemon</td>
      <td>0.091645</td>
      <td>ground</td>
      <td>0.073736</td>
    </tr>
    <tr>
      <th>9</th>
      <td>onions</td>
      <td>0.137449</td>
      <td>green</td>
      <td>0.107114</td>
      <td>virgin</td>
      <td>0.085163</td>
      <td>dried</td>
      <td>0.075665</td>
      <td>green</td>
      <td>0.102845</td>
      <td>cheese</td>
      <td>0.150870</td>
      <td>large</td>
      <td>0.136537</td>
      <td>boneless</td>
      <td>0.127796</td>
      <td>pepper</td>
      <td>0.081649</td>
      <td>celery</td>
      <td>0.070032</td>
    </tr>
    <tr>
      <th>10</th>
      <td>chicken</td>
      <td>0.136259</td>
      <td>vinegar</td>
      <td>0.096605</td>
      <td>extra</td>
      <td>0.084255</td>
      <td>olive</td>
      <td>0.069402</td>
      <td>chili</td>
      <td>0.102704</td>
      <td>green</td>
      <td>0.148715</td>
      <td>sesame</td>
      <td>0.113314</td>
      <td>skinless</td>
      <td>0.117180</td>
      <td>breasts</td>
      <td>0.072407</td>
      <td>purpose</td>
      <td>0.068313</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bell</td>
      <td>0.124041</td>
      <td>fresh</td>
      <td>0.074397</td>
      <td>parmesan</td>
      <td>0.080849</td>
      <td>tomatoes</td>
      <td>0.062320</td>
      <td>tortillas</td>
      <td>0.090132</td>
      <td>bell</td>
      <td>0.139145</td>
      <td>flour</td>
      <td>0.105754</td>
      <td>vegetable</td>
      <td>0.099373</td>
      <td>butter</td>
      <td>0.071125</td>
      <td>carrots</td>
      <td>0.068086</td>
    </tr>
    <tr>
      <th>12</th>
      <td>tomatoes</td>
      <td>0.120329</td>
      <td>fish</td>
      <td>0.072593</td>
      <td>cloves</td>
      <td>0.079385</td>
      <td>seasoning</td>
      <td>0.060247</td>
      <td>cheddar</td>
      <td>0.082945</td>
      <td>purpose</td>
      <td>0.123549</td>
      <td>fresh</td>
      <td>0.104770</td>
      <td>baking</td>
      <td>0.098777</td>
      <td>black</td>
      <td>0.067754</td>
      <td>large</td>
      <td>0.067652</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cloves</td>
      <td>0.116212</td>
      <td>lime</td>
      <td>0.067689</td>
      <td>grated</td>
      <td>0.065107</td>
      <td>cheddar</td>
      <td>0.058565</td>
      <td>chilies</td>
      <td>0.081820</td>
      <td>lime</td>
      <td>0.122732</td>
      <td>mozzarella</td>
      <td>0.095957</td>
      <td>breasts</td>
      <td>0.098549</td>
      <td>parsley</td>
      <td>0.067374</td>
      <td>eggs</td>
      <td>0.067609</td>
    </tr>
    <tr>
      <th>14</th>
      <td>chopped</td>
      <td>0.115483</td>
      <td>vegetable</td>
      <td>0.065785</td>
      <td>lime</td>
      <td>0.058621</td>
      <td>mozzarella</td>
      <td>0.053909</td>
      <td>tomatoes</td>
      <td>0.081759</td>
      <td>powder</td>
      <td>0.116595</td>
      <td>olive</td>
      <td>0.092369</td>
      <td>butter</td>
      <td>0.079509</td>
      <td>dry</td>
      <td>0.064610</td>
      <td>beef</td>
      <td>0.067235</td>
    </tr>
    <tr>
      <th>15</th>
      <td>green</td>
      <td>0.114732</td>
      <td>starch</td>
      <td>0.064323</td>
      <td>cilantro</td>
      <td>0.048782</td>
      <td>grated</td>
      <td>0.046306</td>
      <td>lime</td>
      <td>0.080672</td>
      <td>juice</td>
      <td>0.116008</td>
      <td>purpose</td>
      <td>0.084491</td>
      <td>sodium</td>
      <td>0.077236</td>
      <td>low</td>
      <td>0.060252</td>
      <td>bay</td>
      <td>0.062882</td>
    </tr>
    <tr>
      <th>16</th>
      <td>onion</td>
      <td>0.106438</td>
      <td>water</td>
      <td>0.060938</td>
      <td>oil</td>
      <td>0.048245</td>
      <td>cream</td>
      <td>0.046045</td>
      <td>sour</td>
      <td>0.080259</td>
      <td>milk</td>
      <td>0.106705</td>
      <td>shredded</td>
      <td>0.078137</td>
      <td>cloves</td>
      <td>0.074149</td>
      <td>breast</td>
      <td>0.059395</td>
      <td>milk</td>
      <td>0.061427</td>
    </tr>
    <tr>
      <th>17</th>
      <td>cumin</td>
      <td>0.091953</td>
      <td>sugar</td>
      <td>0.058103</td>
      <td>leaves</td>
      <td>0.044164</td>
      <td>pasta</td>
      <td>0.040633</td>
      <td>beans</td>
      <td>0.072472</td>
      <td>eggs</td>
      <td>0.099099</td>
      <td>basil</td>
      <td>0.072382</td>
      <td>low</td>
      <td>0.072517</td>
      <td>halves</td>
      <td>0.050602</td>
      <td>leaves</td>
      <td>0.060475</td>
    </tr>
    <tr>
      <th>18</th>
      <td>cilantro</td>
      <td>0.079403</td>
      <td>red</td>
      <td>0.056991</td>
      <td>mozzarella</td>
      <td>0.036951</td>
      <td>italian</td>
      <td>0.039979</td>
      <td>chopped</td>
      <td>0.067951</td>
      <td>cilantro</td>
      <td>0.095194</td>
      <td>butter</td>
      <td>0.071348</td>
      <td>onions</td>
      <td>0.070706</td>
      <td>rice</td>
      <td>0.049603</td>
      <td>ginger</td>
      <td>0.058146</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sauce</td>
      <td>0.078895</td>
      <td>scallions</td>
      <td>0.056617</td>
      <td>wine</td>
      <td>0.035994</td>
      <td>sausage</td>
      <td>0.038955</td>
      <td>coriander</td>
      <td>0.067198</td>
      <td>corn</td>
      <td>0.093846</td>
      <td>ricotta</td>
      <td>0.060311</td>
      <td>corn</td>
      <td>0.069142</td>
      <td>thyme</td>
      <td>0.046712</td>
      <td>thyme</td>
      <td>0.052169</td>
    </tr>
    <tr>
      <th>20</th>
      <td>dried</td>
      <td>0.077851</td>
      <td>chili</td>
      <td>0.055695</td>
      <td>onion</td>
      <td>0.035421</td>
      <td>black</td>
      <td>0.037628</td>
      <td>salsa</td>
      <td>0.063082</td>
      <td>large</td>
      <td>0.091795</td>
      <td>garlic</td>
      <td>0.056045</td>
      <td>white</td>
      <td>0.060407</td>
      <td>stock</td>
      <td>0.046481</td>
      <td>cloves</td>
      <td>0.051755</td>
    </tr>
    <tr>
      <th>21</th>
      <td>parsley</td>
      <td>0.063163</td>
      <td>seeds</td>
      <td>0.051120</td>
      <td>mint</td>
      <td>0.032104</td>
      <td>diced</td>
      <td>0.037227</td>
      <td>beef</td>
      <td>0.062215</td>
      <td>baking</td>
      <td>0.091554</td>
      <td>egg</td>
      <td>0.055201</td>
      <td>chopped</td>
      <td>0.057080</td>
      <td>fat</td>
      <td>0.036232</td>
      <td>potatoes</td>
      <td>0.049603</td>
    </tr>
    <tr>
      <th>22</th>
      <td>broth</td>
      <td>0.062468</td>
      <td>carrots</td>
      <td>0.049240</td>
      <td>purple</td>
      <td>0.030860</td>
      <td>oregano</td>
      <td>0.036143</td>
      <td>powder</td>
      <td>0.056900</td>
      <td>unsalted</td>
      <td>0.069346</td>
      <td>beef</td>
      <td>0.053656</td>
      <td>cumin</td>
      <td>0.055539</td>
      <td>soy</td>
      <td>0.034659</td>
      <td>leaf</td>
      <td>0.049564</td>
    </tr>
    <tr>
      <th>23</th>
      <td>leaves</td>
      <td>0.060077</td>
      <td>noodles</td>
      <td>0.046058</td>
      <td>dry</td>
      <td>0.030841</td>
      <td>celery</td>
      <td>0.033957</td>
      <td>ginger</td>
      <td>0.056317</td>
      <td>vanilla</td>
      <td>0.067369</td>
      <td>noodles</td>
      <td>0.053402</td>
      <td>breast</td>
      <td>0.054596</td>
      <td>celery</td>
      <td>0.034149</td>
      <td>bell</td>
      <td>0.048278</td>
    </tr>
    <tr>
      <th>24</th>
      <td>oregano</td>
      <td>0.054819</td>
      <td>sodium</td>
      <td>0.043745</td>
      <td>oregano</td>
      <td>0.025753</td>
      <td>beans</td>
      <td>0.032467</td>
      <td>corn</td>
      <td>0.047847</td>
      <td>fat</td>
      <td>0.065847</td>
      <td>cloves</td>
      <td>0.052762</td>
      <td>fat</td>
      <td>0.054326</td>
      <td>free</td>
      <td>0.033595</td>
      <td>dry</td>
      <td>0.047410</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ginger</td>
      <td>0.053660</td>
      <td>corn</td>
      <td>0.040552</td>
      <td>rosemary</td>
      <td>0.025563</td>
      <td>sauce</td>
      <td>0.031116</td>
      <td>breasts</td>
      <td>0.047448</td>
      <td>broth</td>
      <td>0.065733</td>
      <td>wine</td>
      <td>0.048750</td>
      <td>cream</td>
      <td>0.053227</td>
      <td>cinnamon</td>
      <td>0.029911</td>
      <td>paste</td>
      <td>0.046236</td>
    </tr>
    <tr>
      <th>26</th>
      <td>juice</td>
      <td>0.051967</td>
      <td>boneless</td>
      <td>0.040076</td>
      <td>bread</td>
      <td>0.025144</td>
      <td>crushed</td>
      <td>0.030703</td>
      <td>jack</td>
      <td>0.044704</td>
      <td>boneless</td>
      <td>0.060553</td>
      <td>dry</td>
      <td>0.048346</td>
      <td>large</td>
      <td>0.051773</td>
      <td>mushrooms</td>
      <td>0.029672</td>
      <td>cinnamon</td>
      <td>0.044306</td>
    </tr>
    <tr>
      <th>27</th>
      <td>chili</td>
      <td>0.047257</td>
      <td>cilantro</td>
      <td>0.038926</td>
      <td>olives</td>
      <td>0.024301</td>
      <td>basil</td>
      <td>0.029337</td>
      <td>boneless</td>
      <td>0.043686</td>
      <td>red</td>
      <td>0.060264</td>
      <td>bread</td>
      <td>0.047121</td>
      <td>tomatoes</td>
      <td>0.050526</td>
      <td>leaf</td>
      <td>0.029140</td>
      <td>unsalted</td>
      <td>0.040022</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cayenne</td>
      <td>0.047150</td>
      <td>chinese</td>
      <td>0.038110</td>
      <td>pasta</td>
      <td>0.024180</td>
      <td>sour</td>
      <td>0.028967</td>
      <td>fat</td>
      <td>0.042992</td>
      <td>shredded</td>
      <td>0.058954</td>
      <td>pasta</td>
      <td>0.046721</td>
      <td>dry</td>
      <td>0.050446</td>
      <td>nutmeg</td>
      <td>0.026194</td>
      <td>rice</td>
      <td>0.039919</td>
    </tr>
    <tr>
      <th>29</th>
      <td>thyme</td>
      <td>0.046955</td>
      <td>paste</td>
      <td>0.037757</td>
      <td>plum</td>
      <td>0.022989</td>
      <td>broth</td>
      <td>0.028565</td>
      <td>diced</td>
      <td>0.042698</td>
      <td>sour</td>
      <td>0.058709</td>
      <td>cream</td>
      <td>0.045888</td>
      <td>stock</td>
      <td>0.049304</td>
      <td>carrots</td>
      <td>0.024727</td>
      <td>sliced</td>
      <td>0.039626</td>
    </tr>
    <tr>
      <th>30</th>
      <td>wine</td>
      <td>0.045328</td>
      <td>wine</td>
      <td>0.037643</td>
      <td>flat</td>
      <td>0.022926</td>
      <td>flakes</td>
      <td>0.027388</td>
      <td>skinless</td>
      <td>0.041761</td>
      <td>extract</td>
      <td>0.057543</td>
      <td>starch</td>
      <td>0.045382</td>
      <td>chili</td>
      <td>0.049264</td>
      <td>shallots</td>
      <td>0.024557</td>
      <td>mushrooms</td>
      <td>0.039401</td>
    </tr>
    <tr>
      <th>31</th>
      <td>rice</td>
      <td>0.041858</td>
      <td>brown</td>
      <td>0.037402</td>
      <td>italian</td>
      <td>0.021611</td>
      <td>tortillas</td>
      <td>0.027119</td>
      <td>broth</td>
      <td>0.040958</td>
      <td>skinless</td>
      <td>0.057327</td>
      <td>mushrooms</td>
      <td>0.042995</td>
      <td>eggs</td>
      <td>0.047363</td>
      <td>flat</td>
      <td>0.024033</td>
      <td>seeds</td>
      <td>0.037711</td>
    </tr>
    <tr>
      <th>32</th>
      <td>tomato</td>
      <td>0.041728</td>
      <td>peanut</td>
      <td>0.036325</td>
      <td>leaf</td>
      <td>0.021582</td>
      <td>ricotta</td>
      <td>0.024481</td>
      <td>jalapeno</td>
      <td>0.040954</td>
      <td>brown</td>
      <td>0.056056</td>
      <td>dried</td>
      <td>0.041853</td>
      <td>halves</td>
      <td>0.044294</td>
      <td>vinegar</td>
      <td>0.021668</td>
      <td>diced</td>
      <td>0.037549</td>
    </tr>
    <tr>
      <th>33</th>
      <td>vinegar</td>
      <td>0.040919</td>
      <td>toasted</td>
      <td>0.035690</td>
      <td>spinach</td>
      <td>0.020697</td>
      <td>salsa</td>
      <td>0.022274</td>
      <td>taco</td>
      <td>0.040353</td>
      <td>breasts</td>
      <td>0.054990</td>
      <td>cheddar</td>
      <td>0.040816</td>
      <td>tortillas</td>
      <td>0.043156</td>
      <td>orange</td>
      <td>0.020733</td>
      <td>coriander</td>
      <td>0.036802</td>
    </tr>
    <tr>
      <th>34</th>
      <td>basil</td>
      <td>0.040702</td>
      <td>pork</td>
      <td>0.034992</td>
      <td>feta</td>
      <td>0.020547</td>
      <td>jack</td>
      <td>0.022185</td>
      <td>chile</td>
      <td>0.037935</td>
      <td>egg</td>
      <td>0.048214</td>
      <td>cooking</td>
      <td>0.040547</td>
      <td>cilantro</td>
      <td>0.040693</td>
      <td>unsalted</td>
      <td>0.020145</td>
      <td>grated</td>
      <td>0.034666</td>
    </tr>
    <tr>
      <th>35</th>
      <td>celery</td>
      <td>0.039507</td>
      <td>breasts</td>
      <td>0.034476</td>
      <td>ricotta</td>
      <td>0.019006</td>
      <td>wine</td>
      <td>0.021254</td>
      <td>seasoning</td>
      <td>0.037890</td>
      <td>cheddar</td>
      <td>0.046327</td>
      <td>unsalted</td>
      <td>0.040279</td>
      <td>onion</td>
      <td>0.040544</td>
      <td>grated</td>
      <td>0.019612</td>
      <td>tomato</td>
      <td>0.030095</td>
    </tr>
    <tr>
      <th>36</th>
      <td>diced</td>
      <td>0.039377</td>
      <td>shrimp</td>
      <td>0.034146</td>
      <td>thyme</td>
      <td>0.018897</td>
      <td>parsley</td>
      <td>0.020684</td>
      <td>turmeric</td>
      <td>0.036058</td>
      <td>tortillas</td>
      <td>0.044728</td>
      <td>ginger</td>
      <td>0.039106</td>
      <td>unsalted</td>
      <td>0.036412</td>
      <td>thigh</td>
      <td>0.019199</td>
      <td>peas</td>
      <td>0.030051</td>
    </tr>
    <tr>
      <th>37</th>
      <td>paprika</td>
      <td>0.039151</td>
      <td>dark</td>
      <td>0.033926</td>
      <td>spray</td>
      <td>0.017090</td>
      <td>tomato</td>
      <td>0.020452</td>
      <td>mozzarella</td>
      <td>0.034021</td>
      <td>onion</td>
      <td>0.041539</td>
      <td>extra</td>
      <td>0.038143</td>
      <td>leaves</td>
      <td>0.033698</td>
      <td>thighs</td>
      <td>0.018939</td>
      <td>turmeric</td>
      <td>0.028070</td>
    </tr>
    <tr>
      <th>38</th>
      <td>extra</td>
      <td>0.038843</td>
      <td>peanuts</td>
      <td>0.033119</td>
      <td>shallots</td>
      <td>0.016854</td>
      <td>bacon</td>
      <td>0.020310</td>
      <td>avocado</td>
      <td>0.033977</td>
      <td>sodium</td>
      <td>0.040729</td>
      <td>parsley</td>
      <td>0.037832</td>
      <td>milk</td>
      <td>0.033268</td>
      <td>freshly</td>
      <td>0.018481</td>
      <td>shrimp</td>
      <td>0.027700</td>
    </tr>
    <tr>
      <th>39</th>
      <td>lime</td>
      <td>0.038335</td>
      <td>skinless</td>
      <td>0.032934</td>
      <td>capers</td>
      <td>0.015953</td>
      <td>cooked</td>
      <td>0.020122</td>
      <td>tomato</td>
      <td>0.032353</td>
      <td>low</td>
      <td>0.040598</td>
      <td>spray</td>
      <td>0.037701</td>
      <td>free</td>
      <td>0.033218</td>
      <td>grain</td>
      <td>0.018385</td>
      <td>cardamom</td>
      <td>0.026957</td>
    </tr>
    <tr>
      <th>40</th>
      <td>crushed</td>
      <td>0.038321</td>
      <td>oyster</td>
      <td>0.031194</td>
      <td>cucumber</td>
      <td>0.014118</td>
      <td>breasts</td>
      <td>0.020087</td>
      <td>masala</td>
      <td>0.031173</td>
      <td>fish</td>
      <td>0.038303</td>
      <td>nutmeg</td>
      <td>0.037629</td>
      <td>chilies</td>
      <td>0.032954</td>
      <td>bacon</td>
      <td>0.017854</td>
      <td>ribs</td>
      <td>0.025151</td>
    </tr>
    <tr>
      <th>41</th>
      <td>virgin</td>
      <td>0.037827</td>
      <td>cabbage</td>
      <td>0.029829</td>
      <td>mushrooms</td>
      <td>0.014005</td>
      <td>olives</td>
      <td>0.018183</td>
      <td>monterey</td>
      <td>0.030072</td>
      <td>rice</td>
      <td>0.036761</td>
      <td>italian</td>
      <td>0.036192</td>
      <td>dried</td>
      <td>0.029632</td>
      <td>starch</td>
      <td>0.017437</td>
      <td>masala</td>
      <td>0.025045</td>
    </tr>
    <tr>
      <th>42</th>
      <td>coriander</td>
      <td>0.036516</td>
      <td>leaves</td>
      <td>0.029726</td>
      <td>kalamata</td>
      <td>0.013870</td>
      <td>hot</td>
      <td>0.016850</td>
      <td>parmesan</td>
      <td>0.029592</td>
      <td>onions</td>
      <td>0.034174</td>
      <td>baking</td>
      <td>0.035363</td>
      <td>soda</td>
      <td>0.029417</td>
      <td>bay</td>
      <td>0.016746</td>
      <td>wine</td>
      <td>0.023278</td>
    </tr>
    <tr>
      <th>43</th>
      <td>powder</td>
      <td>0.036082</td>
      <td>thai</td>
      <td>0.029347</td>
      <td>zest</td>
      <td>0.013218</td>
      <td>taco</td>
      <td>0.016077</td>
      <td>garam</td>
      <td>0.028189</td>
      <td>spray</td>
      <td>0.033550</td>
      <td>vinegar</td>
      <td>0.035128</td>
      <td>virgin</td>
      <td>0.028998</td>
      <td>allspice</td>
      <td>0.015601</td>
      <td>sausage</td>
      <td>0.023240</td>
    </tr>
    <tr>
      <th>44</th>
      <td>kosher</td>
      <td>0.035608</td>
      <td>hoisin</td>
      <td>0.029272</td>
      <td>balsamic</td>
      <td>0.013102</td>
      <td>mushrooms</td>
      <td>0.015897</td>
      <td>enchilada</td>
      <td>0.026195</td>
      <td>soda</td>
      <td>0.031164</td>
      <td>spinach</td>
      <td>0.034265</td>
      <td>diced</td>
      <td>0.028943</td>
      <td>bread</td>
      <td>0.015522</td>
      <td>parmesan</td>
      <td>0.022822</td>
    </tr>
    <tr>
      <th>45</th>
      <td>boneless</td>
      <td>0.035311</td>
      <td>white</td>
      <td>0.027111</td>
      <td>garlic</td>
      <td>0.012954</td>
      <td>soup</td>
      <td>0.015311</td>
      <td>lettuce</td>
      <td>0.024838</td>
      <td>cooked</td>
      <td>0.031088</td>
      <td>dark</td>
      <td>0.033817</td>
      <td>bay</td>
      <td>0.028744</td>
      <td>heavy</td>
      <td>0.015497</td>
      <td>garam</td>
      <td>0.022324</td>
    </tr>
    <tr>
      <th>46</th>
      <td>bay</td>
      <td>0.035132</td>
      <td>minced</td>
      <td>0.025879</td>
      <td>chives</td>
      <td>0.012882</td>
      <td>skim</td>
      <td>0.015095</td>
      <td>frozen</td>
      <td>0.024053</td>
      <td>heavy</td>
      <td>0.031075</td>
      <td>tomato</td>
      <td>0.033695</td>
      <td>wine</td>
      <td>0.027278</td>
      <td>arborio</td>
      <td>0.015251</td>
      <td>broth</td>
      <td>0.021978</td>
    </tr>
    <tr>
      <th>47</th>
      <td>white</td>
      <td>0.035089</td>
      <td>mushrooms</td>
      <td>0.025526</td>
      <td>avocado</td>
      <td>0.012648</td>
      <td>sharp</td>
      <td>0.014336</td>
      <td>soup</td>
      <td>0.023741</td>
      <td>hot</td>
      <td>0.027965</td>
      <td>brown</td>
      <td>0.033579</td>
      <td>extra</td>
      <td>0.027182</td>
      <td>soup</td>
      <td>0.015085</td>
      <td>coconut</td>
      <td>0.020376</td>
    </tr>
    <tr>
      <th>48</th>
      <td>vegetable</td>
      <td>0.033995</td>
      <td>cloves</td>
      <td>0.024557</td>
      <td>fat</td>
      <td>0.012510</td>
      <td>lasagna</td>
      <td>0.013389</td>
      <td>paste</td>
      <td>0.023699</td>
      <td>cooking</td>
      <td>0.027575</td>
      <td>white</td>
      <td>0.033477</td>
      <td>sour</td>
      <td>0.026972</td>
      <td>scallions</td>
      <td>0.014807</td>
      <td>stock</td>
      <td>0.020139</td>
    </tr>
    <tr>
      <th>49</th>
      <td>sodium</td>
      <td>0.033304</td>
      <td>beansprouts</td>
      <td>0.024469</td>
      <td>crumbles</td>
      <td>0.012425</td>
      <td>sliced</td>
      <td>0.012864</td>
      <td>cooked</td>
      <td>0.022755</td>
      <td>light</td>
      <td>0.027489</td>
      <td>skim</td>
      <td>0.032144</td>
      <td>masala</td>
      <td>0.026131</td>
      <td>long</td>
      <td>0.014731</td>
      <td>curry</td>
      <td>0.019744</td>
    </tr>
    <tr>
      <th>50</th>
      <td>yellow</td>
      <td>0.033035</td>
      <td>shiitake</td>
      <td>0.024460</td>
      <td>finely</td>
      <td>0.011927</td>
      <td>yellow</td>
      <td>0.012631</td>
      <td>lean</td>
      <td>0.021747</td>
      <td>starch</td>
      <td>0.027416</td>
      <td>lasagna</td>
      <td>0.031608</td>
      <td>buttermilk</td>
      <td>0.025927</td>
      <td>zest</td>
      <td>0.014470</td>
      <td>bread</td>
      <td>0.019551</td>
    </tr>
    <tr>
      <th>51</th>
      <td>beans</td>
      <td>0.031218</td>
      <td>tofu</td>
      <td>0.024145</td>
      <td>fillets</td>
      <td>0.011832</td>
      <td>ham</td>
      <td>0.012314</td>
      <td>tortilla</td>
      <td>0.021227</td>
      <td>celery</td>
      <td>0.027186</td>
      <td>vegetable</td>
      <td>0.031552</td>
      <td>leaf</td>
      <td>0.025575</td>
      <td>fresh</td>
      <td>0.014193</td>
      <td>seed</td>
      <td>0.019521</td>
    </tr>
    <tr>
      <th>52</th>
      <td>carrots</td>
      <td>0.030666</td>
      <td>coconut</td>
      <td>0.023701</td>
      <td>zucchini</td>
      <td>0.011740</td>
      <td>bread</td>
      <td>0.012209</td>
      <td>onion</td>
      <td>0.021160</td>
      <td>granulated</td>
      <td>0.026789</td>
      <td>scallions</td>
      <td>0.031015</td>
      <td>kosher</td>
      <td>0.024114</td>
      <td>rosemary</td>
      <td>0.013999</td>
      <td>egg</td>
      <td>0.019487</td>
    </tr>
    <tr>
      <th>53</th>
      <td>leaf</td>
      <td>0.030281</td>
      <td>juice</td>
      <td>0.022846</td>
      <td>cherry</td>
      <td>0.011653</td>
      <td>monterey</td>
      <td>0.012023</td>
      <td>curry</td>
      <td>0.020948</td>
      <td>breast</td>
      <td>0.026559</td>
      <td>low</td>
      <td>0.030468</td>
      <td>curry</td>
      <td>0.023901</td>
      <td>reduced</td>
      <td>0.013832</td>
      <td>vegetable</td>
      <td>0.019374</td>
    </tr>
    <tr>
      <th>54</th>
      <td>paste</td>
      <td>0.029989</td>
      <td>light</td>
      <td>0.022436</td>
      <td>crushed</td>
      <td>0.011251</td>
      <td>spinach</td>
      <td>0.011933</td>
      <td>cinnamon</td>
      <td>0.020945</td>
      <td>free</td>
      <td>0.026170</td>
      <td>virgin</td>
      <td>0.030459</td>
      <td>garam</td>
      <td>0.023190</td>
      <td>sausage</td>
      <td>0.013557</td>
      <td>nutmeg</td>
      <td>0.019218</td>
    </tr>
    <tr>
      <th>55</th>
      <td>flakes</td>
      <td>0.029983</td>
      <td>broth</td>
      <td>0.022190</td>
      <td>sprigs</td>
      <td>0.011228</td>
      <td>feta</td>
      <td>0.011912</td>
      <td>cardamom</td>
      <td>0.020786</td>
      <td>shrimp</td>
      <td>0.026156</td>
      <td>vanilla</td>
      <td>0.027847</td>
      <td>paste</td>
      <td>0.022650</td>
      <td>leeks</td>
      <td>0.011564</td>
      <td>flat</td>
      <td>0.018808</td>
    </tr>
    <tr>
      <th>56</th>
      <td>skinless</td>
      <td>0.029885</td>
      <td>hot</td>
      <td>0.021416</td>
      <td>peeled</td>
      <td>0.010974</td>
      <td>cajun</td>
      <td>0.011798</td>
      <td>chips</td>
      <td>0.020539</td>
      <td>yolks</td>
      <td>0.026022</td>
      <td>extract</td>
      <td>0.027765</td>
      <td>thyme</td>
      <td>0.022581</td>
      <td>saffron</td>
      <td>0.011217</td>
      <td>clove</td>
      <td>0.018540</td>
    </tr>
    <tr>
      <th>57</th>
      <td>cheese</td>
      <td>0.029726</td>
      <td>peppers</td>
      <td>0.021171</td>
      <td>skim</td>
      <td>0.010799</td>
      <td>pizza</td>
      <td>0.011664</td>
      <td>garlic</td>
      <td>0.020346</td>
      <td>buttermilk</td>
      <td>0.025295</td>
      <td>water</td>
      <td>0.026688</td>
      <td>celery</td>
      <td>0.022435</td>
      <td>almonds</td>
      <td>0.010985</td>
      <td>seasoning</td>
      <td>0.018177</td>
    </tr>
    <tr>
      <th>58</th>
      <td>lemon</td>
      <td>0.029580</td>
      <td>mirin</td>
      <td>0.020110</td>
      <td>dill</td>
      <td>0.010071</td>
      <td>enchilada</td>
      <td>0.011597</td>
      <td>seed</td>
      <td>0.019764</td>
      <td>jack</td>
      <td>0.024178</td>
      <td>pork</td>
      <td>0.026674</td>
      <td>coriander</td>
      <td>0.022392</td>
      <td>cooked</td>
      <td>0.010878</td>
      <td>tumeric</td>
      <td>0.018021</td>
    </tr>
    <tr>
      <th>59</th>
      <td>beef</td>
      <td>0.026624</td>
      <td>spring</td>
      <td>0.019817</td>
      <td>pinenuts</td>
      <td>0.009938</td>
      <td>smoked</td>
      <td>0.011510</td>
      <td>mexican</td>
      <td>0.019666</td>
      <td>grated</td>
      <td>0.023316</td>
      <td>milk</td>
      <td>0.026633</td>
      <td>mushrooms</td>
      <td>0.020774</td>
      <td>sherry</td>
      <td>0.010850</td>
      <td>sticks</td>
      <td>0.017953</td>
    </tr>
    <tr>
      <th>60</th>
      <td>minced</td>
      <td>0.026165</td>
      <td>bean</td>
      <td>0.019732</td>
      <td>reggiano</td>
      <td>0.009430</td>
      <td>thyme</td>
      <td>0.011397</td>
      <td>pasta</td>
      <td>0.019418</td>
      <td>seasoning</td>
      <td>0.023131</td>
      <td>fat</td>
      <td>0.026021</td>
      <td>water</td>
      <td>0.019988</td>
      <td>condensed</td>
      <td>0.010848</td>
      <td>basil</td>
      <td>0.015877</td>
    </tr>
    <tr>
      <th>61</th>
      <td>purple</td>
      <td>0.026104</td>
      <td>peeled</td>
      <td>0.018849</td>
      <td>jalapeno</td>
      <td>0.009347</td>
      <td>mix</td>
      <td>0.011347</td>
      <td>noodles</td>
      <td>0.019084</td>
      <td>halves</td>
      <td>0.023069</td>
      <td>whites</td>
      <td>0.025839</td>
      <td>thighs</td>
      <td>0.019679</td>
      <td>whipping</td>
      <td>0.010657</td>
      <td>bacon</td>
      <td>0.014804</td>
    </tr>
    <tr>
      <th>62</th>
      <td>chilies</td>
      <td>0.025370</td>
      <td>canola</td>
      <td>0.018574</td>
      <td>parmigiano</td>
      <td>0.009246</td>
      <td>zucchini</td>
      <td>0.011331</td>
      <td>mix</td>
      <td>0.018867</td>
      <td>chilies</td>
      <td>0.022895</td>
      <td>toasted</td>
      <td>0.025412</td>
      <td>reduced</td>
      <td>0.019604</td>
      <td>sage</td>
      <td>0.010576</td>
      <td>grain</td>
      <td>0.014591</td>
    </tr>
    <tr>
      <th>63</th>
      <td>chile</td>
      <td>0.024393</td>
      <td>flakes</td>
      <td>0.018463</td>
      <td>eggplant</td>
      <td>0.009020</td>
      <td>beef</td>
      <td>0.011161</td>
      <td>sliced</td>
      <td>0.018763</td>
      <td>sliced</td>
      <td>0.022330</td>
      <td>chinese</td>
      <td>0.025086</td>
      <td>thigh</td>
      <td>0.019585</td>
      <td>sprigs</td>
      <td>0.010256</td>
      <td>lamb</td>
      <td>0.013617</td>
    </tr>
    <tr>
      <th>64</th>
      <td>stock</td>
      <td>0.024287</td>
      <td>stock</td>
      <td>0.017749</td>
      <td>anchovy</td>
      <td>0.008944</td>
      <td>frozen</td>
      <td>0.010908</td>
      <td>sodium</td>
      <td>0.018466</td>
      <td>salsa</td>
      <td>0.022215</td>
      <td>black</td>
      <td>0.024000</td>
      <td>starch</td>
      <td>0.019513</td>
      <td>ribs</td>
      <td>0.010206</td>
      <td>saffron</td>
      <td>0.013570</td>
    </tr>
    <tr>
      <th>65</th>
      <td>shrimp</td>
      <td>0.023372</td>
      <td>cucumber</td>
      <td>0.017703</td>
      <td>dijon</td>
      <td>0.008530</td>
      <td>leaf</td>
      <td>0.010881</td>
      <td>reduced</td>
      <td>0.018379</td>
      <td>orange</td>
      <td>0.022153</td>
      <td>oyster</td>
      <td>0.023908</td>
      <td>parsley</td>
      <td>0.018784</td>
      <td>pork</td>
      <td>0.010083</td>
      <td>long</td>
      <td>0.013379</td>
    </tr>
    <tr>
      <th>66</th>
      <td>breasts</td>
      <td>0.022608</td>
      <td>curry</td>
      <td>0.017389</td>
      <td>penne</td>
      <td>0.008368</td>
      <td>andouille</td>
      <td>0.010705</td>
      <td>refried</td>
      <td>0.018248</td>
      <td>coconut</td>
      <td>0.021851</td>
      <td>minced</td>
      <td>0.023697</td>
      <td>yeast</td>
      <td>0.018723</td>
      <td>ham</td>
      <td>0.009092</td>
      <td>vanilla</td>
      <td>0.013062</td>
    </tr>
    <tr>
      <th>67</th>
      <td>hot</td>
      <td>0.020490</td>
      <td>low</td>
      <td>0.016755</td>
      <td>prosciutto</td>
      <td>0.008347</td>
      <td>boneless</td>
      <td>0.010596</td>
      <td>breast</td>
      <td>0.017839</td>
      <td>whites</td>
      <td>0.021742</td>
      <td>crumbs</td>
      <td>0.022814</td>
      <td>canola</td>
      <td>0.018269</td>
      <td>slices</td>
      <td>0.009005</td>
      <td>yolks</td>
      <td>0.012049</td>
    </tr>
    <tr>
      <th>68</th>
      <td>low</td>
      <td>0.020027</td>
      <td>chile</td>
      <td>0.016640</td>
      <td>pitted</td>
      <td>0.008332</td>
      <td>virgin</td>
      <td>0.010554</td>
      <td>low</td>
      <td>0.016787</td>
      <td>whipping</td>
      <td>0.021718</td>
      <td>light</td>
      <td>0.022028</td>
      <td>rice</td>
      <td>0.017893</td>
      <td>crumbs</td>
      <td>0.008717</td>
      <td>spring</td>
      <td>0.011997</td>
    </tr>
    <tr>
      <th>69</th>
      <td>turmeric</td>
      <td>0.019379</td>
      <td>shaoxing</td>
      <td>0.016239</td>
      <td>pizza</td>
      <td>0.008185</td>
      <td>extra</td>
      <td>0.010507</td>
      <td>yogurt</td>
      <td>0.016136</td>
      <td>jalapeno</td>
      <td>0.021712</td>
      <td>sodium</td>
      <td>0.020398</td>
      <td>cooked</td>
      <td>0.017572</td>
      <td>half</td>
      <td>0.008631</td>
      <td>lean</td>
      <td>0.011887</td>
    </tr>
    <tr>
      <th>70</th>
      <td>parmesan</td>
      <td>0.019220</td>
      <td>honey</td>
      <td>0.016084</td>
      <td>fennel</td>
      <td>0.008149</td>
      <td>skinless</td>
      <td>0.010237</td>
      <td>coconut</td>
      <td>0.014415</td>
      <td>reduced</td>
      <td>0.021705</td>
      <td>yolks</td>
      <td>0.019721</td>
      <td>jack</td>
      <td>0.017002</td>
      <td>threads</td>
      <td>0.008627</td>
      <td>peeled</td>
      <td>0.011871</td>
    </tr>
    <tr>
      <th>71</th>
      <td>jalapeno</td>
      <td>0.018811</td>
      <td>steak</td>
      <td>0.015548</td>
      <td>goat</td>
      <td>0.008120</td>
      <td>worcestershire</td>
      <td>0.009956</td>
      <td>tumeric</td>
      <td>0.014288</td>
      <td>lemon</td>
      <td>0.020684</td>
      <td>lean</td>
      <td>0.019139</td>
      <td>beans</td>
      <td>0.015753</td>
      <td>raisins</td>
      <td>0.008176</td>
      <td>rosemary</td>
      <td>0.011729</td>
    </tr>
    <tr>
      <th>72</th>
      <td>soy</td>
      <td>0.018289</td>
      <td>firm</td>
      <td>0.015241</td>
      <td>sage</td>
      <td>0.007993</td>
      <td>ribs</td>
      <td>0.009596</td>
      <td>sharp</td>
      <td>0.014050</td>
      <td>chocolate</td>
      <td>0.019430</td>
      <td>freshly</td>
      <td>0.017690</td>
      <td>potatoes</td>
      <td>0.014933</td>
      <td>capers</td>
      <td>0.007604</td>
      <td>cabbage</td>
      <td>0.011619</td>
    </tr>
    <tr>
      <th>73</th>
      <td>cooking</td>
      <td>0.018282</td>
      <td>cooked</td>
      <td>0.014814</td>
      <td>arborio</td>
      <td>0.007801</td>
      <td>penne</td>
      <td>0.009084</td>
      <td>leaves</td>
      <td>0.013933</td>
      <td>yellow</td>
      <td>0.019340</td>
      <td>shiitake</td>
      <td>0.017635</td>
      <td>salsa</td>
      <td>0.014439</td>
      <td>dijon</td>
      <td>0.007388</td>
      <td>andouille</td>
      <td>0.011583</td>
    </tr>
    <tr>
      <th>74</th>
      <td>mushrooms</td>
      <td>0.017937</td>
      <td>root</td>
      <td>0.014285</td>
      <td>tarragon</td>
      <td>0.007720</td>
      <td>shrimp</td>
      <td>0.009071</td>
      <td>spinach</td>
      <td>0.013700</td>
      <td>cayenne</td>
      <td>0.017894</td>
      <td>reggiano</td>
      <td>0.017599</td>
      <td>soup</td>
      <td>0.014364</td>
      <td>tarragon</td>
      <td>0.007133</td>
      <td>extract</td>
      <td>0.011556</td>
    </tr>
    <tr>
      <th>75</th>
      <td>sausage</td>
      <td>0.017480</td>
      <td>peas</td>
      <td>0.013476</td>
      <td>broth</td>
      <td>0.007686</td>
      <td>sausages</td>
      <td>0.009064</td>
      <td>ricotta</td>
      <td>0.012676</td>
      <td>parmesan</td>
      <td>0.017855</td>
      <td>parmigiano</td>
      <td>0.017478</td>
      <td>tumeric</td>
      <td>0.014241</td>
      <td>parmesan</td>
      <td>0.007051</td>
      <td>yeast</td>
      <td>0.011160</td>
    </tr>
    <tr>
      <th>76</th>
      <td>pork</td>
      <td>0.017254</td>
      <td>lemongrass</td>
      <td>0.013455</td>
      <td>lasagna</td>
      <td>0.007438</td>
      <td>bay</td>
      <td>0.009030</td>
      <td>lasagna</td>
      <td>0.011875</td>
      <td>powdered</td>
      <td>0.017793</td>
      <td>hoisin</td>
      <td>0.017254</td>
      <td>seed</td>
      <td>0.014035</td>
      <td>mirin</td>
      <td>0.006906</td>
      <td>creole</td>
      <td>0.010171</td>
    </tr>
    <tr>
      <th>77</th>
      <td>flat</td>
      <td>0.016646</td>
      <td>sriracha</td>
      <td>0.013269</td>
      <td>orange</td>
      <td>0.007369</td>
      <td>creole</td>
      <td>0.009001</td>
      <td>italian</td>
      <td>0.011739</td>
      <td>thai</td>
      <td>0.017103</td>
      <td>cottage</td>
      <td>0.016697</td>
      <td>oregano</td>
      <td>0.013921</td>
      <td>couscous</td>
      <td>0.006719</td>
      <td>olives</td>
      <td>0.010055</td>
    </tr>
    <tr>
      <th>78</th>
      <td>olives</td>
      <td>0.016313</td>
      <td>sake</td>
      <td>0.013205</td>
      <td>sea</td>
      <td>0.007304</td>
      <td>corn</td>
      <td>0.008903</td>
      <td>condensed</td>
      <td>0.011390</td>
      <td>thyme</td>
      <td>0.016899</td>
      <td>marinara</td>
      <td>0.016673</td>
      <td>jalapeno</td>
      <td>0.012933</td>
      <td>andouille</td>
      <td>0.006508</td>
      <td>fennel</td>
      <td>0.010014</td>
    </tr>
    <tr>
      <th>79</th>
      <td>fat</td>
      <td>0.016243</td>
      <td>shallots</td>
      <td>0.012838</td>
      <td>baguette</td>
      <td>0.007156</td>
      <td>reggiano</td>
      <td>0.008823</td>
      <td>peppers</td>
      <td>0.011189</td>
      <td>mushrooms</td>
      <td>0.016682</td>
      <td>cinnamon</td>
      <td>0.016555</td>
      <td>paprika</td>
      <td>0.012882</td>
      <td>legs</td>
      <td>0.006383</td>
      <td>basmati</td>
      <td>0.009566</td>
    </tr>
    <tr>
      <th>80</th>
      <td>freshly</td>
      <td>0.015457</td>
      <td>spice</td>
      <td>0.012642</td>
      <td>free</td>
      <td>0.007153</td>
      <td>spaghetti</td>
      <td>0.008669</td>
      <td>turkey</td>
      <td>0.011167</td>
      <td>peanuts</td>
      <td>0.016396</td>
      <td>oregano</td>
      <td>0.016503</td>
      <td>monterey</td>
      <td>0.012688</td>
      <td>honey</td>
      <td>0.006382</td>
      <td>pods</td>
      <td>0.009427</td>
    </tr>
    <tr>
      <th>81</th>
      <td>breast</td>
      <td>0.015306</td>
      <td>wrappers</td>
      <td>0.012357</td>
      <td>romano</td>
      <td>0.007138</td>
      <td>dressing</td>
      <td>0.008465</td>
      <td>blend</td>
      <td>0.011123</td>
      <td>worcestershire</td>
      <td>0.015994</td>
      <td>peanut</td>
      <td>0.015721</td>
      <td>flat</td>
      <td>0.012303</td>
      <td>lamb</td>
      <td>0.006331</td>
      <td>sausages</td>
      <td>0.008947</td>
    </tr>
    <tr>
      <th>82</th>
      <td>spray</td>
      <td>0.014909</td>
      <td>broccoli</td>
      <td>0.012163</td>
      <td>serrano</td>
      <td>0.007076</td>
      <td>marinara</td>
      <td>0.008389</td>
      <td>fish</td>
      <td>0.011004</td>
      <td>noodles</td>
      <td>0.015866</td>
      <td>corn</td>
      <td>0.015040</td>
      <td>arborio</td>
      <td>0.012073</td>
      <td>prosciutto</td>
      <td>0.006220</td>
      <td>boiling</td>
      <td>0.008870</td>
    </tr>
    <tr>
      <th>83</th>
      <td>sesame</td>
      <td>0.014586</td>
      <td>peppercorns</td>
      <td>0.012126</td>
      <td>linguine</td>
      <td>0.006819</td>
      <td>flat</td>
      <td>0.008383</td>
      <td>wedges</td>
      <td>0.010885</td>
      <td>dark</td>
      <td>0.015695</td>
      <td>sharp</td>
      <td>0.014115</td>
      <td>seeds</td>
      <td>0.011945</td>
      <td>peel</td>
      <td>0.006044</td>
      <td>italian</td>
      <td>0.008736</td>
    </tr>
    <tr>
      <th>84</th>
      <td>dry</td>
      <td>0.014361</td>
      <td>ketchup</td>
      <td>0.011730</td>
      <td>grape</td>
      <td>0.006640</td>
      <td>parmigiano</td>
      <td>0.008352</td>
      <td>dressing</td>
      <td>0.010848</td>
      <td>avocado</td>
      <td>0.015650</td>
      <td>yeast</td>
      <td>0.014033</td>
      <td>warm</td>
      <td>0.011849</td>
      <td>bone</td>
      <td>0.006020</td>
      <td>spinach</td>
      <td>0.008651</td>
    </tr>
    <tr>
      <th>85</th>
      <td>seeds</td>
      <td>0.013640</td>
      <td>roasted</td>
      <td>0.011572</td>
      <td>baby</td>
      <td>0.006613</td>
      <td>garlic</td>
      <td>0.007918</td>
      <td>seeds</td>
      <td>0.010069</td>
      <td>mint</td>
      <td>0.015531</td>
      <td>granulated</td>
      <td>0.014020</td>
      <td>active</td>
      <td>0.011044</td>
      <td>veal</td>
      <td>0.005952</td>
      <td>ham</td>
      <td>0.008649</td>
    </tr>
    <tr>
      <th>86</th>
      <td>curry</td>
      <td>0.013444</td>
      <td>choy</td>
      <td>0.011248</td>
      <td>slices</td>
      <td>0.006556</td>
      <td>noodles</td>
      <td>0.007867</td>
      <td>halves</td>
      <td>0.010043</td>
      <td>soup</td>
      <td>0.015395</td>
      <td>pizza</td>
      <td>0.013704</td>
      <td>chiles</td>
      <td>0.010974</td>
      <td>apples</td>
      <td>0.005842</td>
      <td>raisins</td>
      <td>0.008614</td>
    </tr>
    <tr>
      <th>87</th>
      <td>tortillas</td>
      <td>0.013032</td>
      <td>asian</td>
      <td>0.011052</td>
      <td>spaghetti</td>
      <td>0.006531</td>
      <td>cottage</td>
      <td>0.007741</td>
      <td>clove</td>
      <td>0.010041</td>
      <td>chile</td>
      <td>0.015037</td>
      <td>soda</td>
      <td>0.013696</td>
      <td>coconut</td>
      <td>0.010778</td>
      <td>shiitake</td>
      <td>0.005820</td>
      <td>mustard</td>
      <td>0.008430</td>
    </tr>
    <tr>
      <th>88</th>
      <td>zucchini</td>
      <td>0.012891</td>
      <td>worcestershire</td>
      <td>0.010409</td>
      <td>asparagus</td>
      <td>0.006522</td>
      <td>romano</td>
      <td>0.007366</td>
      <td>guacamole</td>
      <td>0.009328</td>
      <td>monterey</td>
      <td>0.014993</td>
      <td>feta</td>
      <td>0.013557</td>
      <td>peas</td>
      <td>0.010723</td>
      <td>drumsticks</td>
      <td>0.005807</td>
      <td>almonds</td>
      <td>0.008402</td>
    </tr>
    <tr>
      <th>89</th>
      <td>peas</td>
      <td>0.012889</td>
      <td>napa</td>
      <td>0.010391</td>
      <td>arugula</td>
      <td>0.006470</td>
      <td>grain</td>
      <td>0.007247</td>
      <td>cottage</td>
      <td>0.009070</td>
      <td>unsweetened</td>
      <td>0.014584</td>
      <td>reduced</td>
      <td>0.013278</td>
      <td>spray</td>
      <td>0.010621</td>
      <td>cajun</td>
      <td>0.005742</td>
      <td>lemon</td>
      <td>0.008382</td>
    </tr>
    <tr>
      <th>90</th>
      <td>plum</td>
      <td>0.012733</td>
      <td>bok</td>
      <td>0.010316</td>
      <td>bulb</td>
      <td>0.006323</td>
      <td>long</td>
      <td>0.007124</td>
      <td>free</td>
      <td>0.008834</td>
      <td>sharp</td>
      <td>0.014000</td>
      <td>shaoxing</td>
      <td>0.013054</td>
      <td>grain</td>
      <td>0.010075</td>
      <td>canned</td>
      <td>0.005471</td>
      <td>okra</td>
      <td>0.008210</td>
    </tr>
    <tr>
      <th>91</th>
      <td>peppers</td>
      <td>0.012677</td>
      <td>chestnuts</td>
      <td>0.010229</td>
      <td>pecorino</td>
      <td>0.006295</td>
      <td>grits</td>
      <td>0.007076</td>
      <td>skim</td>
      <td>0.008512</td>
      <td>diced</td>
      <td>0.013883</td>
      <td>wrappers</td>
      <td>0.012628</td>
      <td>long</td>
      <td>0.010004</td>
      <td>ice</td>
      <td>0.005362</td>
      <td>leeks</td>
      <td>0.008009</td>
    </tr>
    <tr>
      <th>92</th>
      <td>canola</td>
      <td>0.012627</td>
      <td>snow</td>
      <td>0.010036</td>
      <td>shredded</td>
      <td>0.006246</td>
      <td>crumbles</td>
      <td>0.006704</td>
      <td>chipotle</td>
      <td>0.008411</td>
      <td>confectioners</td>
      <td>0.013283</td>
      <td>romano</td>
      <td>0.012252</td>
      <td>meal</td>
      <td>0.009836</td>
      <td>sake</td>
      <td>0.005295</td>
      <td>ghee</td>
      <td>0.007993</td>
    </tr>
    <tr>
      <th>93</th>
      <td>grain</td>
      <td>0.012396</td>
      <td>mint</td>
      <td>0.010025</td>
      <td>crumb</td>
      <td>0.006186</td>
      <td>tortilla</td>
      <td>0.006476</td>
      <td>sticks</td>
      <td>0.008390</td>
      <td>half</td>
      <td>0.013235</td>
      <td>spaghetti</td>
      <td>0.012007</td>
      <td>wheat</td>
      <td>0.009372</td>
      <td>golden</td>
      <td>0.005283</td>
      <td>root</td>
      <td>0.007941</td>
    </tr>
    <tr>
      <th>94</th>
      <td>smoked</td>
      <td>0.012392</td>
      <td>lettuce</td>
      <td>0.009930</td>
      <td>cannellini</td>
      <td>0.005854</td>
      <td>breast</td>
      <td>0.006422</td>
      <td>chiles</td>
      <td>0.008277</td>
      <td>chili</td>
      <td>0.013180</td>
      <td>worcestershire</td>
      <td>0.011631</td>
      <td>turmeric</td>
      <td>0.009265</td>
      <td>chives</td>
      <td>0.005070</td>
      <td>sprigs</td>
      <td>0.007939</td>
    </tr>
    <tr>
      <th>95</th>
      <td>peeled</td>
      <td>0.012387</td>
      <td>sherry</td>
      <td>0.009870</td>
      <td>lettuce</td>
      <td>0.005570</td>
      <td>okra</td>
      <td>0.006385</td>
      <td>shells</td>
      <td>0.008222</td>
      <td>frozen</td>
      <td>0.012965</td>
      <td>kosher</td>
      <td>0.011526</td>
      <td>heavy</td>
      <td>0.009238</td>
      <td>lower</td>
      <td>0.004931</td>
      <td>medium</td>
      <td>0.007905</td>
    </tr>
    <tr>
      <th>96</th>
      <td>masala</td>
      <td>0.012337</td>
      <td>bell</td>
      <td>0.009743</td>
      <td>yogurt</td>
      <td>0.005419</td>
      <td>chips</td>
      <td>0.006360</td>
      <td>pinto</td>
      <td>0.008036</td>
      <td>pecans</td>
      <td>0.012816</td>
      <td>powdered</td>
      <td>0.010935</td>
      <td>frozen</td>
      <td>0.009195</td>
      <td>marsala</td>
      <td>0.004885</td>
      <td>warm</td>
      <td>0.007757</td>
    </tr>
    <tr>
      <th>97</th>
      <td>corn</td>
      <td>0.012130</td>
      <td>grain</td>
      <td>0.009468</td>
      <td>leeks</td>
      <td>0.005321</td>
      <td>provolone</td>
      <td>0.006231</td>
      <td>adobo</td>
      <td>0.007802</td>
      <td>peanut</td>
      <td>0.012569</td>
      <td>broccoli</td>
      <td>0.010594</td>
      <td>canned</td>
      <td>0.008975</td>
      <td>apricot</td>
      <td>0.004676</td>
      <td>cooked</td>
      <td>0.007267</td>
    </tr>
    <tr>
      <th>98</th>
      <td>cooked</td>
      <td>0.012062</td>
      <td>star</td>
      <td>0.008791</td>
      <td>cooking</td>
      <td>0.005266</td>
      <td>pesto</td>
      <td>0.006168</td>
      <td>ghee</td>
      <td>0.007506</td>
      <td>enchilada</td>
      <td>0.012238</td>
      <td>honey</td>
      <td>0.010386</td>
      <td>rosemary</td>
      <td>0.008943</td>
      <td>bouillon</td>
      <td>0.004670</td>
      <td>crumbs</td>
      <td>0.007234</td>
    </tr>
    <tr>
      <th>99</th>
      <td>garam</td>
      <td>0.011817</td>
      <td>flank</td>
      <td>0.008681</td>
      <td>romaine</td>
      <td>0.004878</td>
      <td>turkey</td>
      <td>0.006149</td>
      <td>plain</td>
      <td>0.007468</td>
      <td>condensed</td>
      <td>0.012199</td>
      <td>chives</td>
      <td>0.010371</td>
      <td>plain</td>
      <td>0.008459</td>
      <td>wedge</td>
      <td>0.004627</td>
      <td>shiitake</td>
      <td>0.007148</td>
    </tr>
  </tbody>
</table>
</div>




```python
rows = [{'species': 'Tetraodon lineatus', 'age': 1, 'length': 8.0, 'weight': 9.0},
        {'species': 'Tetraodon fahaka', 'age': 1, 'length': 8.0, 'weight': 9.0}]
df = df.append(rows, ignore_index=True)
```


```python
def PC_content(pca_model = pca80, number_of_PC=10, number_of_components=10):
    df_total = pd.DataFrame()
    for PC in range(number_of_PC):
        ind = np.argsort(-pca_model.components_[PC])[:number_of_components]
        ind_bottom = np.argsort(-pca80.components_[PC])[-number_of_components:]
        df_ = pd.DataFrame({'Ingredient':terms[ind], 'Score':[pca_model.components_[PC][x] for x in ind]})
        df_ = pd.concat([df_,pd.DataFrame([{'Ingredient':'----------','Score':'----------'}])])
        df_bottom = pd.DataFrame({'Ingredient':terms[ind_bottom], 'Score':[pca_model.components_[PC][x] for x in ind_bottom]})
        df_ = pd.concat([df_,df_bottom])
        df_ = add_top_column(df_, "PC {}".format(PC))
        df_total = pd.concat([df_total,df_],axis=1)
    return df_total
PC_content()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">PC 0</th>
      <th colspan="2" halign="left">PC 1</th>
      <th colspan="2" halign="left">PC 2</th>
      <th colspan="2" halign="left">PC 3</th>
      <th colspan="2" halign="left">PC 4</th>
      <th colspan="2" halign="left">PC 5</th>
      <th colspan="2" halign="left">PC 6</th>
      <th colspan="2" halign="left">PC 7</th>
      <th colspan="2" halign="left">PC 8</th>
      <th colspan="2" halign="left">PC 9</th>
    </tr>
    <tr>
      <th></th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
      <th>Ingredient</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pepper</td>
      <td>0.523607</td>
      <td>sauce</td>
      <td>0.474086</td>
      <td>fresh</td>
      <td>0.558289</td>
      <td>cheese</td>
      <td>0.359403</td>
      <td>cheese</td>
      <td>0.290647</td>
      <td>fresh</td>
      <td>0.363356</td>
      <td>cheese</td>
      <td>0.526328</td>
      <td>chicken</td>
      <td>0.538314</td>
      <td>chicken</td>
      <td>0.479741</td>
      <td>onions</td>
      <td>0.594707</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ground</td>
      <td>0.361922</td>
      <td>soy</td>
      <td>0.261672</td>
      <td>olive</td>
      <td>0.221516</td>
      <td>pepper</td>
      <td>0.318516</td>
      <td>ground</td>
      <td>0.280666</td>
      <td>pepper</td>
      <td>0.233058</td>
      <td>sauce</td>
      <td>0.342029</td>
      <td>oil</td>
      <td>0.303247</td>
      <td>ground</td>
      <td>0.202891</td>
      <td>green</td>
      <td>0.395903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>garlic</td>
      <td>0.258455</td>
      <td>oil</td>
      <td>0.255458</td>
      <td>cheese</td>
      <td>0.194892</td>
      <td>bell</td>
      <td>0.131593</td>
      <td>chicken</td>
      <td>0.17387</td>
      <td>sugar</td>
      <td>0.215038</td>
      <td>ground</td>
      <td>0.26187</td>
      <td>flour</td>
      <td>0.237056</td>
      <td>broth</td>
      <td>0.167949</td>
      <td>fresh</td>
      <td>0.144382</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fresh</td>
      <td>0.247066</td>
      <td>sesame</td>
      <td>0.193111</td>
      <td>juice</td>
      <td>0.14489</td>
      <td>parmesan</td>
      <td>0.095975</td>
      <td>cumin</td>
      <td>0.152478</td>
      <td>chicken</td>
      <td>0.21041</td>
      <td>oil</td>
      <td>0.238259</td>
      <td>powder</td>
      <td>0.20145</td>
      <td>white</td>
      <td>0.124195</td>
      <td>water</td>
      <td>0.140545</td>
    </tr>
    <tr>
      <th>4</th>
      <td>oil</td>
      <td>0.225165</td>
      <td>rice</td>
      <td>0.182921</td>
      <td>chopped</td>
      <td>0.133097</td>
      <td>red</td>
      <td>0.092441</td>
      <td>cilantro</td>
      <td>0.139846</td>
      <td>butter</td>
      <td>0.195371</td>
      <td>parmesan</td>
      <td>0.173779</td>
      <td>salt</td>
      <td>0.201422</td>
      <td>boneless</td>
      <td>0.114295</td>
      <td>butter</td>
      <td>0.116424</td>
    </tr>
    <tr>
      <th>5</th>
      <td>black</td>
      <td>0.200506</td>
      <td>ginger</td>
      <td>0.135998</td>
      <td>lemon</td>
      <td>0.114092</td>
      <td>green</td>
      <td>0.088739</td>
      <td>onions</td>
      <td>0.133142</td>
      <td>chopped</td>
      <td>0.184577</td>
      <td>eggs</td>
      <td>0.159981</td>
      <td>broth</td>
      <td>0.196694</td>
      <td>skinless</td>
      <td>0.104954</td>
      <td>parsley</td>
      <td>0.095025</td>
    </tr>
    <tr>
      <th>6</th>
      <td>red</td>
      <td>0.173708</td>
      <td>garlic</td>
      <td>0.129364</td>
      <td>tomatoes</td>
      <td>0.099469</td>
      <td>onions</td>
      <td>0.085838</td>
      <td>shredded</td>
      <td>0.126227</td>
      <td>flour</td>
      <td>0.182289</td>
      <td>soy</td>
      <td>0.153966</td>
      <td>purpose</td>
      <td>0.145914</td>
      <td>wine</td>
      <td>0.099634</td>
      <td>flour</td>
      <td>0.080278</td>
    </tr>
    <tr>
      <th>7</th>
      <td>olive</td>
      <td>0.154176</td>
      <td>onions</td>
      <td>0.121855</td>
      <td>parsley</td>
      <td>0.094211</td>
      <td>shredded</td>
      <td>0.079754</td>
      <td>cream</td>
      <td>0.111774</td>
      <td>cream</td>
      <td>0.174245</td>
      <td>sugar</td>
      <td>0.153048</td>
      <td>olive</td>
      <td>0.145514</td>
      <td>sodium</td>
      <td>0.093258</td>
      <td>tomatoes</td>
      <td>0.074155</td>
    </tr>
    <tr>
      <th>8</th>
      <td>salt</td>
      <td>0.142398</td>
      <td>chicken</td>
      <td>0.114661</td>
      <td>basil</td>
      <td>0.090868</td>
      <td>chicken</td>
      <td>0.079438</td>
      <td>sauce</td>
      <td>0.105296</td>
      <td>sauce</td>
      <td>0.169985</td>
      <td>grated</td>
      <td>0.151882</td>
      <td>garlic</td>
      <td>0.140513</td>
      <td>lemon</td>
      <td>0.091645</td>
      <td>ground</td>
      <td>0.073736</td>
    </tr>
    <tr>
      <th>9</th>
      <td>onions</td>
      <td>0.137449</td>
      <td>green</td>
      <td>0.107114</td>
      <td>virgin</td>
      <td>0.085163</td>
      <td>dried</td>
      <td>0.075665</td>
      <td>green</td>
      <td>0.102845</td>
      <td>cheese</td>
      <td>0.15087</td>
      <td>large</td>
      <td>0.136537</td>
      <td>boneless</td>
      <td>0.127796</td>
      <td>pepper</td>
      <td>0.081649</td>
      <td>celery</td>
      <td>0.070032</td>
    </tr>
    <tr>
      <th>0</th>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
      <td>----------</td>
    </tr>
    <tr>
      <th>0</th>
      <td>cream</td>
      <td>-0.048325</td>
      <td>powder</td>
      <td>-0.088161</td>
      <td>green</td>
      <td>-0.08229</td>
      <td>cinnamon</td>
      <td>-0.102169</td>
      <td>butter</td>
      <td>-0.135727</td>
      <td>cumin</td>
      <td>-0.048089</td>
      <td>chicken</td>
      <td>-0.080952</td>
      <td>black</td>
      <td>-0.051596</td>
      <td>cumin</td>
      <td>-0.124275</td>
      <td>chicken</td>
      <td>-0.091728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>purpose</td>
      <td>-0.049665</td>
      <td>purpose</td>
      <td>-0.098166</td>
      <td>salt</td>
      <td>-0.089083</td>
      <td>cumin</td>
      <td>-0.118491</td>
      <td>large</td>
      <td>-0.137763</td>
      <td>seeds</td>
      <td>-0.055478</td>
      <td>tomatoes</td>
      <td>-0.084862</td>
      <td>soy</td>
      <td>-0.05537</td>
      <td>flour</td>
      <td>-0.141632</td>
      <td>purple</td>
      <td>-0.098146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>extract</td>
      <td>-0.053961</td>
      <td>cream</td>
      <td>-0.099527</td>
      <td>onions</td>
      <td>-0.105974</td>
      <td>lime</td>
      <td>-0.123837</td>
      <td>red</td>
      <td>-0.140862</td>
      <td>tomatoes</td>
      <td>-0.058044</td>
      <td>onions</td>
      <td>-0.089859</td>
      <td>sugar</td>
      <td>-0.061979</td>
      <td>cheese</td>
      <td>-0.144645</td>
      <td>juice</td>
      <td>-0.107242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>milk</td>
      <td>-0.059031</td>
      <td>black</td>
      <td>-0.115411</td>
      <td>soy</td>
      <td>-0.115482</td>
      <td>lemon</td>
      <td>-0.126376</td>
      <td>purpose</td>
      <td>-0.156556</td>
      <td>cloves</td>
      <td>-0.063852</td>
      <td>red</td>
      <td>-0.089956</td>
      <td>fresh</td>
      <td>-0.086649</td>
      <td>garlic</td>
      <td>-0.149002</td>
      <td>corn</td>
      <td>-0.112201</td>
    </tr>
    <tr>
      <th>4</th>
      <td>baking</td>
      <td>-0.059533</td>
      <td>pepper</td>
      <td>-0.128559</td>
      <td>flour</td>
      <td>-0.130339</td>
      <td>cilantro</td>
      <td>-0.129898</td>
      <td>olive</td>
      <td>-0.178225</td>
      <td>garlic</td>
      <td>-0.073379</td>
      <td>juice</td>
      <td>-0.090428</td>
      <td>juice</td>
      <td>-0.088741</td>
      <td>tomatoes</td>
      <td>-0.16079</td>
      <td>black</td>
      <td>-0.114467</td>
    </tr>
    <tr>
      <th>5</th>
      <td>eggs</td>
      <td>-0.066511</td>
      <td>flour</td>
      <td>-0.137922</td>
      <td>powder</td>
      <td>-0.195647</td>
      <td>ginger</td>
      <td>-0.158165</td>
      <td>flour</td>
      <td>-0.188857</td>
      <td>extra</td>
      <td>-0.104226</td>
      <td>bell</td>
      <td>-0.116409</td>
      <td>bell</td>
      <td>-0.105188</td>
      <td>green</td>
      <td>-0.162426</td>
      <td>cilantro</td>
      <td>-0.115482</td>
    </tr>
    <tr>
      <th>6</th>
      <td>vanilla</td>
      <td>-0.067746</td>
      <td>butter</td>
      <td>-0.141176</td>
      <td>pepper</td>
      <td>-0.203849</td>
      <td>juice</td>
      <td>-0.191526</td>
      <td>sugar</td>
      <td>-0.237697</td>
      <td>virgin</td>
      <td>-0.105047</td>
      <td>lime</td>
      <td>-0.120947</td>
      <td>ground</td>
      <td>-0.13712</td>
      <td>cilantro</td>
      <td>-0.165946</td>
      <td>sauce</td>
      <td>-0.152874</td>
    </tr>
    <tr>
      <th>7</th>
      <td>flour</td>
      <td>-0.081321</td>
      <td>salt</td>
      <td>-0.181661</td>
      <td>sugar</td>
      <td>-0.20708</td>
      <td>sugar</td>
      <td>-0.220829</td>
      <td>oil</td>
      <td>-0.238624</td>
      <td>ground</td>
      <td>-0.229245</td>
      <td>cilantro</td>
      <td>-0.127093</td>
      <td>red</td>
      <td>-0.152382</td>
      <td>chili</td>
      <td>-0.188832</td>
      <td>lime</td>
      <td>-0.177601</td>
    </tr>
    <tr>
      <th>8</th>
      <td>butter</td>
      <td>-0.082038</td>
      <td>cheese</td>
      <td>-0.212065</td>
      <td>sauce</td>
      <td>-0.213586</td>
      <td>fresh</td>
      <td>-0.424464</td>
      <td>salt</td>
      <td>-0.247133</td>
      <td>olive</td>
      <td>-0.248549</td>
      <td>green</td>
      <td>-0.134722</td>
      <td>sauce</td>
      <td>-0.197569</td>
      <td>oil</td>
      <td>-0.213678</td>
      <td>powder</td>
      <td>-0.182614</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sugar</td>
      <td>-0.163628</td>
      <td>ground</td>
      <td>-0.378365</td>
      <td>ground</td>
      <td>-0.354903</td>
      <td>ground</td>
      <td>-0.437808</td>
      <td>pepper</td>
      <td>-0.304312</td>
      <td>oil</td>
      <td>-0.337388</td>
      <td>pepper</td>
      <td>-0.138121</td>
      <td>pepper</td>
      <td>-0.285337</td>
      <td>powder</td>
      <td>-0.387671</td>
      <td>onion</td>
      <td>-0.285034</td>
    </tr>
  </tbody>
</table>
</div>


