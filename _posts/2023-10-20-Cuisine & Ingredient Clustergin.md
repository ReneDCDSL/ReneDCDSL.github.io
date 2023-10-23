---
layout: post
title: "Recipe Ingredients Clustering"
subtitle: "Clustering the ingredients from recipes from Yummly"
#date:
skills: "[Clustering] [Unsupervised Learning] [PCA] [K-means] [LDA]"
background: '/img/posts/twitter_SA/twitter_SA.jpg'
link: 'https://github.com/ReneDCDSL/Twitter_Sentiment_Analysis'
---
<style>
  /* Style for tables */
  table {
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 20px;
      border: 1px solid #ccc;
      font-size: 14px;
 }

  table th,
  table td {
      padding: 8px 12px;
      text-align: center;
      border-bottom: 1px solid #ccc;
  }
  /*background color for columns and rows' names */
  table th {
      background-color: #f0f0f0;
  }

  /* Style for code blocks */
  pre {
      background-color: #f4f4f4;
      border: 1px solid #ddd;
      padding: 10px;
      margin-bottom: 20px;
      overflow-x: auto;
      line-height: 1.4;
      font-size: 14px;
      border-radius: 4px;
  }

  pre code {
      display: block;
      padding: 0;
      margin: 0;
      font-family: Monaco, monospace;
  }

  /* Style for inline code */
  code {
      background-color: #f4f4f4;
      padding: 2px 4px;
      border-radius: 4px;
      font-family: Monaco, monospace;
      font-size: 14px;
  }

  /* Style for headers */
  h1, h2, h3, h4, h5, h6 {
      margin-top: 1.5em;
      margin-bottom: 0.5em;
  }

  /* Style for lists */
  ul, ol {
      margin-left: 20px;
      margin-bottom: 20px;
  }

  /* Style for links */
  a {
      text-decoration: none;
  }

  a:hover {
      text-decoration: underline;
  }
</style>


# Cuisine and Ingredients clustering

## Imports


```python
# Basic Tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn
import itertools
from IPython.display import display, HTML
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
df = pd.read_json('./data/kaggle_recipe-ingredients-dataset/train.json')
#df = pd.read_json('./data/Yummly/train.json')
```

## Exploratory Data Analysis


```python
print('Dataset shape: ', df.shape)
df.head()
```

    Dataset shape:  (39774, 4)
    




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




```python
# Missing ingredient list
print('Missing set of ingredients: ', sum(df['ingredients'].isna()))

# Missing cuisine
print('Missing meal origin: ', sum(df['cuisine'].isna()))

print('Number of unique meals: ', sum(df['id'].value_counts()))
```

    Missing set of ingredients:  0
    Missing meal origin:  0
    Number of unique meals:  39774
    

## Text process

### Text preprocessing


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

#### Common stopwords


```python
# Vectorizer
count_vect = CountVectorizer(stopwords = ENGLISH_STOP_WORDS)

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

### K-means


```python
# 1st Model
# Computation: 25 clusters(~6min)

number_of_clusters=25
km = KMeans(n_clusters = number_of_clusters)
km.fit(counts)
```




    KMeans(n_clusters=25)




```python
# Clusters observation

# Find centroids
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# Liss of features
terms = count_vect.get_feature_names_out()

print("Top terms per cluster:")
for i in range(number_of_clusters):
    top_15_words = [terms[ind] for ind in order_centroids[i, :15]]
    print("Cluster {}: {}".format(i, ' '.join(top_15_words)))
```

    Top terms per cluster:
    Cluster 0: sugar flour baking salt butter purpose all powder eggs large vanilla unsalted extract soda milk
    Cluster 1: fresh cilantro lime salt chopped juice garlic onion pepper oil tomatoes chilies ground jalapeno corn
    Cluster 2: ground salt oil powder cumin garlic onions ginger coriander seeds chili leaves green masala tomatoes
    Cluster 3: fresh oil pepper olive salt garlic chopped ground black cloves parsley tomatoes cheese lemon juice
    Cluster 4: pepper bell red salt oil garlic green ground fresh black olive onions tomatoes onion chopped
    Cluster 5: fresh lemon juice salt oil pepper olive garlic ground butter sugar grated parsley cloves orange
    Cluster 6: oil olive extra virgin salt pepper garlic fresh ground cloves black tomatoes red cheese parsley
    Cluster 7: salt sugar water butter oil milk cheese fresh rice white onions garlic cream lime green
    Cluster 8: cheese parmesan mozzarella pepper sauce grated garlic fresh salt oil ricotta ground shredded olive basil
    Cluster 9: flour salt all purpose oil water eggs sugar butter milk pepper vegetable yeast large dry
    Cluster 10: sauce oil soy pepper chicken garlic corn starch rice sugar ginger sesame water onions salt
    Cluster 11: pepper green onions bell chicken garlic salt celery oil chopped sauce rice tomatoes ground black
    Cluster 12: powder pepper ground garlic salt black chili cumin oil onion chicken cheese corn cilantro onions
    Cluster 13: fresh sauce lime fish sugar garlic oil cilantro juice rice leaves red pepper chicken coconut
    Cluster 14: chicken boneless oil skinless garlic pepper salt onions breasts broth fresh breast sauce tomatoes halves
    Cluster 15: cheese cream shredded tortillas green cheddar onions sour ground beans corn tomatoes chicken sauce salsa
    Cluster 16: oil sesame sauce soy garlic rice ginger onions pepper sugar seeds salt green vinegar fresh
    Cluster 17: ground pepper salt garlic fresh oil cumin black ginger cloves cinnamon chicken coriander onions olive
    Cluster 18: sauce soy garlic oil sugar onions salt water rice pepper ginger vinegar chicken green fresh
    Cluster 19: chicken broth salt pepper oil fresh garlic sodium chopped olive fat black ground white cloves
    Cluster 20: sugar vanilla cream milk butter extract egg large eggs salt yolks flour chocolate water ground
    Cluster 21: pepper salt ground black garlic onions oil water butter sauce white red onion chicken fresh
    Cluster 22: dried pepper garlic salt oregano oil ground tomatoes olive black cheese basil red onions onion
    Cluster 23: cheese pepper salt butter parmesan grated ground milk black garlic cream eggs fresh large flour
    Cluster 24: oil olive garlic pepper salt tomatoes fresh cloves onions cheese red wine black parsley onion
    

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
# Liss of features
terms2 = count_vect2.get_feature_names_out()

for i in range(number_of_clusters):
    top_15_words = [terms2[ind] for ind in order_centroids2[i, :15]]
    print("Cluster {}: {}".format(i, ' '.join(top_15_words)))
```

    Cluster 0: fat sodium broth free low cooking spray reduced dry wine dried thyme cream parsley boneless
    Cluster 1: mozzarella parmesan shredded grated ricotta basil pasta italian skim tomato noodles lasagna spinach dried seasoning
    Cluster 2: extra virgin leaves vinegar lemon wine basil kosher large parsley dry sea bread olives dried
    Cluster 3: cream sour shredded tortillas cheddar salsa beans cumin corn chili jack chilies seasoning fat taco
    Cluster 4: cream potatoes large bread leaves unsalted pork broth kosher parsley dry seasoning coconut wine carrots
    Cluster 5: corn starch soy ginger sesame vinegar wine boneless scallions skinless pork broth sodium breasts chili
    Cluster 6: lime chilies jalapeno avocado cumin purple chile chili corn wedges leaves kosher tortillas orange oregano
    Cluster 7: lime fish coconut leaves paste thai ginger curry chili basil shrimp brown chile peanuts noodles
    Cluster 8: vanilla extract cream purpose large cinnamon brown unsalted baking chocolate egg corn granulated light heavy
    Cluster 9: bell celery yellow seasoning broth sausage shrimp leaves parsley diced thyme bay cayenne dried tomato
    Cluster 10: masala garam cumin ginger coriander chili paste leaves turmeric chilies seed tumeric lemon seeds tomato
    Cluster 11: cumin seeds coriander ginger leaves seed cinnamon chili turmeric mustard curry coconut chilies cardamom tumeric
    Cluster 12: large egg yolks whites cream purpose unsalted vanilla lemon extract chocolate corn spray fat heavy
    Cluster 13: parsley leaf flat wine lemon extra virgin dry bay thyme large leaves broth carrots celery
    Cluster 14: sesame soy ginger seeds vinegar toasted scallions carrots chili brown noodles wine pork sodium minced
    Cluster 15: beef broth carrots tomato wine potatoes stock thyme lean parsley dried paste bay purpose leaves
    Cluster 16: dried oregano basil thyme parsley tomato leaves diced wine bell crushed bay broth parmesan celery
    Cluster 17: corn tortillas cumin chili lime beans chilies shredded jalapeno avocado jack salsa cream cheddar chile
    Cluster 18: lemon parsley zest grated leaves large cream shrimp cinnamon orange wine dry unsalted mint kosher
    Cluster 19: purpose baking large unsalted dry yeast cream warm buttermilk corn cinnamon active potatoes kosher lemon
    Cluster 20: soy ginger vinegar pork carrots scallions brown chili mirin wine leaves chinese sodium noodles dark
    Cluster 21: baking soda purpose buttermilk large unsalted vanilla cream corn extract brown cinnamon granulated yellow meal
    Cluster 22: vinegar wine cider balsamic leaves mustard brown basil kosher purple dried cucumber lemon bell apple
    Cluster 23: beans cumin diced chili shredded broth corn seasoning salsa kidney cheddar bell dried beef leaves
    Cluster 24: parmesan grated parsley cream wine dry basil broth large bread pasta dried leaves mushrooms purpose
    

Could remove bit more stopwords (fat, sodium, free, low, large, dried, meal...) but at the same time, some words are more common in specific region. For example, maybe most regions say salt but american could be used to say sodium instead, or 'free' as in free-range chicken/eggs may be something more present in western culture so it still carries some meaning and distinction.

--------------

### PCA

#### Global


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

    Original shape: (39774, 3010)
    Reduced shape: (39774, 2)
    


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

#plt.savefig('recipes_pca.png',dpi=300,bbox_inches='tight')
```


    
![png](\img\posts\Cuisine\output_34_0.png)
    


The recipes are globally packed up, although we can see some distinctive regions (pink squares at the bottom, turquoise hexagons at the top). The other great thing is that, from the few clusters we can see, they don't have too big intra-cluster variance, we can see clear grouping of them.

#### Focus on few cuisines


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
ax.set_title("4 Cuisines' cluster",fontsize=16)

#fig.savefig('recipes_4_selected_pca.png',dpi=300,bbox_inches='tight')
```


    
![png](\img\posts\Cuisine\output_37_0.png)
    


#### Centroids


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
ax.set_title("Plot of centroids for each cuisine",fontsize=16)

#fig.savefig('recipes_centroids_pca.png',dpi=300,bbox_inches='tight')
```


    
![png](\img\posts\Cuisine\output_40_0.png)
    


#### Ingredients association 


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
      <td>-0.000019</td>
      <td>-0.000010</td>
      <td>-2.898915e-05</td>
      <td>0.000019</td>
      <td>-0.000010</td>
      <td>0.000016</td>
      <td>0.000042</td>
      <td>0.000039</td>
      <td>-0.000028</td>
      <td>0.000085</td>
      <td>...</td>
      <td>0.000047</td>
      <td>-0.000016</td>
      <td>-0.000008</td>
      <td>-0.000005</td>
      <td>-0.001142</td>
      <td>-0.000050</td>
      <td>-0.000034</td>
      <td>0.000319</td>
      <td>0.012927</td>
      <td>-0.000048</td>
    </tr>
    <tr>
      <th>second</th>
      <td>-0.000001</td>
      <td>-0.000025</td>
      <td>6.023890e-07</td>
      <td>-0.000143</td>
      <td>-0.000065</td>
      <td>-0.000025</td>
      <td>-0.000031</td>
      <td>-0.000093</td>
      <td>0.000003</td>
      <td>-0.000102</td>
      <td>...</td>
      <td>-0.000034</td>
      <td>0.000008</td>
      <td>0.000011</td>
      <td>0.000022</td>
      <td>-0.006708</td>
      <td>-0.000169</td>
      <td>-0.000017</td>
      <td>-0.000575</td>
      <td>-0.000123</td>
      <td>-0.000044</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 3010 columns</p>
</div>




```python
# Ingredients selection
select_ingredients = np.empty([18,], dtype=object)
select_ingredients[:5] = pca_df.iloc[:, np.argsort(pca_df.loc['first'])[-5:]].columns
select_ingredients[5:10] = pca_df.iloc[:, np.argsort(pca_df.loc['first'])[:5]].columns
select_ingredients[10:14] = pca_df.iloc[:, np.argsort(pca_df.loc['second'])[-4:]].columns
select_ingredients[14:] = pca_df.iloc[:, np.argsort(pca_df.loc['second'])[:4]].columns

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

#fig.savefig('ingredients_pca.png',dpi=300)
```


    
![png](\img\posts\Cuisine\output_43_0.png)
    



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

#fig.savefig('ingredients_20_pca.png',dpi=300)
```


    
![png](\img\posts\Cuisine\output_44_0.png)
    


### LDA

https://highdemandskills.com/lda-clustering/


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

    lda.components_.shape: (25, 3010)
    


```python
sorting = np.argsort(lda.components_, axis=1)[:,::-1]
feature_names = np.array(count_vect.get_feature_names_out())
```


```python
mglearn.tools.print_topics(topics=range(25),
                           feature_names=feature_names,
                           sorting=sorting,
                           topics_per_chunk=5,
                           n_words=10)
```

    topic 0       topic 1       topic 2       topic 3       topic 4       
    --------      --------      --------      --------      --------      
    pepper        wine          fresh         fat           lemon         
    bell          white         sauce         cooking       juice         
    green         fresh         fish          spray         fresh         
    onions        dry           lime          sodium        orange        
    red           butter        coconut       low           zest          
    garlic        oil           cilantro      free          lime          
    salt          chicken       sugar         broth         grated        
    tomatoes      olive         rice          reduced       sugar         
    oil           salt          garlic        chicken       water         
    rice          mushrooms     red           chopped       mint          
    
    
    topic 5       topic 6       topic 7       topic 8       topic 9       
    --------      --------      --------      --------      --------      
    egg           flour         onions        ground        ground        
    large         all           bay           sauce         pepper        
    yolks         purpose       carrots       beef          fresh         
    potatoes      salt          pork          pepper        cumin         
    whites        water         celery        salt          salt          
    eggs          butter        thyme         garlic        garlic        
    salt          oil           salt          onions        oil           
    russet        eggs          dried         tomato        ginger        
    gold          sugar         leaves        black         yogurt        
    yukon         milk          garlic        hot           paprika       
    
    
    topic 10      topic 11      topic 12      topic 13      topic 14      
    --------      --------      --------      --------      --------      
    baking        sauce         sesame        fresh         ground        
    flour         soy           oil           oil           cinnamon      
    powder        oil           soy           olive         sugar         
    salt          garlic        sauce         tomatoes      almonds       
    all           ginger        rice          onion         nutmeg        
    purpose       sugar         seeds         purple        raisins       
    eggs          pepper        sugar         pepper        brown         
    butter        sesame        onions        salt          allspice      
    sugar         rice          toasted       garlic        apples        
    soda          water         garlic        dried         golden        
    
    
    topic 15      topic 16      topic 17      topic 18      topic 19      
    --------      --------      --------      --------      --------      
    corn          vinegar       seeds         sugar         cheese        
    shrimp        mustard       powder        cream         parmesan      
    steak         pepper        salt          vanilla       grated        
    oil           salt          oil           extract       mozzarella    
    starch        cider         chili         milk          basil         
    frozen        red           coriander     butter        pasta         
    pepper        sugar         cumin         chocolate     garlic        
    medium        white         onions        heavy         fresh         
    broccoli      black         ground        eggs          italian       
    peas          dijon         leaves        brown         dried         
    
    
    topic 20      topic 21      topic 22      topic 23      topic 24      
    --------      --------      --------      --------      --------      
    chicken       cheese        olive         cheese        cilantro      
    boneless      bread         oil           cream         salt          
    skinless      pepper        pepper        shredded      lime          
    breasts       butter        extra         cheddar       garlic        
    breast        salt          virgin        sour          pepper        
    halves        ground        garlic        tortillas     fresh         
    broth         eggs          salt          green         onion         
    oil           black         red           onions        oil           
    pepper        milk          fresh         salsa         chilies       
    in            grated        tomatoes      jack          chopped       
    
    
    
