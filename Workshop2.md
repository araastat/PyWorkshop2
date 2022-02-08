---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Python Data Analysis Workshop II


## Why Python?


Python is one of the most popular general-purpose scripting languages in use today. It has over *40K* user-contributed packages for various purposes, including software prototyping and development, GUI development, web programming and scientific programming (about 2400). It is one of the best "glue" languages, that can link together functionalities from different software platforms. 


### The PyData stack

```python
from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()
```

* Numpy (starting life as Numeric), c. 1995
* Scipy (2001)
* Matplotlib (early 2000s)
* Sympy (2007)

These formed the core of the Python Data Stack. They were meant to be an open-source replacement for **Matlab**, targeting engineers and *numerical* data. 

* pandas
* statsmodels
* scikit-learn
* scikit-image

These packages are more recent developments keeping in mind current demands and the need to compete with R


## A quick review

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

```python
import numpy as np
import pandas as pd
pd.__version__
```

```python
cars = pd.read_csv('data/mtcars.csv')
cars.head()
```

```python
cars['kmpg']=cars['mpg']*1.6
cars['guzzlers'] = cars['mpg']<25
cars.sort('mpg', ascending=False)

```

```python
cars['mpg'].describe()
```

```python
cars[cars['cyl']==4]
```

```python
cars.groupby(['cyl','gear'])['mpg'].agg(['count',np.mean, np.median])
```

```python
values = [5,9,38,5,3]
vals = pd.Series(values)
vals.index = pd.Index(['a','b','c','d','e'])
vals2 = vals[['a','b','c','d','f','g']]
df = pd.DataFrame({'one':vals,'two':vals2})
df
```

```python
df.fillna(method='ffill')
```

```python
df2 = pd.DataFrame(np.random.randn(5,3), index=pd.Index(['a','b','c','d','e']))
df2.columns = ['A','B','C']
df.join(df2, how='outer')
```

```python
movies = pd.read_csv('data/movies/movies.dat',sep='::',header=None, 
                     names=['MovieID','Title','Category'])
movies.shape
```

```python
ratings = pd.read_csv('data/movies/ratings.dat',sep='::',header=None,
                      names=['RaterID','MovieID','Rating','Timestamp'])
ratings.shape
```

```python
dat = pd.merge(ratings,movies)
dat.head()
```

```python
users = pd.read_csv('data/movies/users.dat', header=None, sep='::',
                    names=['RaterID','Gender','Age','Occupation','Zip'])
dat = pd.merge(users,dat)
dat.iloc[:,:10].head()
dat.columns
```

Let's do the data reading "Pythonically"

```python
f = open('data/movies/ratings.dat','r')
x = f.readlines()
f.close()
x[:5]
```

```python
dat = [u.strip('\n').split('::') for u in x]
dat[:5]
```

```python
dat = pd.DataFrame([[int(u) for u in d] for d in dat])
dat.columns = pd.Index(['RaterID','MovieID','Rating','Timestamp'])
dat.head()
```

### Tips


imap for parallel processing
yield/next generators for large data processing

import codecs
codecs.utf8

json and utf for messily formatted files

unicodecsv package for reading and parsing messy csv files

Be VERY CAREFUL with null values. It doesn't automatically get transformed to NaN

json great for nested data, not great for columnar data

You can explicitly type columns using numpy.array 

chardet will try to _guess_ unicoding of data







# Statistical modeling


The main package for statistical modeling is <code>statsmodels</code> which has good integration with <code>pandas</code>. Some amount of modeling can also be done using <code>numpy</code> and <code>scipy</code>

```python
import statsmodels.api as sm
from patsy import dmatrices
```

```python
cars['cyl'] = cars['cyl']
cars.head()
```

```python
y,X = dmatrices('mpg~disp+wt+C(cyl)', data=cars,return_type='dataframe')
```

```python
X
```

```python
mod = sm.OLS(y,X)
res=mod.fit()
print res.summary()
```

```python
res.params
```

```python
res.pvalues
```

```python
dir(res)
```

```python
import statsmodels.formula.api as smf
mod1 = smf.ols('mpg~disp+hp+C(cyl)',data=cars)
mod1.fit().summary()
```

```python
mod1.fit().pvalues
```

```python
preds=mod1.fit().predict()
```

```python
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
```

```python
%pylab inline
d = pd.DataFrame({'y':cars['mpg'],'ypred':preds})
```

```python
plt.plot(d.ypred,d.y,'ro')
```

```python
sns.set(palette="Purples_r")
mpl.rc('figure',figsize=(5,5))
d=pd.DataFrame({'y':cars.mpg, 'ypred':preds})
```

```python
sns.lmplot('y','ypred',d)
```

### Logistic regression

```python
iris = pd.read_csv('data/iris.csv')
iris.head()

```

```python
iris['class2']=iris['class'].map(lambda x: x=='Iris-setosa')
```

```python
iris.columns
```

```python
iris['y']=1
iris['y'][iris['class2']]=0
mod_binom = smf.glm('y~sepal_length+sepal_width',data=iris, 
                    family=sm.families.Binomial())
```

```python
mod_binom.fit().summary()
```

```python
sm.stats.anova_lm(mod1.fit())
```

```python
cars.mpg.hist()
```

```python
import statsmodels
statsmodels.graphics.gofplots.qqplot(cars.mpg,line='s')
```

### Hypothesis testing

```python
from scipy import stats
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=25, scale=10, size=500)
stats.ttest_ind(rvs1,rvs2)
```

```python
stats.pearsonr(cars.mpg, cars.wt)
```

```python
cars.plot('mpg','wt',kind='scatter', title='MPG vs wt',grid=True);
```

## Machine Learning


### Scikits-learn

Methods include:

+ Cluster analysis
+ Dimension reduction
+ Generalized linear models
+ Support Vector Machines
+ Nearest neighbors
+ Decision Trees
+ Ensemble methods
+ Discriminant analysis
+ Cross-validation
+ Transformations

```python
import sklearn as sk
from sklearn.cross_validation import cross_val_score
```

```python
from sklearn import neighbors, datasets
#iris2=datasets.load_iris()
n_neighbors=5
X = np.array(iris.loc[:150,'sepal_length':'sepal_width'])[:-1,:]

```

```python
def convert2num(x):
    if x=='Iris-virginica':
        y=0
    elif x=='Iris-setosa':
        y=1
    else:
        y=2
    return y

y = iris['class'].map(convert2num)[:-1]
```

```python
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X,y)
```

```python
ypred1=clf.predict(X)
ypred1
```

```python
cv1 = cross_val_score(clf, X,y)
cv1
```

```python
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(X,y)
```

```python
ypred2 = clf.predict(X)
cv2 = cross_val_score(clf,X,y)
cv2
```

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(X,y)
ypred3 = clf.predict(X)
cv3 = cross_val_score(clf, X,y)
cv3
```

```python
cv1.mean()
```

```python
cv2.mean()
```

```python
cv3.mean()
```

```python
from ggplot import *
```

```python
print ggplot(cars, aes('mpg', 'qsec')) +\
    geom_point(colour='steelblue') + facet_wrap('cyl')+ \
    scale_x_continuous(breaks=[10,20,30],labels=['horrible','ok','awesome'])
```

```python
from bokeh.sampledata.iris import flowers
from bokeh.plotting import *

output_file("iris.html", title="iris.py example")

colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

flowers['color'] = flowers['species'].map(lambda x: colormap[x])

#setting the name kwarg will give this scatter plot a user
#friendly id, and the corresponding embed.js will have a nice name
#too

scatter(flowers["petal_length"], flowers["petal_width"],
        color=flowers["color"], fill_alpha=0.2, size=10, name="iris")

show()
```

## Python as glue

```python
%load_ext rmagic
```

```python
%R X=c(1,4,5,7); sd(X);mean(X)
```

```R
Y = c(2,4,3,9)
print(summary(lm(Y~X)))
```

```python
%R plot(X,Y)
```

There is also a SQL magic, found [here](https://github.com/catherinedevlin/ipython-sql)

```R
library(randomForest)
rf=randomForest(iris[,1:4],iris[,5])
print(rf)
```

```python
len(iris['class'][:150])
```

```python
preds = pd.DataFrame({'orig':iris['class'][:150], 
                      'knn':ypred1, 'DT':ypred2, 'RF':ypred3})
```

```python
pd.crosstab(preds['orig'],preds['knn'])
```

```python
pd.crosstab(preds['orig'],preds['DT'])
```

```python
pd.crosstab(preds['orig'],preds['RF'])
```

## Bootstrapping

```python
x = np.arange(0, 101)
y = 2*x + np.random.normal(0, 10, 101)
 
# Add the column of ones for the intercept
X = sm.add_constant(x, prepend=False)
 
# Plot the data
plt.clf()
plt.plot(x, y, 'bo', markersize=10)
 
```

```python
# Define the OLS models
mod1 = sm.OLS(y, X)
 
# Fit the OLS model
results = mod1.fit()
 
# Get the fitted values
yHat = results.fittedvalues
 
# Get the residuals
resid = results.resid
 
```

```python
# Set bootstrap size
n = 10000
 
t0 = time.time()
# Save the slope
b1 = np.zeros( (n) )
b1[0] = results.params[0]
 
for i in np.arange(1, 10000):
    residBoot = np.random.permutation(resid)
    yBoot = yHat + residBoot
    modBoot = sm.OLS(yBoot, X)
    resultsBoot = modBoot.fit()
    b1[i] = resultsBoot.params[0]
```

```python
b1[:10]
```

```python
np.mean(b1)
```

```python
np.std(b1)
```

```python

```
