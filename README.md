[![Open in Jupyter](https://img.shields.io/badge/Open%20in-Jupyter-blue.svg?logo=jupyter)](https://github.com/Ender17133/eBay_CarSales/blob/main/Basics.ipynb)
# Exploring eBay Car Sales Data
In this project, I am going to work with dataset of used cars from *eBay Kleinanzeigen* (classified section of German eBay webpage).

You can access the original dataset from [here](https://data.world/data-society/used-cars-data)

The original dataset differs from the dataset that I am working with in this analysis due to two reasons:
 - 50,000 data points were selected from the full dataset to ensure high performance. 
 - Dataset was dirtied. (The original dataset is cleaner. Dataset was dirtied for a training of data cleaning skills.)
 
 Dictionary: 
  - **dateCrawled** - When this ad was first crawled. 
  - **name** - Name of the car.
  - **seller** - Whether the seller is private or dealer. 
  - **offerType** - The type of listing. 
  - **price** - The price on the ad to sell the car. 
  - **abtest** - Whether the listing is included in an A/B test. 
  - **vehicleType** - Type of the vehicle. 
  - **yearOfRegistration** - The year in which the car was first registered. 
  - **gearbox** - The transmission type. 
  - **powerPS** - The power of the car in PS. 
  - **model** - The car model name. 
  - **odometer** - How many kilometers the car has driven. 
  - **monthOfRegistration** - The month in which the car was first registered. 
  - **fuelType** - Type of fuel that car uses. 
  - **brand** - The brand of the car. 
  - **notRepairedDamage** - If the car has a damage which is not yet repaired. 
  - **dateCreated** - The date on which the eBay listing was created. 
  - **nrOfPictures** - The number of pictures in the ad. 
  - **postalCode** - The postal code for the location of the vehicle. 
  - **lastSeenOnline** - When the crawler saw this ad last online.
  
  My goal of the project is to clean the data and analyze the car listings in this dataset. 

## Introduction - importing libraries and the dataset 


```python
import pandas as pd
import numpy as np

# autos = pd.read_csv('autos.csv') 
# We can't read the file with default UTF-8 encoding

autos = pd.read_csv('autos.csv', encoding = 'Latin-1')
# Another encoding type also works if you want to try: 
# autos = pd.read_csv('autos.csv', encoding = 'Windows-1252')


```


```python
autos # The data is very messy and raw. I should start cleaning it. 
autos.info() # There are also missing values in few columns. 
autos.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 20 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   dateCrawled          50000 non-null  object
     1   name                 50000 non-null  object
     2   seller               50000 non-null  object
     3   offerType            50000 non-null  object
     4   price                50000 non-null  object
     5   abtest               50000 non-null  object
     6   vehicleType          44905 non-null  object
     7   yearOfRegistration   50000 non-null  int64 
     8   gearbox              47320 non-null  object
     9   powerPS              50000 non-null  int64 
     10  model                47242 non-null  object
     11  odometer             50000 non-null  object
     12  monthOfRegistration  50000 non-null  int64 
     13  fuelType             45518 non-null  object
     14  brand                50000 non-null  object
     15  notRepairedDamage    40171 non-null  object
     16  dateCreated          50000 non-null  object
     17  nrOfPictures         50000 non-null  int64 
     18  postalCode           50000 non-null  int64 
     19  lastSeen             50000 non-null  object
    dtypes: int64(5), object(15)
    memory usage: 7.6+ MB
    




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
      <th>dateCrawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offerType</th>
      <th>price</th>
      <th>abtest</th>
      <th>vehicleType</th>
      <th>yearOfRegistration</th>
      <th>gearbox</th>
      <th>powerPS</th>
      <th>model</th>
      <th>odometer</th>
      <th>monthOfRegistration</th>
      <th>fuelType</th>
      <th>brand</th>
      <th>notRepairedDamage</th>
      <th>dateCreated</th>
      <th>nrOfPictures</th>
      <th>postalCode</th>
      <th>lastSeen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$5,000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150,000km</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>0</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-26 18:57:24</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manuell</td>
      <td>102</td>
      <td>golf</td>
      <td>70,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>35394</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-12 16:58:10</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$4,350</td>
      <td>control</td>
      <td>kleinwagen</td>
      <td>2007</td>
      <td>automatik</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>nein</td>
      <td>2016-03-12 00:00:00</td>
      <td>0</td>
      <td>33729</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-04-01 14:38:50</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,350</td>
      <td>test</td>
      <td>kombi</td>
      <td>2003</td>
      <td>manuell</td>
      <td>0</td>
      <td>focus</td>
      <td>150,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>nein</td>
      <td>2016-04-01 00:00:00</td>
      <td>0</td>
      <td>39218</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
  </tbody>
</table>
</div>



We can see several issues in the dataset:
 - 1. There are a lot of null values in some columns.  
 - 2. Column names are also not clear, as columns are written in camelcase. 

## Cleaning Column Names



```python
autos.columns
```




    Index(['dateCrawled', 'name', 'seller', 'offerType', 'price', 'abtest',
           'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model',
           'odometer', 'monthOfRegistration', 'fuelType', 'brand',
           'notRepairedDamage', 'dateCreated', 'nrOfPictures', 'postalCode',
           'lastSeen'],
          dtype='object')



 - I will change the columns from camelcase to snakecase
 - I will change the wordings of the columns so they will be more accurate and understandable. 


```python
autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'ab_test',
       'vehicle_type', 'registration_year', 'gearbox', 'power_ps', 'model',
       'odometer', 'registration_month', 'fuel_type', 'brand',
       'unrepaired_damage', 'ad_created', 'num_photos', 'postal_code',
       'last_seen']
autos.head() # Columns are cleaner now. 
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
      <th>date_crawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offer_type</th>
      <th>price</th>
      <th>ab_test</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>num_photos</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$5,000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150,000km</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>0</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-26 18:57:24</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manuell</td>
      <td>102</td>
      <td>golf</td>
      <td>70,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>35394</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-12 16:58:10</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$4,350</td>
      <td>control</td>
      <td>kleinwagen</td>
      <td>2007</td>
      <td>automatik</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>nein</td>
      <td>2016-03-12 00:00:00</td>
      <td>0</td>
      <td>33729</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-04-01 14:38:50</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,350</td>
      <td>test</td>
      <td>kombi</td>
      <td>2003</td>
      <td>manuell</td>
      <td>0</td>
      <td>focus</td>
      <td>150,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>nein</td>
      <td>2016-04-01 00:00:00</td>
      <td>0</td>
      <td>39218</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
  </tbody>
</table>
</div>



## Initial Exploration and Cleaning
Now let's do some basic data exploration: 
- Text columns where all or almost all values are the same should be removed, as they are not useful for the analysis. 
- Numeric data stored in text format which can be cleaned and converted. 


```python
autos.describe(include = 'all')
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
      <th>date_crawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offer_type</th>
      <th>price</th>
      <th>ab_test</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>num_photos</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000</td>
      <td>50000</td>
      <td>50000</td>
      <td>50000</td>
      <td>50000</td>
      <td>50000</td>
      <td>44905</td>
      <td>50000.000000</td>
      <td>47320</td>
      <td>50000.000000</td>
      <td>47242</td>
      <td>50000</td>
      <td>50000.000000</td>
      <td>45518</td>
      <td>50000</td>
      <td>40171</td>
      <td>50000</td>
      <td>50000.0</td>
      <td>50000.000000</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>48213</td>
      <td>38754</td>
      <td>2</td>
      <td>2</td>
      <td>2357</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>245</td>
      <td>13</td>
      <td>NaN</td>
      <td>7</td>
      <td>40</td>
      <td>2</td>
      <td>76</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39481</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2016-04-02 11:37:04</td>
      <td>Ford_Fiesta</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$0</td>
      <td>test</td>
      <td>limousine</td>
      <td>NaN</td>
      <td>manuell</td>
      <td>NaN</td>
      <td>golf</td>
      <td>150,000km</td>
      <td>NaN</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-04-03 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-04-07 06:17:27</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>3</td>
      <td>78</td>
      <td>49999</td>
      <td>49999</td>
      <td>1421</td>
      <td>25756</td>
      <td>12859</td>
      <td>NaN</td>
      <td>36993</td>
      <td>NaN</td>
      <td>4024</td>
      <td>32424</td>
      <td>NaN</td>
      <td>30107</td>
      <td>10687</td>
      <td>35232</td>
      <td>1946</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2005.073280</td>
      <td>NaN</td>
      <td>116.355920</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.723360</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>50813.627300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>105.712813</td>
      <td>NaN</td>
      <td>209.216627</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.711984</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>25779.747957</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1000.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1067.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1999.000000</td>
      <td>NaN</td>
      <td>70.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>30451.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2003.000000</td>
      <td>NaN</td>
      <td>105.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>49577.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2008.000000</td>
      <td>NaN</td>
      <td>150.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>71540.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9999.000000</td>
      <td>NaN</td>
      <td>17700.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>99998.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Columns `seller`, `offer_type` and `num_photos` look odd. These columns should ve investigated further. 


```python
autos['seller'].value_counts()

```




    privat        49999
    gewerblich        1
    Name: seller, dtype: int64




```python
autos['offer_type'].value_counts()
```




    Angebot    49999
    Gesuch         1
    Name: offer_type, dtype: int64




```python
autos['num_photos'].value_counts()
```




    0    50000
    Name: num_photos, dtype: int64



 - 1. `seller` and `offer_type` are columns where nearly all of the values are the same. 
 - 2. `num_photos` has 0 for every column. 
 
 We are going to drop these 3 columns as they are not useful for the data analysis.


```python
autos = autos.drop(['seller', 'num_photos', 'offer_type'], axis = 1)
```


```python
autos.columns # Columns are sucessfully dropped. 
```




    Index(['date_crawled', 'name', 'price', 'ab_test', 'vehicle_type',
           'registration_year', 'gearbox', 'power_ps', 'model', 'odometer',
           'registration_month', 'fuel_type', 'brand', 'unrepaired_damage',
           'ad_created', 'postal_code', 'last_seen'],
          dtype='object')



Let's look which columns have numeric data written in the text format. I am going to clean data of such columns and turn it into numeric type.


```python
autos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 17 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   date_crawled        50000 non-null  object
     1   name                50000 non-null  object
     2   price               50000 non-null  object
     3   ab_test             50000 non-null  object
     4   vehicle_type        44905 non-null  object
     5   registration_year   50000 non-null  int64 
     6   gearbox             47320 non-null  object
     7   power_ps            50000 non-null  int64 
     8   model               47242 non-null  object
     9   odometer            50000 non-null  object
     10  registration_month  50000 non-null  int64 
     11  fuel_type           45518 non-null  object
     12  brand               50000 non-null  object
     13  unrepaired_damage   40171 non-null  object
     14  ad_created          50000 non-null  object
     15  postal_code         50000 non-null  int64 
     16  last_seen           50000 non-null  object
    dtypes: int64(4), object(13)
    memory usage: 6.5+ MB
    

It can be observed that columns `price` and `odometer` columns have numeric values written in text form.

Let's start from `price` column.


```python
autos['price'].value_counts() # We should remove $ and , signs 
autos['price'] = autos['price'].str.replace('$','').str.replace(',','').astype(int)

```

    C:\Users\Beibarys Nyussupov\AppData\Local\Temp\ipykernel_46968\222734509.py:2: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.
      autos['price'] = autos['price'].str.replace('$','').str.replace(',','').astype(int)
    


```python
autos['price'].head() # Completed!

```




    0    5000
    1    8500
    2    8990
    3    4350
    4    1350
    Name: price, dtype: int32



Now, let's change values of `odometer` column.


```python
autos['odometer'].value_counts()
```




    150,000km    32424
    125,000km     5170
    100,000km     2169
    90,000km      1757
    80,000km      1436
    70,000km      1230
    60,000km      1164
    50,000km      1027
    5,000km        967
    40,000km       819
    30,000km       789
    20,000km       784
    10,000km       264
    Name: odometer, dtype: int64




```python
autos['odometer'] = autos['odometer'].str.replace(',','').str.replace('km','').astype(int)
```


```python
autos['odometer'].head() # Completed!
```




    0    150000
    1    150000
    2     70000
    3     70000
    4    150000
    Name: odometer, dtype: int32



We cleaned values of `odometer` column and transformed these values into int-64 form. However, let's make a change to the name of the odometer column to make accurate description of these values.


```python
autos.rename({'odometer':'odometer_km'}, axis = 1, inplace = True)

```


```python
autos['odometer_km'].head()
```




    0    150000
    1    150000
    2     70000
    3     70000
    4    150000
    Name: odometer_km, dtype: int32



## Translating German words to English words



```python
autos.head()
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
      <th>date_crawled</th>
      <th>name</th>
      <th>price</th>
      <th>ab_test</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer_km</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>5000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150000</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>8500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150000</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-26 18:57:24</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>8990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manuell</td>
      <td>102</td>
      <td>golf</td>
      <td>70000</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>35394</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-12 16:58:10</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>4350</td>
      <td>control</td>
      <td>kleinwagen</td>
      <td>2007</td>
      <td>automatik</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70000</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>nein</td>
      <td>2016-03-12 00:00:00</td>
      <td>33729</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-04-01 14:38:50</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>1350</td>
      <td>test</td>
      <td>kombi</td>
      <td>2003</td>
      <td>manuell</td>
      <td>0</td>
      <td>focus</td>
      <td>150000</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>nein</td>
      <td>2016-04-01 00:00:00</td>
      <td>39218</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
  </tbody>
</table>
</div>




```python
autos['vehicle_type'].unique()
```




    array(['bus', 'limousine', 'kleinwagen', 'kombi', nan, 'coupe', 'suv',
           'cabrio', 'andere'], dtype=object)



First column that has entries with German language is - `vehicle_type`. It is an important column and we need to change entries written in German to English language.

`Kleinwagen` - Compact/Subcompact
    
`Kombi` - Wagon

`andere` - other

We also have NA values which should be grouped into other type of cars too.




```python
before = autos['vehicle_type'].value_counts()
```


```python
before
```




    limousine     12859
    kleinwagen    10822
    kombi          9127
    bus            4093
    cabrio         3061
    coupe          2537
    suv            1986
    andere          420
    Name: vehicle_type, dtype: int64




```python
autos['vehicle_type'] = autos['vehicle_type'].replace('kleinwagen', 'compact/subcompact')
autos['vehicle_type'] = autos['vehicle_type'].replace('kombi', 'wagon')
autos['vehicle_type'] = autos['vehicle_type'].replace('andere', 'other')
autos['vehicle_type'] = autos['vehicle_type'].fillna('undefined')
        
```


```python
after = autos['vehicle_type'].value_counts().sort_values(ascending = False)

```


```python
after
```




    limousine             12859
    compact/subcompact    10822
    wagon                  9127
    undefined              5095
    bus                    4093
    cabrio                 3061
    coupe                  2537
    suv                    1986
    other                   420
    Name: vehicle_type, dtype: int64



Everything in `vehichle_type` column was cleaned. Now every vehicle type is clear and understandable in English language. 

Limousine, compact/subcompact, wagon, bus and cabrio are vehicle types with the most number of car listings on the website.

The next two columns that we need to clean are `gearbox` and `unrepaired_damage`

`automatik` - automatic

`manuell` - manual 

`nein` - no

`ja` - yes


```python
autos['gearbox'].unique()
```




    array(['manuell', 'automatik', nan], dtype=object)




```python
autos['gearbox'].value_counts()
```




    manuell      36993
    automatik    10327
    Name: gearbox, dtype: int64




```python
autos['gearbox'] = autos['gearbox'].replace('manuell', 'manual')
autos['gearbox'] = autos['gearbox'].replace('automatik', 'automatic')
autos['gearbox'] = autos['gearbox'].fillna('undefined')
```


```python
autos['gearbox'].value_counts()
```




    manual       36993
    automatic    10327
    undefined     2680
    Name: gearbox, dtype: int64




```python
autos['unrepaired_damage'].unique()
```




    array(['nein', nan, 'ja'], dtype=object)




```python
autos['unrepaired_damage'] = autos['unrepaired_damage'].replace('nein', 'no')
autos['unrepaired_damage'] = autos['unrepaired_damage'].replace('ja', 'yes')
autos['unrepaired_damage'] = autos['unrepaired_damage'].fillna('undefined')
```


```python
autos['unrepaired_damage'].value_counts()
```




    no           35232
    undefined     9829
    yes           4939
    Name: unrepaired_damage, dtype: int64



The rest two columns were cleaned and entries were transformed into understandable English language. Most of the cars on the market are on manual engine and do not have unrepaired damage.

## Exploring the Odometer and Price Columns


```python
autos['odometer_km'].value_counts().sort_index(ascending = True)
```




    5000        967
    10000       264
    20000       784
    30000       789
    40000       819
    50000      1027
    60000      1164
    70000      1230
    80000      1436
    90000      1757
    100000     2169
    125000     5170
    150000    32424
    Name: odometer_km, dtype: int64




```python
autos['odometer_km'].describe()
```




    count     50000.000000
    mean     125732.700000
    std       40042.211706
    min        5000.000000
    25%      125000.000000
    50%      150000.000000
    75%      150000.000000
    max      150000.000000
    Name: odometer_km, dtype: float64



Mileage values of cars are rounded, which means that sellers of vehicles had to choose from pre-existing options on the market to define the number of kilometres driven by each car. Furthermore, there are no listed cars with 0 mileage on the market. However it can be a drawback of prexisting options on eBay, as if there are cars that have a mileage below 5000 kilometres - sellers still had to choose 5000 due to limited options. 


```python
autos['price'].unique().shape # Number of unique values in price column
```




    (2357,)




```python
autos['price'].value_counts().head(20).sort_index(ascending = True)
```




    0       1421
    300      384
    500      781
    600      531
    650      419
    700      395
    750      433
    800      498
    850      410
    900      420
    950      379
    999      434
    1000     639
    1200     639
    1500     734
    2000     460
    2200     382
    2500     643
    3500     498
    4500     394
    Name: price, dtype: int64



Again, values are very rounded. However, given that there are 2357 unique values in the price column, it means that sellers are just tended to round numbers and they are not limited to pre-existing options on eBay webpage.


```python
autos['price'].describe()

```




    count    5.000000e+04
    mean     9.840044e+03
    std      4.811044e+05
    min      0.000000e+00
    25%      1.100000e+03
    50%      2.950000e+03
    75%      7.200000e+03
    max      1.000000e+08
    Name: price, dtype: float64




```python
autos['price'].value_counts().sort_index(ascending = True).head(50)
```




    0      1421
    1       156
    2         3
    3         1
    5         2
    8         1
    9         1
    10        7
    11        2
    12        3
    13        2
    14        1
    15        2
    17        3
    18        1
    20        4
    25        5
    29        1
    30        7
    35        1
    40        6
    45        4
    47        1
    49        4
    50       49
    55        2
    59        1
    60        9
    65        5
    66        1
    70       10
    75        5
    79        1
    80       15
    89        1
    90        5
    99       19
    100     134
    110       3
    111       2
    115       2
    117       1
    120      39
    122       1
    125       8
    129       1
    130      15
    135       1
    139       1
    140       9
    Name: price, dtype: int64




```python
autos['price'].value_counts().sort_index(ascending = False).head(20)
```




    99999999    1
    27322222    1
    12345678    3
    11111111    2
    10000000    1
    3890000     1
    1300000     1
    1234566     1
    999999      2
    999990      1
    350000      1
    345000      1
    299000      1
    295000      1
    265000      1
    259000      1
    250000      1
    220000      1
    198000      1
    197000      1
    Name: price, dtype: int64




```python
(len(autos[autos['price'] > 350000]) / len(autos)) * 100 # Percentage of cars with a cell price higher than 350 000 USD

```




    0.027999999999999997



There are plenty of car listings with a prices below 30 dollars and 1421 listings with 0 prices. There are also 14 listings with prices above 350 000 dollars. 

Assuming that eBay is an auction website, there can be cars that where the opening equals to 1 dollar. However, I am going to remove all car listings with prices above 350 000 $ as prices increase regularly and then jummp up to very high and less realistic numbers. 


```python
autos = autos[autos['price'].between(1, 350000)]
```


```python
autos['price'].describe() # We removed the outliers in prices
```




    count     48565.000000
    mean       5888.935591
    std        9059.854754
    min           1.000000
    25%        1200.000000
    50%        3000.000000
    75%        7490.000000
    max      350000.000000
    Name: price, dtype: float64



## Exploring the date columns
Columns with date information: 
 - `date_crawled` 
 - `last_seen`
 - `ad_created`
 - `registration_month`
 - `registration_year`
 
 There is a mix of information that was created by crawler and information that was produced by website itself. Let's explore these columns a bit more. 


```python
autos[['date_crawled', 'ad_created', 'last_seen']][0:5]
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
      <th>date_crawled</th>
      <th>ad_created</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-26 17:47:46</td>
      <td>2016-03-26 00:00:00</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-04-04 13:38:56</td>
      <td>2016-04-04 00:00:00</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-26 18:57:24</td>
      <td>2016-03-26 00:00:00</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-12 16:58:10</td>
      <td>2016-03-12 00:00:00</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-04-01 14:38:50</td>
      <td>2016-04-01 00:00:00</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
  </tbody>
</table>
</div>




```python
(autos['date_crawled'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = True)) * 100
```




    2016-03-05    2.532688
    2016-03-06    1.404304
    2016-03-07    3.601359
    2016-03-08    3.329558
    2016-03-09    3.308967
    2016-03-10    3.218367
    2016-03-11    3.257490
    2016-03-12    3.691959
    2016-03-13    1.566972
    2016-03-14    3.654896
    2016-03-15    3.428395
    2016-03-16    2.960980
    2016-03-17    3.162772
    2016-03-18    1.291053
    2016-03-19    3.477813
    2016-03-20    3.788737
    2016-03-21    3.737259
    2016-03-22    3.298672
    2016-03-23    3.222485
    2016-03-24    2.934212
    2016-03-25    3.160712
    2016-03-26    3.220426
    2016-03-27    3.109235
    2016-03-28    3.486050
    2016-03-29    3.409863
    2016-03-30    3.368681
    2016-03-31    3.183363
    2016-04-01    3.368681
    2016-04-02    3.547823
    2016-04-03    3.860805
    2016-04-04    3.648718
    2016-04-05    1.309585
    2016-04-06    0.317101
    2016-04-07    0.140019
    Name: date_crawled, dtype: float64



According to the data, the website was crawled daily in a period of one month from April to March 2016.


```python
(autos['last_seen'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = True))* 100
```




    2016-03-05     0.107073
    2016-03-06     0.432410
    2016-03-07     0.539483
    2016-03-08     0.741275
    2016-03-09     0.959539
    2016-03-10     1.066612
    2016-03-11     1.237517
    2016-03-12     2.378256
    2016-03-13     0.889529
    2016-03-14     1.260167
    2016-03-15     1.587563
    2016-03-16     1.645218
    2016-03-17     2.808607
    2016-03-18     0.735097
    2016-03-19     1.583445
    2016-03-20     2.065273
    2016-03-21     2.063214
    2016-03-22     2.137342
    2016-03-23     1.853186
    2016-03-24     1.976732
    2016-03-25     1.921137
    2016-03-26     1.680222
    2016-03-27     1.564913
    2016-03-28     2.085864
    2016-03-29     2.234119
    2016-03-30     2.477093
    2016-03-31     2.378256
    2016-04-01     2.279419
    2016-04-02     2.491506
    2016-04-03     2.520334
    2016-04-04     2.448265
    2016-04-05    12.476063
    2016-04-06    22.180583
    2016-04-07    13.194688
    Name: last_seen, dtype: float64



`last_seen` column has a data recorded by crawler which shows the last date on which any car listing was seen. Looking on these values can help us with determining on what day the listing was removed, possibly because the car was sold. 

Last three days of March (05, 06, 07) have disproportinate `last_seen` percentages, which are around 10 times bigger from other days. It is very unlikely that these 3 days experienced a rapid hike in sales and it is more likely that these values are related to crawling period ending and don't show spike in in car sales.


```python
(autos['ad_created'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = True)) * 100
```




    2015-06-11    0.002059
    2015-08-10    0.002059
    2015-09-09    0.002059
    2015-11-10    0.002059
    2015-12-05    0.002059
                    ...   
    2016-04-03    3.885514
    2016-04-04    3.685782
    2016-04-05    1.181921
    2016-04-06    0.325337
    2016-04-07    0.125605
    Name: ad_created, Length: 76, dtype: float64




```python
print('Number of dates crawled:', autos['date_crawled'].str[:10].unique().shape)
print('Number of dates ad_created:', autos['ad_created'].str[:10].unique().shape)
(autos['ad_created'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = True)) * 100
```

    Number of dates crawled: (34,)
    Number of dates ad_created: (76,)
    




    2015-06-11    0.002059
    2015-08-10    0.002059
    2015-09-09    0.002059
    2015-11-10    0.002059
    2015-12-05    0.002059
                    ...   
    2016-04-03    3.885514
    2016-04-04    3.685782
    2016-04-05    1.181921
    2016-04-06    0.325337
    2016-04-07    0.125605
    Name: ad_created, Length: 76, dtype: float64



There is a variety of dates when car listings were created. Some of them are  within 1-2 months of the listing date, but some other dates are old and few of them even fall within around 8-9 months.

We can also turn dates in `date_crawled`, `ad_created` and `last_seen` columns to uniform numeric data. 


```python
autos['date_crawled'] = autos['date_crawled'].str.replace('-','').str.split(' ').str[0].astype(int)
autos['ad_created'] = autos['ad_created'].str.replace('-','').str.split(' ').str[0].astype(int)
autos['last_seen'] = autos['last_seen'].str.replace('-','').str.split(' ').str[0].astype(int)
```


```python
autos['date_crawled'].head()
```




    0    20160326
    1    20160404
    2    20160326
    3    20160312
    4    20160401
    Name: date_crawled, dtype: int32




```python
autos['ad_created'].head()
```




    0    20160326
    1    20160404
    2    20160326
    3    20160312
    4    20160401
    Name: ad_created, dtype: int32




```python
autos['last_seen'].head()
```




    0    20160406
    1    20160406
    2    20160406
    3    20160315
    4    20160401
    Name: last_seen, dtype: int32




```python
autos.head()
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
      <th>date_crawled</th>
      <th>name</th>
      <th>price</th>
      <th>ab_test</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer_km</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20160326</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>5000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manual</td>
      <td>158</td>
      <td>andere</td>
      <td>150000</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>no</td>
      <td>20160326</td>
      <td>79588</td>
      <td>20160406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20160404</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>8500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatic</td>
      <td>286</td>
      <td>7er</td>
      <td>150000</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>no</td>
      <td>20160404</td>
      <td>71034</td>
      <td>20160406</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20160326</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>8990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manual</td>
      <td>102</td>
      <td>golf</td>
      <td>70000</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>no</td>
      <td>20160326</td>
      <td>35394</td>
      <td>20160406</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20160312</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>4350</td>
      <td>control</td>
      <td>compact/subcompact</td>
      <td>2007</td>
      <td>automatic</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70000</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>no</td>
      <td>20160312</td>
      <td>33729</td>
      <td>20160315</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20160401</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>1350</td>
      <td>test</td>
      <td>wagon</td>
      <td>2003</td>
      <td>manual</td>
      <td>0</td>
      <td>focus</td>
      <td>150000</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>no</td>
      <td>20160401</td>
      <td>39218</td>
      <td>20160401</td>
    </tr>
  </tbody>
</table>
</div>




```python
autos['registration_year'].describe()
```




    count    48565.000000
    mean      2004.755421
    std         88.643887
    min       1000.000000
    25%       1999.000000
    50%       2004.000000
    75%       2008.000000
    max       9999.000000
    Name: registration_year, dtype: float64



It is more likely that the date when the car was first registered will indicate the age of the car. Some strange values can be seen in above statistics. Minimum registration year is `1000` when cars even weren't used at that time and maximum is `9999` - a year which we even didn't reach yet. 

## Dealing with Incorrect Registration Year Data

Because a car can't be registered after the listing was seen, any vehicle with a registration year above 2016 is definitely inaccurate. However determining the earliest valid year is more complex. It is more likely to be somewhere in 1900s. 

Let's count the number of listings with cars that fall outside the 1900-2016 interval and see if it's safe to tremove those rows entirely, or if we need more custom logic.



```python
((~autos["registration_year"].between(1900,2016)).sum() / autos.shape[0]) * 100
```




    3.8793369710697



As the inaccurate dates are less than 4% of the data, we can safely remove such rows.


```python
autos = autos[autos["registration_year"].between(1900,2016)]
(autos["registration_year"].value_counts(normalize = True, dropna = False).sort_index(ascending = False).head(25))*100
```




    2016    2.613483
    2015    0.839742
    2014    1.420278
    2013    1.720186
    2012    2.806281
    2011    3.476789
    2010    3.403954
    2009    4.466485
    2008    4.744971
    2007    4.877788
    2006    5.719672
    2005    6.289497
    2004    5.790364
    2003    5.781796
    2002    5.325507
    2001    5.646837
    2000    6.760781
    1999    6.205951
    1998    5.062017
    1997    4.179431
    1996    2.941239
    1995    2.628478
    1994    1.347443
    1993    0.910435
    1992    0.792614
    Name: registration_year, dtype: float64




```python
(autos['registration_year'].value_counts(normalize = True).head(25))*100
```




    2000    6.760781
    2005    6.289497
    1999    6.205951
    2004    5.790364
    2003    5.781796
    2006    5.719672
    2001    5.646837
    2002    5.325507
    1998    5.062017
    2007    4.877788
    2008    4.744971
    2009    4.466485
    1997    4.179431
    2011    3.476789
    2010    3.403954
    1996    2.941239
    2012    2.806281
    1995    2.628478
    2016    2.613483
    2013    1.720186
    2014    1.420278
    1994    1.347443
    1993    0.910435
    2015    0.839742
    1992    0.792614
    Name: registration_year, dtype: float64



As inaccurate rows were removed, it can be seen that most of the cars were first registered in past 22 years.

## Exploring Price by Brand


```python
(autos['brand'].value_counts(normalize = True)) * 100
```




    volkswagen        21.126368
    bmw               11.004477
    opel              10.758124
    mercedes_benz      9.646323
    audi               8.656627
    ford               6.989996
    renault            4.714980
    peugeot            2.984083
    fiat               2.564212
    seat               1.827296
    skoda              1.640925
    nissan             1.527388
    mazda              1.518819
    smart              1.415994
    citroen            1.400998
    toyota             1.270324
    hyundai            1.002549
    sonstige_autos     0.981127
    volvo              0.914719
    mini               0.876159
    mitsubishi         0.822604
    honda              0.784045
    kia                0.706926
    alfa_romeo         0.664082
    porsche            0.612669
    suzuki             0.593389
    chevrolet          0.569825
    chrysler           0.351321
    dacia              0.263490
    daihatsu           0.250637
    jeep               0.227073
    subaru             0.214220
    land_rover         0.209936
    saab               0.164949
    jaguar             0.156381
    daewoo             0.149954
    trabant            0.139243
    rover              0.132816
    lancia             0.107110
    lada               0.057839
    Name: brand, dtype: float64




```python
(autos['brand'].value_counts(normalize = True).head(7))*100
```




    volkswagen       21.126368
    bmw              11.004477
    opel             10.758124
    mercedes_benz     9.646323
    audi              8.656627
    ford              6.989996
    renault           4.714980
    Name: brand, dtype: float64



There are a wide range of car brands that do not have significant percentages of listings, that's why I have filtered down the data to the manufacturers that have more than 4% of total listings.

German transport manufacturers represent top 5 brands with around 60% of overall listings. Volkswagen is the most popular car brand in the market having approximately 21% of overall listings. 



```python
brand_counts = (autos['brand'].value_counts(normalize = True)) * 100
top_brands = brand_counts[brand_counts > 4].index
print(top_brands)
```

    Index(['volkswagen', 'bmw', 'opel', 'mercedes_benz', 'audi', 'ford',
           'renault'],
          dtype='object')
    


```python
mean_prices = {}
for brand in top_brands: 
    names = autos[autos['brand'] == brand]
    price = names['price'].mean()
    mean_prices[brand] = int(price)
    
mean_prices
```




    {'volkswagen': 5402,
     'bmw': 8332,
     'opel': 2975,
     'mercedes_benz': 8628,
     'audi': 9336,
     'ford': 3749,
     'renault': 2474}



According to the data: 
        
        - Audi, Bmw and Mercedes Benz are more expensive car brands. 
        - Ford, Openal and Renault are less expensive car manufacturers. 
        - Folkswagen - It is a car brand that is "something in between", 
        which also explains the popularity of given brand.

## Exploring mileage




```python
bmp_series = pd.Series(mean_prices).sort_values(ascending = False)
price_df = pd.DataFrame(bmp_series, columns = ['mean_price($)'])
price_df

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
      <th>mean_price($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>audi</th>
      <td>9336</td>
    </tr>
    <tr>
      <th>mercedes_benz</th>
      <td>8628</td>
    </tr>
    <tr>
      <th>bmw</th>
      <td>8332</td>
    </tr>
    <tr>
      <th>volkswagen</th>
      <td>5402</td>
    </tr>
    <tr>
      <th>ford</th>
      <td>3749</td>
    </tr>
    <tr>
      <th>opel</th>
      <td>2975</td>
    </tr>
    <tr>
      <th>renault</th>
      <td>2474</td>
    </tr>
  </tbody>
</table>
</div>




```python
mean_mileage = {}
for brand in top_brands: 
    names = autos[autos['brand'] == brand]
    mileages = names['odometer_km'].mean()
    mean_mileage[brand] = int(mileages)

bmm_series = pd.Series(mean_mileage).sort_values(ascending  = False)


mileage_df = pd.DataFrame(bmm_series, columns = ['mean_mileage(km)'])
mileage_df



    

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
      <th>mean_mileage(km)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bmw</th>
      <td>132572</td>
    </tr>
    <tr>
      <th>mercedes_benz</th>
      <td>130788</td>
    </tr>
    <tr>
      <th>opel</th>
      <td>129310</td>
    </tr>
    <tr>
      <th>audi</th>
      <td>129157</td>
    </tr>
    <tr>
      <th>volkswagen</th>
      <td>128707</td>
    </tr>
    <tr>
      <th>renault</th>
      <td>128071</td>
    </tr>
    <tr>
      <th>ford</th>
      <td>124266</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Common dataframe 
mileage_df['mean_price($)'] = price_df

mileage_df
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
      <th>mean_mileage(km)</th>
      <th>mean_price($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bmw</th>
      <td>132572</td>
      <td>8332</td>
    </tr>
    <tr>
      <th>mercedes_benz</th>
      <td>130788</td>
      <td>8628</td>
    </tr>
    <tr>
      <th>opel</th>
      <td>129310</td>
      <td>2975</td>
    </tr>
    <tr>
      <th>audi</th>
      <td>129157</td>
      <td>9336</td>
    </tr>
    <tr>
      <th>volkswagen</th>
      <td>128707</td>
      <td>5402</td>
    </tr>
    <tr>
      <th>renault</th>
      <td>128071</td>
      <td>2474</td>
    </tr>
    <tr>
      <th>ford</th>
      <td>124266</td>
      <td>3749</td>
    </tr>
  </tbody>
</table>
</div>



The range of kilometers passed doesn't really vary by car brand as prices do. Although, there is a weak trend that more expensive cars have more mileage than less expensive cars.
