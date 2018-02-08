# [Taxi trip time prediction](https://www.kaggle.com/c/nyc-taxi-trip-duration)

This is a Kaggle competition.

## Introduction

For an intelligent transportation system, when we hope to go somewhere, it is possible to have different routes. But which one is fastest
would be very important to us. This competition requires us to predict the trip time given the starting point and ending point.

## Data observation
![data information](https://github.com/SeanCsc/Taxi-trip-time-prediction/blob/master/other/data_info.jpg)

This could bring me what the data look like and what the features it have. The location and time would the core for this competition.
## Data Preprocessing and Exploration
Possible informative features: time(month, week, day, hour), distance, weather, passenger numbers. So firstly, I would extract these features from the raw data.

<1> data clean-up. I noticed that the some records show that the duration time is between 1s to 980 hours. It may be a good trial to exclude some outliers. Also, for the location, I would exclude the points that are outside NY city. 

```python
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]
```

<2> Deal with the trip duration. Because the metrics is using log, so I tried log transformation for the trip duration. Then I plot the histogram.

```python
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
plt.hist(train['log_trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()
sns.distplot(train["log_trip_duration"], bins =100)
```
<3> We can also check the relationship between the trip duration and passenger numbers. But the results shows that there are not too much relationship between them.

## Feature Engineering
<1> Extract the distanc: harvesine distance -> manhantan distance -> direction
Thanks to [Beluga]https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367 , we can get the distance feature with the location.
```python
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
```
<2> Visualize the pick-up location as cluster (k-means)
An advantage to use k-means:we can find common things for the scattered location
<3> Date extraction
<4> Deal with categorical features : one-hot coding

```python
get_dummies
```
<5> Check
But there is an mismatch (which is because there are some numbers that haven't shown in the training)



## Model selection: XGBoost is my first trial. 
use grid-search to find the hyperparameters.







## update:
<1> Data enrichment: use other sources of data
<2> Ensemble
