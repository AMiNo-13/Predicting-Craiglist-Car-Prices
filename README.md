# Predicting Craiglist Car Prices

## Introduction

When we want to sell used cars, one of the biggest problems is deciding reasonable selling prices for the cars. An effective way to solve this problem is to use a machine-learning model that can predict car prices.

We used Python code and libraries to build a price-prediction model. The project is an implementation of the project Predicting the Prices of Used Cars.

The goal of this project is to explore the dataset and discuss some interesting observations through visualizations and train machine learning models to fit and predict the prices of the used cars using supervised learning.


## Libraries Used:

To assist us in our research, we utilized the following libraries:

- Pandas - data cleaning and analysis
- Matplotlib - data visualization
- Seaborn - data visualization
- Numpy - data cleaning and analysis
- Math - data analysis
- Random - data cleaning and analysis
- Pickle - data transfer and storing
- statsmodels.formula.api - to create a model from a formula and dataframe.  
- Plotly.express - data visualization
- Re - regular expression searches


## Project objectives

The main objectives of this project are as follows:
Data collection and cleaning
Analyzing the data
Identify relevant machine-learning algorithms for the project.
Build price-prediction models based on the chosen algorithms.
Validate the models.
Identify the most appropriate model.

## Dataset: "Used Cars Dataset"

We used Kaggle dataset for this project. The data comes from Craigslist in the USA and provides information on car sales. It contains more 500,000 vehicles and has 25 columns.

#### Data Columns:

- identry: ID
- url: listing URL
- region: craigslist region
- region_url: region URL
- price: entry price
- year: entry year
- manufacturer: manufacturer of vehicle
- model: model of vehicle
- condition: condition of vehicle
- cylinders: number of cylinders
- fuel: fuel type
- odometer: miles traveled by vehicle
- title_status: title status of vehicle
- transmission: transmission of vehicle
- vin: vehicle identification number
- drive: type of drive
- size: size of vehicle
- type: generic type of vehicle
- paint_color: color of vehicle
- image_url: image URL
- description: listed description of vehicle
- county: useless column left in by mistake
- state: state of listing
- lat: latitude of listing
- long: longitude of listing

## Data Cleaning and Feature Selection:

Dataset has so many missing values in almost all variable, except for a few. Many columns have nothing to do with the prices of the car, such as “id”, “url”, “image_url”, so we removed those columns. We wanted to fix missing values, because it is important to be handled as they could lead to wrong prediction or classification for any given model being used. We removed outliers and zero values of the price column. As locations do not have major correlation to the prices, we choose to remove ‘long’ and ‘lat’ as well. Since it is difficult to impute accurately, we dropped columns that didn’t have a year or odometer value because they are both heavily correlated to the other and are both missing. We also dropped model column in the end since the information was close to manufacture column, it had many missing values, and it was very difficult to find the value. We did not start dropping columns in the beginning because we needed them to determine the missing value for the other columns with missing values, so we imputed missing values and used np.select function.

Since we had many missing values we used np.select function using the keywords that we are looking for, then we imputed for the remaining missing values. For example, for manufacturer column we used np.select from the description column since we could find our missing value answers in the descriptions. We also took the same approach for the odometer column, but for odometer we choose values between 250 and 300,000 because they were many dirty values and missing information. We took the same approach for the year column and age to determine the average odometer for that car. For the paint column we imputed the color values from manufacturer column to replace are missing values. Again we used the same approach for vehicle cylinders, transmission, drive, fuel, size, type, and condition. For VIN column we changed its categorical value to 0 and 1 stating that if it has a VIN number or not.

For our feature engineering we used the description column to add more features. We made features such as cash sale or financed from terms found in the description column. This helped dial in our model and make up some ground because of not being able to use the model column since it was so dirty.

We found the polynomial regression models performed the best since they created a lot of interaction features. This helped unearth new relationships that helped the model perform better. Our best model had 3250 features and an R^2 of 0.78.

## Analysis

Based on our data, we came across many findings. We found that there was a statistical difference of mean selling price between the various 50 states. 
specifically, we found that used car prices were slightly higher in the states neighboured on Canada (such as Alaska, Idaho, Washington, Montana, North Dakota). Depending on the car condition, price was generally higher for cars that were new and like new conditions. In general, we found that there was statistical significance in including nearly all of the available catagorical features that were provided in the initial dataset. This resulted in our preprocessed data having over 144 columns, prior to any interaction generation.

#### Based on our model we identified three key features:
Odometer , Age, and year

Our EDA and visualizations can be viewed in detail in our notebook, but here are some of our visualization that we did for our model:

### Car Type Price Median Resale 

![Price_car_type](https://user-images.githubusercontent.com/62824675/93013192-f6851500-f55a-11ea-92e0-a673f413bc50.png)

The highest median resale price belongs to pickup trucks, followed by trucks. Coupes, vans, and SUVS have lower median resale prices than the trucks. Pickup trucks and trucks also have the largest spread of resale prices.

### Car Popularity:

![Manuf_Count](https://user-images.githubusercontent.com/62824675/93013496-e28ee280-f55d-11ea-906f-60abffff093c.png)

The most popular car manufactures were Ford, Chevrolet, Toyota, Honda, and Nissan


### Color Popularity 

![Unknown](https://user-images.githubusercontent.com/62824675/93013502-f76b7600-f55d-11ea-985b-4d54274b594b.png)

As we can see the most popular color is white followed by black and silver color. 

### Price and Odometer Relation

![Unknown](https://user-images.githubusercontent.com/62824675/93014259-7ebbe800-f564-11ea-9bdf-b35ca6b4765a.png)

As expected, higher odometer tends to have lower prices, while lower odometer tends to be more expensive.

### Correlation Between Age, Price, Odometer

![Correlations](https://user-images.githubusercontent.com/62824675/93015033-bc237400-f56a-11ea-98d2-4d418c86bf6a.png)

Based on the figure we can say highest correlation is between age and price, and the most negative is price and odometer

### Top Listing States

![States_listed_count ](https://user-images.githubusercontent.com/62824675/93015037-c6457280-f56a-11ea-95a1-2ad125cef373.png)

California, Florida, Texas, New York, and Oregon are top used car listing states.



## Conclusion

Based on our analysis, we can conclude that most new and like-new cars tend to be more expensive, while cars with fair and salvage conditions tend to be much cheaper. The price depends on the age, odometer, and the type of vehicle. The highest correlation is between odometer and price (negative relationship). Planning strategically when a car is bought and sold makes it possible to lose as little money as possible.

We found that using polynomial models of degree two on all available features, including OHEs, lead to the greatest results. We infer that this is due to the nature of categorical interactions. Adding context to the sale, such as color, manufacturer, condition, etc, provides our model with a greater understanding of the true underlying value of an online car listing.

## Future Steps

Some additional steps that we would like to explore are as follows:
- Clean up the "model" feature that was provided in the original dataset. We believe that if we had accurate manufacturer - model information that our regression estimates would be significantly improved. Unfortunately, the data as provided from kaggle is highly unreliable and requires extensive cleaning and imputing.

- Include popularity/review data on manufacturer by state, taken from either the cars.com or kellybluebook apis.

https://drive.google.com/file/d/1dnrDq6ahVozGRNKaQnhxBooNfnvStIUC/view?usp=sharing
