# Machine Learning Final Project - Ford Vehicle Price Estimator
I chose to create a publicly available machine learning dataset and deployed to Heroku. 
### Dataset
My original plan was to make a housing price database that would allow you to see the price of your house by inputing a number of characteristics (number of bedrooms, bathrooms, lot size, etc.). I found a pretty decent dataset, however after working with it I discovered that there are way too many attributes to be effective. In order to get an estimate a user would have to enter nearly 100 details about their home, many that would be difficult to measure. I decided to search for another dataset, one with a lot of rows and fewer attributes. The dataset I landed on was one describing Ford vehicles and the price the were sold for. It included characteristics such as mileage, model, year, fuel type, etc. 

### Model

I tested three different models on my data: linear regression, decision tree regression, and random forest regression. I used RMSE, R<sup>2</sup>
