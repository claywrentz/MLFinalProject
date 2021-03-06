# Machine Learning Final Project - Ford Vehicle Price Estimator
I chose to create a publicly available machine learning dataset and deployed to Heroku. 
### Dataset
My original plan was to make a housing price database that would allow you to see the price of your house by inputing a number of characteristics (number of bedrooms, bathrooms, lot size, etc.). I found a pretty decent dataset, however after working with it I discovered that there are way too many attributes to be effective. In order to get an estimate a user would have to enter nearly 100 details about their home, many that would be difficult to measure. I decided to search for another dataset, one with a lot of rows and fewer attributes. The dataset I landed on was one describing Ford vehicles and the price the were sold for. It included characteristics such as mileage, model, year, fuel type, etc. 

### Model

I tested three different models on my data: linear regression, decision tree regression, and random forest regression. I used RMSE, R<sup>2</sup>, and adjusted  R<sup>2</sup> to evaluate my models. Adjusted R<sup>2</sup> is supposed to help account for the number of parameters used to get a prediction. In all of the evaluations the R<sup>2</sup> and adjusted  R<sup>2</sup> values are very similar, and I think this is because the number of parameters is pretty low. The random forest regression ended up performing the best with an R<sup>2</sup> of 0.93 and adjusted  R<sup>2</sup> of 0.93.

### Deployment

I compacted the code for my model into a python file along with the streamlit code to deploy locally. I deployed publicly using Heroku. I used a different strategy for deploying then I used previously. In the past I deployed with Heroku CLI, but with this project I connected my GitHub directly. 

### Takeaways and Outlook
Through this project I was able to experiment with new types of machine learning models. I haven't done a whole lot of regression models, I usually do classification. The random forest regressor was interesting to implement and I learned that its especially good for data with high dimensionality - although my dataset didn't have high dimensionality. This is because it only works with a subset of the data when fitting the model. I was also able to get more familiar with Streamlit. I implemented a lot more features than my last project with streamlit so I am now a lot more confident with it. It's very enjoyable to use. I also just got more familiar with the deployment process in general. I now feel confident in being able to deploy something I am working on, something I had never done before this class. Going forward I would like to make this particular project more broad. The dataset I used was specific to Ford vehicles and didn't even have all the models. I think this would be interesting to try to apply to a broader database of cars and create a more intricate user interface.


