# Stock prediction tool

## Input Data

The raw data only include "open", "high", "low" and "close" column.
The target of the homework is to tell me when should I buy or sell a stock.
So I pre-process the data by labeling "1" or "0" of each day to indicate if the stock price will be higher in the next day.

Now, I have 4 features and 1 label for each row data.

## Used model

I used the LSTM model to learn the stock behavior and classify the data into 2 class. And to make the LSTM can learn the trend of the stock price, I provide 10 days data for each input. So, actually, there are 10days*4feature for the LSTM to learn the relation.

## Predict stock

To preserve the trend during predicting the testing test, I will concat the input data into the original training data and predict the last 20 days label results.

After I get the possiblity of these 20 days, I re-scaler the result possibility to range 0 to 1. If the scalered result is higher than 0.5, it means the stock price will get higher in the next day.