# Intraday-Stock-Trading-Predictions-DS
# DATA SCIENCE INTERNSHIP REPORT

## TOPIC:

## INTRADAY STOCK TRADING PREDICTION USING

## MACHINE LEARNING

## TEAM MEMBERS:

## ASAVARI NEMADE

## ROHIT JADHAV

## SATVIK ATMAKURI

## SHREENIDHI HIPPARAGI

## SMRUTI RANJAN BEHERA


## CONTENT

- 1. ABSTRACT....................................................................................................
- 2. INTRODUCTION...........................................................................................
- 3. METHODOLOGY..........................................................................................
- 4. DATASETS & FEATURES............................................................................
- 5. EXPLORATORY DATA ANALYSIS...........................................................
- 6. MODELING..................................................................................................
- 7. RESULTS......................................................................................................
- 8. ACTUAL DEPLOYED MODEL..................................................................
- 9. REFERENCES..............................................................................................


ABSTRACT

Accurate stock price prediction is a significant benefit to the stock investors. The
most important need of any stockholder is to know the fluctuations of stock prices
in the financial market. The future stock value of any company is determined by
stock market prediction. This is an important prediction as it facilitates their
decision in investing or de-invest in the stock market. A successful prediction of
the stock’s future price could result in a significant profit. Hence investors prefer
a precise stock price prediction. There are many different approaches that help in
forecasting stock prices. This report focuses on how the predictive power of
machine learning models can reap financial benefits for investors who trade based
on future price prediction. The project focuses on predicting the next-minute price
movement of MSFT and NIFTY 50 using Support Vector Machine (SVM) model
which is implemented in this project to predict the stock trends and its variants
(linear, poly, rbf & sigmoid) are compared to determine which model gives the
best results. The model is trained by the usage of actual historical data. It is
concluded that support vector machine with polynomial kernel performs the best
among all of our models.

Keywords **_–_** Stock Market Forecasting, Machine Learning, Support Vector
Machine (SVM)


INTRODUCTION

In the modern world, with great advancement of Computer Science and rapidly
growing silicon industry, stocks are one of the major assets people are counting
upon. Majority of people, with sound knowledge of the market, statistics and lot
of ‘gut’ feelings, are investing their hard-earned money into company shares. And
then we have people, who are termed as ‘Risk Takers’, who believe in the
understanding of commerce, current affairs, and mathematics. They are the major
players in the world of intra-day trading. One can categorize it to be one of the
riskiest investments in stocks market, quite equivalent to gambling. Therefore,
the ability to precisely predict the price movement of stocks is the key to
profitability in trading. Many investors spend time actively trading stocks in hope
of outperforming the market, colloquially referred to as a passive investment. In
light of the increasing availability of financial data, prediction of price movement
in the financial market with machine learning has become a topic of interests for
both investors and researchers alike. Insights about price movements from the
models could help investors make more educated decisions. In this project, we
aim to focus on making short term price movements prediction using the
timeseries data of stock price. Such predictions will then be used to generate
short-term trading strategies to capitalize on small price movements in highly
liquid stocks.
The most important need of any stockholder is to know the fluctuations of
stock prices in the financial market. This is an important prediction as it facilitates
their decision in investing or de-invest in the stock market. SVM (System Vector
Machine) is the methodology used in this report to predict stock trends. Here the
Model is trained by the usage of actual historical data. This SVM learning theory
gives us a one-step prediction.


METHODOLOGY

Problem Definition

Dataset Explanation

```
EDA (Data-
preprocessing,
Visualiation)
```
Model Training

Model Testing

Results and Deployment


DATASET EXPLANATION

Stock market data can be interesting to analyse and as a further incentive, strong
predictive models can have large financial payoff. We have considered two
datasets.

The first one contains Microsoft's (MSFT) stock data for the last 35 years (from
1986 to date) with daily interval.

The second one contains Nifty 50 stock data for 3 months (from 01/01/2021 to
31/03/2021) with an interval of 1-minute.

Original format of the dataset: CSV

A brief explanation of every column in the dataset is as follows:

Date - in format: YYMMDD

Time- in format: HH:MM

open = Open Price of stock = The price at which stock opened

close =Close price of stock = The price at which stock closed

High = The highest price the stock touched

Low = The lowest price the stock touched

Adj Close - adjusted close price adjusted for both dividends and splits.

Volume - the number of shares that changed hands during a given day

We believe insights from this data can be used to build useful price forecasting
algorithms to aid investment.


Nifty 50 one-minute stock dataset

```
Microsoft’s daily stock dataset
```

EXPLORATORY DATA ANALYSIS (EDA)

Exploratory Data Analysis refers to the process of performing action or
investigations on data to detect outliers, if detected, check for the actions to
remove them or check if the outliers are not by mistake but in our case the outliers
are reality. EDA also consists of statistics and graphical representation. In short
EDA refers to exploration (or explanation of the data) of the given data for the
future usage. It is a good practice that we need to understand our data and gather
as many insights out of the data.

MSFT.csv contains 7 features/columns namely Date, Open, High, Low, Close,
Adj Close and Volume. DataFrame.csv contains 7 features/ columns namely
Type, Date, Time, Open, High, Low and Close.

```
EDA on MSFT dataset
```
Detection of outliers: -

Outliers are extreme low or extreme high values in the data. In MSFT dataset we
had observed many outliers, but the removal of outliers will lead to loss of many
data from the dataset. The outliers in our dataset were indeed a reality so removal
of real data will hamper the output accuracy.


➢ Plots for detecting outliers

Outliers in Open column Outliers in High column

Outliers in Low column Outliers in High Column

These Box plots represent the shape, maximum, minimum values and outliers.
By observing the box plot, the outliers here are extreme high-end side of the
values. We calculated Q3 (third quartile) and Q1 (first quartile) and also inter
quartile range (middle 50% values i.e., Q3-Q1). The outliers were found out
by the equation Q3 + 1.5 (Inter quartile range).


➢ Line plots

Line plots are used to describe time series data.

```
Line chart (Year vs Open)
```
This chart shows the values of open column against the year. We can observe that
values of open were increasing from 2016.

The stock value kept on increasing as we can see the price of stock was near to
zero in 1986 and gradually kept on increasing. Year 1998, 1999, and 2000 were
better compared to previous year. Now to date the value of stock has reached 250
USD.

To understand more about data, we divided the data into many samples, one such
sample consisted data from end of 2017 to end of 2019.


```
Line Plot after sampling (2017 to 2019)
```
Market went down in the end of 2018 but started to recover in 2019 and kept on
going high till the end of 2019

➢ Correlation plot (heat map)

Correlation checks for the linear dependency of any two variables among
themselves. When correlation value is 1 we say they are positively correlated.

```
Heat Map
```

From the heat map we can observe that Open, High, Low, Close are positively
correlated to each other. The value of Open, Close, High and Low are also
positively correlated to the corresponding year.

```
EDA on Data Frame dataset
```
Detection of outliers: -

No outliers were found in the dataset.

➢ Plots for detecting outliers

No outlier is seen as all the data points obeys the rule of Q3+ 1.5(Inter quartile
range) and Q1-1.5(Inter quartile range).

➢ Line Plots

Line plots are used to describe time series data.


Line Plot (Date vs close)

2021 started good but started to go down in mid Jan and went all time low value
at the end of January. February took a good leap ahead and continued to leap but
end of month was not good as it fell down to its month’s lowest value. March
took the leap forward and fell in mid-March and took a leap ahead we can observe
month ending high.

Line plot (Date vs low on left side and Date vs high on right side)

Similar trends can be seen in both the line plots and the plot explanation remains
the same to that of plot (Date vs close)

To understand the dataset, we sampled the dataset into many samples (no
particular sample size was used – sample size was randomly chosen).


➢ Cat plot

Cat plots are relatively new addition to the library Seaborn. This plot tries to
simplify plots involving categorical variables and many numerical variables. Cat
plot can handle 8 different plots currently available in Seaborn (such as strip plot,
boxplot, bar plot, scatterplot etc.).

The sample size was of 1100 data points. The features used to plot were: -

1. X-axis = minutes
2. Y-axis =close
3. Column = Date
4. Hue = hour (coloured dots).

Cat plot

The colour codes used for hours are as follows:

1. Blue for 9 hours
2. Orange for 10 hours
3. Green for 11 hours
4. Red for 12 hours
5. Violet for 13 hours


6. Brown for 14 hours
7. Pink for 15 hours.

The plots convey the information about the close value on x minutes, y hours and
z day.

Date: - 2021 - 01 - 01

At 14 hours the Stock went high, looking at the minute axis the stock at 14 hours
to 15 hours (60 mins) in all minute’s stocks were high. Stock opened at 9.16 and
at 9 hour 59 minutes the price went high when compared to the stock price in 9
to 10 hours. Stocks were pretty good at starting of 15 hours but achieved all-time
low of that day on 15 hours 22 min and slightly recovered at 15 hours 32 minute
(end of the day)


MODELING

We used Support Vector Machine (SVM) for building the model.

Support Vector Machine:

Support Vector is a supervised machine learning algorithm which can be used
for both classification and regression challenges. Here we plot data as a point in
n-dimensional space, with the value of each feature being the value of a
particular coordinate. Next, we perform classification by finding the hyper-
plane that differentiated the two classes very well.

### OBJECTIVE:

The objective of the support vector machine algorithm is to find a hyperplane in
an N-dimensional space that distinctly classifies the data points.

HYPERPLANES AND SUPPORT VECTOR:

Hyperplanes are the boundaries which help in classifying the data points. Data
points which are on the either side of the hyperplane can be classified into
different classes. The dimension of the hyperplane depends on the number of
features.


Support vectors are the data points that are closer and are parallel to the
hyperplane. The maximization of the margin of the respective classifier depends
on the support vectors. These points help us in building our SVM model.

With the help of hyperplane and support vector, we can affect how well SVM
will handle outliers and the data points.

Hard Margin: Forms when the support vectors which are very close to the
hyperplane.

Soft Margin: Forms when the support vectors which are far from the
hyperplane.

### BASIC PARAMETERS:

### 1) C:

➢ Regularization parameter
➢ Float, default=1.
2) Kernel:
➢ Specifies the kernel type to be used in the algorithm.
➢ linear, poly, rbf, sigmoid, precomputed
3) Degree:
➢ Degree of the polynomial kernel function(poly)
➢ Int, default=


4) Gamma:
➢ if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var ())
as value of gamma.
➢ if ‘auto’, uses 1 / n_features.
➢ {‘scale’, ’auto’} or float, default=’scale’

### SVM KERNEL FUNCTIONS:

SVM algorithms use a set of mathematical functions that are defined as the
kernel. The function of kernel is to take data as input and transform it into the
required form.

TYPES:

1. Linear
    ➢ It is the most basic type of kernel, usually one dimensional in
       nature.
    ➢ Used when data is linearly separable.
    ➢ Function: F(x, xj) = sum( x.xj)
2. Polynomial
    ➢ It represents the similarity of vectors in training set of data in a
       feature space over polynomials of the original variables used in
       kernel.
    ➢ It is popular in image processing.
    ➢ Function: F (x, xj) = (x. xj+1) ^d
3. Radial Basis Function
    ➢ It is a general-purpose kernel
    ➢ Used when there is no prior knowledge about the data.
    ➢ It is one of the most preferred and used kernel functions in svm.
    ➢ Function: F (x, xj) = exp (-gamma * ||x - xj||^2)


4. Sigmoid
    ➢ This kernel function is similar to a two-layer perceptron model of
       the neural network, which works as an activation function for
       neurons.
    ➢ Function: F (x, xj) = tanh ( **αxay + c)**


RESULTS

### TRAINING THE MODEL AND CALCULATING THE ACCURACY:

```
1) Training Data for machine learning is a key input to algorithm that
comprehend from such data and memorize the information for future
prediction.
2) More than 70% of the data is used for training and the remaining data is
used for testing and finding the accuracy if the model built.
3) ‘train_test_split’ from sklearn is used for splitting the data.
MSFT
➢ Kernel used in the data set for building the model is ‘poly’ of degree 8
➢ Accuracy for Close:
```
```
➢ Plotting original and predicted data
```

➢ Accuracy for High:

➢ Plotting original and predicted data

➢ Accuracy for Low:

➢ Plotting original and predicted data


Data Frame

➢ Kernel used in the data set for building the model is ‘poly’ of degree 8
➢ Accuracy for Close:

➢ Plotting original and predicted data

```
➢ Accuracy for High:
```

➢ Plotting original and predicted data

➢ Accuracy for Low:

➢ Plotting original and predicted data


ACTUAL DEPLOYED MODEL

```
Homepage:
Intraday-stock.herokuapp.com
```

### MSFT TAB


### DATAFRAME TAB


REFERENCES

1) Tomar, Ayushi & Ghosh, Bikramaditya & Manjunath, Chinthakunta &
Addapalli, V & Krishna, (2020) – “Employing Deep Learning in Intraday
Stock Trading.”
https://www.researchgate.net/publication/347840605_Employing_Deep_
Learning_In_Intraday_Stock_Trading
2) https://towardsdatascience.com/stock-predictions-intraday-trading-
e27064884c57
3) https://towardsdatascience.com/support-vector-machine-introduction-to-
machine-learning-algorithms-934a444fca47
4) https://dataaspirant.com/svm-kernels/#t- 1608054630727
5) https://www.academia.edu/44538799/Stock_Closing_Price_Prediction_us
ing_Machine_Learning_SVM_Model
6) "Stock price prediction using support vector regression on daily and up to
the minute prices”
https://www.sciencedirect.com/science/article/pii/S2405918818300060


