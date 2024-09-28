# DATA422_2024Fall_Team_3
# README
The objective for this dataset is to create a predictive model of the 4th and 5th legs in EUROUSD movements. The data was acquired by hand, sourced from TradingView, and is organized across three main table. The first table is the legs data, which consists of the following variables: leg1, leg2, leg3, leg4 and leg5. The five legs make up what is called the Elliott wave theory and is used to predict trend. Each variable contains numerical data representing different phases of market movement, known as legs, where the goal is to predict leg4 and leg5 using prior legs and data from the other tables. The second table has four variables: slope1, slope2, slope3 and slope4. Each variable contains numerical slope data corresponding to each leg. Moreover, the slope indicates the steepness of price movements for each associated leg. The third table contains miscellaneous variables, such as price, EMA (exponential moving average), structure, wick, and body. Price reflects the current market value, which is vital for trend analysis. EMA helps detect momentum, structure identified common chart patterns, and wick measures the volatility of price within the hour. Furthermore, body represents the strength of price movement within the hour.  
Regarding the amount of data, the legs dataset contains 1100 observations and contains five variables, one for each leg of the Elliott wave theory. The slopes dataset includes 1100 observations and has four variables: four slopes for the first four legs. The miscellaneous dataset has 1100 observations and five variables, which are complementary market variables for each observation. The proposed data wrangling techniques will be discussed next. The tables can be joined on the common id field, ensuring that legs, slopes, and market indicators are aligned for each observation. The reasoning is that joining on id ensures that each set of legs is paired with its respective slopes and market data, which is paramount for creating a predictive model. If any missing values exist, we will exclude the incomplete rows from the dataset. For feature engineering, creating an interaction between legs and slopes will uncover meaningful relationships that will be beneficial. Additional features from the miscellaneous dataset will be incorporated to improve model accuracy. To close, the combined and cleaned data will be stored in a structured database, CSV if needed, to ensure compatibility with model training.
