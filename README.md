# data-cleaning
Data cleaning require different steps, which depends on the type of data. The code has been prepared to clean daily or monthly timeseries. The all_max.csv file has daily tempertaure data.
The various steps followed are mentioned below:
Step1:Upload data
Step2:Remove duplicate values based on the date of observation
Step3:Determine number of null values (NaN)
Step4:Remove rows with missing values (i) if any row has a missing value (ii) if all rows have missing value (iii) if a specific number of missing values are found in a row (iv) if rows of a specific column has missing value (iv) substitute null values by data of previous year (suitable for monthly data) (v) imputation by linear interpolation (vi) drop rows if the number of consecutive null values goes above 5
Step5:Create box-plot of the clean data for each year to check the data variation
Additionally the missing values can be replaced by the most frequent value (mode), median or mean of the data
