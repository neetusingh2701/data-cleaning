import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime
csv2=pd.read_csv("all_max.csv")
csv2.head()
print(csv2.shape)
print(csv2.dtypes)
#remove duplicate values
mod_csv=csv2.drop_duplicates(subset=['date'])
#number of null values
for col in csv2.columns:
    pct_missing = np.mean(csv2[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
#remove rows with missing values
#remove rows with missing values
mod_csv2=csv2.dropna()
mod_csv2.head()
#remove rows with all values missing
mod_csv3=csv2.dropna(how='all')
mod_csv3.head()
#remove rows which contain all NaN values in the selected columns
mod_csv4=csv2.dropna(how='all',subset=['T_max'])
mod_csv4.head()
#remove rows which contain a specific number of non NaN values
mod_csv5=csv2.dropna(thresh=2)
mod_csv5.head()
#remove rows which contain a specific number of non NaN values in the selected columns
mod_csv6=csv2.dropna(thresh=1,subset=['T_max'])
mod_csv6.head()
#no. of consecutive missing values
csv2['consec_NaN']=csv2.T_max.isnull().astype(int).groupby(csv2.T_max.notnull().astype(int).cumsum()).cumsum()
csv2.head(10)
#date coversion
csv2['new_date'] =  pd.to_datetime(csv2['date'],
                              format='%d-%m-%Y')
csv2['month'] = csv2['new_date'].dt.month
csv2['year']=csv2['new_date'].dt.year
# pick missing values from previous year data (suitable for monthly data showing seasonal variation)
csv2['Tmax_new']=csv2['T_max']
#interpolate missing values (suitable for daily data)
csv3=csv2.interpolate()
csv2['Tmax_new2']=np.where(csv2['year']<1981,csv2['Tmax_new'].interpolate(method="linear"),csv2['Tmax_new']) 
for row in csv2.index:
    if(csv2.iloc[row,3]>0 and csv2.iloc[row,3]<5):
        csv2.iloc[row,8]=csv2.iloc[row-12,8]
print(csv2.iloc[360:400,1:9])
# drop rows with number of consecutive missing values >5
csv7=csv2[csv2.consec_NaN<6]
csv7['consec_NaN2']=csv7['consec_NaN']
for row2 in range(len(csv7)):
    if(csv7.iloc[row2,3]>4):
        csv7.iloc[row2-4:row2,9]=5
csv8=csv7[csv7.consec_NaN2<5]
csv8.tail(20)

# conversion of single time series to multiple time series based on the year
csv_annual=csv3.groupby('year')
m=pd.unique(csv3['year'])
n=len(m)
csv_year2=pd.DataFrame({'Sno':range(366)})
for row in range(n):
    year = m[row]
    csv_year44=csv_annual.get_group(year) 
    csv_year4=csv_year44['Tmax_new']
    #csv_year4.head()
    if(year%4!=0):
        csv_year4.index=(csv3.iloc[0:365,2].index)
    else:
        csv_year4.index=(csv3.iloc[0:366,2].index)
    csv_year2.insert(row,m[row],csv_year4,True)
csv_year2.tail()

#box plot of data 
csv_year2.iloc[:,0:21].boxplot(rot=45,figsize=(9,9))
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()

#line plot
plt.plot(csv_year2.iloc[:,0:21])
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()

#distribution of data
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(csv_year2.iloc[:,4])
plt.show()

#Finding outliers using the quantiles
sample=pd.DataFrame({'Tmax':csv_year2.iloc[0:366,1]})
#sample2['Tmax']=csv_year2.iloc[:,1]
#sample['Sno']=csv_year2.iloc[:,21]
per25 = sample['Tmax'].quantile(0.25)
per75 = sample['Tmax'].quantile(0.75)
#print(per25)
#Finding upper and lower limit
iqr=per75-per25
UL = per75 + 1.5 * iqr
LL = per25 - 1.5 * iqr
#Removing Outliers
df1=sample[sample.Tmax<UL]
df2=sample[sample.iloc[:,0]>LL]
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(csv_year2.iloc[0:366,1])
plt.subplot(2,2,2)
sns.boxplot(csv_year2.iloc[0:366,1])
plt.subplot(2,2,3)
sns.distplot(df1['Tmax'])
plt.subplot(2,2,4)
sns.boxplot(df1['Tmax'])
plt.show()
new_df_cap = sample.copy()
new_df_cap['Tmax'] = np.where(
    new_df_cap['Tmax'] > UL,
    UL,
    np.where(
        new_df_cap['Tmax'] < LL,
        LL,
        new_df_cap['Tmax']
    )
)
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(sample['Tmax'])
plt.subplot(2,2,2)
sns.boxplot(sample['Tmax'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['Tmax'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['Tmax'])
plt.show()
                           
#replace by most frequent value


#replace by mean value

                             
#replace by assuming a normal distribution
