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

#box plot of data for non leap years:
csv_year=csv3[csv3.year%4!=0]
n=len(pd.unique(csv_year['year']))
row2=0
csv_year2=pd.DataFrame({'T_mm':csv3.iloc[0:366,2]})
for row in range(n):
    csv_year4=csv3.iloc[row2:row2+366,2]
    row2=row2+366   
    csv_year4.index=(csv3.iloc[0:366,2].index)
    csv_year2.insert(row+1,'Tmm',csv_year4,True)
    print(row2)
csv_year2.tail(20)
plt.boxplot(csv_year2.iloc[1:n])

#box plot for leap years:
csv_year=csv3[csv3.year%4==0]
n=len(pd.unique(csv_year['year']))
row2=0
csv_year2=pd.DataFrame({'T_mm':csv3.iloc[0:366,2]})
for row in range(n):
    csv_year4=csv3.iloc[row2:row2+366,2]
    row2=row2+366   
    csv_year4.index=(csv3.iloc[0:366,2].index)
    csv_year2.insert(row+1,'Tmm',csv_year4,True)
    print(row2)
csv_year2.tail(20)
plt.boxplot(csv_year2.iloc[1:n])

#distribution of data

#replace by most frequent value

                             
                             
#replace by mean value

                             
#replace by assuming a normal distribution
