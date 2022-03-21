## data collected from livechennai.com

##import libraries
import pandas as pd
import numpy as np
import datetime
import requests
import bs4
import lxml

df=pd.DataFrame(columns=[0,1,2])
df_list=list()
for yno in range(2018,2022):
   for mno in range(1,13):
     tab=requests.get('https://www.livechennai.com/get_goldrate_history.asp?monthno='+str(mno)+'&yearno='+str(yno))
     bsoup=bs4.BeautifulSoup(tab.text,'html.parser')
     monthly_table=bsoup.find('table',class_="table-price")
     data=monthly_table.find_all('td')
     ts_list=[x.getText() for x in data][3:]
     ts_list=[x.replace('\r\n\t,','') for x in ts_list]
     n_cols=3
     ts_array=np.array(ts_list).reshape(int(len(ts_list)/n_cols),n_cols)
     month_df=pd.DataFrame(ts_array)
     df=pd.concat([df,month_df])

#name the df columns
df.columns=['Date',	'Pure Gold (24 k)',	'Standard Gold (22 K)']

#convert to 'Date' columns to datetime object
df['Date']=pd.to_datetime(df['Date'])

#remove comma from columns 'Pure Gold (24 k)' and 'Standard Gold (22 K)' if anything exist
df['Pure Gold (24 k)'] = df['Pure Gold (24 k)'].str.replace(",","").astype(float)
df['Standard Gold (22 K)'] = df['Standard Gold (22 K)'].str.replace(",","").astype(float)

#datetime for file name
today=datetime.datetime.today()
month=today.strftime("%B")
day=today.strftime("%d")

#save dataframe to csv file called ''gold_rate.csv'
df.to_csv('gold_rate_'+str(day)+'_'+str(month)+'.csv',index=False)