import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 
#%matplotlib inline
#↑If you use jupyter, remove # and use it
import codecs
import os
import codecs
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class my_directory_p:
    def __init__(self,pass_out):
        self.pass_out = pass_out
    def pass_o(self):
        return self.pass_out
    def pass_out_new(self):
        # Create the folder "output" if it does not exist.
        data_dir = self.pass_out
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(u.pass_o()+"I created a folder!")
    def imp_data(self):
        image_file_path = './data/simulated_data/Medical_Device_simulated_data.csv'
        import pandas as pd
        with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
                df_md = pd.read_table(file, delimiter=",")
        return df_md

#define the value of self here
u = my_directory_p("./20211128_output/")
u.pass_out_new()
df_r=u.imp_data()
u.pass_o()

df_rename=df_r.rename(columns={'作業日': "day"})
pd1=df_rename.set_index(["day","Unnamed: 0"])
#Add an all ones value for counting
pd1['count_p'] = 1
day_sum=pd1.sum(level=0)

#Call the database log data and simulation data
image_file_path = './data/simulated_data/logbackup_simulated_data.csv'
image_file_path2 = './data/simulated_data/periodic_inspection_simulated_data.csv'

import pandas as pd
with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
                logbackup = pd.read_table(file, delimiter=",")
with codecs.open(image_file_path2, "r", "Shift-JIS", "ignore") as file:
                periodic_inspection = pd.read_table(file, delimiter=",")

logbackup_d=logbackup.rename(columns={'TIMESTAMP': 'day'})
day1_m=logbackup_d["day"]
str_date =logbackup_d["day"].astype(str)
dpc_y = logbackup_d["day"].str[:4]
dpc_m = logbackup_d["day"].str[5:7]
dpc_d = logbackup_d["day"].str[8:10]
dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d
dpc_date=pd.to_datetime(dpc_date_p)
print(dpc_date)
df_m_4_9=logbackup_d
df_m_4_9d=df_m_4_9.drop('day', axis=1)
df_m_4_9d["day"]=dpc_date
print(df_m_4_9d)
#dayの('-', '/')を入れ替えたいので以下の方法を利用した
df=df_m_4_9d
df['day'] = df['day'].astype(str)
#day の　ー　を　/　に直すソースコード
def day_str_replace(df1):
    sorted_df3r=df1['day'].str.replace('-', '/')
    sorted_df3r
    df1_non_day=df1.drop('day', axis=1)
    df1_non_day['day']=sorted_df3r
    sorted_df3_d=df1_non_day
    return sorted_df3_d
day_str_re=day_str_replace(df)

#The data inspection date of the periodic inspection is set to day.
periodic_inspection_p=periodic_inspection.rename(columns={'点検日': 'day'})

#day as a number with day of the week data
s_date=day_sum
df_date=s_date.reset_index()
df_date['day'] = pd.to_datetime(df_date['day'])
df_date['曜日']=df_date['day'].apply(lambda x:x.weekday())
#Sort by date
sorted_df = df_date.sort_values(['day'])
sorted_df2p=sorted_df.reset_index()
sorted_df2=sorted_df2p.drop('index', axis=1)
df_date=sorted_df2
df_date['day'] = pd.to_datetime(df_date['day'])
df_date['day'] = df_date['day'].astype(str)
sorted_df3_1=df_date
df1 = sorted_df3_1
sorted_df3r=df1['day'].str.replace('-', '/')
df1_non_day=df1.drop('day', axis=1)
df1_non_day['day']=sorted_df3r
sorted_df3_d=df1_non_day
sorted_df3_d

def Sort_by_date(sorted_df2):
    df_date=sorted_df2
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['day'] = df_date['day'].astype(str)
    sorted_df3_1=df_date
    df1 = sorted_df3_1
    sorted_df3r=df1['day'].str.replace('-', '/')
    sorted_df3r
    df1_non_day=df1.drop('day', axis=1)
    df1_non_day['day']=sorted_df3r
    sorted_df3_d=df1_non_day
    return sorted_df3_d

log_data=Sort_by_date(day_str_re)
periodic_inspection=Sort_by_date(periodic_inspection_p)
day_i=sorted_df3_d
day1_m=day_i["day"]
str_date =day_i["day"].astype(str)
dpc_y = day_i["day"].str[:4]
dpc_m = day_i["day"].str[5:7]
dpc_d = day_i["day"].str[8:10]
dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d
dpc_date=pd.to_datetime(dpc_date_p)
day1h=dpc_date_p[0]
day1l=dpc_date_p[len(dpc_date)-1]
start = datetime.strptime(str(day1h), '%Y-%m-%d').date()
end   = datetime.strptime(str(day1l), '%Y-%m-%d').date()

def daterange(_start, _end):
    for n in range((_end - _start).days):
        yield _start + timedelta(n)
day_all=[]
for i in daterange(start, end):
    print (i)
    day_all.append(i)
s= pd.Series(day_all)
ss=s.astype(str)
s_new = ss.str.replace('-', '/')
df_rk=day_sum
df_b = df_r
df_b_1 = df_b
df_b_1_p = df_b_1["管理番号"].values
dpc_d_p = df_b_1_p.tolist()
dpc_d = list(dict.fromkeys(dpc_d_p))
dpc=[]
for i in dpc_d:
 dpc.append(i)
np.nan_to_num(dpc)
for i in dpc:
 Department_name = i
 print(i)
df_0=df_b.fillna(0)

def ward_name(data,colum1,colum2,count_t1,count_t2):
        df_d = data
        df_d['count_p'] = 1
        df_1=df_d.reset_index()
        df_2 = df_1.set_index([colum1,colum2])
        #ここでDepartment_nameが設定されている↓
        hd_1_set = df_2.xs(Department_name, level=0)
        #ここでDepartment_nameが設定されている↑
        send_dt_dp = hd_1_set.groupby(count_t1).count()
        send_dt_d2 = hd_1_set.groupby([count_t1,count_t2]).count()
ward_name(df_0,"管理番号","作業日","count_p","発行　受付者")

def make_pd_column_value(df_b,column_v):
    df_b_1 = df_b
    df_b_1_p = df_b_1[column_v].values
    dpc_d_p = df_b_1_p.tolist()
    dpc_d = list(dict.fromkeys(dpc_d_p))
    dpc=[]
    for i in dpc_d:
     dpc.append(i)
    np.nan_to_num(dpc)
    for i in dpc:
     Department_name = i
     print(i)
    return dpc
scp=make_pd_column_value(df_b,"作業報告種類")
day_t=make_pd_column_value(df_b,"作業日")

def make_pd_column_value(df_b,column_v):
    df_b_1 = df_b
    df_b_1_p = df_b_1[column_v].values
    dpc_d_p = df_b_1_p.tolist()
    dpc_d = list(dict.fromkeys(dpc_d_p))
    dpc=[]
    for i in dpc_d:
     dpc.append(i)
    np.nan_to_num(dpc)
    for i in dpc:
     Department_name = i
    return dpc
scp=make_pd_column_value(df_b,"作業報告種類")

day_l=[]
scope_l=[]
coun_l=[]
no_l=[]
def ward_name(data,colum1,colum2,count_t2):
        df_d = data
        df_1=df_d.reset_index()
        df_2 = df_1.set_index([colum1,colum2])        
        for i in scp:
            scp_1 = df_2.xs(i, level=0)
            send_dt_d2 = scp_1.groupby([count_t2]).count()
            for n, (dday, coun) in enumerate(zip(send_dt_d2.index,send_dt_d2['index'])):
                no_l.append(n)
                day_l.append(dday)
                coun_l.append(coun)
                scope_l.append(i)
ward_name(df_b,"作業報告種類","作業日","作業日")

students = [ no_l ,day_l,coun_l,scope_l]
dfObj = pd.DataFrame(students, index= ['index' , 'day', 'count',"Type"]).T
dfObj2=dfObj.drop(columns=["index"])
Type=dfObj.drop(columns=["index"])
scp=make_pd_column_value(df_b,"発行　受付者")
day_l=[]
scope_l=[]
coun_l=[]
no_l=[]
scp=make_pd_column_value(df_b,"発行　受付者")
ward_name(df_b,"発行　受付者","作業日","作業日")
students = [ no_l ,day_l,coun_l,scope_l]
dfObjs = pd.DataFrame(students, index= ['index' , 'day', 'sen_count',"sen_1"]).T
dfObjs2=dfObjs.drop(columns=["index"])
Worker=dfObjs.drop(columns=["index"])

scp=make_pd_column_value(df_b,"機器基本情報::機器名称")
day_l=[]
scope_l=[]
coun_l=[]
no_l=[]
ward_name(df_b,"機器基本情報::機器名称","作業日","作業日")
students = [ no_l ,day_l,coun_l,scope_l]
dfObjy = pd.DataFrame(students, index= ['index' , 'day', 'user_count',"Equipment_name"]).T
dfObjy2=dfObjy.drop(columns=["index"])
Equipment=dfObjy.drop(columns=["index"])
df_d_T=pd.merge(df_rk,Type,on=["day"], how='outer')
df_d_T_W=pd.merge(Worker,df_d_T,on=["day"], how='outer')
df_d_T_W_E=pd.merge(df_d_T_W,Equipment,on=["day"], how='outer')

def day_get_dummies(df_d_T_W_E):
    df_d_T_W_E_dey=df_d_T_W_E["day"]
    df_d_T_W_E_drop_dey=df_d_T_W_E.drop(columns=["day"])
    df=pd.get_dummies(df_d_T_W_E_drop_dey)
    df["day"]=df_d_T_W_E_dey
    df_d_T_W_E1=df.groupby('day').sum()
    return df_d_T_W_E1
    
df_d_T_W_E2=day_get_dummies(df_d_T_W_E)
df=df_d_T_W_E2.reset_index()

s_m_5_8_d=day_get_dummies(log_data)
s_s_5_8_d=day_get_dummies(periodic_inspection)
df_spd=pd.merge(s_m_5_8_d,s_s_5_8_d,on=["day"], how='outer')
df_s_spd=pd.merge(df,df_spd,on=["day"], how='outer')
df_s_spd2=df_s_spd.groupby('day').sum()
s_new2=pd.DataFrame(s_new,columns=["day"])
df_s_spd3=pd.merge(s_new2,df_s_spd2,on=["day"], how='outer')
allday=df_s_spd3

def Sort_by_date(allday):
    s_date=allday
    df_date=s_date.reset_index()
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['曜日']=df_date['day'].apply(lambda x:x.weekday())
    sorted_df = df_date.sort_values(['day'])
    sorted_df2p=sorted_df.reset_index()
    sorted_df2=sorted_df2p.drop('index', axis=1)
    df_date=sorted_df2
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['day'] = df_date['day'].astype(str)
    sorted_df3=df_date
    df1 = sorted_df3
    sorted_df3r=df1['day'].str.replace('-', '/')
    df1_non_day=df1.drop('day', axis=1)
    df1_non_day['day']=sorted_df3r
    sorted_df3_d=df1_non_day
    return sorted_df3_d

sorted_df3_2nd=Sort_by_date(send2_allday)
send2_ps_date=sorted_df3_2nd.fillna(0)
send2_ps_date.to_csv(r""+u.pass_o()+'pre_data.csv', encoding = 'shift-jis')
day_only=send2_ps_date["day"]
send2_p1=send2_ps_date.drop(["day"], axis=1)# 削除
send2_p=send2_p1.fillna(0)
df_one_hot_encoded = pd.get_dummies(send2_p)
df_t=df_one_hot_encoded
day_o=pd.concat([day_only, df_t], axis=1)
df_one_hot_encoded_p=day_o
df=df_one_hot_encoded_p
Fix_the_columns=df.loc[:,~df.columns.duplicated()]
df_one_hot_encoded=Fix_the_columns
apprix_df=df_one_hot_encoded
apprix_1 = apprix_df.iloc[:200000,:]
apprix_2 = apprix_df.iloc[200001:400000,:]
apprix_3 = apprix_df.iloc[400001:700000,:]
apprix_4 = apprix_df.iloc[700001:,:]
df=apprix_1
df1=df.groupby('day').sum()
df=apprix_2
df2=df.groupby('day').sum()
df1_2=pd.concat([df1, df2])
df=apprix_3
df3=df.groupby('day').sum()
df=apprix_4
df4=df.groupby('day').sum()
df3_4=pd.concat([df3, df4])
df1_4=pd.concat([df1_2, df3_4])
df_b1=df1_4
df_b1['flg']=df_b1['target'].where(df_b1['target'] > 0, 0)
df_b2=df_b1
df_b2['flg']=df_b1['flg'].replace([1,2,3,4.0,5,6,7,8,9,10,27.0,12.0,16.0,24.0,180.0,160.0],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
test_c=df_b2.drop(['target'], axis=1)
df_one_hot_encoded=test_c.rename(columns={'flg':"target"})
df_one_hot_encoded.to_csv(r""+u.pass_o()+'df_one_hot_encoded.csv', encoding = 'shift-jis')
image_file_path2 = u.pass_o()+'df_one_hot_encoded_adjustment.csv'
with codecs.open(image_file_path2, "r", "Shift-JIS", "ignore") as file:
                df_one_hot_encoded_adjustment = pd.read_table(file, delimiter=",")
df_one_hot_encoded=df_one_hot_encoded_adjustment
def future_prediction_day(dfdata,culum_u,n_day):
        def Create_Description_X(dtt):
            train_data = dtt
            X = train_data.drop(culum_u, axis=1)
            return X
        Xp = Create_Description_X(dfdata)
        def Objective_variable_creationY(Ymoto):
            targek = Ymoto
            Y=targek.T
            return Y
        targetk = dfdata[culum_u].values
        Yp = Objective_variable_creationY(targetk)
        Y = pd.DataFrame({'target':Yp})
        X = pd.DataFrame(Xp)
        Y_future = Y[n_day:]
        Y_future2 =Y_future.reset_index()
        X_future = X[:-n_day]
        X_future2 =X_future.reset_index()       
        Y_X_future2 =pd.concat([Y_future2, X_future2], axis=1)
        Y_X_future2
        Y_X_future2_o=Y_X_future2
        return Y_X_future2_o
test1 = future_prediction_day(df_one_hot_encoded,"target",1)
test2 = future_prediction_day(df_one_hot_encoded,"target",2)
test3 = future_prediction_day(df_one_hot_encoded,"target",3)
test4 = future_prediction_day(df_one_hot_encoded,"target",4)
test5 = future_prediction_day(df_one_hot_encoded,"target",5)
test6 = future_prediction_day(df_one_hot_encoded,"target",6)
test7 = future_prediction_day(df_one_hot_encoded,"target",7)
test30 = future_prediction_day(df_one_hot_encoded,"target",30)
test1.drop("index", axis=1)
test2.drop("index", axis=1)
test3.drop("index", axis=1)
test4.drop("index", axis=1) 
test5.drop("index", axis=1) 
test6.drop("index", axis=1) 
test7.drop("index", axis=1) 
test30.drop("index", axis=1) 
tes=test7.drop("index", axis=1)
def make_Base_line(df,numb):
	columns_name = df.columns
	Serial_number= [x for x in range(len(columns_name))]
	mean_1=df[columns_name[numb]].mean()
	lst = [df[columns_name[numb]].mean()] * len(df.index)     
	label = df[str(columns_name[numb])]
	pred = lst
	mae = mean_absolute_error(label, pred)
	rmse = np.sqrt(mean_squared_error(label, pred))
	index1 = ["mean", 'MAE_mean', 'RMSE_mean']
	columns1 =[str(columns_name[numb])]
	Calculation=pd.DataFrame(data=[df[columns_name[numb]].mean(),format(mae),format(rmse)], index=index1, columns=columns1)
	return Calculation

Calculation=make_Base_line(tes,0)
test1.drop("index", axis=1)
Calculation1=make_Base_line(test1,1)
test2.drop("index", axis=1)
Calculation2=make_Base_line(test2,1)
test3.drop("index", axis=1)
Calculation3=make_Base_line(test3,1)
test4.drop("index", axis=1) 
Calculation4=make_Base_line(test4,1)
test5.drop("index", axis=1) 
Calculation5=make_Base_line(test5,1)
test6.drop("index", axis=1) 
Calculation6=make_Base_line(test6,1)
test7.drop("index", axis=1) 
Calculation7=make_Base_line(test7,1)
tes=test30.drop("index", axis=1)
tes1=test7.drop("index", axis=1)
test_c=tes1
merge_data=test_c
# Isolate the objective variable
X = merge_data.drop("target",axis=1).values
y = merge_data["target"].values
columns_name = merge_data.drop("target",axis=1).columns
def Test_data_and_training_data_split(df,X,Y):
             N_train = int(len(df) * 0.8)
             N_test = len(df) - N_train
             X_train, X_test, y_train, y_test = \
                train_test_split(X, Y, test_size=N_test,shuffle=False,random_state=42)
             return X_train, X_test, y_train, y_test
# Execute a function that separates data for training and data for testing.
X_train, X_test, y_train, y_test = Test_data_and_training_data_split(merge_data,X,y)
X_train = pd.DataFrame(X_train, columns=columns_name)
X_test = pd.DataFrame(X_test, columns=columns_name)
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
y_trainp = pd.DataFrame(y_train)
X_trainp = pd.DataFrame(X_train)
train=pd.concat([y_trainp, X_trainp], axis=1)
merge_data_p=train.rename(columns={0:"target"})
X = merge_data_p.drop("target",axis=1).values
y = merge_data_p["target"].values
columns_name = merge_data_p.drop("target",axis=1).columns
def Test_data_and_training_data_split(df,X,Y):
             N_train = int(len(df) * 0.8)
             N_test = len(df) - N_train
             X_train, X_test, y_train, y_test = \
                train_test_split(X, Y, test_size=N_test,shuffle=False,random_state=42)
             return X_train, X_test, y_train, y_test
# Execute a function that separates the data for training from the data for validation.
X_train,X_val, y_train,y_val = Test_data_and_training_data_split(merge_data_p,X,y)
X_train = pd.DataFrame(X_train, columns=columns_name)
X_val = pd.DataFrame(X_val, columns=columns_name)
#training verification Combine test data vertically
y_trainp = pd.DataFrame(y_train)
X_trainp = pd.DataFrame(X_train)
train=pd.concat([y_trainp, X_trainp], axis=1)
y_valp = pd.DataFrame(y_val)
X_valp = pd.DataFrame(X_val)
val=pd.concat([y_valp, X_valp], axis=1)
train_vol=pd.concat([train, val])
order_of_things=train_vol.rename(columns={0:"target"})
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
marge_data_out=pd.concat([order_of_things, test_df])
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
y_trainp = pd.DataFrame(y_train)
X_trainp = pd.DataFrame(X_train)
train=pd.concat([y_trainp, X_trainp], axis=1)
y_valp = pd.DataFrame(y_val)
X_valp = pd.DataFrame(X_val)
val=pd.concat([y_valp, X_valp], axis=1)
test_vol=pd.concat([train, val])
order_of_things=test_vol.rename(columns={0:"target"})
test_df_n_d=test_df.drop("day", axis=1)
order_of_things_day=order_of_things["day"]
order_of_things_d=order_of_things
order_of_things=order_of_things_d.drop("day", axis=1)
X = order_of_things.drop("target",axis=1).values
Y = order_of_things["target"].values
columns_name = order_of_things.drop("target",axis=1).columns
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = 0.9, random_state=42)
X_res, Y_res = sm.fit_sample(X, Y)
y_valp = pd.DataFrame(Y_res)
X_valp = pd.DataFrame(X_res, columns=columns_name)
val=pd.concat([y_valp, X_valp], axis=1)
SMOTE_dfp=val.rename(columns={0:"target"})
SMOTE_df=SMOTE_dfp.sample(frac=1, random_state=0)
marge_data_over_sampling=pd.concat([SMOTE_df, test_df])
test_classification=marge_data_over_sampling.drop("day", axis=1)

from pycaret.classification import *
exp_name = setup(SMOTE_df, target = "target",test_data=test_df_n_d,silent=True,fold_strategy='timeseries',data_split_shuffle=False,session_id=42)
best = compare_models()
