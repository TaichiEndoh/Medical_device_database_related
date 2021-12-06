import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 
%matplotlib inline
import codecs
import os
import codecs
import datetime
from sklearn.model_selection import train_test_split

class my_directory_p:
    def __init__(self,pass_out):
        #self.day_im = day_im
        self.pass_out = pass_out


    def pass_o(self):
        return self.pass_out

    def pass_out_new(self):
        # フォルダ「output」が存在しない場合は作成する
        data_dir = self.pass_out
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(u.pass_o()+"フォルダ作成しました！！")
        
    def imp_data(self):
        #ubuntu 
        image_file_path = './data/Medical_Device_simulated_data.csv'
        import pandas as pd
        with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
                df_r9 = pd.read_table(file, delimiter=",")
        return df_r9

#ここでselfの値を定義する
#define the value of self here
u = my_directory_p("./20211128_output/")
u.pass_out_new()
df_r=u.imp_data()
u.pass_o()

df_r2=df_r.rename(columns={'作業日': "day"})
pd1=df_r2.set_index(["day","Unnamed: 0"])
#カウント用の全てが1の値を追加する
pd1['count_p'] = 1
go_t=pd1.sum(level=0)


image_file_path = './data/logbackup_simulated_data.csv'
image_file_path2 = './data/periodic_inspection_simulated_data.csv'

import pandas as pd
with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
                df_m_5_9 = pd.read_table(file, delimiter=",")
with codecs.open(image_file_path2, "r", "Shift-JIS", "ignore") as file:
                df_s_5_9 = pd.read_table(file, delimiter=",")

df_m_5_9d=df_m_5_9.rename(columns={'TIMESTAMP': 'day'})
send2_d=df_m_5_9d
day1_m=send2_d["day"]
str_date =send2_d["day"].astype(str)
#date time に変更したい
#文字列を削除してくっつけたい
dpc_y = send2_d["day"].str[:4]
dpc_m = send2_d["day"].str[5:7]
dpc_d = send2_d["day"].str[8:10]
dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d
dpc_date=pd.to_datetime(dpc_date_p)
print(dpc_date)
df_m_4_9=send2_d
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
log_4_9dv=day_str_replace(df)

#定期点検のデータ点検日　を　day とする
periodic_inspection_p=df_s_5_9.rename(columns={'点検日': 'day'})

#dayを曜日データで数字にする　ソースコード
s_date=go_t
df_date=s_date.reset_index()
df_date['day'] = pd.to_datetime(df_date['day'])
df_date['曜日']=df_date['day'].apply(lambda x:x.weekday())
#日付順に並び替える
sorted_df = df_date.sort_values(['day'])
sorted_df2p=sorted_df.reset_index()
sorted_df2=sorted_df2p.drop('index', axis=1)
#datetime　からstr変換
df_date=sorted_df2
df_date['day'] = pd.to_datetime(df_date['day'])
df_date['day'] = df_date['day'].astype(str)

sorted_df3_1=df_date
#dayの('-', '/')を入れ替えたいので以下の方法を利用した
df1 = sorted_df3_1

sorted_df3r=df1['day'].str.replace('-', '/')
sorted_df3r
df1_non_day=df1.drop('day', axis=1)
df1_non_day['day']=sorted_df3r
sorted_df3_d=df1_non_day
sorted_df3_d


def Sort_by_date(sorted_df2):

    df_date=sorted_df2
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['day'] = df_date['day'].astype(str)

    sorted_df3_1=df_date
    #dayの('-', '/')を入れ替えたいので以下の方法を利用した
    df1 = sorted_df3_1
    sorted_df3r=df1['day'].str.replace('-', '/')
    sorted_df3r
    df1_non_day=df1.drop('day', axis=1)
    df1_non_day['day']=sorted_df3r
    sorted_df3_d=df1_non_day
    return sorted_df3_d


log_data=Sort_by_date(log_4_9dv)
periodic_inspection=Sort_by_date(periodic_inspection_p)
print(log_data)
print(periodic_inspection)

day_i=sorted_df3_d
print(day_i)
send2_d=day_i
#print(send2_d)
day1_m=send2_d["day"]

#print(day1_m)
#date time に変更した！！！！
#重複の削除

str_date =send2_d["day"].astype(str)
#date time に変更したい
#文字列を削除してくっつけたい
dpc_y = send2_d["day"].str[:4]
dpc_m = send2_d["day"].str[5:7]
dpc_d = send2_d["day"].str[8:10]
#print(type(dpc_m))
#print(dpc_d)
dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d

#print(dpc_date)
dpc_date=pd.to_datetime(dpc_date_p)
print(dpc_date)


#ここは手動にする必要がある
#pandasの最初と最後を入力　すると日付を作ってくれる
day1h=dpc_date_p[0]
#print(day1h)
day1l=dpc_date_p[80]
#print(day1l)

from datetime import datetime
from datetime import timedelta

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

df_rk=go_t
#データから管理番号を取り出す
df_b = df_r
#print(df_b)
df_b_1 = df_b
df_b_1_p = df_b_1["管理番号"].values
print(df_b_1_p)

dpc_d_p = df_b_1_p.tolist()
#以下のソースコードで重なる内容を消す
dpc_d = list(dict.fromkeys(dpc_d_p))
#print(dpc_d)
dpc=[]
for i in dpc_d:
 #print(i)
 dpc.append(i)
np.nan_to_num(dpc)
for i in dpc:
 Department_name = i
 print(i)
df_0=df_b.fillna(0)

#ここからモジュールにしていく
def ward_name(data,colum1,colum2,count_t1,count_t2):
        df_d = data
        #カウントするための　count_pを作成されている
        df_d['count_p'] = 1
        df_1=df_d.reset_index()
        df_2 = df_1.set_index([colum1,colum2])
        print(df_2)
        #ここでDepartment_nameが設定されている↓
        hd_1_set = df_2.xs(Department_name, level=0)
        #ここでDepartment_nameが設定されている↑
        print("hd_1_set",hd_1_set)
        send_dt_dp = hd_1_set.groupby(count_t1).count()
        print("send_dt_dp",send_dt_dp)
        send_dt_d2 = hd_1_set.groupby([count_t1,count_t2]).count()
        print(send_dt_d2)
        

ward_name(df_0,"管理番号","作業日","count_p","発行　受付者")

def make_pd_column_value(df_b,column_v):
    df_b_1 = df_b
    df_b_1_p = df_b_1[column_v].values
    dpc_d_p = df_b_1_p.tolist()
    #以下のソースコードで重なる値をを消す Eliminate overlapping values in the following source code
    dpc_d = list(dict.fromkeys(dpc_d_p))
    #print(dpc_d)
    dpc=[]
    for i in dpc_d:
     #print(i)
     dpc.append(i)
    np.nan_to_num(dpc)
    for i in dpc:
     Department_name = i
     print(i)
    return dpc
scp=make_pd_column_value(df_b,"作業報告種類")
print("scp",scp)
day_t=make_pd_column_value(df_b,"作業日")
print("day_t",day_t)

#スコープのPandas作成
def make_pd_column_value(df_b,column_v):
    df_b_1 = df_b
    df_b_1_p = df_b_1[column_v].values
    dpc_d_p = df_b_1_p.tolist()
    #以下のソースコードで重なる値をを消す Eliminate overlapping values in the following source code
    dpc_d = list(dict.fromkeys(dpc_d_p))
    #print(dpc_d)
    dpc=[]
    for i in dpc_d:
     #print(i)
     dpc.append(i)
    np.nan_to_num(dpc)
    for i in dpc:
     Department_name = i
     print(i)
    return dpc
scp=make_pd_column_value(df_b,"作業報告種類")
print("scp",scp)
#scpを利用してグループ化してカウントした値を返す Use scp to group and return counted values

day_l=[]
scope_l=[]
coun_l=[]
no_l=[]

def ward_name(data,colum1,colum2,count_t2):
        df_d = data
        df_1=df_d.reset_index()
        #マルチインデックスを作成　Create a multi-index
        df_2 = df_1.set_index([colum1,colum2])
        #重なる値をを消したものをspcとした場合　要素を iに入力しマルチインデックスで検索 Enter the value of i after removing overlapping values and search with multi index.
        for i in scp:
            #print(i)
            scp_1 = df_2.xs(i, level=0)
            #以下でカウントして
            send_dt_d2 = scp_1.groupby([count_t2]).count()
            #pandas.DataFrameの任意の2列から辞書生成　以下の場合　もしpandasのカラムに'No.'があればそこの値をリストとして取り出す
            #以下のソースコードで
            print("send_dt_d2",send_dt_d2)
            for n, (dday, coun) in enumerate(zip(send_dt_d2.index,send_dt_d2['index'])):
                print("n",n)
                print("dday",dday)
                print("coun",coun)
                print("i",i)
                no_l.append(n)
                day_l.append(dday)
                coun_l.append(coun)
                scope_l.append(i)

ward_name(df_b,"作業報告種類","作業日","作業日")

#dataframeにするためのリストの値を設定　Set the value of the list to be a dataframe.
students = [ no_l ,day_l,coun_l,scope_l]
# Creating a dataframe object from listoftuples
dfObj = pd.DataFrame(students, index= ['index' , 'day', 'count',"Type"]).T

dfObj2=dfObj.drop(columns=["index"])
Type=dfObj.drop(columns=["index"])

scp=make_pd_column_value(df_b,"発行　受付者")
print("scp",scp)
day_l=[]
scope_l=[]
coun_l=[]
no_l=[]
scp=make_pd_column_value(df_b,"発行　受付者")
print("scp",scp)
ward_name(df_b,"発行　受付者","作業日","作業日")

#dataframeにするためのリストの値を設定　Set the value of the list to be a dataframe.
students = [ no_l ,day_l,coun_l,scope_l]
# Creating a dataframe object from listoftuples
dfObjs = pd.DataFrame(students, index= ['index' , 'day', 'sen_count',"sen_1"]).T

dfObjs2=dfObjs.drop(columns=["index"])
Worker=dfObjs.drop(columns=["index"])

#ユーザーの値
scp=make_pd_column_value(df_b,"機器基本情報::機器名称")
print("scp",scp)
day_l=[]
scope_l=[]
coun_l=[]
no_l=[]
ward_name(df_b,"機器基本情報::機器名称","作業日","作業日")

#dataframeにするためのリストの値を設定　Set the value of the list to be a dataframe.
students = [ no_l ,day_l,coun_l,scope_l]
# Creating a dataframe object from listoftuples
dfObjy = pd.DataFrame(students, index= ['index' , 'day', 'user_count',"Equipment_name"]).T
dfObjy2=dfObjy.drop(columns=["index"])
Equipment=dfObjy.drop(columns=["index"])

#ここが作業報告の内容
df_rk
Type
Worker
Equipment

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
print(df)

#log と　定期点検データの活用
sorted_df3_d

s_m_5_8_d=day_get_dummies(log_data)
s_s_5_8_d=day_get_dummies(periodic_inspection)
df_spd=pd.merge(s_m_5_8_d,s_s_5_8_d,on=["day"], how='outer')
df_s_spd=pd.merge(df,df_spd,on=["day"], how='outer')

df_s_spd2=df_s_spd.groupby('day').sum()
s_new2=pd.DataFrame(s_new,columns=["day"])
df_s_spd3=pd.merge(s_new2,df_s_spd2,on=["day"], how='outer')
send22=df_s_spd3
send2_allday=send22

#dayを曜日データで数字にする　ソースコード
def Sort_by_date(send2_allday):
    s_date=send2_allday
    df_date=s_date.reset_index()
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['曜日']=df_date['day'].apply(lambda x:x.weekday())
    #日付順に並び替える
    sorted_df = df_date.sort_values(['day'])
    sorted_df2p=sorted_df.reset_index()
    sorted_df2=sorted_df2p.drop('index', axis=1)
    sorted_df2
    #datetime　からstr変換
    df_date=sorted_df2
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['day'] = df_date['day'].astype(str)

    sorted_df3=df_date
    #dayの('-', '/')を入れ替えたいので以下の方法を利用した
    df1 = sorted_df3

    sorted_df3r=df1['day'].str.replace('-', '/')
    sorted_df3r
    df1_non_day=df1.drop('day', axis=1)
    df1_non_day['day']=sorted_df3r
    sorted_df3_d=df1_non_day
    sorted_df3_d
    return sorted_df3_d

sorted_df3_2nd=Sort_by_date(send2_allday)
sorted_df3_2nd

send2_ps_date=sorted_df3_2nd.fillna(0)
send2_ps_date.to_csv(r""+u.pass_o()+'pre_data.csv', encoding = 'shift-jis')
#dayのみ取り出しておく　series
day_only=send2_ps_date["day"]

#ワンホットエンコードのために　dayを消す
send2_p1=send2_ps_date.drop(["day"], axis=1)# 削除
send2_p=send2_p1.fillna(0)

#ワンホットエンコードの実施
df_one_hot_encoded = pd.get_dummies(send2_p)
df_t=df_one_hot_encoded
#ワンホットエンコードした後に　日付をくっつける
day_o=pd.concat([day_only, df_t], axis=1)
df_one_hot_encoded_p=day_o
df=df_one_hot_encoded_p
Fix_the_columns=df.loc[:,~df.columns.duplicated()]
df_one_hot_encoded=Fix_the_columns
df_one_hot_encoded

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
df1_4
df1_4_allday=df1_4
df_d=df1_4_allday
df_b1=df_d
df_b1['flg']=df_b1['target'].where(df_b1['target'] > 0, 0)

df_b2=df_b1
df_b2['flg']=df_b1['flg'].replace([1,2,3,4.0,5,6,7,8,9,10,27.0,12.0,16.0,24.0,180.0,160.0],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
test_c=df_b2.drop(['target'], axis=1)
df_one_hot_encoded=test_c.rename(columns={'flg':"target"})

df_one_hot_encoded.to_csv(r""+u.pass_o()+'df_one_hot_encoded.csv', encoding = 'shift-jis')
#データベースのデータ消えて使えなかったので諦め↑

image_file_path2 = u.pass_o()+'df_one_hot_encoded_adjustment.csv'
with codecs.open(image_file_path2, "r", "Shift-JIS", "ignore") as file:
                df_one_hot_encoded_adjustment = pd.read_table(file, delimiter=",")

#調整した値を入力
df_one_hot_encoded_adjustment

df_one_hot_encoded=df_one_hot_encoded_adjustment
#pandasの中のcolumns（target）をn_day分ずらす変数　これによってn_day先を予測することができるfor文で回すことで調節が可能
#variable that shifts columns (target) in pandas by n_days This allows us to predict n_days ahead, and can be adjusted by turning it with a for statement.
def future_prediction_day(dfdata,culum_u,n_day):

        def Create_Description_X(dtt):
            train_data = dtt
            X = train_data.drop(culum_u, axis=1)# 削除
            #説明変数作成
            #print("X.shape",X.shape)
            return X
        #説明変数Create_Description_X実施これでXに値が入るはず
        Xp = Create_Description_X(dfdata)
        #print("Create_Description_X 後の X.shape",X.shape)
        #目的変数入力用関数
        def Objective_variable_creationY(Ymoto):
            targek = Ymoto
            Y=targek.T
            return Y
        #目的変数作成
        targetk = dfdata[culum_u].values
        #目的変数作成関数利用
        Yp = Objective_variable_creationY(targetk)
        #pandsにもどす
        Y = pd.DataFrame({'target':Yp})
        #pandasに戻す
        X = pd.DataFrame(Xp)
        #目的変数の最初の行をn_day分削除
        #print(Y)
        Y_future = Y[n_day:]
        print(n_day,"日分削除した目的変数↓")
        print("Y_future",Y_future)
        Y_future2 =Y_future.reset_index()
        #説明変数の最後の行をn_day分削除
        X_future = X[:-n_day]
        X_future2 =X_future.reset_index()       
        Y_X_future2 =pd.concat([Y_future2, X_future2], axis=1)
        Y_X_future2
        Y_X_future2_o=Y_X_future2
        return Y_X_future2_o

test1 = future_prediction_day(df_one_hot_encoded,"target",1)
print(test1)
test2 = future_prediction_day(df_one_hot_encoded,"target",2)
print(test2)
test3 = future_prediction_day(df_one_hot_encoded,"target",3)
print(test3)
test4 = future_prediction_day(df_one_hot_encoded,"target",4)
print(test4)
test5 = future_prediction_day(df_one_hot_encoded,"target",5)
print(test5)
test6 = future_prediction_day(df_one_hot_encoded,"target",6)
print(test6)
test7 = future_prediction_day(df_one_hot_encoded,"target",7)
print(test7)
test30 = future_prediction_day(df_one_hot_encoded,"target",30)
print(test30)

#この後にAI解析実施予定
test1.drop("index", axis=1)
test2.drop("index", axis=1)
test3.drop("index", axis=1)
test4.drop("index", axis=1) 
test5.drop("index", axis=1) 
test6.drop("index", axis=1) 
test7.drop("index", axis=1) 
test30.drop("index", axis=1) 

tes=test7.drop("index", axis=1)# 削除

def make_Base_line(df,numb):
	import pandas as pd
	import numpy as np
	# Mean Absolute Error(MAE)用
	from sklearn.metrics import mean_absolute_error
	# Root Mean Squared Error(RMSE)
	from sklearn.metrics import mean_squared_error
	#以下のでカラムの値を取り出して名前を入れれる You can extract the value of the column and put the name in it as follows
	columns_name = df.columns
	#print(columns_name)
	Serial_number= [x for x in range(len(columns_name))]
	#print("Serial_number",Serial_number)
	#print(columns_name[numb])
	mean_1=df[columns_name[numb]].mean()
	#print("columns_name and mean",mean_1)
	lst = [df[columns_name[numb]].mean()] * len(df.index)     
	## label data
	label = df[str(columns_name[numb])]
	## AI predicted data
	pred = lst
	# MAE計算
	mae = mean_absolute_error(label, pred)
	#print('MAE : {:.3f}'.format(mae))
	# {:.3f}で小数点以下は3桁で表示
	# RMSE計算
	rmse = np.sqrt(mean_squared_error(label, pred))
	#print('RMSE : {:.3f}'.format(rmse))
	index1 = ["mean", 'MAE_mean', 'RMSE_mean']
	columns1 =[str(columns_name[numb])]
	Calculation=pd.DataFrame(data=[df[columns_name[numb]].mean(),format(mae),format(rmse)], index=index1, columns=columns1)
	#print(Calculation)
	return Calculation

Calculation=make_Base_line(tes,0)
Calculation

test1.drop("index", axis=1)
Calculation1=make_Base_line(test1,1)
print("test1",Calculation1)
test2.drop("index", axis=1)
Calculation2=make_Base_line(test2,1)
print("test2",Calculation2)
test3.drop("index", axis=1)
Calculation3=make_Base_line(test3,1)
print("test3",Calculation3)
test4.drop("index", axis=1) 
Calculation4=make_Base_line(test4,1)
print("test4",Calculation4)
test5.drop("index", axis=1) 
Calculation5=make_Base_line(test5,1)
print("test5",Calculation5)
test6.drop("index", axis=1) 
Calculation6=make_Base_line(test6,1)
print("test6",Calculation6)
test7.drop("index", axis=1) 
Calculation7=make_Base_line(test7,1)
print("test7",Calculation7)

tes=test30.drop("index", axis=1)

#ここの調整で30日後などを選べる
test7
tes1=test7.drop("index", axis=1)
tes1
test_c=tes1
test_c
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

    
# shape 確認
print("train shape", X_train.shape)
print("test shape", X_test.shape)
print("validation shape", X_val.shape)
X_test_df = pd.DataFrame(X_test)
# shape 確認
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)
print("y_validation shape", y_val.shape)
y_test_df = pd.DataFrame(y_test)
print("y_test describe",y_test_df.describe())
print("not_ y_test describe",(~y_test_df.duplicated()).sum())
#y_test_df.value_counts().plot(kind="bar")
print("y_test_df.duplicated().sum()",y_test_df.duplicated().sum())
#print(y_test_df[y_test_df.duplicated()])  
#テストデータをtest_df　とする
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
test_df


#テストデータをtest_df　とする
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
test_df
#pd.DataFrame　に戻して　縦に　train　val　結合していく
y_trainp = pd.DataFrame(y_train)
X_trainp = pd.DataFrame(X_train)
train=pd.concat([y_trainp, X_trainp], axis=1)
train

y_valp = pd.DataFrame(y_val)
X_valp = pd.DataFrame(X_val)
val=pd.concat([y_valp, X_valp], axis=1)
val
test_vol=pd.concat([train, val])
test_vol
#yの目的変数のカラムが0になってるので　target に変化 
order_of_things=test_vol.rename(columns={0:"target"})
order_of_things

test_df_n_d=test_df.drop("day", axis=1)

order_of_things_day=order_of_things["day"]
order_of_things_d=order_of_things
order_of_things=order_of_things_d.drop("day", axis=1)

#オーバーサンプリング marge_data_over_sampling
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
#ここでシャッフルさせる
SMOTE_df=SMOTE_dfp.sample(frac=1, random_state=0)
print(SMOTE_df.shape)

marge_data_over_sampling=pd.concat([SMOTE_df, test_df])
print("marge_data_over_sampling.shape")
print(marge_data_over_sampling.shape)
marge_data_over_sampling

test_classification=marge_data_over_sampling.drop("day", axis=1)
test_classification


from pycaret.classification import *
exp_name = setup(SMOTE_df, target = "target",test_data=test_df_n_d,silent=True,fold_strategy='timeseries',data_split_shuffle=False,session_id=42)
best = compare_models()






