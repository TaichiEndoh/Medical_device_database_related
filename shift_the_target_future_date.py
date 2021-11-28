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


#ここの調整で30日後などを選べる
test7
tes1=test7.drop("index", axis=1)
tes1
#test_c=df_r9.drop("Unnamed: 0", axis=1)
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

#オーバーサンプリング利用して解析
from pycaret.classification import *
exp_name = setup(SMOTE_df, target = "target",test_data=test_df_n_d,silent=True,fold_strategy='timeseries',data_split_shuffle=False,session_id=42)
best = compare_models()
