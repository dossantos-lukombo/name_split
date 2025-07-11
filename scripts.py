import pandas as pd
import numpy as np
import requests
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def embed_text_model(text:str,model='nomic-embed-text'):
    url = "http://localhost:11434/api/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": text
    }
    
    print(data)

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    embedding = response.json()["embedding"]
    return embedding

def split_dataset(df:pd.DataFrame)->dict:
    X_train,X_test,y_train,y_test = train_test_split(
        df["index"].to_frame(),
        df["gender"].to_frame(),
        test_size=0.4,
        shuffle=True
    )
    
    print(train_test_split(
        df["index"],
        df["gender"],
        test_size=0.4,
        shuffle=True
    ))
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def open_data(path_file:str):
    df = pd.read_csv(path_file)
    df.reset_index(inplace=True)
    return df

#Cleaning dataset
def etl_data(df:pd.DataFrame):
    
    df = df.dropna()
    df['gender'].replace('boy',1,inplace=True)
    df['gender'].replace('girl',0,inplace=True)
    
    df["gender"] = df["gender"].astype(int)
    df["index"] = df["index"].astype(int)
        
    return df


def train_dateset(X_train,Y_train):

    print(f"X_train type: {X_train.shape}")
    print(f"Y_train type: {Y_train.shape}")
    model = LogisticRegression()
    
   
    print("X_train : \n",X_train)
    return model.fit(X_train,Y_train)
    
    
def test_model(model,X_test,dataframe_origin):
    print("X_test: \n",X_test)
    
    # print(model.predict(X_test))
    predicted = model.predict(X_test)
    
    model_ans = []
    model_eval = []
    name = []
    name_retrieve= []
    
    for pred_value in predicted:
        # print(pred_value)
        if pred_value==1:
            model_ans.append("boy")
        else:
            model_ans.append("girl")

    for v in dataframe_origin["index"]:
        if v in X_test["index"]:
            name_retrieve.append(dataframe_origin.iloc[v]["name"])
            
    
    df_model_ans = pd.DataFrame({
        "val_name": X_test["index"],
        "name_retrieve": name_retrieve,
        "pred": model_ans
    })
    
    
    # for value in df_model_ans["val_name"]:
    #     if value in dataframe_origin["index"]:
    #         if dataframe_origin.iloc[value]["gender"] == "boy":
    #             model_eval.append(True)
    #             name.append(dataframe_origin.iloc[value]["name"])
                
    #         if dataframe_origin.iloc[value]["gender"] == "girl":
    #             model_eval.append(True)
    #             name.append(dataframe_origin.iloc[value]["name"])
    #         else:
    #             model_eval.append(False)
    #             name.append(dataframe_origin.iloc[value]["name"])
            
   
   
    # print("name size: ",len(name))
    # print("value_name: ",len(X_test["index"]))
    # print("model_gender: ",len(model_ans))
    # print("model_eval: ",len(model_eval))
    # df = pd.DataFrame({
    #     "name":name,
    #     "value_name":X_test["index"],
    #     "model_gender": model_ans,
    #     "model_eval":model_eval
    # })
    
    
    return df_model_ans

print(f"dataset:${open_data("data/babynames-clean.csv").head()}")

df_origin = open_data("data/babynames-clean.csv")
df_to_split=etl_data(df_origin)

data_splitted = split_dataset(df_to_split)


print("trainning...")
model=train_dateset(data_splitted["X_train"],data_splitted["y_train"])
print("trained !")
test_model(model,data_splitted['X_test'],df_origin).to_csv("data/model_resultat.csv",index=True)