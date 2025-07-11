import pandas as pd
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
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

def split_dataset(df:pd.DataFrame)->dict[str,list]:
    X_train,X_test,y_train,y_test = train_test_split(
        df["name"],
        df["gender"],
        test_size=0.4,
        shuffle=True
    )
    
    print(train_test_split(
        df["name"],
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
    return df

#Cleaning dataset
def etl_data(df:pd.DataFrame):
    
    df = df.dropna()
    df['gender'].replace('boy',1,inplace=True)
    df['gender'].replace('girl',0,inplace=True)
    
    df["gender"] = df["gender"].astype(int)
    names_vector:list = []
    count=0
    for name in df["name"]:
        names_vector.append(float(np.mean(embed_text_model(name))))
        count+=1
        print(count)
    df["name"] = names_vector
    print("df[name].head():\n",df["name"].head())
        
    return df


def train_dateset(X_train,Y_train):

    print(f"X_train type: {X_train.shape}")
    print(f"Y_train type: {Y_train.shape}")
    model = LogisticRegression()
    
    print("X_train : \n",X_train)
    return model.fit(X_train,Y_train)
    
    
def test_model(model,X_test):
    
    print(model.predict(X_test))

print(f"dataset:${open_data("data/babynames-clean.csv").head()}")


df_to_split=etl_data(open_data("data/babynames-clean.csv"))

data_splitted = split_dataset(df_to_split)

# print("data_splitted:\n",data_splitted)

print("trainning...")
model=train_dateset(data_splitted["X_train"],data_splitted["y_train"])
print("trained !")
test_model(model,data_splitted['X_test'])