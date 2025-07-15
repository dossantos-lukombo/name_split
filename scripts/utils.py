import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def open_data(path_file:str):
    df = pd.read_csv(path_file)
    df.reset_index(inplace=True)
    return df


#Cleaning dataset
def clean_data(df:pd.DataFrame):
    
    df = df.dropna()
    df['gender'].replace('boy',1,inplace=True)
    df['gender'].replace('girl',0,inplace=True)
    
    df["gender"] = df["gender"].astype(int)
        
    return df

# def convert_dataset(dataframe:pd.DataFrame):
#     df_to_trained = pd.DataFrame()
#     df_to_trained['name_value'] = dataframe['name'].apply(base26_to_dec)
#     df_to_trained['gender'] = dataframe['gender'].copy()
    
#     return df_to_trained
    

def split_dataset(df:pd.DataFrame)->dict:
    X_train,X_test,y_train,y_test = train_test_split(
        df["name"],
        df["gender"],
        test_size=0.3,
    )
    
    print(train_test_split(
        df["name"],
        df["gender"],
        test_size=0.3,
    ))
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
        
    
def train_dateset(X_train,Y_train):
    
    print("X_train : \n",X_train)
    print("y_train : \n",Y_train)
    
    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2,4))),
            ('clf',   LogisticRegression(max_iter=500, random_state=42))
        ]
    )
        
    return pipeline.fit(X_train, Y_train.values.ravel())
    
def test_model(model,X_test):

    print("X_test: \n",X_test)
    predicted = model.predict(X_test)
    
    model_ans = []
    
    print("prediction : \n",predicted)
    
    for pred_value in predicted:
        if pred_value==1:
            model_ans.append("boy")
        else:
            model_ans.append("girl")
            
    print("predicted : \n",predicted)
    # name_retrieved = X_test['name_value'].apply(dec_to_base26)
    
    df_model_ans = pd.DataFrame({
        "name": X_test,
        "pred": model_ans
    })
    
    
    return df_model_ans

