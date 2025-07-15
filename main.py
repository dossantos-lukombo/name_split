from scripts.utils import *



if __name__ == "__main__":
    print(f"dataset:${open_data("./data/babynames-clean.csv").head()}")

    df_origin = open_data("./data/babynames-clean.csv")
    df_cleaned = clean_data(df_origin)
    # df_to_split = convert_dataset(df_cleaned)
    # print(df_to_split.head())
    data_splitted = split_dataset(df_cleaned)

    print("trainning...")
    model=train_dateset(data_splitted["X_train"],data_splitted["y_train"])
    # model=train_dateset(df_to_split[['name_value']],df_to_split[['gender']])
    print("trained !")
    test_model(model,data_splitted['X_test']).to_csv("./data/model_resultat.csv")
    # test_model(model,df_to_split).to_csv("./data/model_resultat.csv")