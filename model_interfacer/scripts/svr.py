# -*- coding: utf-8 -*-

import pandas as pd
df_class_pt = pd.DataFrame([1 ,2 ,3 ,4 ,5])
                           # , columns=['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN'])
# index=[0, 0.33, 0.67, 1])
print(df_class_pt)


def save_df(df, filename):
    df.to_csv(filename, mode="w", sep=',')


def read_df(filename):
    # df = pd.read_csv(filename, sep=',', na_values=".",index_col=0, encoding = "ISO-8859-1")
    # df = pd.read_csv(filename, sep=',', na_values=".", encoding="ISO-8859-1")
    df = pd.read_csv(filename, sep=',', na_values=".", encoding="utf-8")
    return df


name = "이병현"
speech_kr = "설렁탕 먹었어"
speech_en = "I ate Seolleongtang"

dataset = [[name, speech_en, speech_kr],[name, speech_en, speech_kr]]
print(dataset)
data = pd.DataFrame(dataset, columns=["name", "speech_en", "speech_kr"])
# data = pd.DataFrame(dataset)
# df_class_pt = pd.DataFrame([[name], [speech_kr], [speech_en]] columns=['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN'])
# index=[0, 0.33, 0.67, 1])
print(data)
#
save_df(data, "data.csv")
print(read_df("data.csv"))