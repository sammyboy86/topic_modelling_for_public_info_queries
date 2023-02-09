# -*- coding: utf-8 -*-
from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd
import numpy as np
import time
#%% paths


pathBHR = '../archivos/csv/Bot_history_report.csv'
pathPrueba1 = '../archivos/csv/Prueba 1 ITAM.csv'
pathPalabrasLlave = '../archivos/csv/Copia de Palabras llave RFP.csv'
pathTraining = '../archivos/csv/Training_data.csv'
pathOutput = 'temp/out2.csv'
pathRes = 'temp/res2.csv'
pathDur = 'temp/dur2.csv'
#%% cositas de archivos
df = pd.read_csv('../archivos/csv/Prueba 1 ITAM.csv')
print(df)
serie = list(df['Descripción'])
f = open(pathOutput, "w", encoding='utf8');f.write('num,label,name,start_pos,end_pos,proba' + '\n');f.close() #reiniciamos nuestro archivo de output
f = open(pathOutput, "a", encoding='utf8')

# aplicamos el modelo que nos da una lista de entidades nombradas y escribimos al archivo
startTime = time.time() 
tagger = SequenceTagger.load("flair/ner-spanish-large") # load tagger
for index, sen in enumerate(serie):
    sentence = Sentence(sen)
    tagger.predict(sentence) # predict NER tags
    for entity in sentence.get_spans('ner'): # iterate over entities and print
        f.write(str(index) + ',' +  str(entity.get_label("ner").value) + ',' + str(entity.text) + ',' + str(entity.start_position) + ',' +  str(entity.end_position) + ',' +  str(entity.get_label("ner").score) + '\n')
f.close()

endTime = time.time()
duration = endTime - startTime # get the execution time

with open(pathDur, 'w') as the_file:
    the_file.write(str(duration) + " seconds")
    
serie1 = serie.copy()
#%% reemplazamos las palabras del modelo por #####
df2 = pd.read_csv(pathOutput)
for j in range(len(serie)):
    for i in list(df2.query('num=='+str(j))['name']):
        serie1[j] = serie1[j].replace(i, '### ')
#%% reemplazamos las palabras de regex por #####
pass
#%% construimos el csv
df3 = pd.DataFrame(list(zip(serie, serie1)), columns = ['original', 'censurado'])
df3.to_csv(pathRes, index=False, encoding='utf-8')
print(df3)