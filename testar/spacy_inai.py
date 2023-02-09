import spacy
import es_dep_news_trf
import pandas as pd
import re
import time
import multiprocessing as mp

pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn'

nlp = spacy.load("es_dep_news_trf")
nlp = es_dep_news_trf.load()

df_entradas=pd.read_feather('cdas_o22_analisis_texto/samuel/data/solicitudes.feather') # cargar base
tamanio_df = df_entradas.shape[0]
columna_interes= "DESCRIPCIONSOLICITUD"
#columna_interes= "Descripción"
num_entradas_por_bloque = 10

num_corridas=0
if((tamanio_df%num_entradas_por_bloque!=0)):
    num_corridas=int(tamanio_df/num_entradas_por_bloque) + 1
else:
    num_corridas=int(tamanio_df/num_entradas_por_bloque)

lim_inferior=0
lim_sup=num_entradas_por_bloque

for i in range(num_corridas):
    df_entradas_100 = df_entradas.loc[lim_inferior:lim_sup, [columna_interes]]

    num_palabras = sum([len(texto.split()) for texto in df_entradas_100[columna_interes]])
    tamanio_archivo = df_entradas_100.memory_usage(deep=True).sum()

    #%% reemplazamos las palabras de regex por #####
    def testar_regex(texto):

        #correo electronico
        regex=re.sub("([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", '#####', texto)

        #clave de elector
        regex=re.sub("[A-Z]{6}[0-9]{6,}(H|M)[0-9]{3}", '#####', regex)

        #curp
        regex=re.sub("[A-Z]{4}[0-9]{6}[A-Z]{6}([A-Z][0-9]|0[0-9])", '#####', regex)

        #RFC
        regex=re.sub("[A-Z,Ñ,&]{3,4}([0-9]{2})(0[1-9]|1[0-2])(0[1-9]|1[0-9]|2[0-9]|3[0-1])[A-Z|\d]{3}", '#####', regex)

        #numero cartilla militar
        regex=re.sub("(A|B|C)([0-9]{7,8}|\-[0-9]{7,8})", '#####', regex)

        #numero tel
        regex=re.sub("[0-9]{8,}|[0-9]{3} [0-9]{7}|[0-9]{2,3}(\-|.)[0-9]{4}(\-|.)[0-9]{4}", '#####', regex)

        return regex

    startTime = time.time() 
    df_entradas_100[columna_interes] = df_entradas_100[columna_interes].apply(lambda x: testar_regex(x))

    pool = mp.Pool(mp.cpu_count())

    df_entradas_100['entidades'] = pool.map(nlp, df_entradas_100['DESCRIPCIONSOLICITUD'])

    pool.close()

    df_entradas_100['entidades'] = df_entradas_100['entidades'].apply(lambda X: [(entidad.text, entidad.pos_) for entidad in X])

    df_entradas_100['testado'] = df_entradas_100['entidades'].apply(lambda x: [pair[0] if pair[1]!="PROPN" else "####" for pair in x])
    endTime = time.time()

    df_entradas_100.to_csv('out/bloque' + str(i))
    print('Total entradas: ' + str(df_entradas_100.shape[0]) + '\nTotal palabras: ' + str(num_palabras) + '\nTotal memoria: ' + str(tamanio_archivo/100) + " KB" + '\nTiempo de corrida: ' + str(int((endTime - startTime)/60)) + ' min ' + str((endTime - startTime)%60) + ' seg')

    lim_inferior=lim_sup
    if((lim_sup+num_entradas_por_bloque)>tamanio_df):
        lim_sup=tamanio_df
    else:
        lim_sup=lim_sup+num_entradas_por_bloque
