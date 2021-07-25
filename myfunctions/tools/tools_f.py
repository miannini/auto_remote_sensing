# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:14:42 2021

@author: Marcelo
"""
import pandas as pd
import numpy as np
import datetime

class Tools:
    def corr_num(x):
        if len(x.split('_'))==2 and len(x.split('_')[1])==1:   
            new_num = x.split('_')[0]+'_0'+x.split('_')[1]
        elif len(x.split('_'))==3 and len(x.split('_')[2])==1:   
            new_num = x.split('_')[0]+'_'+x.split('_')[1]+'_0'+x.split('_')[2] 
        elif 'T9' in x:
            new_num = 'T9'
        else:
            new_num = x
        return new_num
    
    def chem_correct_num(base,comp):
            if pd.isna(comp) == False and pd.isna(base) == True: #esta evaluando toda la coluna
                base2 = int(comp)
            else:
                base2 = base
            return base2

    def finca_cor(finca, potrero):
            if finca == 'EL RECODO': #finca
                try: 
                    number = float(potrero)
                    if number < 10:
                        return 'RC' + '0' + potrero
                    else:
                        return 'RC'  + potrero
                except:
                    if 'T' in potrero:
                        return potrero[-1] + potrero[0]
            elif finca == 'JUNCAL':
                return 'Juncalito-Lote_' + potrero
            elif finca == 'PARAISO':
                return 'Paraiso-Lote_' + potrero
            elif finca == 'LA ISLA':
                return'La_Isla-Lote_' + potrero           
            elif finca == 'MANGA LARGA':
                return 'La_Isla-Lote_ML_' + potrero
            elif finca == 'RANCHO': #posiblemente es T
                number = float(potrero)
                if number < 10:
                    return 'RANCHO' + '0' + potrero
                else:
                    return 'RANCHO' + potrero
    
    #hato nombre_c
    def hato_cor(hato, variable):
        if variable == 'lag_1': #finca
            return hato + '_1'
        elif variable == 'lag_2': #finca
            return hato + '_2'
        elif variable == 'lag_3': #finca
            return hato + '_3'
        else:
            return hato
                
    def lista_dates(df,column,offset):
        min_date = min(df[column]); max_date = max(df[column])
        #create list of dates with interval
        len_dates = int((max_date-min_date).days / offset)
        dates_list = []
        for n in range(0,len_dates+1):
            if n == 0:
                fecha = min_date
            else:
                fecha = fecha + datetime.timedelta(days=offset)
            dates_list.append(fecha)
        return dates_list
    
    def self_join(dates_list, data, list_cols): #'lote_id','name_c','band'
        base = data.loc[:,list_cols]
        base.drop_duplicates(inplace=True)
        base2=pd.DataFrame()    
        for m in dates_list:
            base_t = base.copy()
            base_t['date'] = m
            base2 = pd.concat([base2,base_t],ignore_index=True, axis=0)
        return base2
    
    def run_sum_reset(data, defa=0):
        v = pd.Series(data) #full_seguimiento2['dias_de_pastoreo'])
        n = v==0
        a = ~n
        c = a.cumsum()
        index = c[n].index  # need the index for reconstruction after the np.diff
        d = pd.Series(np.diff(np.hstack(([0.], c[n]))), index=index)
        v[n] = -d
        result = v.cumsum()
        if defa==0:
            result[result == 0] = np.nan
        return result
    
    #lenta la funcion //pensar cambio a np.where
    def actividades(lote_chg, actividad, animales, date, actividad_change):
        ultimo_pastoreo = 0
        dias_sin_pastoreo = 0
        dias_de_pastoreo = 0
        ultimo_animales = 0
        if lote_chg==True:
            if actividad =='PASTOREO': #primer pastoreo en primer registro
                ultimo_pastoreo = date
                dias_sin_pastoreo = (date - ultimo_pastoreo).days
                dias_de_pastoreo = 1
                ultimo_animales = animales 
        #mismo lote
        else:
            #hay pastoreo
            if actividad =='PASTOREO':
                ultimo_pastoreo = date
                ultimo_animales = animales
                dias_sin_pastoreo = (date - ultimo_pastoreo).days
                #dias seguidos pastoreo
                if actividad_change == False:   #primer pastoreo en set de fechas
                    dias_de_pastoreo = 1
                else:
                    dias_de_pastoreo = dias_de_pastoreo + 1 #pastoreo continuo
    
            else: 
                dias_sin_pastoreo = 1# (date - ultimo_pastoreo).days 3running difference
    
        return ([ultimo_pastoreo, dias_sin_pastoreo, dias_de_pastoreo, ultimo_animales])  #dias_sin_pastoreo,


    #funcion lenta
    def otras_actividades(lote_chg, actividad, actividades):
        ultima_fumigada = 0
        ultima_siembra = 0
        ultima_renovada = 0
        ultima_desbrosada = 0
        ultima_abonada = 0
        ultima_enmienda = 0
        ultima_rotobo = 0
        if lote_chg==True:
            if actividad =='FUMIGADA':
                ultima_fumigada = 0
            if actividad =='SIEMBRA':
                ultima_siembra = 0
            if actividad =='RENOVADA':
                ultima_renovada = 0
            if actividad =='DESBROSADA':
                ultima_desbrosada = 0
            if actividad =='ABONADA':
                ultima_abonada = 0
            if actividad =='ENMIENDA':
                ultima_enmienda = 0
            if actividad =='ROTOBO':
                ultima_rotobo = 0
        #mismo lote
        else:
            #hay FUMIGADA
            if actividad =='FUMIGADA':
                ultima_fumigada = 0
            else: 
                ultima_fumigada = 1
            #hay SIEMBRA
            if actividad =='SIEMBRA':
                ultima_siembra = 0
            else: 
                ultima_siembra = 1
            #hay SIEMBRA
            if actividad =='RENOVADA':
                ultima_renovada = 0
            else: 
                ultima_renovada = 1
            #hay SIEMBRA
            if actividad =='DESBROSADA':
                ultima_desbrosada = 0
            else: 
                ultima_desbrosada = 1
            #hay SIEMBRA
            if actividad =='ABONADA':
                ultima_abonada = 0
            else: 
                ultima_abonada = 1
            #hay SIEMBRA
            if actividad =='ENMIENDA':
                ultima_enmienda = 0
            else: 
                ultima_enmienda = 1
            #hay SIEMBRA
            if actividad =='ROTOBO':
                ultima_rotobo = 0
            else: 
                ultima_rotobo = 1
                
        return ([ultima_fumigada, ultima_siembra, ultima_renovada, ultima_desbrosada, ultima_abonada, ultima_enmienda, ultima_rotobo]) 

    def merge_dicts(a, b, path=None):
        "merges b into a"
        if path is None: path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    Tools.merge_dicts(a[key], b[key], path + [str(key)]) #o podria ser con self
                elif a[key] == b[key]:
                    pass # same leaf value
                else:
                    raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                a[key] = b[key]
        return a