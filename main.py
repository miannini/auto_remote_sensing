# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 13:12:09 2021

@author: Marcelo
"""

#from google.cloud import storage
#from google.oauth2 import service_account
#import re
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import io
import os
import json
import shutil
import time
from pathlib import Path
import sqlalchemy as sa
from datetime import datetime, date #, timedelta
import geopandas as gpd
from myfunctions import Ext_shape
from myfunctions.tools import GCP_Functions
from myfunctions.tools import Tools
from myfunctions.temp_stats import Stats_charts
from myfunctions import Satellite_proc


def process_image(event, context):
######################################       DATABASE SQL    ##############################################

    engine = sa.create_engine("mysql+pymysql://" + os.environ['DB_USER']+ ":" + os.environ['PASSWORD'] + "@" + os.environ['IP']  + "/" + "lv_test")
    
    ################################ READ CLIENTES 
    ''' usar SQL en lugar de API '''
    sql = "select ID_cliente, sentinel_zone from lv_test.Finca"
    clientes = pd.read_sql(sql, engine)
    clientes = clientes[clientes['sentinel_zone'].notnull()]
    clientes = clientes[clientes['sentinel_zone'].str.contains("/")]
    clientes.drop_duplicates(inplace=True)
    
    sql_lote = "select * from lv_test.lotes"
    lotes = pd.read_sql(sql_lote, engine)
    ########################## SHAPEFILES IN CLOUD STORAGE
    
    
         
    ##############################################################################################
    for clien in clientes.ID_cliente:
        folder_cliente = 'ID_CLIENTE-'+str(clien)
        zona = clientes.sentinel_zone.str.split('/',expand=False).str.join('') #''' esto hacerlo por numeracion o tuple '''
        ###########################    BBOX     AOI    ###########################################
        pref = folder_cliente + '/BBOX/'
        bbox_list = GCP_Functions.list_all_blobs('shapefiles-storage', prefix=pref, delimiter='/') #
        bbox_file = [s for s in bbox_list if '.shp' in s][0]
        #aoi = gpd.read_file(GCP_Functions.open_blob('shapefiles-storage',bbox_file))
        ''' temporal download of shapefiles '''
        for blob in bbox_list:
            if len(blob.split('/')[-1])>0:
                GCP_Functions.download_blob('shapefiles-storage', blob, blob.split('/')[-1])
        #Area of Interest, based on shapefile
        aoi = gpd.read_file(bbox_file.split('/')[-1])  
        
        # el CRS y EPSG son terminos geo-espaciales, para definir referencia base de esfera a plano
        #aoi.crs = {'init':'epsg:32618', 'no_defs': True}
        aoi.set_crs('epsg:32618', inplace=True)  #nueva forma
        aoi_universal= aoi.copy()
        aoi_universal.to_crs(4326, inplace=True) #nueva forma
        
        footprint = None    # el footprint es para hallar imagenes satelitales que contengan esta area
        for i in aoi_universal['geometry']:                             #area
            footprint = i
        
        ''' removal of shapefiles ESTO AL FINAL'''
        my_dir = os.getcwd()
        for fname in os.listdir(my_dir):
            if fname.startswith("big_box"):
                #print('to_delete')
                os.remove(os.path.join(my_dir, fname))
        del(aoi_universal, bbox_file, bbox_list,pref)
        ##############################   LOTES y FINCAS AOI y Shapes    #######################################
        shape_folder= 'shapefiles/' 
        fincas = GCP_Functions.list_gcs_directories('shapefiles-storage', folder_cliente+'/'+"ID_FINCA")
        for f in fincas:
            destination = shape_folder #shape_folder + f
            ''' crear folder de shapefiles temporal '''
            Path(destination).mkdir(parents=True, exist_ok=True) #create folder
            #list and download files in corresponding folders
            objetos = list(GCP_Functions.list_all_blobs('shapefiles-storage',prefix=f,delimiter='/')) 
            for n in objetos:
                ''' sin descargar, solo read '''
                GCP_Functions.download_blob('shapefiles-storage', n, destination + n.split('/')[-1])
          
        ### shapes de cada lote
        todos_lotes, todos_lotes_loc = Ext_shape.merge_shapes2(shape_folder)    #folder_name,analysis_area 
        aoig_near = todos_lotes_loc
        #lote_aoi_loc = todos_lotes_loc
        print("[info] lotes totales incluyendo de archivo externo = {}".format(len(todos_lotes)))
        
        ''' remove shapefiles folder  '''
        shutil.rmtree(shape_folder, ignore_errors=True)
        
        #restart indexes
        aoig_near.reset_index(drop=True, inplace=True)
        #lote_aoi_loc.reset_index(drop=True, inplace=True)   
        todos_lotes.reset_index(drop=True, inplace=True)
        #export to geojson
        '''  si esto ya esta en storage, no hacerlo '''
        '''
        Path(shape_folder+analysis_area+'/multiples_json/').mkdir(parents=True, exist_ok=True)
        for i in range(0,len(todos_lotes)):
            temp = todos_lotes[todos_lotes.index == i] #filter geodataframe, keeping same format to export
            temp.to_file(shape_folder+analysis_area+'/multiples_json/'+todos_lotes.iloc[i,0]+'.geojson', driver ='GeoJSON')
        '''
            
        ''' los nombres corregidos solo para cliente 1 ... de hecho siempre deberian ser los mismos en SQL y JSON  '''    
        ### corregir nombres lotes y unir con ID
        aoig_near['name'] = aoig_near['name'].apply(lambda x: Tools.corr_num(x))
        aoig_near = aoig_near.merge(lotes, how='left', left_on=['name'], right_on=['NOMBRE_LOTE'])
        aoig_near.rename(columns={'ID_LOTE': 'lote_id'}, inplace=True)
        aoig_near.drop(columns=['NOMBRE_LOTE','DESCRIPCION'], inplace=True)
        
        ''' esto se trajo de PNG '''
        #aoig_near = pd.merge(aoig_near,lotes.iloc[:,[0,2]], left_on='name', right_on='NOMBRE_LOTE')
        #aoig_near['lote_id']=aoig_near['ID_LOTE']
        #aoig_near.drop(columns=['ID_LOTE','NOMBRE_LOTE'], inplace=True)
        #aoig_near.set_index('lote_id', inplace=True); aoig_near.sort_index(inplace=True)
        
        todos_lotes['name'] = todos_lotes['name'].apply(lambda x: Tools.corr_num(x))
        todos_lotes = todos_lotes.merge(lotes.iloc[:,1:], how='left', left_on=['name'], right_on=['NOMBRE_LOTE'])
        todos_lotes.drop(columns=['NOMBRE_LOTE'], inplace=True)
        
        ''' pensar si esto va aqui, o solo 1 vez para las fincas en SQL '''
        ### leer departamentos y municipios
        #basado en archivo externo con shapes de los municipios y dptos de Colombia
        '''
        mpos = gpd.read_file(shape_folder+'/Colombia/mpos/MGN_MPIO_POLITICO.shp')
        #estandarizar coordenadas a un mismo sistema de referencia
        mpos.crs = {'init':'epsg:4326', 'no_defs': True}
        mpos = gpd.overlay(mpos, todos_lotes[0:1] , how='intersection') #lote_aoi, aoi_universal
        #traer solo el municipio y departamento mas cercano
        municipio, departamento = mpos.loc[0,'MPIO_CNMBR'], mpos.loc[0,'DPTO_CNMBR']
        '''
        del(shape_folder,fincas,objetos, todos_lotes_loc)
        ################################################# CROP de Imagenes y calculo indices   ######################
        ### Recortar y procesar imagenes satelitales
        output_folder='Data/Output_Images/' + folder_cliente + '/'
        png_folder='Data/PNG_Images/' + folder_cliente + '/'
        json_folder = 'Data/JSON/'
        #contabilizar tiempo
        start = time.time() 
        #inicializar listas y dataframes
        bucket='satellite_storage'
        #list of folders in Raw_images / zone /
        fechas = list(sorted(GCP_Functions.list_gcs_directories(bucket='satellite_storage', prefix='Raw_images/'+zona+'/')))
        
        for dire in fechas[0:1]:
            print(dire)
            start_local = time.time() 
            #dire=fechas[0]
            print("[INFO] Date to Analyze = {}".format(dire.split('/')[-2]))
            onlyfiles = GCP_Functions.list_all_blobs(bucket, prefix=dire, delimiter='/') #  
            not_analyze = ['TCI','AOT','SCL','WVP']
            for ba in onlyfiles:
                if not any (x in ba for x in not_analyze): #Bands not in list of NOT ANALYZR
                    #print(ba)
                    #ba2=ba
                    skip = Satellite_proc.crop_sat(ba,aoi,zona[0],output_folder, bucket)
                if any (x in ba for x in not_analyze):   #Bands to omit, send to archive
                    print(ba)
                    ### COPY ORIGINAL FILE TO COLDLINE / ARCHIVED
                    GCP_Functions.copy_blobs_sate(bucket, ba, 'archived_sentinel_raw', "/".join(ba.split('/')[1:]))
                    ### DELETE ORIGINAL FILE IN MONITORING BUCKET
                    GCP_Functions.delete_blob(bucket, ba)
            if skip == True:
                print("[INFO] fecha {}, zona {} recortada ... skip".format(dire.split('/')[-2],zona[0]))
                valid='No'
                #write log to DB at end of process  
                elapsed_local = time.time()  - start_local
                print("Time spent by date is: ",elapsed_local) 
                processed_df = pd.DataFrame(data={'ID_cliente' : [clien], 'zone':[zona[0]], 'image_date' : [dire.split('/')[-2]] ,'processed_date' : [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], 'elapsed_time':[elapsed_local] , 'valid_data' : [valid] , 'Clear_of_Cloud_prc' : [0]})
                try:
                    processed_df.to_sql('monitoreo_imagenes_process', engine,  if_exists='append', index=False)
                except:
                    print('record already in DB')
                    pass
                continue
            
            ''' removal of bands local ... provisional'''
            #my_dir = os.getcwd()
            for fname in os.listdir(my_dir):
                if fname.startswith(dire.split('/')[-2]):
                    #print('to_delete')
                    os.remove(os.path.join(my_dir, fname))
            
            #calculate indexes, clouds and all others
            meta, cld_pxl_count = Satellite_proc.band_calc(dire.split('/')[-2], zona[0],output_folder,png_folder,bucket) #calculate NDVI, crop='grass'
            plt.close("all")
            
            ''' removal of bands local ... provisional'''
            #my_dir = os.getcwd()
            for fname in os.listdir(my_dir):
                if fname.startswith(dire.split('/')[-2]):
                    #print('to_delete')
                    os.remove(os.path.join(my_dir, fname))
                     
            #database
            #analysis_date=output_folder+zona[0]+'/'+ dire.split('/')[-2]+'/'
            if meta==0:
                valid='No'
                #elapsed_local = time.time()  - start_local
                #print("[INFO] Date, zone with no data = {} - {}".format(dire.split('/')[-2],zona[0]))
                #print("Time spent by date is: ",elapsed_local) 
                #next
            else:
                valid='Yes'
            
                size_flag, datag, short_ordenado, short_resume, dictio_all = Stats_charts.data_g(dire.split('/')[-2], aoig_near, todos_lotes, output_folder, zona[0], bucket) #//aoig_near
            
                if size_flag:
                    print(date)
            '''
            else:
                pd.DataFrame(big_proto.append(datag )) 
                resumen_bandas = pd.concat([resumen_bandas,short_resume])
                table_bandas = pd.concat([table_bandas,short_ordenado])
            '''
            #clouds database
            #if count_of_clouds.empty:
            #count_of_clouds = pd.DataFrame(data={'date' : [dire.split('/')[-2]], 'clear_pxl_count':[cld_pxl_count]})
            #else:
                #count_of_clouds = count_of_clouds.append(pd.DataFrame(data={'date' : [dire.split('/')[-2]], 'clear_pxl_count':[cld_pxl_count]}))
            print("[INFO] Date, zone Analyzed = {} - {}".format(dire.split('/')[-2],zona[0]))
            
            elapsed_local = time.time()  - start_local
            print("Time spent by date is: ",elapsed_local) 
        
        
            #write log to DB at end of process  
            processed_df = pd.DataFrame(data={'ID_cliente' : [clien], 'zone':[zona[0]], 'image_date' : [dire.split('/')[-2]] ,'processed_date' : [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], 'elapsed_time':[elapsed_local] , 'valid_data' : [valid] , 'Clear_of_Cloud_prc' : [cld_pxl_count]})
            try:
                processed_df.to_sql('monitoreo_imagenes_process', engine,  if_exists='append', index=False)
            except:
                print('record already in DB')
                pass
            
            ################################################ REPORTE a CV o SQL
            #exportar datos CSV o JSON o SQL
            #convert from pivot table to dataframe
            '''version resumida de 'short_ordenado' enviar a database usando SQL
            short_resume convertir a json, ordenado por Poly, indicador, date y subir a un json modificable en GCLOUD
            datag agregar a un archivo plano que crezca, por cada cliente / o mejor usar el data dentro de temp_stats ne JSON
            '''
            if meta!=0:
                start=time.time()
                flattened = pd.DataFrame(short_ordenado.to_records())
                #corregir titulos de las columnas, para qutar parentesis
                flattened.columns = [hdr.replace("('", "").replace("')", "").replace("', '", ".") for hdr in flattened.columns]    
                #._cldmsk cambiado a ._ind por nueva forma de detectar nubes
                flattened['cld_percentage']=flattened["sum_value.ind"]/flattened["count_pxl.ind"]
                clouds_cover_all = np.sum(flattened.cld_percentage)/len(flattened)
                
                ''' aqui incluid processed_df, agregar cobertura de nubes sobre lotes y enviar ahi si a SQL '''
                #check if all lotes are covered by clouds
                if clouds_cover_all !=1.0:
                    flattened['area_factor']= (flattened["count_pxl.bm"]/flattened["count_pxl.ind"])*((100*flattened["count_pxl.bm"])/flattened["area"]) #mayor a 1 se debe reducir, menor a 1 se debe sumar
                    #Biomass corrected value
                    flattened['biomass_corrected'] = flattened["mean_value.bm"]*(flattened["area"]/(100*100))
                    
                    #organizar por lote_id y reemplazar poly
                    flattened = pd.merge(flattened, aoig_near.loc[:,['name','lote_id']], left_on='name', right_on='name')# (type_df, time_df, left_index=True, right_on='Project')
                    ''' revisar esto, para unir con sql lotes '''
                    #flattened['poly'] = flattened['DESCRIPCION'] #lote_id
                    #flattened.drop(columns=['DESCRIPCION'], inplace=True)
                    #flattened.rename(columns={'poly':'lote_id'}, inplace=True)
                    #flattened.sort_values(by=['lote_id', 'date'], inplace=True) #ahora llamar lote_id
                    
                    
                    #define columns, rename as per API and format dates 
                    reduced_columns=['lote_id','date','mean_value.bm','mean_value.cp','mean_value.ndf','mean_value.lai','mean_value.ndvi','cld_percentage','area_factor','biomass_corrected']
                    reduced_flat = flattened.loc[:,flattened.columns.isin(reduced_columns)]
                    reduced_flat.rename(columns={'lote_id':'ID_lote','date':'fecha','mean_value.bm':'Mean_BM','mean_value.cp':'Mean_CP','mean_value.ndf':'Mean_NDF','mean_value.lai':'Mean_LAI','mean_value.ndvi':'Mean_NDVI'}, inplace=True)
                    reduced_flat['fecha'] = pd.to_datetime(reduced_flat['fecha'], format="%Y%m%d").dt.strftime('%Y-%m-%d')
                    #reorder columns as per API
                    cols = reduced_flat.columns.tolist()
                    cols = cols[-1:] + cols[0:1] + cols[2:4] + cols[5:6] + cols[4:5] + cols[6:-1]
                    reduced_flat = reduced_flat[cols] 
                    ''' si se corrige arriba, no se necesita esto  '''
                    #reduced_flat = pd.merge(reduced_flat, lotes.iloc[:,[0,2]], left_on='ID_lote', right_on='NOMBRE_LOTE')
                    #reduced_flat['ID_lote']=reduced_flat['ID_LOTE']
                    #reduced_flat.drop(columns=['ID_LOTE','NOMBRE_LOTE'], inplace=True)
                    #SEND TO SQL
                    #merge lotes name with ID and send to SQL
                    try:
                        reduced_flat.to_sql('Lotes_variables', engine,  if_exists='append', index=False)       
                    except:
                        print('Error on uploading to SQL')
                    #upload to SQL database calling API 
                    '''
                    aqui voy
                    en archivo ready, cambiar poly a lote_id [usar python, no es tan facil como cambiar titulo]
                    crear json desde aqui, si es necesario
                    cliente, finca, lote_id, [nombre_lote, banda], fecha, [min, mean, max, std, kurt, stats, percs...]
                    '''
                    
                    #organizar por lote_id y reemplazar poly
                    short_resume.reset_index(inplace=True)
                    short_resume = pd.merge(short_resume, aoig_near.loc[:,['name','lote_id']], left_on='name', right_on='name')
                    
                    #short_resume['poly'] = short_resume['name']
                    short_resume.drop(columns=['poly','DESCRIPCION'], inplace=True)
                    #short_resume.rename(columns={'poly':'lote_id'}, inplace=True)
                    short_resume.sort_values(by=['lote_id', 'date', 'band'], inplace=True)
                    short_resume['client']  = clien
                    j = (short_resume.groupby(['client','ID_FINCA','lote_id','band','date'], as_index=False)
                                 .apply(lambda x: x[['mean_value','sum_value','min_value','max_value','std_dev','count_pxl','perc_10',
                                                     'perc_20','perc_30','perc_40','perc_50','perc_60','perc_70','perc_80',
                                                     'perc_90']].to_dict('r'))
                                 .reset_index()
                                 .rename(columns={None:'tide-Data'})
                                 .to_json(orient='records'))
                    '''
                    j = (short_resume.groupby(['client','ID_FINCA','lote_id','band','date'], as_index=False)
                                 .apply(lambda x: x[['mean_value','sum_value','min_value','max_value','std_dev','count_pxl','perc_10',
                                                     'perc_20','perc_30','perc_40','perc_50','perc_60','perc_70','perc_80',
                                                     'perc_90']].to_dict('r'))
                                 .reset_index()
                                 .rename(columns={None:'Tide-Data'})
                                 .to_json(orient='records'))
                    '''
                    
                    print(json.dumps(json.loads(j), indent=2, sort_keys=True))
                    ''' est JSON debe ser incremental y mejor organizado  '''
                    GCP_Functions.upload_string(bucket, json.dumps(json.loads(j), indent=2, sort_keys=True), 'application/json', json_folder+dire.split('/')[-2]+'_resumed_data.json')
                    '''upload dictio all'''
                    #dictio_all
                    #esto enviarlo directo a storage como JSON  
                    #flattened.to_csv (r'../Data/Database/'+analysis_area+'/resumen_lotes_medidas'+Date_Fin+'.csv', index = False, header=True)
                    #short_resume.to_csv (r'../Data/Database/'+analysis_area+'/resumen_vertical_lotes_medidas'+Date_Fin+'.csv', index = True, header=True)
                    
                    #del(flattened, short_resume, short_ordenado, datag, reduced_flat, processed_df, meta, onlyfiles) #j
                    print("[INFO] data table exported as CSV")
                    end = time.time()
                    print("time to process dataframes and json",end - start)
                    
                    
                    ############################################ IMAGENES PNG   #########################################################
                    
                    #obtener fechas basado en nombres, organizar y extraer inicial, media y final
                    start = time.time() 
                    arr = GCP_Functions.list_all_blobs(bucket,output_folder+zona[0]+'/'+ dire.split('/')[-2]+'/',delimiter='/')
                    indexes = ["ndvi","atsavi","lai","bm","cp","ndf"]
                    aoig_reduced = pd.merge(aoig_near, flattened.loc[:,['name','cld_percentage']], left_on='name', right_on='name')
                    aoig_reduced = aoig_reduced.loc[aoig_reduced['cld_percentage'] <= 0.5]
                    aoig_reduced.reset_index(drop=True, inplace=True)
                    '''  esto esta demorando mucho tiempo  '''
                    if len(aoig_reduced)>=1:
                        for inde in arr:
                            if any (x in inde for x in indexes):
                                print(inde)
                                Satellite_proc.small_area_crop_plot(dire.split('/')[-2],aoig_reduced,zona[0], inde, output_folder, png_folder, bucket)    
                    else:
                        print('All areas covered by clouds')
                    end = time.time()
                    print("Time to process lotes PNG images", end - start)
                    del(arr, indexes)
    return print('exito')