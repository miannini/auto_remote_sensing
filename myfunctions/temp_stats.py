# -*- coding: utf-8 -*-
#from __future__ import division
import numpy.ma as ma
import pandas as pd
import numpy as np
import os
import rasterstats as rs
from myfunctions.tools import GCP_Functions
from rasterio.io import MemoryFile
#import seaborn as sns



def reduct(x):return x

def data_bruta(data,x):
    VAR = False
    BB = data[x].get('reduct')
    data2 = ma.getdata(BB)[BB.mask == VAR]
    poly = np.repeat(x , len(data2))
    dataf = {'poly':poly, 'data':data2}
    dataf = pd.DataFrame(dataf) 
    return dataf
    #pd.DataFrame(data_bruta(0))
def full_append(data,x):
    FA = [data_bruta(data,ii) for ii in x]
    return pd.concat(FA)

class Stats_charts:
    def data_g(data_i, aoig_near,todos_lotes,output_folder,analysis_area, bucket):
        
        analysis_date=output_folder+analysis_area+'/'+ data_i+'/'
        
        ''' list bands in GCP bucket foldet '''
        bands_avail = list(sorted(GCP_Functions.list_all_blobs(bucket, prefix=analysis_date, delimiter='/')))
        '''
        #bands_avail
        bands_avail = []
        for root, dirs, files in os.walk(analysis_date): #output_folder+analysis_area):
            for file in files:
                if file.endswith(".tif") and "_lote" not in file:
                    bands_avail.append(file[8:])
        bands_avail = list(set(bands_avail))
        '''
        big = []
        dictio_all = {}
        for band in bands_avail:
            #band=bands_avail[1]
            with MemoryFile(GCP_Functions.open_blob(bucket,band)) as memfile:
                with memfile.open() as src:
                    affine= src.transform
                    array= src.read(1)
                data = rs.zonal_stats(aoig_near, array,
                #data = rs.zonal_stats(aoig_near, analysis_date+band,  #"_NDVI.tif"
                                      stats="count",
                                      add_stats={'reduct': reduct}, affine=affine) #layer=1)
                '''  data[poly][reduct]  tiene el masked con los valores de cada poligono y banda
                este se deberia guardar en un JSON grande, para cada banda y fecha, y asi poder graficar los pixeles
                despues si se quiere desde otra plataforma para mostrar raw data, o suavizar o extender en la forma o algo'''
                porygons = full_append(data, range(0,len(aoig_near))) #range(0,len(aoig_near))
                Ndate = np.shape(porygons)[0]
                dateAssign = np.repeat( data_i , Ndate)
                band_name = band.split('/')[-1].split('_')[-1].split('.')[0]
                datag = { 'date':dateAssign,'poly':porygons['poly'],'data_pixel':porygons['data'],'band':band_name }
                datag = pd.DataFrame(datag)
                big.append(datag)
                src.close()
                dictio_all['band_'+band_name] = data
                #memfile.close()
        size_flag =  np.shape(datag)[0] == 0
        #df resumen
        short_resume = pd.DataFrame()
        for n in range(0,len(big)):
            temp_resume = big[n].groupby(['date','poly','band']).agg(mean_value=('data_pixel', 'mean'), sum_value=('data_pixel', 'sum'), min_value=('data_pixel', 'min'), 
                                                                     max_value=('data_pixel', 'max'), 
                                                                     std_dev=('data_pixel', 'std'), count_pxl=('data_pixel', 'count'), 
                                                                     perc_10 = ('data_pixel', lambda x: np.percentile(x,10)),
                                                                     perc_20 = ('data_pixel', lambda x: np.percentile(x,20)),
                                                                     perc_30 = ('data_pixel', lambda x: np.percentile(x,30)),
                                                                     perc_40 = ('data_pixel', lambda x: np.percentile(x,40)),
                                                                     perc_50 = ('data_pixel', lambda x: np.percentile(x,50)),
                                                                     perc_60 = ('data_pixel', lambda x: np.percentile(x,60)),
                                                                     perc_70 = ('data_pixel', lambda x: np.percentile(x,70)),
                                                                     perc_80 = ('data_pixel', lambda x: np.percentile(x,80)),
                                                                     perc_90 = ('data_pixel', lambda x: np.percentile(x,90)) )
            short_resume = pd.concat([short_resume,temp_resume]) 
            ''' aqui podria ir un dict the resumenes estadisticos, reemplazando short_resume(flattened y reduced_flat) y short ordenado  '''
        #unir datos lotes
        short_resume = short_resume.merge(todos_lotes, left_on='poly', right_index=True, how="left")
        short_resume = short_resume.drop(['geometry'], axis=1)
        short_ord = pd.pivot_table(short_resume, index=['date',"poly","name","x","y","area"],columns="band", values=['mean_value','std_dev','sum_value','count_pxl',
                                                                                                                     'min_value','perc_10','perc_20','perc_30','perc_40',
                                                                                                                     'perc_50','perc_60','perc_70','perc_80','perc_90','max_value'])
        return size_flag, datag, short_ord, short_resume, dictio_all
