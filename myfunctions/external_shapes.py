# -*- coding: utf-8 -*-
#from shapely.geometry import Polygon
import geopandas as gpd
#import folium 
import pandas as pd
#import shapefile
import os
#from pyproj import Proj, transform
from scipy.spatial import distance_matrix

class Ext_shape:
    def merge_shapes(shape_folder,lote_aoi,lote_aoi_loc,folder_name,analysis_area):
        #loop to read shapefiles
        todos = gpd.GeoDataFrame() #columns=['geometry','x','y']
        todos_loc = gpd.GeoDataFrame()
        for root, dirs, files in os.walk(shape_folder+folder_name):
            for file in files:
                #print (root,dirs,files)
                crs = {'init': 'epsg:32618'}
                if file.endswith(".shp"):
                    shape_path = os.path.join(root, file)
                    temp_file = gpd.read_file(shape_path)
                    temp_file.columns = map(str.lower, temp_file.columns) #lower case column names
                    #maybe join name of area with name of file
                    file_name = file.split(".")[0].split("-")[0]
                    if file_name == 'big_box':
                        continue
                    if 'name' not in temp_file:
                        temp_file['name'] = 'no_name' + "-" + temp_file.index.map(str)
                    if file_name.upper() != "POLYGON":
                        temp_file['name'] = file_name + "-" + temp_file['name'].map(str)
                    #local
                    temp_file_loc= temp_file.to_crs(32618)
                    temp_file_loc["x"] = temp_file_loc.centroid.map(lambda p: p.x)
                    temp_file_loc["y"] = temp_file_loc.centroid.map(lambda p: p.y)
                    temp_file_loc["area"] = temp_file_loc.area
                    temp_file_loc = temp_file_loc[['name','geometry','x','y','area']]
                    todos_loc = gpd.GeoDataFrame( pd.concat( [temp_file_loc,todos_loc], ignore_index=True) , crs=crs)
                    #universal
                    #temp_file= temp_file.to_crs(32618)
                    temp_file["x"] = temp_file.centroid.map(lambda p: p.x)
                    temp_file["y"] = temp_file.centroid.map(lambda p: p.y)
                    temp_file["area"] = temp_file_loc['area'] #copy area from local in meters
                    temp_file = temp_file[['name','geometry','x','y','area']]
                    todos = gpd.GeoDataFrame( pd.concat( [temp_file,todos], ignore_index=True) , crs=crs)
        coords = todos_loc[["x","y"]]
        coords.index = todos_loc["name"]
        distances =  pd.DataFrame(distance_matrix(coords.values, coords.values), index=coords.index, columns=coords.index)
        indexes = []
        for n in range(0,len(distances)):
            temp_min = distances.iloc[:,n].nsmallest(10)           
            temp_min = temp_min[(temp_min != 0)].index
            temp_min = list(temp_min)
            indexes.append(temp_min)
        headers = ['close_1','close_2','close_3','close_4','close_5','close_6','close_7','close_8','close_9']
        indexes = pd.DataFrame(indexes, columns=headers)
        indexes["area"] = todos['area']
        indexes.index = todos_loc["name"]
        
        todos_loc = gpd.GeoDataFrame( pd.concat( [lote_aoi_loc,todos_loc], ignore_index=True) , crs=crs)
        todos = gpd.GeoDataFrame( pd.concat( [lote_aoi,todos], ignore_index=True) , crs=crs)
        indexes.to_csv (r'../Data/Database/'+analysis_area+'/lotes_cercanos.csv', index = True, header=True)
        return todos, todos_loc
    
    def merge_shapes2(shape_folder): #folder_name, analysis_area
        #loop to read shapefiles
        todos = gpd.GeoDataFrame() #columns=['geometry','x','y']
        todos_loc = gpd.GeoDataFrame()
        for root, dirs, files in os.walk(shape_folder): #+folder_name
            for file in files:
                #print (root,dirs,files)
                #crs = {'init': 'epsg:32618'}
                if file.endswith(".shp"):
                    shape_path = os.path.join(root, file)
                    temp_file = gpd.read_file(shape_path)
                    temp_file.columns = map(str.lower, temp_file.columns) #lower case column names
                    #maybe join name of area with name of file
                    file_name = file.split(".")[0].split("-")[0]
                    if file_name == 'big_box':
                        continue
                    if 'name' not in temp_file:
                        temp_file['name'] = 'no_name' + "-" + temp_file.index.map(str)
                    if file_name.upper() != "POLYGON":
                        temp_file['name'] = file_name + "-" + temp_file['name'].map(str)
                    #local
                    temp_file_loc= temp_file.to_crs(32618)
                    temp_file_loc["x"] = temp_file_loc.centroid.map(lambda p: p.x)
                    temp_file_loc["y"] = temp_file_loc.centroid.map(lambda p: p.y)
                    temp_file_loc["area"] = temp_file_loc.area
                    temp_file_loc = temp_file_loc[['name','geometry','x','y','area']]
                    todos_loc = gpd.GeoDataFrame( pd.concat( [temp_file_loc,todos_loc], ignore_index=True))# , crs=crs)
                    todos_loc.set_crs('epsg:32618', inplace=True) 
                    #universal
                    #temp_file= temp_file.to_crs(32618)
                    temp_file["x"] = temp_file.centroid.map(lambda p: p.x)
                    temp_file["y"] = temp_file.centroid.map(lambda p: p.y)
                    temp_file["area"] = temp_file_loc['area'] #copy area from local in meters
                    temp_file = temp_file[['name','geometry','x','y','area']]
                    todos = gpd.GeoDataFrame( pd.concat( [temp_file,todos], ignore_index=True))# , crs=crs)
                    todos.to_crs('epsg:4326', inplace=True) 
        coords = todos_loc[["x","y"]]
        coords.index = todos_loc["name"]
        distances =  pd.DataFrame(distance_matrix(coords.values, coords.values), index=coords.index, columns=coords.index)
        indexes = []
        for n in range(0,len(distances)):
            temp_min = distances.iloc[:,n].nsmallest(10)           
            temp_min = temp_min[(temp_min != 0)].index
            temp_min = list(temp_min)
            indexes.append(temp_min)
        headers = ['close_1','close_2','close_3','close_4','close_5','close_6','close_7','close_8','close_9']
        indexes = pd.DataFrame(indexes, columns=headers)
        indexes["area"] = todos['area']
        indexes.index = todos_loc["name"]
        
        todos_loc = gpd.GeoDataFrame(todos_loc)#, crs=crs)
        todos = gpd.GeoDataFrame(todos)#, crs=crs)
        #indexes.to_csv (r'../Data/Database/'+analysis_area+'/lotes_cercanos.csv', index = True, header=True)
        return todos, todos_loc
