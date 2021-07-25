# -*- coding: utf-8 -*-
import numpy as np
import io
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
#from rasterio.plot import show
#from osgeo import gdal_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import interpolation
import cv2
#from varname import nameof, Wrapper, varname
import inspect
from sklearn.preprocessing import MinMaxScaler #, RobustScaler
from myfunctions.tools import MidpointNormalize
from myfunctions.tools import Satellite_tools
from myfunctions.tools import GCP_Functions
#Define custom fucntions to compare bands
#more complex than rest

x_width = 768*2    #16km width
y_height = 768*2   #16km height
#bucket = 'satellite_storage'    
class Satellite_proc:
    def crop_sat(name, aoi, analysis_area, output_folder, bucket): #file name, aoi, zona, output
        with MemoryFile(GCP_Functions.open_blob(bucket,name)) as memfile:
            with memfile.open() as src:
                out_image, out_transform = rio.mask.mask(src, aoi.geometry, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
                src.close()
            date,band = name.split('/')[-1].split("_")[1][0:8] , name.split('/')[-1].split("_")[-1].split('.')[0]
            newname = output_folder+analysis_area+'/'+date+'/'+date+"_"+band+".tif"
            #in case of asymmetryc shapes  
            dif_dims = out_image.shape[1] - out_image.shape[2]
            max_dim = max(out_image.shape[1],out_image.shape[2])
            dif_perc = abs(dif_dims)/max_dim
            
            ''' if band == B02, B03 or B04 and np.sum(out_image) == 0 close images, skip=True, entry in DB with no data and delete bucket folder '''
            sum_values = np.sum(out_image) #+ np.sum(red) + np.sum(green)
            print('Pixels with data in area of interest: ', sum_values)
            if sum_values ==0 :
                print("No data for client: {}, in zone {}, for date {}".format(output_folder.split('/')[-2],analysis_area,date))
                #goto end
                ### DELETE ORIGINAL FILE IN MONITORING BUCKET
                ''' Eliminar solo despues de pasar por todos los clientes que usen esa area '''
                GCP_Functions.delete_blob(bucket, name)
                skip = True
                return skip
            
            if dif_perc > 0.05:
                skip = True
            else:
                skip = False
            if skip == False:
                if dif_dims < 0 :
                    out_image = out_image[:,:,abs(dif_dims):]
                elif dif_dims > 0 :
                    out_image = out_image[:,abs(dif_dims):,:]
                #out_image =  out_image_c      
                
                scale = x_width/len(out_image[0])
                if scale > 1.1:
                    data = out_image[0]
                    data_interpolated = interpolation.zoom(data,scale)
                    data_interpolated = np.expand_dims(data_interpolated, axis=0)
                    out_image = data_interpolated
                '''aqui voy ... abrir blob destino en GCP con rasterio y hacer write directo'''
                #with MemoryFile() as memfile:
                    #with memfile.open(newname,"w", **out_meta) as dest:
                #with MemoryFile(GCP_Functions.open_blob(bucket,newname)) as memfile:
                #    with memfile.open() as src:
                with rio.open(newname.split('/')[-1], "w", **out_meta) as dest:
                #with rio.open(newname, "w", **out_meta) as dest:
                    dest.write(out_image)
                    #GCP_Functions.upload_blob(bucket,newname.split('/')[-1],newname) ### este si esta funcionando, pero genera archivo
                    ''' antes de close hacerle un read y dejarlo en variable  / Diccionario '''
                    dest.close()
                ### UPLOAD TO GCP STORAGE
                GCP_Functions.upload_blob_file(bucket,newname,newname.split('/')[-1])
                print(("[INFO] Archivo {}, procesada".format(name.split('/')[-1])))
                ### COPY ORIGINAL FILE TO COLDLINE / ARCHIVED
                ''' evaluar si se necessita archive de original o solo de recortado '''
                GCP_Functions.copy_blobs_sate(bucket, name, 'archived_sentinel_raw', "/".join(name.split('/')[1:]))
                ### DELETE ORIGINAL FILE IN MONITORING BUCKET
                ''' Eliminar solo despues de pasar por todos los clientes que usen esa area '''
                GCP_Functions.delete_blob(bucket, name)
        return skip
    
    def area_crop(date,aoi2,analysis_area,source, destination, output_folder): #"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" 
        with rio.open(output_folder+analysis_area+'/'+date[:8]+source) as src:
            try:
                out_image, out_transform = rio.mask.mask(src, aoi2.geometry,crop=True)
            except:
                out_image, out_transform = rio.mask.mask(src, aoi2,crop=True)  
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})  
            src.close()
        with rio.open(output_folder+analysis_area+'/'+date[:8]+destination, "w", **out_meta) as dest:
            dest.write(out_image)
            dest.close()
    '''        
    def cld_msk(date, clouds_data, ind_mask, analysis_area,output_folder):
        #define construction fixed band at 512 pxl and use always same
        base = rio.open(output_folder+analysis_area+'/'+date[:8]+"B04.tif")
        x_width = base.width
        y_height = base.height
        base.close()
        #cloud mask generate file
        #src = output_folder+analysis_area+'/'+"CLOUD_BASE.tif"
        src = output_folder+analysis_area+'/'+date[:8]+"B04.tif"
        data = clouds_data[:,:,ind_mask]
        data = data[:,:,0]
        scale = x_width/len(data[0])
        #data = data[0]
        data_interpolated = interpolation.zoom(data,scale)
        data_interpolated = np.expand_dims(data_interpolated, axis=0)
        save = gdal_array.LoadFile(src) 
        save = gdal_array.SaveArray(data_interpolated, output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        save = gdal_array.SaveArray(data_interpolated, output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        save = None
        return x_width, y_height
    '''
    
    def band_calc(date, analysis_area,output_folder,png_folder,bucket): 
        route = output_folder+analysis_area+'/'+date+'/'+date
        #route = route[1:]
        #msk_cloud = rio.open(output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif")       
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B01.tif")) as memfile:
            b1 = memfile.open()
            b1r = b1.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B02.tif")) as memfile:
            b2 = memfile.open()
            blue = b2.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B03.tif")) as memfile:
            b3 = memfile.open()
            green =b3.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B04.tif")) as memfile:
            b4 = memfile.open()
            red = b4.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B05.tif")) as memfile:
            b5 = memfile.open()
            b5r = b5.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B07.tif")) as memfile:
            b7 = memfile.open()
            b7r = b7.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B08.tif")) as memfile:
            b8 = memfile.open()
            nir = b8.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B8A.tif")) as memfile:
            b8a = memfile.open()
            b8ar = b8a.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B09.tif")) as memfile:
            b9 = memfile.open()
            b9r = b9.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B11.tif")) as memfile:
            b11 = memfile.open()
            b11r = b11.read()
        with MemoryFile(GCP_Functions.open_blob(bucket,route +"_B12.tif")) as memfile:
            b12 = memfile.open()
            b12r = b12.read()
        
        #Close opened bands
        bands = [b1,b2,b3,b4,b5,b7,b8,b8a,b9,b11,b12]# b6, b10, msk_cloud
        for ban in bands:
            ban.close()
        
        #Sum content of blue, red and green ... if 0, go to end and skip
        #entry in DB of no data
        #message print
        sum_values = np.sum(blue)+ np.sum(red) + np.sum(green)
        if sum_values ==0 :
            print("No data for client: {}, in zone {}, for date {}".format(output_folder.split('/')[-2],analysis_area,date))
            #goto end
            return 0, 1
        
        #20m convert
        def scale_20(band):
            scale = x_width/len(band[0])
            data = band[0]
            data_interpolated = interpolation.zoom(data,scale)
            data_interpolated = np.expand_dims(data_interpolated, axis=0)
            return data_interpolated
        
        def mirror_dims(band):
            dif_dims = band.shape[1] - band.shape[2]
            max_dim = max(band.shape[1],band.shape[2])
            dif_perc = abs(dif_dims)/max_dim
            if dif_perc > 0.05:
                skip = True
            else:
                skip = False
            if skip == False:
                if dif_dims < 0 :
                    band = band[:,:,abs(dif_dims):]
                elif dif_dims > 0 :
                    band = band[:,abs(dif_dims):,:]
                return band     
                
                
        
        b1r_c = scale_20(mirror_dims(b1r))
        b5r_c = scale_20(mirror_dims(b5r))
        #b6r_c = scale_20(mirror_dims(b6r))
        b7r_c = scale_20(mirror_dims(b7r))
        b8ar_c = scale_20(mirror_dims(b8ar))
        b9r_c = scale_20(mirror_dims(b9r))
        #b10r_c = scale_20(mirror_dims(b10r))
        b11r_c = scale_20(mirror_dims(b11r))
        b12r_c = scale_20(mirror_dims(b12r))
        
        '''
        blue=492 +/- 66, green=560 +/- 36, red=664 +/- 31, nir=833 +/- 106  [10 meter] RGB and NIR
        b5=704.1 +/- 15, b6=740.5 +/- 15, b7=782.8 +/- 20, b8a=864.7 +/- 21 [20 meter] Veg rededge / narrow NIR
        b11=1613 +/- 91, b12=2202 +/- 175                                   [20 meter] SWIR
        b1=442.7 +/- 21, b9=945 +/- 20, b10=1373 +/- 31                     [60 meter] Cloud & Atm
        b1=coastal aerosol, b5-7=vegetation red edge, b8a=narrow NIR, b9=water vapour, b10=SWIR-cirrus, b11-12=SWIR 
        '''
        
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (nir.astype(float)-red.astype(float))/(nir+red) #normalized difference vegetation index
        ndwi = (green.astype(float)-nir.astype(float))/(nir+green) #normalized difference water index
        rededge = b5r_c.astype(float)/red
        #rededge2 = (b5r_c.astype(float) - red.astype(float))/(b5r_c.astype(float)+red)
        ccci = ((rededge.astype(float)-nir.astype(float))/(nir.astype(float)+rededge.astype(float)))/ndvi.astype(float) #canopy chlorophyll content index
        cig = (nir.astype(float)/green) -1 #chlorophyll  index green
        X=0.08; a=1.22; b=0.03; L=0.5
        atsavi = a*((nir-a*red-b)/(a*nir+red-a*b+X*(1+a*a))) #adjusted transformed soil-adjusted Vegetation index
        savi = (1+L)*(nir.astype(float)-red.astype(float))/(nir+red+L) #soil adjusted vegetation index
        ndmi = (nir.astype(float)-b11r_c.astype(float))/(nir+b11r_c) #normalized diferenve moisture index
        gvmi = ((nir+0.1)-(b11r_c+0.02))/((nir+0.1)+(b11r_c+0.02)) #Global vegetation moisture index
        cvi = nir.astype(float) * (red.astype(float)/(green.astype(float)*green.astype(float))) # chlorophyll vegetation index
        dswi = (nir.astype(float)-green.astype(float)) / (b11r_c+red) # 	Disease water stress index
        lai=0.001*np.exp(8.7343*ndvi.astype(float)) #leaf area index
        bm = 451.99*lai.astype(float) + 1870.7 #biomass
        bwdrvi = (0.1*nir.astype(float)-blue.astype(float))/(0.1*nir+blue) #blue wide dynamic range vegetation index
        bri = ((1/green.astype(float))-(1/b5r_c.astype(float)))/nir.astype(float)  #brwoning reflectance index
        #s2 CP and NDF from paper, b1=560, b2=665, b3=865, b4=2202
        cp = 7.63-54.39*green.astype(float)/10000 + 31.37*red.astype(float)/10000 +21.23*b8ar_c.astype(float)/10000 -25.33*b12r_c.astype(float)/10000 #Crude Protein
        ndf = 54.49+207.53*green.astype(float)/10000 -193.99*red.astype(float)/10000 -54.19*b8ar_c.astype(float)/10000 +111.15*b12r_c.astype(float)/10000 #Neutral Detergent Fiber
        
        #cld_pxl_count = (np.count_nonzero(cld==0))/(cld[0].shape[0]*cld[0].shape[1])
        
        #custom own cloud detection algorithm
        ndgr = (green.astype(float)-red.astype(float))/(green+red) #normalized difference green/red
        tao = 0.2
        cld_custom = np.where( ((b11r_c/10000>tao) & ( ((green/10000>0.175) & (ndgr>0)) | (green/10000>0.39))) , 1, 0)
        cld_shadow = np.where( (green/10000<0.319) & (b8ar_c/10000<0.166) & (((green-b7r_c/10000<0.027) & (b9r_c-b11r_c/10000>-0.097)) | ((green-b7r_c/10000>0.027) & (b9r_c-b11r_c/10000>0.021))), 1, 0)
        cld_shadow2 = np.where( (green/10000<0.319) & (b5r_c/b11r_c>4.33) & (green/10000<0.525) & (b1r_c/b5r_c>1.184), 1, 0)                      
        cld_shd_tot = np.where( (cld_shadow2==1) |(cld_shadow==1) , 1, 0)     
           
        #CV para quitar ruido (pequenos puntos) y ampliar bordes para quitar sombras y zona de duda
        cld2 = cld_custom[0,:,:]
        cld2 = cld2.astype('uint8')
        kernel = np.ones((3,3),np.uint8)
        #kernel2 = np.ones((5,5),np.uint8)
        erosion = cv2.erode(cld2,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 3)
        cldcustom = np.expand_dims(dilation, axis=0)
        
        cldall = np.where(cld_shd_tot==1, 2, np.where(cldcustom==1, 1, 0)) #1=cloud, 2=shadow  
        cld_pxl_count = (np.count_nonzero(cldcustom==0))/(cldcustom[0].shape[0]*cld_custom[0].shape[1])
        
        indexes = [ndvi,ndwi,ccci,cig,atsavi,savi,ndmi,gvmi,cvi,dswi,lai,bm,bwdrvi,bri,cp,ndf]
        
        #corrections based on clouds
        def remove_clouds(index):
            index[cldcustom>=1] = None #>1
        for ind in indexes:
            remove_clouds(ind)
        
        indexes = [ndvi,ndwi,ccci,cig,atsavi,savi,ndmi,gvmi,cvi,dswi,lai,bm,bwdrvi,bri,cp,ndf,cldcustom]
        #indexes.append(cld_custom) #//no trae el nombre correctamente para TIF

        #cp[cp<5] = None
        #ndf[ndf>60] = None
        cp[ndvi<0.35] = None
        ndf[ndvi<0.35] = None
        
        
        def retrieve_name(var):
            callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var]
        
        #np.nanpercentile(cvi,90)

        def plot_figura(index, cmap, **kwargs): #, vmin=-1, vmax=1
            ext=False
            cen = False
            for key,value in kwargs.items():
                if key == 'vmin':
                    vmin1=value
                    ext=True
                elif key == 'vmax':
                    vmax1=value
                    ext=True
                elif key == 'vcen':
                    vcen1=value
                    ext=True
                    cen = True
                        
            fig = plt.figure(figsize=(10,10))
            #area chart
            ax = plt.gca()
            if cen == True:
                im = ax.imshow(index[0], cmap=cmap, clim=(vmin1, vmax1), norm=MidpointNormalize(midpoint=vcen1,vmin=vmin1, vmax=vmax1))
            elif ext==True:
                im = ax.imshow(index[0], cmap=cmap, vmin=vmin1, vmax=vmax1)
            else:
                im = ax.imshow(index[0], cmap=cmap, vmin=np.nanpercentile(index,1), vmax=np.nanpercentile(index,99)) #'RdYlGn'
            #plot
            plt.title(str(retrieve_name(index)[0])+' in Analysis area')
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            #to locate colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ''' enviar plot a GCP '''
            # temporarily save image to buffer
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches='tight',dpi=300 , format='png')
            #plt.savefig(output_folder+analysis_area+'/'+date[:8]+"_"+str(retrieve_name(index)[0])+".png")
            # upload buffer contents to gcs
            image_name = png_folder+analysis_area+'/'+date +'/' + date+"_"+str(retrieve_name(index)[0])+".png"
            GCP_Functions.upload_string(bucket, buf.getvalue(),'image/png', image_name)
            #blob.upload_from_string(buf.getvalue(),content_type='image/png')
            buf.close()
            plt.clf()
        
        #cmaps = 'RdYlGn', 'RdYlBu','nipy_spectral_r'
        plot_figura(ndvi,'RdYlGn',vmin=-1,vmax=1)
        plot_figura(ndwi,'RdYlBu', vmin=-1, vmax=0.4,vcen=0)
        plot_figura(ccci,'RdYlGn')
        plot_figura(cig,'RdYlGn')
        plot_figura(atsavi,'RdYlGn', vmin=0, vmax=1)
        plot_figura(savi,'RdYlGn')
        plot_figura(dswi,'RdYlGn')
        plot_figura(lai,'nipy_spectral_r', vmin=0, vmax=3)
        plot_figura(bm,'nipy_spectral_r', vmin=2000, vmax=3500)
        plot_figura(bwdrvi,'jet')
        plot_figura(bri,'terrain')
        plot_figura(cp,'RdYlGn', vmin=5, vmax=18)
        plot_figura(ndf,'RdYlGn', vmin=30, vmax=60)
        plot_figura(cldcustom,'gray',vmin=0,vmax=1)
        plot_figura(cldall,'copper',vmin=0,vmax=2)
        
        #RGB     
        rgb=np.zeros((green.shape[1],green.shape[2],3))
        bands = [red, green, blue]
        for i,b in enumerate(bands):
            #copy of original RGB bands
            transformed_band = b.astype(float).copy()
            transformed_band[cld_custom>=1] = None
            #stats of bands color without clouds to normalize colors
            tr_avg = np.nanmean(transformed_band)
            tr_std= np.nanstd(transformed_band)
            #scale
            scaler = MinMaxScaler()
            x_sample = [tr_avg-2*tr_std, tr_avg+2*tr_std] #rango de AVG +/- STD DEV
            scaler.fit(np.array(x_sample)[:, np.newaxis]) #ajustado al rango de cada banda
            #reshape to apply same transformation to all rows and cols of image
            ascolumns = transformed_band[0,:,:].reshape(-1,1)
            t = scaler.transform(ascolumns) #fit_
            transformed = t.reshape(transformed_band.shape)
            #include again the clouds, but with a mask yellow over it
            #transformed[cld_custom>=1] = b 
            #include in big RGB file
            rgb[...,i] = transformed#_band
            
        rgb2 = (rgb.copy()*255).astype('uint8')
        #Reverse Red-Blue Channels as open cv will reverse again upon writing image on disk
        ''' enviar bytes a GCP  probar con buf'''
        newname=png_folder+analysis_area+'/'+date+'/'+date+"_True_Color.jpg"
        #buf = io.BytesIO()
        #cv2.imwrite(buf,rgb2[...,::-1])
        #GCP_Functions.upload_string(bucket, buf.getvalue(),'image/jpg', newname) #upload_blob_file(bucket,newname,buf.getvalue())
        cv2.imwrite(newname.split('/')[-1],rgb2[...,::-1])
        GCP_Functions.upload_blob_file(bucket,newname,newname.split('/')[-1])
        
        #export indexes as tif format
        meta = b4.meta
        meta.update(driver='GTiff')
        meta.update(dtype=rio.float32)
        def export_tif(index):
            newname = output_folder+analysis_area+'/'+date+'/'+date+"_"+str(retrieve_name(index)[0])+".tif"
            with rio.open(newname.split('/')[-1], 'w', **meta) as dst:
                dst.write(index.astype(rio.float32))
                dst.close()
                b4.close()
                ''' enviar a GCP  '''
            ### UPLOAD TO GCP STORAGE
            GCP_Functions.upload_blob_file(bucket,newname,newname.split('/')[-1])
            print(("[INFO] Archivo {}, procesada".format(newname.split('/')[-1])))
     
        for ind in indexes:
            export_tif(ind)
        
        #Close opened bands
        #bands = [b1,b2,b3,b4,b5,b7,b8,b8a,b9,b11,b12]# b6, b10, msk_cloud
        #for ban in bands:
        #    ban.close()
        
        #store expanded bands // but can be more space required
        '''
        #should create a dictionary of bands vs real name band to save with same names
        expanded_bands = [b1r_c, b5r_c, b6r_c, b7r_c, b8ar_c, b9r_c, b10r_c, b11r_c, b12r_c]
        for exb in expanded_bands:
            export_tif(exb)
        '''
        #label:end
        return meta, cld_pxl_count  
        
    def trns_coor(area,meta):
        try:
            x, y = meta['transform'][2]+area[0]*10, meta['transform'][5]+area[1]*-10
        except:
            x, y = meta[0]['transform'][2]+area[0]*10, meta[0]['transform'][5]+area[1]*-10    
        return x, y
    
    def mosaic_files(unzipped_folder,analysis_area,qs):
        src_files_to_mosaic = []
        for fp in qs:
            src = rio.open(fp)
            src_files_to_mosaic.append(src)
        mosaic, out_trans = merge(src_files_to_mosaic)
        #show(mosaic, cmap='terrain')
        try:
            route = qs[0].split("/")[0]
            name = qs[0].split("/")[-1].split(".")[0]  
            folder_safe = qs[0].split("/")[-5] 
        except:
            route = qs[0].split("\\")[0]
            name = qs[0].split("\\")[-1].split(".")[0]
            folder_safe = qs[0].split("\\")[-5]
        name1 = name.split("_")[0][:3]+"_"+name.split("_")[1]+"_"+name.split("_")[2]
        # Copy the metadata
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update({"driver": "GTiff", #"JP2ECW", 
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                         }
                        ) 
        # Write the mosaic raster to disk
        with rio.open(unzipped_folder+analysis_area+"/mosaic/"+name1+".tif", "w", **out_meta) as dest: #".jp2" ".tif"
            dest.write(mosaic)
            dest.close()
        src.close()
        return route, name1, folder_safe
    
    #function to plot indexes by lote
    def small_area_crop_plot(date,aoig_near,analysis_area, source, output_folder, png_folder, bucket): #aoi2,"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" ,lote_name
        #analysis_date=output_folder+analysis_area+'/'+ date+'/'
        with MemoryFile(GCP_Functions.open_blob(bucket,source)) as memfile:
            with memfile.open() as src:
                #with rio.open(source) as src:
                indexes = ["ndvi","atsavi","lai","bm","cp","ndf"]
                #incluir aqui el loop para recortar los lotes, evitando I/O por cada lote
                for n, geo in enumerate (aoig_near.geometry):
                    out_image, out_transform = rio.mask.mask(src, [geo],crop=True)  
                    lote_name = aoig_near['lote_id'][n] 
                    #out_image, out_transform = rio.mask.mask(src, aoi2,crop=True) 
                    indx=source.split('/')[-1].split('_')[-1].split('.')[0]
                    if indx==indexes[0]: #ndvi
                        Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder, png_folder, lote_name,indx,'RdYlGn',bucket, vmin=-1,vmax=1)
                    elif indx==indexes[1]: #atsavi
                        Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder, png_folder, lote_name,indx,'RdYlGn',bucket, vmin=0,vmax=1)
                    elif indx==indexes[2]: #lai
                        Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder, png_folder, lote_name,indx,'nipy_spectral_r',bucket, vmin=0,vmax=3)
                    elif indx==indexes[3]: #bm
                        Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder, png_folder, lote_name,indx,'nipy_spectral_r',bucket, vmin=2000,vmax=3500)
                    elif indx==indexes[4]: #cp
                        Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder, png_folder, lote_name,indx,'RdYlGn',bucket, vmin=5,vmax=18)
                    elif indx==indexes[5]: #ndf
                        Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder, png_folder, lote_name,indx,'RdYlGn',bucket, vmin=30,vmax=60)
                    
                src.close()
                plt.close("all")
            
            #output_folder+analysis_area+'/'+date[:8]+destination
