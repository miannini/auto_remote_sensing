# -*- coding: utf-8 -*-
import numpy as np
#import rasterio as rio
#from rasterio.mask import mask
#from osgeo import gdal_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from scipy.ndimage import interpolation
#from myfunctions.tools import GCP_Functions
from .GCP_functions import GCP_Functions
#from myfunctions.tools import MidpointNormalize
#import cv2
import io
#import math
#from rasterio.mask import mask
#Define custom fucntions to compare bands
#more complex than rest

import matplotlib.colors as colors



class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class Satellite_tools:
    def trns_coor(area,meta):
        try:
            x, y = meta['transform'][2]+area[0]*10, meta['transform'][5]+area[1]*-10
        except:
            x, y = meta[0]['transform'][2]+area[0]*10, meta[0]['transform'][5]+area[1]*-10    
        return x, y
    
    def plot_figura2(image, analysis_area, date, output_folder, png_folder, lote_name,index_str, cmap, bucket, **kwargs): #, vmin=-1, vmax=1
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
                        
            fig = plt.figure(figsize=(3,3))
            #area chart
            ax = plt.gca()
            #remove 0 to None
            image[image==0] = None
            if cen == True:
                im = ax.imshow(image[0], cmap=cmap, clim=(vmin1, vmax1), norm=MidpointNormalize(midpoint=vcen1,vmin=vmin1, vmax=vmax1))
            elif ext==True:
                im = ax.imshow(image[0], cmap=cmap, vmin=vmin1, vmax=vmax1)
            else:
                im = ax.imshow(image[0], cmap=cmap, vmin=np.nanpercentile(image,1), vmax=np.nanpercentile(image,99)) #'RdYlGn'
            #plot
            plt.title(index_str+' in ' + str(lote_name))
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            #to locate colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ''' enviar plot a GCP '''
            # temporarily save image to buffer
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches='tight',dpi=100 , format='png')
            #plt.savefig(png_folder+analysis_area+'/'+date[:8]+"_"+str(lote_name)+index_str+".png",bbox_inches='tight',dpi=100)
            image_name = png_folder+analysis_area+'/lotes/'+str(lote_name)+'/'+index_str +'/'+str(lote_name)+"_"+index_str+'_'+date+".png"
            GCP_Functions.upload_string(bucket, buf.getvalue(),'image/png', image_name)
            #blob.upload_from_string(buf.getvalue(),content_type='image/png')
            buf.close()
            plt.clf()
            plt.close("all")