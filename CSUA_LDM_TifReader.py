from osgeo import gdal, osr
import numpy as np
import os

gdal.UseExceptions()

class Tif_Read_and_Write(object):
    def __init__(self):
        super(Tif_Read_and_Write, self).__init__()

    def Tif_Read(self, input_data_path):
        dataset = gdal.Open(input_data_path)
        im_width = dataset.RasterXSize  # image width in pixels
        im_height = dataset.RasterYSize  # image height in pixels
        im_proj = dataset.GetProjection()  # projection coordinate system
        im_Geotrans = dataset.GetGeoTransform()  # affine georeferencing transform
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # read the image data as a NumPy array
        del dataset  # release the dataset object
        return im_data, im_proj, im_Geotrans


    def Numpy_to_Tif(self, array_data, output_path, top_left_lon, top_left_lat, pixel_width, pixel_height, epsg_code=None, prj_info=None, nodata_value=np.nan):

        if len(array_data.shape) == 3:  # determine whether the input is multi-band (3-D) or single-band (2-D)
            im_bands, rows, cols = array_data.shape
        else:
            im_bands, (rows, cols) = 1, array_data.shape
        # compute the georeferencing transform
        geotransform = (top_left_lon, pixel_width, 0, top_left_lat, 0, pixel_height)     # pixel_height is signed to account for north-up vs. south-up images
        # create the output GeoTIFF dataset
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(output_path, cols, rows, im_bands, gdal.GDT_Float32)
        # set the georeferencing transform
        output_dataset.SetGeoTransform(geotransform)
        # set the projection
        output_srs = osr.SpatialReference()
        if prj_info is not None:
            output_srs.ImportFromWkt(prj_info)
        elif epsg_code is not None:
            output_srs.ImportFromEPSG(epsg_code)
        else:
            raise Exception('Please provide either epsg_code or prj_info.')
        
        output_dataset.SetProjection(output_srs.ExportToWkt())
        # write raster data to the output bands
        if im_bands == 1:
            output_band = output_dataset.GetRasterBand(1)
            output_band.SetNoDataValue(nodata_value)
            output_band.WriteArray(array_data)
        else:
            for i in range(im_bands):
                output_band = output_dataset.GetRasterBand(i + 1)
                output_band.SetNoDataValue(nodata_value)
                output_band.WriteArray(array_data[i])
        # build overview pyramids
        output_dataset.BuildOverviews("BILINEAR", [2, 4, 8, 16])  # using bilinear resampling
        # close the output dataset
        del output_dataset