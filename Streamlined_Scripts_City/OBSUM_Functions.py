#The purpose of this script is to contain all the functions we will use to execute OBSUM
#The code was taken from Houcai Guo, Dingqi Ye, Hanzeyu Xu, Lorenzo Bruzzone

import numpy as np
from scipy.optimize import lsq_linear
from skimage.transform import downscale_local_mean, resize
import rasterio


#Define the read raster function
def read_raster(raster_path):
    dataset = rasterio.open(raster_path)
    raster_profile = dataset.profile
    raster = dataset.read()
    #We order the dimensions to be "Row, Column, Band" instead of "Band, Row, Column"
    raster = np.transpose(raster, (1, 2, 0))
    raster = raster.astype(np.dtype(raster_profile["dtype"]))

    return raster, raster_profile

#Define the write raster function
def write_raster(raster, raster_profile, raster_path):
    raster_profile["dtype"] = str(raster.dtype)
    raster_profile["height"] = raster.shape[0]
    raster_profile["width"] = raster.shape[1]
    raster_profile["count"] = raster.shape[2]

    #Reorder the bands back to "Band, Row, Column"
    image = np.transpose(raster, (2, 0, 1))
    dataset = rasterio.open(raster_path, mode = 'w', **raster_profile)
    dataset.write(image)
    dataset.close()

class OBSUM:
    #Create the OBSUM object with initialized parameters
    def __init__(self, F_tb, C_tb, C_tp, F_tb_class, F_tb_objects,
                 class_num=5, scale_factor=30, win_size=15,
                 OL_RC_percent=5,
                 similar_win_size=31, similar_num=30,
                 min_val=0.0, max_val=1.0):
        self.F_tb = F_tb.astype(np.float32)
        self.C_tb = C_tb.astype(np.float32)
        self.C_tp = C_tp.astype(np.float32)
        #Calculating the coarse image temporal change
        self.delta_C = self.C_tp - self.C_tb
        self.F_tb_class = F_tb_class
        self.F_tb_objects = F_tb_objects
        self.class_num = class_num
        self.scale_factor = scale_factor
        self.win_size = win_size
        self.OL_RC_percent = OL_RC_percent
        self.similar_win_size = similar_win_size
        self.similar_num = similar_num
        self.min_val = min_val
        self.max_val = max_val
    

    #Now determine the classes of each object within the image
    def refine_classification_using_objects(self):
        """
        Refine the land-cover classification map using the segmented image objects.
        """
        #Create this array to store the refined classification results
        refined_class = np.empty(shape=self.F_tb_class.shape, dtype=np.uint8)

        #We essentially want to get all of the class labels from the classification map

        #Create an array containing all of the unique object indices found in the self.F_tb_objects segmentation map
        object_indices = np.unique(self.F_tb_objects)
        #Now the code loops through each unique object index, with object_classes getting the class labels from the classification map where the object mask is True
        for object_idx in object_indices:
            #This line creates a binary mask, identifying which pixels in the F_tb_objects array belong to the current object
            object_mask = self.F_tb_objects == object_idx

            #We then find the instances where the binary mask is true and retrieve the class labels from the classification map
            object_classes = self.F_tb_class[object_mask]

            #If there is only one pixel belonging to the current object (the object is a single pixel), then we set the class equal to the class of the singular pixel
            if np.count_nonzero(object_mask) == 1:
                object_class = object_classes[0]
            else:
                #Otherwise the most frequent class label is chosen as the class of the entire object
                object_class = np.argmax(np.bincount(object_classes.squeeze())[1:]) + 1
                
            refined_class[object_mask] = object_class
    
        self.F_tb_class = refined_class
        print(f"Refined land-cover classification map!")

    def calculate_class_fractions(self):
        """
        Calculate the fractions of the land-cover classes inside each coarse pixel.

        Returns
        -------
        C_fractions : array_like, (C_row, C_col, class_num) shaped
            Fractions of the land cover classes.
        """
        #Initialize a zero matrix with the same number of rows and columns in the coarse grid, with another dimension based on the number of classes
        C_fractions = np.zeros(shape=(self.C_tp.shape[0], self.C_tp.shape[1], self.class_num), dtype=np.float32)
        #Loop through the rows and the columns, essentially each coarse pixel, and extract a corresponding subsection of the fine-resolution classification map
        for row_idx in range(self.C_tp.shape[0]):
            for col_idx in range(self.C_tp.shape[1]):
                #self.scale_factor is the size of each coarse pixel in terms of fine pixels
                #The code line below extracts a subarray of fine-classified pixels corresponding to the current coarse pixel
                F_class_pixels = self.F_tb_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                                                 col_idx * self.scale_factor:(col_idx+1) * self.scale_factor]
                #The loop counts the number of pixels in F_class_pixels that are classified as a specific class in class_idx
                #We then divide the count by self.scale_factor^2 which is the total number of fine pixels in a coarse pixel
                #We then store the result
                for class_idx in range(self.class_num):
                    pixel_num = np.count_nonzero(F_class_pixels == class_idx)
                    C_fractions[row_idx, col_idx, class_idx] = pixel_num / (self.scale_factor * self.scale_factor)

        #This is the fraction of each class in every coarse pixel
        return C_fractions
    
    #We are using this function to unmix the coarse pixel and solve it through optimization
    def unmix_window(self, C_values, C_fractions, lower_bound, upper_bound):
        """
        Use a constrained least squares method to unmix the coarse pixel to get the reflectance of the each class
        in the predicted image covered by the central coarse pixel of the moving window.

        Parameters
        ----------
        C_values : array_like, (win_size**2, 1) shaped
            Reflectances of the coarse pixels in the moving window.
        C_fractions : array_like, (win_size**2, class_num) shaped
            Fractions of all the classes inside each coarse pixel.
        lower_bound : float
            Lower bound in lsq estimation.
        lower_bound : float
            Upper bound in lsq estimation.
        Returns
        -------
        result : array_like, (class_num, 1) shaped
            The unmixed changes.
        """
        lsq = lsq_linear(C_fractions, C_values,
                         bounds=(lower_bound, upper_bound), method = "bvls", max_iter=100)
        
        result = lsq.x

        return result
    
    def calculate_distances_in_coarse_pixel(self):
        """
        For each fine pixel within a coarse pixel, calculate its distance to the central fine pixel.
        """
        #Define the number of rows and columns
        rows = np.linspace(start=0, stop=self.scale_factor - 1, num=self.scale_factor)
        cols = np.linspace(start=0, stop=self.scale_factor - 1, num=self.scale_factor)
        #We create two 2D arrays that represent the grid coordinates of the fine pixels in the coarse pixels
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        #We get the row and column indices of the central fine pixel within the coarse pixel and calculate the distance from the grid coordinates of every pixel
        central_row = self.scale_factor // 2
        central_col = self.scale_factor // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        # distances = np.concatenate([distances for i in range(self.C_tp.shape[0])], axis = 0)
        # distances = np.concatenate([distances for i in range(self.C_tp.shape[1])], axis = 1)
        distances = np.tile(distances, (self.C_tp.shape[0], self.C_tp.shape[1]))

        #Normalize to [1, 1+sqrt(2)]
        distances = 1 + distances / (self.scale_factor // 2)

        #Return the normalized sitances of all fine pixels within each coarse pixel to the central fine pixel
        return distances

    def calculate_object_homogeneity_index(self):
        """
        For each fine pixel, calculate the object homogeneity index inside the local window (size is one coarse pixel).
        """
        #Create an array to hold the OHI (Object Homogeneity Index)
        object_homo_index = np.zeros(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=np.float32)
        #Add padding around the edges, with half the size of a coarse pixel
        F_tb_objects_pad = np.pad(self.F_tb_objects, pad_width=((self.scale_factor // 2, self.scale_factor // 2),
                                                                (self.scale_factor // 2, self.scale_factor // 2)),
                                mode = "reflect")
        
        #Loop through every fine pixel and retrieve the object label of the current fine pixel
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):
                current_object = self.F_tb_objects[row_idx, col_idx]
                #Extract a local window centered on the current fine pixel
                pixel_objects = F_tb_objects_pad[row_idx:row_idx + self.scale_factor,
                                                 col_idx:col_idx + self.scale_factor]
                
                #Count how many pixels in the local window belong to the same object as the current fine pixel
                #Calculates the total number of pixels in the local window that belong to the same object as the fine pixel
                object_homo_index[row_idx, col_idx] = np.count_nonzero(pixel_objects == current_object) / np.square(self.scale_factor)

        return object_homo_index
    
    
    def calculate_similar_pixel_distances(self):
        """
        Calculate similar pixels' distances to the central target pixel within a given window size, specifically designed for a local window
        """
        rows = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        cols = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        central_row = self.similar_win_size // 2
        central_col = self.similar_win_size // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        #Normalize to [1, 1+sqrt(2)]
        distances = 1 + distances / (self.similar_win_size // 2)

        return distances

    def select_similar_pixels(self):
        """
        Select similar pixels for pixel-level residual compensation.
        """
        F_tb_pad = np.pad(self.F_tb,
                          pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                     (self.similar_win_size // 2, self.similar_win_size // 2),
                                     (0, 0)),
                                     mode="reflect")
        
        F_tb_similar_weights = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.similar_num),
                                        dtype = np.float32)
        F_tb_similar_indices = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.similar_num),
                                        dtype = np.uint32)
        
        distances = self.calculate_similar_pixel_distances().flatten()
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):
                central_pixel_vals = self.F_tb[row_idx, col_idx, :]
                neighbor_pixel_vals = F_tb_pad[row_idx:row_idx + self.similar_win_size,
                                               col_idx:col_idx + self.similar_win_size, :]
                D = np.mean(np.abs(neighbor_pixel_vals - central_pixel_vals), axis=2).flatten()
                similar_indices = np.argsort(D)[:self.similar_num]
                similar_distances = 1 + distances[similar_indices] / (self.similar_win_size // 2)
                similar_weights = (1 / similar_distances) / np.sum(1 / similar_distances)

                F_tb_similar_indices[row_idx, col_idx, :] = similar_indices
                F_tb_similar_weights[row_idx, col_idx, :] = similar_weights

        return F_tb_similar_indices, F_tb_similar_weights
    
    def object_based_spatial_unmixing(self):
        """
        Object-Based Spatial Unmixing Model.

        Returns
        -------
        F_tp_OBSUM : array_like
            The predicted fine image at tp.
        """
        ###########################################################
        #                      Initialization                     #
        ###########################################################
        #What we are predicting
        F_tp_OBSUM = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.C_tp.shape[2]),
                              dtype=self.C_tp.dtype)
        
        #Refine the image classes using image objects
        self.refine_classification_using_objects()

        #Calculate the class fractions
        C_fractions = self.calculate_class_fractions()

        #Calculate the object residual index which is used to perform the object-level residual compensation (OL-RC)
        #Currently (128, 112)
        distances_in_C = self.calculate_distances_in_coarse_pixel()
        #Currently (1920, 1776)
        object_homo_index = self.calculate_object_homogeneity_index()
        object_residual_index = object_homo_index / distances_in_C

        #Pad the coarse temporal change and the fraction maps since the unmixing process is based on a local window
        delta_C_pad = np.pad(self.delta_C, pad_width=((self.win_size // 2, self.win_size // 2),
                                                      (self.win_size // 2, self.win_size // 2), (0, 0)), mode="reflect")
        
        C_fractions_pad = np.pad(C_fractions, pad_width=((self.win_size // 2, self.win_size // 2),
                                                         (self.win_size // 2, self.win_size // 2), (0, 0)), mode="reflect")
        
        object_indices = np.unique(self.F_tb_objects)

        #Select similar pixels for pixel level residual compensation (PL-RC)
        F_tb_similar_indices, F_tb_similar_weights = self.select_similar_pixels()
        print("Selected similar pixels!")

        ####################################################################
        #    Apply the Object-Based Spatial Unmixing Model band-by-band    #
        ####################################################################
        for band_idx in range(self.C_tp.shape[2]):
            lower_bound = np.min(delta_C_pad[:, :, band_idx])
            upper_bound = np.max(delta_C_pad[:, :, band_idx])

            SU_prediction = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=self.C_tp.dtype)

            ###########################################################
            #              1. Object-level unmxing (OL-U)             #
            ###########################################################
            for row_idx in range(self.C_tp.shape[0]):
                for col_idx in range(self.C_tp.shape[1]):
                    C_pixels_win = delta_C_pad[row_idx:row_idx + self.win_size, 
                                               col_idx:col_idx + self.win_size,
                                               band_idx]
                    
                    C_fractions_win = C_fractions_pad[row_idx:row_idx + self.win_size,
                                                      col_idx:col_idx + self.win_size, :]
                    
                    #Unmix the local window
                    F_values = self.unmix_window(C_pixels_win.flatten(),
                                                 C_fractions_win.reshape(C_pixels_win.shape[0] * C_pixels_win.shape[1],
                                                                         self.class_num),
                                                                         lower_bound, upper_bound)
                    
                    #Get the class information of the central coarse pixel of the window
                    C_classes = self.F_tb_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                                                col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor]
                    
                    #Assign the reflectances of the fine pixels inside current coarse pixel class by class
                    for class_idx in range(self.class_num):
                        class_mask = C_classes == class_idx
                        
                        #Create a block variable
                        r_start = row_idx * self.scale_factor
                        r_end = (row_idx + 1) * self.scale_factor
                        c_start = col_idx * self.scale_factor
                        c_end = (col_idx + 1) * self.scale_factor

                        block = SU_prediction[r_start:r_end, c_start:c_end]
                        block[class_mask] = F_values[class_idx]

                        SU_prediction[r_start:r_end,
                                      c_start:c_end] = block
                        
            for object_idx in object_indices:
                #Assign the unmixed value
                object_mask = self.F_tb_objects == object_idx
                object_value = np.mean(SU_prediction[object_mask])
                F_tp_OBSUM[:, :, band_idx][object_mask] = self.F_tb[:, :, band_idx][object_mask] + object_value
            
            print(f"Finished initial prediction of band {band_idx}!")

            F_tp_OBSUM[F_tp_OBSUM > self.max_val] = self.max_val
            F_tp_OBSUM[F_tp_OBSUM < self.min_val] = self.min_val

            ###########################################################
            #      2. Object-level residual compensation (OL-RC)      #
            ###########################################################
            #Calculate coarse residuals and downscale them to fine scale using bi-cubic interpolation
            C_tp_prediction = downscale_local_mean(F_tp_OBSUM[:, :, band_idx],
                                                   factors=(self.scale_factor, self.scale_factor))
            C_residuals = self.C_tp[:, :, band_idx] - C_tp_prediction
            F_residuals = resize(C_residuals, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order = 3)

            for object_idx in object_indices:
                object_mask = self.F_tb_objects == object_idx
                object_residuals = F_residuals[object_mask]

                #Residual selection
                residual_indices = object_residual_index[object_mask]
                indices = (residual_indices >= np.percentile(residual_indices, 100 - self.OL_RC_percent)).nonzero()[0]
                selected_residuals = object_residuals[indices]

                #Use weighted residuals
                selected_weights = residual_indices[indices]
                selected_weights = selected_weights / np.sum(selected_weights)
                residual = np.sum(selected_weights * selected_residuals)

                #Assign the predicted residual
                F_tp_OBSUM[:, :, band_idx][object_mask] = (
                    F_tp_OBSUM[:, :, band_idx][object_mask] + np.int32(residual)
                ) 
            
            print(f"Finished object-level residual compensation of band {band_idx}!")

            F_tp_OBSUM[F_tp_OBSUM > self.max_val] = self.max_val
            F_tp_OBSUM[F_tp_OBSUM < self.min_val] = self.min_val

            ###########################################################
            #      3. Pixel-level residual compensation (PL-RC)       #
            ###########################################################
            C_tp_prediction = downscale_local_mean(F_tp_OBSUM[:, :, band_idx],
                                                   factors=(self.scale_factor, self.scale_factor))
            C_residuals = self.C_tp[:, :, band_idx] - C_tp_prediction
            F_residuals = resize(C_residuals, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order = 3)
            F_residuals_pad = np.pad(F_residuals,
                                     pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                                (self.similar_win_size // 2, self.similar_win_size // 2)),
                                                mode = "reflect")
            
            for row_idx in range(F_residuals.shape[0]):
                for col_idx in range(F_residuals.shape[1]):
                    neighbor_pixel_residuals = F_residuals_pad[row_idx:row_idx + self.similar_win_size,
                                                               col_idx:col_idx + self.similar_win_size]
                    #Similar pixels
                    similar_indices = F_tb_similar_indices[row_idx, col_idx, :]
                    similar_residuals = neighbor_pixel_residuals.flatten()[similar_indices]
                    similar_weights = F_tb_similar_weights[row_idx, col_idx, :]

                    #Use weighted residuals
                    residual = np.sum(similar_residuals * similar_weights)

                    #Assign the predicted residual
                    F_tp_OBSUM[row_idx, col_idx, band_idx] = (
                        F_tp_OBSUM[row_idx, col_idx, band_idx] + np.int32(residual)
                    )
            
            print(f"Finished final prediction of band {band_idx}!")

            F_tp_OBSUM[F_tp_OBSUM > self.max_val] = self.max_val
            F_tp_OBSUM[F_tp_OBSUM < self.min_val] = self.min_val

        return F_tp_OBSUM

    






    
    
                
        



        