# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:41:15 2022

@author: 
    Buki: olubukola.ishola@okstate.edu
    Data and Computational Geoscientist

about:
    patchify_BukiMod is a modification to patchify to easily slice image data into 2D or 3D subvolumes of specific shape.
    It can also put the slices back into the original image.
    This is a useful tool in preparing image data for use in machine learning deployment. 
    
credit: 
    Patchify: https://pypi.org/project/patchify/
    DigitalSreeni: https://www.youtube.com/c/DigitalSreeni/
   
"""
"""
code requirement(s):
    install patchify:  "pip install patchify"
    see more information here: https://pypi.org/project/patchify/https://pypi.org/project/patchify/
   
"""

from patchify import patchify, unpatchify
import numpy as np
import sys

def patch(image,z_axis,x_axis,y_axis):
    
    '''
    Description:
        slices 2D or 3D image volume into 2D or 3D subvolumes of specific shape.
    
    Parameters: 
        image (Array of uint8): the image file to slice into subvolumes. If array is not int8, code auto converts to int8.
        z_axis (int): Intended subvolume thickness or number of stacks of 2D images. For 2D images, this will be "1"
        x_axis (int): Intended subvolume height.
        y_axis (int): Intended subvolume width
    
    Returns:
        (Array of uint8): Subvolumes stcaked into a single 3D array.
    '''
    if  len(image.shape)<2:
        print('Error::::::::::::: A 2D or 3D array is expected')
        sys.exit()  

    if  len(image.shape)>3:
        print('Error::::::::::::: A 2D or 3D array is expected')
        sys.exit()  
        
    if len(image.shape)==2:
        print('2D array detected...expanding dimension to 3D array...', flush=True)
        image = np.reshape(image,(1, image.shape[0], image.shape[1]))
     
        if z_axis > image.shape[0]:
            print('Input image is 2D but with 3D slicing subvolume inputs... input error on z-axis autocorrected to 2D...', flush=True)
            z_axis = image.shape[0]

    if  y_axis >  image.shape[-1]:
        print('Error:::::::::::::width of the patch is more than the width of the actual image being sliced')
        sys.exit()

    if  x_axis >  image.shape[-2]:
        print('Error:::::::::::::height of the patch is more than the height of the actual image being sliced')
        sys.exit()
        
    if  z_axis >  image.shape[-3]:
        print('Error:::::::::::::thickness of the patch is more than the thickness of the actual image being sliced')
        sys.exit()        
                    
    print('Converting input image to int8...', flush=True)
    image = image.astype(np.uint8)
    
    print('Extracting and calculating parameters from inputs...', flush=True)
    z = image.shape[0]
    z = z/z_axis
    z = int(np.ceil(z))
    z = z*z_axis
    
    x = image.shape[1]
    x = x/x_axis
    x = int(np.ceil(x))
    x = x*x_axis
    
    y = image.shape[2]
    y = y/y_axis
    y = int(np.ceil(y))
    y = y*y_axis

    print('Padding image with white edges to enable perfect slices...', flush=True)
    padded_img = np.full((z,x,y),255)
    padded_img = padded_img.astype(np.uint8)
    padded_img[:image.shape[0],:image.shape[1],:image.shape[2]] = image
    
    print('Slicing into subvolumes...', flush=True)
    patches_img = patchify(padded_img,(z_axis,x_axis,y_axis), step=(z_axis,x_axis,y_axis))
    
    print('Stacking subvolumes...', flush=True)
    collated = []
    for i in range(patches_img.shape[0]):
        for j in range (patches_img.shape[1]):
            for k in range (patches_img.shape[2]):
        
                single_patch = patches_img[i,j,k,:,:,:]
                collated.append(single_patch)

    print('Converting output into array...', flush=True)
    collated = np.array(collated)
    
    if z_axis == 1:
        print('2D slices subvolume detected. Adjusting output array sd appropiate...', flush=True)
        collated = np.reshape(collated,(collated.shape[0], collated.shape[2], collated.shape[3]))
    return collated

def unpatch(patched_img, large_img_dim):
    
    '''
    Description:
        Combines  2D or 3D image subvolumes into full 2D or 3D volumes of original shape.
    
    Parameters: 
        patched_img (Array of uint8): the image file containing slices to combine into volume.
        large_img_dim (list): The shape of the original image file. Expecting 2D (2 numbers seperated by comma) or 3D array (3 numbers seperated by comma) 
        
    Returns:
        (Array of uint8): A 3D array of the combined subvolume.
        Note: If you get this error (cannot reshape array of size 4718592 into shape ....) please check the large_img_dim, its likely wrong.
    '''
    
    if len(patched_img.shape) == 3:
        print('3D array detected for patches...expanding dimension to 4D array', flush=True)
        patched_img = np.reshape(patched_img,(patched_img.shape[0], 1, patched_img.shape[1], patched_img.shape[2]))
    
    check2D = 0    
    if len(large_img_dim)==2:
        print('2D array detected for original image...expanding dimension to 3D array...', flush=True)
        check2D = 1
        large_img_dim = [1, large_img_dim[0], large_img_dim[1]]
        
    print('Converting input images to int8...', flush=True)
    patched_img = patched_img.astype(np.uint8)
    
    print('Extracting and calculating parameters from inputs...', flush=True)
    z_axis = int(np.ceil(large_img_dim[0]/patched_img.shape[1]))
    x_axis = int(np.ceil(large_img_dim[1]/patched_img.shape[2]))
    y_axis = int(np.ceil(large_img_dim[2]/patched_img.shape[3]))
    
    print('Reshaping image...', flush=True)
    recontsructed_image = np.reshape(patched_img,(z_axis,x_axis,y_axis,patched_img.shape[1],patched_img.shape[2],patched_img.shape[3]))
    
    print('Combining slices...', flush=True) 
    recontsructed_image = unpatchify(recontsructed_image, (patched_img.shape[1]*z_axis,patched_img.shape[2]*x_axis,patched_img.shape[3]*y_axis))
    
    print('Removing padding slices...', flush=True)
    recontsructed_image = recontsructed_image[:large_img_dim[0],:large_img_dim[1],:large_img_dim[2]]
    
    if check2D == 1:
        print('2D array detected...expanding dimension to 3D array...', flush=True)
        recontsructed_image = np.reshape(recontsructed_image,(recontsructed_image.shape[1], recontsructed_image.shape[2]))    
    
    return recontsructed_image
