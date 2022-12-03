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
   
code requirement(s):
    install patchify:  "pip install patchify"
    see more information here: https://pypi.org/project/patchify/https://pypi.org/project/patchify/
"""

from patchify import patchify, unpatchify
import numpy as np
import logging

logger = logging.getLogger(__name__)


def patch(image, z_axis, x_axis, y_axis):
    """
    slices 2D or 3D image volume into 2D or 3D subvolumes of specific shape.

    Parameters
    ----------
    image : Array of uint8
        The image file to slice into subvolumes. If array is not int8, code auto converts to int8..
    z_axis : int
        Intended subvolume thickness or number of stacks of 2D images. For 2D images, this will be "1".
    x_axis : int
        Intended subvolume height.
    y_axis : int
        Intended subvolume width.

    Raises
    ------
    ValueError
        A 2D or 3D patch is expected with sizes of subvolume less than main volume.

    Returns
    -------
    collated_patches : Array of uint8
        Subvolumes stcaked into a single 3D array.

    """

    if len(image.shape) < 2:
        raise ValueError("A 2D or 3D array is expected.")

    if len(image.shape) > 3:
        raise ValueError("A 2D or 3D array is expected.")

    if len(image.shape) == 2:
        logging.debug("2D array detected...expanding dimension to 3D array.")
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))

        if z_axis > image.shape[0]:
            logging.debug(
                "Input image is 2D but with 3D slicing subvolume inputs... input error on z-axis autocorrected to 2D."
            )
            z_axis = image.shape[0]

    if y_axis > image.shape[-1]:
        raise ValueError(
            "Width of the patch is more than the width of the actual image being sliced."
        )

    if x_axis > image.shape[-2]:
        raise ValueError(
            "Height of the patch is more than the height of the actual image being sliced."
        )

    if z_axis > image.shape[-3]:
        raise ValueError(
            "Thickness of the patch is more than the thickness of the actual image being sliced."
        )

    logging.debug("Converting input image to int8.")
    image = image.astype(np.uint8)

    logging.debug("Extracting and calculating parameters from inputs.")
    new_z_axis = image.shape[0]
    new_z_axis = new_z_axis / z_axis
    new_z_axis = int(np.ceil(new_z_axis))
    new_z_axis = new_z_axis * z_axis

    new_x_axis = image.shape[1]
    new_x_axis = new_x_axis / x_axis
    new_x_axis = int(np.ceil(new_x_axis))
    new_x_axis = new_x_axis * x_axis

    new_y_axis = image.shape[2]
    new_y_axis = new_y_axis / y_axis
    new_y_axis = int(np.ceil(new_y_axis))
    new_y_axis = new_y_axis * y_axis

    logging.debug("Padding image with white edges to enable perfect slices.")
    padded_img = np.full((new_z_axis, new_x_axis, new_y_axis), 255)
    padded_img = padded_img.astype(np.uint8)
    padded_img[:image.shape[0], :image.shape[1], :image.shape[2]] = image

    logging.debug("Slicing into subvolumes.")
    patched_img = patchify(padded_img, (z_axis, x_axis, y_axis),
                           step=(z_axis, x_axis, y_axis))

    logging.debug("Stacking subvolumes.")
    collated_patches = []
    for i in range(patched_img.shape[0]):
        for j in range(patched_img.shape[1]):
            for k in range(patched_img.shape[2]):

                single_patch = patched_img[i, j, k, :, :, :]
                collated_patches.append(single_patch)

    logging.debug("Converting output into array.")
    collated_patches = np.array(collated_patches)

    if z_axis == 1:

        logging.debug(
            "2D slices subvolume detected. Adjusting output array as appropiate."
        )
        collated_patches = np.reshape(
            collated_patches,
            (collated_patches.shape[0], collated_patches.shape[2],
             collated_patches.shape[3]))
    return collated_patches


def unpatch(patched_img, large_img_dim):
    """
    Combines  2D or 3D image subvolumes into full 2D or 3D volumes of original shape.

    Parameters
    ----------
    patched_img : Array of uint8
        The image file containing slices to combine into volume.
    large_img_dim : list
        The shape of the original image file. Expecting 2D (2 numbers seperated by comma) or 3D array (3 numbers seperated by comma).

    Returns
    -------
    recontsructed_image : Array of uint8
        A 3D array of the combined subvolume.

    """

    if len(patched_img.shape) == 3:
        logging.debug(
            "3D array detected for patches...expanding dimension to 4D array.")
        patched_img = np.reshape(patched_img,
                                 (patched_img.shape[0], 1,
                                  patched_img.shape[1], patched_img.shape[2]))

    check2D = 0
    if len(large_img_dim) == 2:
        logging.debug(
            "2D array detected for original image...expanding dimension to 3D array."
        )
        check2D = 1
        large_img_dim = [1, large_img_dim[0], large_img_dim[1]]

    logging.debug("Converting input images to int8.")
    patched_img = patched_img.astype(np.uint8)

    logging.debug("Extracting and calculating parameters from inputs.")
    z_axis = int(np.ceil(large_img_dim[0] / patched_img.shape[1]))
    x_axis = int(np.ceil(large_img_dim[1] / patched_img.shape[2]))
    y_axis = int(np.ceil(large_img_dim[2] / patched_img.shape[3]))

    logging.debug("Reshaping image.")
    recontsructed_image = np.reshape(
        patched_img, (z_axis, x_axis, y_axis, patched_img.shape[1],
                      patched_img.shape[2], patched_img.shape[3]))

    logging.debug("Combining slices")
    recontsructed_image = unpatchify(
        recontsructed_image,
        (patched_img.shape[1] * z_axis, patched_img.shape[2] * x_axis,
         patched_img.shape[3] * y_axis))

    logging.debug("Removing padding slices")
    recontsructed_image = recontsructed_image[:large_img_dim[
        0], :large_img_dim[1], :large_img_dim[2]]

    if check2D == 1:
        logging.debug("2D array detected...expanding dimension to 3D array.")
        recontsructed_image = np.reshape(
            recontsructed_image,
            (recontsructed_image.shape[1], recontsructed_image.shape[2]))
    return recontsructed_image
