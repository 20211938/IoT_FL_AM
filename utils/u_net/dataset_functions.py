import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import os

from utils.u_net.image_processing import *


def create_dataset(clientIdentifierDict, imagePath0, imagePath1, npyPath, tileSize = 128):
    '''
    Creates a dataset of clients using the post-spreading and post-fusion images
    based on clientIdentifierDict.
    
    Returns:
    - datasetImageDict: Dictionary of tiled images keyed by clientID with dimension (nTiles, tileSize, tileSize, 2)
    - datasetMaskDict : Dictionary of tiled masks keyed by clientID with dimension (nTiles, tileSize, tileSize)
    
    '''
    datasetImageDict, datasetMaskDict = {}, {}
    
    for clientID in clientIdentifierDict:
        fileNum = 0
        clientImages, clientMasks = None, None
        
        print(f'\n{clientID}...')
        total_files = len(clientIdentifierDict[clientID])
        
        for idx, fileName in enumerate(clientIdentifierDict[clientID]):
            try:
                im0 = Image.open(imagePath0 + fileName + '.jpg')
                im1 = Image.open(imagePath1 + fileName + '.jpg')
                segmentationMask = np.load(npyPath + fileName + '.npy')
                
                splitImages, splitSegmentationMask = preprocess_image(im0, im1,
                                                                      segmentationMask,
                                                                      tileSize)
      
                
                if fileNum == 0:
                    clientImages, clientMasks = splitImages, splitSegmentationMask
                else:
                    clientImages = tf.concat([clientImages, splitImages], 0)
                    clientMasks = tf.concat([clientMasks, splitSegmentationMask], 0)
                fileNum += 1
                
                # 진행 상황 출력 (10개마다 또는 마지막 파일)
                if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                    print(f'  처리 중: {idx + 1}/{total_files} 파일 완료 (현재 타일 수: {clientImages.shape[0]})')
                    
            except Exception as e:
                print(f'  경고: {fileName} 처리 중 오류 발생: {e}')
                continue
        
        print(f'Contains {fileNum} images...')
        if clientImages is not None:
            print('Tiled Image Tensor Shape: ', clientImages.shape)
            print('Tiled Mask Shape: ', clientMasks.shape)
            datasetImageDict[clientID] = clientImages
            datasetMaskDict[clientID] = clientMasks
        else:
            print(f'  경고: {clientID}에 유효한 이미지가 없습니다.')
    
    return datasetImageDict, datasetMaskDict


def unwrap_client_data(imageDict, maskDict, clientList):
    '''
    Takes all clients in clientList and combines their data into a single tensor.
    
    '''
    unwrappedImages = imageDict[clientList[0]]
    unwrappedMasks = maskDict[clientList[0]]
    for i in range(1,len(clientList)):
        unwrappedImages = tf.concat([unwrappedImages, imageDict[clientList[i]]], 0)
        unwrappedMasks = tf.concat([unwrappedMasks, maskDict[clientList[i]]], 0)
    return unwrappedImages, unwrappedMasks


def create_dataset_lazy(clientIdentifierDict, imagePath0, imagePath1, npyPath, tileSize=128):
    '''
    Creates a dictionary of file paths for each client (lazy loading).
    Does not load data into memory.
    
    Returns:
    - filePathDict: Dictionary of file paths keyed by clientID
    - imagePath0, imagePath1, npyPath: Paths for image loading
    - tileSize: Tile size for preprocessing
    '''
    filePathDict = {}
    
    for clientID in clientIdentifierDict:
        filePathDict[clientID] = clientIdentifierDict[clientID]
        print(f'{clientID}: {len(clientIdentifierDict[clientID])}개 파일')
    
    return filePathDict, imagePath0, imagePath1, npyPath, tileSize


def load_and_preprocess_file(fileName, imagePath0, imagePath1, npyPath, tileSize):
    '''
    Loads and preprocesses a single file.
    This function will be called by tf.data.Dataset.
    '''
    try:
        im0 = Image.open(imagePath0 + fileName + '.jpg')
        im1 = Image.open(imagePath1 + fileName + '.jpg')
        segmentationMask = np.load(npyPath + fileName + '.npy')
        
        splitImages, splitSegmentationMask = preprocess_image(im0, im1,
                                                              segmentationMask,
                                                              tileSize)
        

        
        return splitImages, splitSegmentationMask
    except Exception as e:
        print(f'  경고: {fileName} 처리 중 오류 발생: {e}')
        # Return empty tensors with correct shape on error
        empty_images = tf.zeros((0, tileSize, tileSize, 2), dtype=tf.float32)
        empty_masks = tf.zeros((0, tileSize, tileSize), dtype=tf.int32)
        return empty_images, empty_masks


def create_tf_dataset(fileList, imagePath0, imagePath1, npyPath, tileSize, batch_size=32, shuffle=True):
    '''
    Creates a tf.data.Dataset from a list of file names.
    Data is loaded lazily during training.
    Returns dataset and dataset size.
    '''
    # Convert file list to tensor
    fileListTensor = tf.constant(fileList, dtype=tf.string)
    
    # Create dataset from file names
    dataset = tf.data.Dataset.from_tensor_slices(fileListTensor)
    
    # Wrapper function for loading files (captures paths in closure)
    def load_file_wrapper(fileName):
        # Decode the tensor to string
        fileName_str = fileName.numpy().decode('utf-8') if isinstance(fileName, tf.Tensor) else fileName
        images, masks = load_and_preprocess_file(fileName_str, imagePath0, imagePath1, npyPath, tileSize)
        return images, masks
    
    # Use py_function to call Python function (for file I/O)
    dataset = dataset.map(
        lambda x: tf.py_function(
            func=load_file_wrapper,
            inp=[x],
            Tout=(tf.float32, tf.int32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set output shapes for the dataset (important for batching)
    dataset = dataset.map(
        lambda images, masks: (tf.ensure_shape(images, [None, tileSize, tileSize, 2]), 
                              tf.ensure_shape(masks, [None, tileSize, tileSize])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Unbatch to get individual tiles
    dataset = dataset.flat_map(lambda images, masks: tf.data.Dataset.from_tensor_slices((images, masks)))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Calculate dataset size (number of tiles)
    dataset_size = get_dataset_size(fileList, imagePath0, imagePath1, npyPath, tileSize)
    
    return dataset, dataset_size


def get_dataset_size(fileList, imagePath0, imagePath1, npyPath, tileSize):
    '''
    Calculates the total number of tiles for a client without loading all data.
    This is needed for weighted averaging in federated learning.
    '''
    total_tiles = 0
    for fileName in fileList:
        try:
            im0 = Image.open(imagePath0 + fileName + '.jpg')
            segmentationMask = np.load(npyPath + fileName + '.npy')
            
            # Calculate number of tiles
            rightcrop = im0.size[0] // tileSize * tileSize
            bottomcrop = im0.size[1] // tileSize * tileSize
            n_tiles = (rightcrop // tileSize) * (bottomcrop // tileSize)
    
            
            total_tiles += n_tiles
        except Exception as e:
            continue
    
    return total_tiles