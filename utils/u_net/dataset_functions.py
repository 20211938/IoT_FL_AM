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


def get_augmentation_factor_by_defect_type_ratio_for_unet(
    defect_type_ratio,
    very_high_threshold=0.20,   # 20% 이상
    high_threshold=0.10,        # 10% 이상
    mid_high_threshold=0.05,    # 5% 이상
    mid_threshold=0.02,         # 2% 이상
    low_threshold=0.01,         # 1% 이상
    very_high_aug_factor=1,     # 매우 높은 비율: 1배
    high_aug_factor=2,          # 높은 비율: 2배
    mid_high_aug_factor=3,      # 중간-높은 비율: 3배
    mid_aug_factor=4,           # 중간 비율: 4배
    low_aug_factor=5,           # 낮은 비율: 5배
    very_low_aug_factor=5       # 매우 낮은 비율: 5배 (상한 고정)
):
    '''
    U-Net 세그멘테이션용으로 보수적으로 조정한 augmentation_factor 계산 함수.
    - 결함 비율이 높을수록 augmentation_factor를 작게 유지
    - 가장 희귀한 결함도 최대 5배까지만 증강
    '''
    if defect_type_ratio >= very_high_threshold:
        return very_high_aug_factor
    elif defect_type_ratio >= high_threshold:
        return high_aug_factor
    elif defect_type_ratio >= mid_high_threshold:
        return mid_high_aug_factor
    elif defect_type_ratio >= mid_threshold:
        return mid_aug_factor
    elif defect_type_ratio >= low_threshold:
        return low_aug_factor
    else:
        return very_low_aug_factor


def compute_defect_type_ratios_for_unet(npyPath, clientIdentifierDict):
    '''
    전체 마스크에서 결함 유형별 픽셀 비율 계산 (U-Net용).
    0, 1 레이블은 정상/배경으로 보고 제외하고 나머지만 결함 타입으로 집계.
    '''
    defect_type_pixel_counts = {}
    total_defect_pixels = 0

    for clientID, file_list in clientIdentifierDict.items():
        for file_name in file_list:
            npy_file = os.path.join(npyPath, file_name + '.npy') if not npyPath.endswith('.npy') else npyPath
            if not os.path.exists(npy_file):
                continue
            try:
                mask = np.load(npy_file)
            except Exception:
                continue

            unique_vals, counts = np.unique(mask, return_counts=True)
            for v, c in zip(unique_vals, counts):
                if v in [0, 1]:
                    continue
                defect_type_pixel_counts[int(v)] = defect_type_pixel_counts.get(int(v), 0) + int(c)
                total_defect_pixels += int(c)

    if total_defect_pixels > 0:
        defect_type_ratios = {k: v / total_defect_pixels for k, v in defect_type_pixel_counts.items()}
    else:
        defect_type_ratios = {k: 0.0 for k in defect_type_pixel_counts.keys()}

    print("U-Net용 결함 유형별 픽셀 비율:", defect_type_ratios)
    return defect_type_ratios


def random_geometric_augment_unet(image_tiles, mask_tiles):
    '''
    U-Net용 간단 기하학 증강 함수.
    - 0, 90, 180, 270도 중 하나로 회전
    - 좌우/상하 플립을 랜덤하게 적용
    image_tiles: (N, H, W, 2)
    mask_tiles : (N, H, W)
    '''
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    flip_lr = tf.less(tf.random.uniform([]), 0.5)
    flip_ud = tf.less(tf.random.uniform([]), 0.5)

    imgs = image_tiles
    masks = mask_tiles

    # 회전
    imgs = tf.image.rot90(imgs, k=k)
    masks = tf.image.rot90(tf.expand_dims(masks, -1), k=k)
    masks = tf.squeeze(masks, -1)

    # 플립
    def _maybe_flip_lr(x):
        return tf.cond(flip_lr, lambda: tf.image.flip_left_right(x), lambda: x)

    def _maybe_flip_ud(x):
        return tf.cond(flip_ud, lambda: tf.image.flip_up_down(x), lambda: x)

    imgs = _maybe_flip_lr(imgs)
    masks = _maybe_flip_lr(tf.expand_dims(masks, -1))
    masks = tf.squeeze(masks, -1)

    imgs = _maybe_flip_ud(imgs)
    masks = _maybe_flip_ud(tf.expand_dims(masks, -1))
    masks = tf.squeeze(masks, -1)

    return imgs, masks


def create_dataset_with_augmentation(clientIdentifierDict,
                                     imagePath0,
                                     imagePath1,
                                     npyPath,
                                     tileSize=128):
    '''
    결함 비율 기반 보수적 증강을 포함한 U-Net용 데이터셋 생성 함수.

    동작 개요:
    1) 전체 마스크(npy)를 훑어서 결함 타입별 픽셀 비율 계산
    2) 각 파일에서 주요 결함 타입을 찾고, 그 타입의 비율에 따라 augmentation_factor 결정
    3) preprocess_image로 생성된 타일을 augmentation_factor만큼 랜덤 기하학 증강하여 증가

    Returns:
    - datasetImageDict: {clientID: (nTiles_aug, tileSize, tileSize, 2)}
    - datasetMaskDict : {clientID: (nTiles_aug, tileSize, tileSize)}
    '''
    datasetImageDict, datasetMaskDict = {}, {}

    # 1) 전체 결함 비율 계산
    defect_type_ratios = compute_defect_type_ratios_for_unet(npyPath, clientIdentifierDict)

    for clientID, file_list in clientIdentifierDict.items():
        print(f'\n{clientID} (with augmentation)...')
        clientImages_list = []
        clientMasks_list = []

        total_files = len(file_list)

        for idx, fileName in enumerate(file_list):
            try:
                im0_path = imagePath0 + fileName + '.jpg'
                im1_path = imagePath1 + fileName + '.jpg'
                npy_file = npyPath + fileName + '.npy'

                if (not os.path.exists(im0_path) or
                        not os.path.exists(im1_path) or
                        not os.path.exists(npy_file)):
                    continue

                im0 = Image.open(im0_path)
                im1 = Image.open(im1_path)
                raw_mask = np.load(npy_file)

                # 이 파일의 주요 결함 타입 계산 (0,1 제외)
                defect_pixels = raw_mask[~np.isin(raw_mask, [0, 1])]
                if defect_pixels.size == 0:
                    main_defect_type = None
                    defect_ratio = 0.0
                else:
                    unique_vals, counts = np.unique(defect_pixels, return_counts=True)
                    main_defect_type = int(unique_vals[np.argmax(counts)])
                    defect_ratio = defect_type_ratios.get(main_defect_type, 0.0)

                augmentation_factor = get_augmentation_factor_by_defect_type_ratio_for_unet(defect_ratio)
                augmentation_factor = max(1, int(augmentation_factor))

                # 기본 타일 생성
                base_tiles_img, base_tiles_mask = preprocess_image(im0, im1, raw_mask, tileSize)

                # augmentation_factor 만큼 증강 (1회는 원본)
                all_imgs = [base_tiles_img]
                all_masks = [base_tiles_mask]

                for _ in range(augmentation_factor - 1):
                    aug_imgs, aug_masks = random_geometric_augment_unet(base_tiles_img, base_tiles_mask)
                    all_imgs.append(aug_imgs)
                    all_masks.append(aug_masks)

                all_imgs = tf.concat(all_imgs, axis=0)
                all_masks = tf.concat(all_masks, axis=0)

                clientImages_list.append(all_imgs)
                clientMasks_list.append(all_masks)

                if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                    print(f'  처리 중: {idx + 1}/{total_files} 파일 완료 '
                          f'(현재 타일 수: {sum(t.shape[0] for t in clientImages_list)})')

            except Exception as e:
                print(f'  경고: {fileName} 증강 처리 중 오류 발생: {e}')
                continue

        if len(clientImages_list) == 0:
            print(f'  경고: {clientID}에 유효한(또는 증강 가능한) 이미지가 없습니다.')
            continue

        clientImages = tf.concat(clientImages_list, axis=0)
        clientMasks = tf.concat(clientMasks_list, axis=0)

        print(f'{clientID}: 최종 타일 수 (증강 포함) = {clientImages.shape[0]}')
        datasetImageDict[clientID] = clientImages
        datasetMaskDict[clientID] = clientMasks

    return datasetImageDict, datasetMaskDict