import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import os
from sklearn.model_selection import KFold
from torch.utils.data import random_split
from utils.patch_cnn.classifier import save_patch_cnn_model


def federated_averaging_defect_classifier(model, SERVER_ROUNDS, LOCAL_EPOCHS, LOCAL_BATCH_SIZE,
                                         LOCAL_LEARNING_RATE, clientIDs, imageDict, labelDict,
                                         num_classes, device=None,
                                         early_stopping_accuracy=None, freeze_epochs=3,
                                         train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    결함 유형 분류 모델용 연합학습 (FedAvg) - 분산 평가 방식
    
    Args:
        train_ratio: 학습 데이터 비율 (기본값: 0.6)
        val_ratio: 검증 데이터 비율 (기본값: 0.2)
        test_ratio: 테스트 데이터 비율 (기본값: 0.2)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # 분류용 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 각 클라이언트의 데이터 개수 계산
    nk = {}
    total_samples = 0
    for clientID in clientIDs:
        if clientID in labelDict:
            nk[clientID] = len(labelDict[clientID])
            total_samples += nk[clientID]

    # 가중치 비율 계산
    proportionsDict = {}
    for clientID in clientIDs:
        if clientID in nk:
            proportionsDict[clientID] = nk[clientID] / total_samples
        else:
            proportionsDict[clientID] = 0.0

    # 디버깅: 비율 출력
    print(f"\n클라이언트 데이터 개수 및 가중치 비율:")
    for clientID in clientIDs:
        if clientID in nk:
            print(f"  {clientID}: {nk[clientID]}개 샘플, 비율: {proportionsDict[clientID]:.4f}")
    
    lossDict = {clientID: [] for clientID in clientIDs}
    accuracyDict = {clientID: [] for clientID in clientIDs}
    testLoss = []
    testAccuracy = []
    
    # 각 클라이언트의 테스트 데이터셋 저장 (분산 평가용)
    clientTestLoaders = {}
    
    serverStateDict = deepcopy(model.state_dict())
    
    model_save_dir = 'patch_cnn_models'
    os.makedirs(model_save_dir, exist_ok=True)
    
    for round_num in range(SERVER_ROUNDS):
        # 일정 라운드 후 Backbone 해제
        if round_num == freeze_epochs:
            print(f"라운드 {round_num + 1}: Backbone 해제, 전체 모델 학습")
            model.unfreeze_backbone()
        
        print('=' * 60)
        print(f'------ Server Round {round_num} ------')
        print('=' * 60)
        
        clientStateDicts = {}
        
        for clientID in clientIDs:
            if clientID not in imageDict:
                continue
            
            print(f'\nRunning local updates for {clientID}...')
            
            clientModel = deepcopy(model)
            clientModel.load_state_dict(serverStateDict)
            clientModel = clientModel.to(device)
            clientModel.train()
            
            optimizer = optim.Adam(
                [p for p in clientModel.parameters() if p.requires_grad],
                lr=LOCAL_LEARNING_RATE
            )
            
            # 클라이언트 데이터셋 (텐서로 변환)
            clientImages = imageDict[clientID].to(device)
            clientLabels = labelDict[clientID].to(device)
            
            # Train/Validation/Test 분할 (60% train, 20% validation, 20% test)
            clientDataset = torch.utils.data.TensorDataset(clientImages, clientLabels)
            total_size = len(clientDataset)
            
            # 비율에 따라 크기 계산
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)
            test_size = total_size - train_size - val_size  # 나머지가 test
            
            # 최소 크기 보장
            if test_size < 1:
                test_size = 1
                val_size = max(1, val_size - 1)
                train_size = total_size - val_size - test_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                clientDataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드
            )
            
            trainLoader = torch.utils.data.DataLoader(
                train_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=True
            )
            valLoader = torch.utils.data.DataLoader(
                val_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=False
            )
            
            # 첫 라운드에만 테스트 데이터로더 생성 및 저장
            if round_num == 0:
                testLoader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=False
                )
                clientTestLoaders[clientID] = testLoader
            
            round_losses = []
            round_accuracies = []
            
            for epoch in range(LOCAL_EPOCHS):
                # 학습 단계
                clientModel.train()
                epoch_loss = 0.0
                
                for batch_images, batch_labels in trainLoader:
                    optimizer.zero_grad()
                    
                    outputs = clientModel(batch_images)  # (B, num_classes)
                    
                    loss = criterion(outputs, batch_labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(trainLoader)
                
                # Validation 단계 (정확도 평가)
                clientModel.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_images, batch_labels in valLoader:
                        outputs = clientModel(batch_images)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0
                
                round_losses.append(avg_loss)
                round_accuracies.append(accuracy)
                
                print(f'Epoch {epoch + 1}/{LOCAL_EPOCHS} - Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
            
            lossDict[clientID].extend(round_losses)
            accuracyDict[clientID].extend(round_accuracies)
            
            print(f'Saving local updates for {clientID}...')
            clientStateDicts[clientID] = deepcopy(clientModel.state_dict())
        
        # 서버 업데이트
        print('\nPerforming Server Update...')
        updatedServerStateDict = {}
        
        for key in serverStateDict.keys():
            param = serverStateDict[key]
            
            if param.dtype in [torch.float32, torch.float64]:
                temp = torch.zeros_like(param)
                for clientID in clientIDs:
                    if clientID in clientStateDicts:
                        temp += proportionsDict[clientID] * clientStateDicts[clientID][key]
                updatedServerStateDict[key] = temp
            elif param.dtype == torch.int64:
                if 'num_batches_tracked' in key:
                    temp = torch.zeros_like(param, dtype=torch.int64)
                    for clientID in clientIDs:
                        if clientID in clientStateDicts:
                            temp += clientStateDicts[clientID][key].to(torch.int64)
                    updatedServerStateDict[key] = temp
                else:
                    if clientIDs and clientIDs[0] in clientStateDicts:
                        updatedServerStateDict[key] = clientStateDicts[clientIDs[0]][key].clone()
            else:
                if clientIDs and clientIDs[0] in clientStateDicts:
                    updatedServerStateDict[key] = clientStateDicts[clientIDs[0]][key].clone()
        
        print('Done...')
        
        model.load_state_dict(updatedServerStateDict)
        serverStateDict = updatedServerStateDict
        
        # 분산 테스트 평가: 각 클라이언트가 자신의 테스트 데이터로 평가
        print('\nEvaluating Test Set Performance (Distributed Testing)...')
        model.eval()
        
        client_test_accuracies = {}
        client_test_losses = {}
        client_test_counts = {}
        
        with torch.no_grad():
            for clientID in clientIDs:
                if clientID not in clientTestLoaders:
                    continue
                
                testLoader = clientTestLoaders[clientID]
                test_losses = []
                test_correct = 0
                test_total = 0
                
                for batch_images, batch_labels in testLoader:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    batch_outputs = model(batch_images)
                    
                    batch_loss = criterion(batch_outputs, batch_labels)
                    test_losses.append(batch_loss.item())
                    
                    _, batch_predicted = torch.max(batch_outputs.data, 1)
                    test_total += batch_labels.size(0)
                    test_correct += (batch_predicted == batch_labels).sum().item()
                
                test_loss = sum(test_losses) / len(test_losses) if len(test_losses) > 0 else 0.0
                test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0.0
                
                client_test_accuracies[clientID] = test_accuracy
                client_test_losses[clientID] = test_loss
                client_test_counts[clientID] = test_total
                
                print(f'  {clientID}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}% ({test_total}개 샘플)')
        
        # 서버가 가중 평균 계산
        weighted_test_loss = 0.0
        weighted_test_accuracy = 0.0
        total_test_samples = 0
        
        for clientID in clientIDs:
            if clientID in client_test_accuracies:
                weight = proportionsDict.get(clientID, 0.0)
                weighted_test_loss += weight * client_test_losses[clientID]
                weighted_test_accuracy += weight * client_test_accuracies[clientID]
                total_test_samples += client_test_counts[clientID]
        
        testLoss.append(weighted_test_loss)
        testAccuracy.append(weighted_test_accuracy)
        
        print(f'\n가중 평균 테스트 성능:')
        print(f'  Test Loss: {weighted_test_loss:.4f}')
        print(f'  Test Accuracy: {weighted_test_accuracy:.2f}% (총 {total_test_samples}개 샘플)')
        print('Done...\n')
        
        # 모델 저장
        model_save_path = os.path.join(model_save_dir, f'round_{round_num}.pth')
        save_patch_cnn_model(model, model_save_path, num_classes=num_classes)
        print(f'모델이 {model_save_path}에 저장되었습니다.\n')
        
        if early_stopping_accuracy is not None and weighted_test_accuracy >= early_stopping_accuracy:
            print(f'=' * 60)
            print(f'조기 종료: 서버 정확도 {weighted_test_accuracy:.2f}%가 목표 정확도 {early_stopping_accuracy:.2f}%에 도달했습니다!')
            print(f'=' * 60)
            break
    
    return model, serverStateDict, lossDict, testLoss, accuracyDict, testAccuracy