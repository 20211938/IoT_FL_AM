import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from utils.cnn.dataset_functions import unwrap_client_data


def federated_averaging(model,
                        SERVER_ROUNDS, LOCAL_EPOCHS,
                        LOCAL_BATCH_SIZE,
                        LOCAL_LEARNING_RATE,
                        clientIDs, imageDict, labelDict,
                        testImages, testLabels,
                        device=None):
    """
    PyTorch 기반 FedAvg 알고리즘 실행
    
    Args:
        model: 전역 CNN 모델 (PyTorch)
        SERVER_ROUNDS: 서버 라운드 수
        LOCAL_EPOCHS: 로컬 에포크 수
        LOCAL_BATCH_SIZE: 로컬 배치 크기
        LOCAL_LEARNING_RATE: 로컬 학습률
        clientIDs: 클라이언트 ID 리스트
        imageDict: 클라이언트별 이미지 딕셔너리
        labelDict: 클라이언트별 레이블 딕셔너리
        testImages: 테스트 이미지 텐서
        testLabels: 테스트 레이블 텐서
        device: PyTorch device
    
    Returns:
        model: 학습된 모델
        serverStateDict: 서버 가중치
        lossDict: 클라이언트별 손실 딕셔너리
        testLoss: 테스트 손실 리스트
        accuracyDict: 클라이언트별 정확도 딕셔너리
        testAccuracy: 테스트 정확도 리스트
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    testImages = testImages.to(device)
    testLabels = testLabels.to(device)
    
    # 손실 함수 및 메트릭
    criterion = nn.CrossEntropyLoss()
    
    # 각 클라이언트의 데이터 개수 계산
    nk = [len(labelDict[clientID]) for clientID in clientIDs]
    
    # 가중 평균을 위한 비율 계산
    proportionsDict = {clientIDs[i]: nk[i] / sum(nk) for i in range(len(clientIDs))}
    
    # 손실 및 정확도 기록
    lossDict = {clientID: [] for clientID in clientIDs}
    accuracyDict = {clientID: [] for clientID in clientIDs}
    testLoss = []
    testAccuracy = []
    
    # 서버 가중치 초기화
    serverStateDict = deepcopy(model.state_dict())
    
    for round_num in range(SERVER_ROUNDS):
        print('=' * 60)
        print(f'------ Server Round {round_num} ------')
        print('=' * 60)
        
        clientStateDicts = {}
        
        # 클라이언트 업데이트
        for clientID in clientIDs:
            print(f'\nRunning local updates for {clientID}...')
            
            # 클라이언트 모델 생성 및 서버 가중치 복사
            clientModel = deepcopy(model)
            clientModel.load_state_dict(serverStateDict)
            clientModel = clientModel.to(device)
            clientModel.train()
            
            # 옵티마이저 설정
            optimizer = optim.Adam(clientModel.parameters(), lr=LOCAL_LEARNING_RATE)
            
            # 클라이언트 데이터 가져오기
            clientImages = imageDict[clientID].to(device)
            clientLabels = labelDict[clientID].to(device)
            
            # 유효하지 않은 레이블 필터링 (-1 또는 범위를 벗어난 값)
            valid_mask = (clientLabels >= 0) & (clientLabels < num_classes)
            if valid_mask.sum() == 0:
                print(f"경고: {clientID}에 유효한 데이터가 없습니다. 건너뜁니다.")
                continue
            
            clientImages = clientImages[valid_mask]
            clientLabels = clientLabels[valid_mask]
            
            # DataLoader 생성
            clientDataset = torch.utils.data.TensorDataset(clientImages, clientLabels)
            clientLoader = torch.utils.data.DataLoader(
                clientDataset, batch_size=LOCAL_BATCH_SIZE, shuffle=True
            )
            
            # 로컬 학습
            round_losses = []
            round_accuracies = []
            
            for epoch in range(LOCAL_EPOCHS):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_images, batch_labels in clientLoader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = clientModel(batch_images)
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # 통계
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                
                avg_loss = epoch_loss / len(clientLoader)
                accuracy = 100 * correct / total
                
                round_losses.append(avg_loss)
                round_accuracies.append(accuracy)
                
                print(f'Epoch {epoch + 1}/{LOCAL_EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # 평균 손실 및 정확도 저장
            lossDict[clientID].extend(round_losses)
            accuracyDict[clientID].extend(round_accuracies)
            
            print(f'Saving local updates for {clientID}...')
            clientStateDicts[clientID] = deepcopy(clientModel.state_dict())
        
        # 서버 업데이트 (가중 평균)
        print('\nPerforming Server Update...')
        updatedServerStateDict = {}
        
        for key in serverStateDict.keys():
            temp = torch.zeros_like(serverStateDict[key])
            for clientID in clientIDs:
                temp += proportionsDict[clientID] * clientStateDicts[clientID][key]
            updatedServerStateDict[key] = temp
        
        print('Done...')
        
        # 전역 모델에 가중치 할당
        print('Assigning current server state to the global model...')
        model.load_state_dict(updatedServerStateDict)
        serverStateDict = updatedServerStateDict
        
        # 테스트 세트 성능 평가
        print('Evaluating Test Set Performance...')
        model.eval()
        with torch.no_grad():
            testOutputs = model(testImages)
            test_loss = criterion(testOutputs, testLabels).item()
            _, testPredicted = torch.max(testOutputs.data, 1)
            test_total = testLabels.size(0)
            test_correct = (testPredicted == testLabels).sum().item()
            test_accuracy = 100 * test_correct / test_total
        
        testLoss.append(test_loss)
        testAccuracy.append(test_accuracy)
        
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        print('Done...\n')
    
    return model, serverStateDict, lossDict, testLoss, accuracyDict, testAccuracy