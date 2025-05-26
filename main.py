import numpy as np
from csv_processor import CSVProcessor
from crypto_processor import CryptographicProcessor
from data_encryptor import DataEncryptor
from linear_interpolator import LinearInterpolator
from logistic_regression import LogisticRegression


def main():
    """메인 실행 함수"""
    print("=== PiHEAAN 센서 데이터 처리 시스템 ===")
    
    # 로지스틱 회귀 파라미터 
    initial_weights = np.random.randn(300) * 0.0001
    initial_bias = 0.0
    threshold = 0.5
    
    # CSV 파일 경로
    csv_path = "./dataset/df_final_timestamp.csv"
    
    try:
        # 1. 각 클래스 초기화
        print("\n1. 시스템 초기화...")
        csv_processor = CSVProcessor()
        crypto_processor = CryptographicProcessor()
        encryptor = DataEncryptor(crypto_processor)
        interpolator = LinearInterpolator(crypto_processor)
        logistic_regression = LogisticRegression(crypto_processor)
        
        # 2. CSV 데이터 로드
        print("\n2. CSV 데이터 로드...")
        sensor_data, nan_masks, y_labels, timestamps = csv_processor.load_csv_to_sensor_data(
            csv_path,
            label_column=None,
            timestamp_column='SensorTime',
            missing_value=-999
        )
        
        # 데이터 확인
        print(f"센서 데이터 형태: {sensor_data.shape}")
        print(f"NaN 마스크 형태: {nan_masks.shape}")
        print(f"타임스탬프 개수: {len(timestamps)}")
        
        # 라벨이 없으면 랜덤 생성
        if y_labels is None:
            y_labels = np.random.choice([0, 1], size=sensor_data.shape[1], p=[0.7, 0.3])
            print("랜덤 라벨 생성 완료")
        
        # 3. 데이터 암호화
        print("\n3. 센서 데이터 암호화...")
        encrypted_data, plaintext_nan_masks = encryptor.encrypt_sensor_data(sensor_data, nan_masks)
        print(f"암호화된 센서 데이터 개수: {len(encrypted_data)}")
        
        # 4. 선형 보간
        print("\n4. 선형 보간 수행...")
        interpolated_data = interpolator.interpolate_all_sensors(encrypted_data, plaintext_nan_masks)
        print(f"보간된 센서 데이터 개수: {len(interpolated_data)}")
        
        # 5. 로지스틱 회귀 학습 및 예측
        print("\n5. 로지스틱 회귀 학습 및 예측...")
        anomaly_results = logistic_regression.train_and_predict(
            interpolated_data, 
            initial_weights, 
            initial_bias, 
            timestamps,
            y_labels, 
            learning_rate=0.001, 
            num_steps=10, 
            threshold=threshold
        )
        
        # 6. 결과 출력 및 평가
        print("\n6. 결과 분석...")
        evaluate_results(anomaly_results, y_labels)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


def evaluate_results(anomaly_results, y_labels):
    """결과 평가 및 출력"""
    print("\n=== 최종 이상치 탐지 결과 ===")
    
    # 처음 20개 결과 출력
    for i, (timestamp, is_anomaly) in enumerate(anomaly_results):
        if i < len(y_labels):
            actual = "ANOMALY" if y_labels[i] == 1 else "NORMAL"
            predicted = "ANOMALY" if is_anomaly else "NORMAL"
            match = "✓" if (y_labels[i] == 1) == is_anomaly else "✗"
            print(f"{i:3d}: {timestamp} - Actual: {actual:7s}, Predicted: {predicted:7s} {match}")
    
    # 정확도 계산
    if len(anomaly_results) > 0:
        correct = 0
        total = min(len(anomaly_results), len(y_labels))
        
        for i in range(total):
            predicted_anomaly = anomaly_results[i][1]
            actual_anomaly = y_labels[i] == 1
            if predicted_anomaly == actual_anomaly:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\n전체 정확도: {accuracy:.2f}% ({correct}/{total})")
        
        # 추가 통계
        true_positives = sum(1 for i in range(total) 
                           if anomaly_results[i][1] and y_labels[i] == 1)
        false_positives = sum(1 for i in range(total) 
                            if anomaly_results[i][1] and y_labels[i] == 0)
        true_negatives = sum(1 for i in range(total) 
                           if not anomaly_results[i][1] and y_labels[i] == 0)
        false_negatives = sum(1 for i in range(total) 
                            if not anomaly_results[i][1] and y_labels[i] == 1)
        
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        
        # 정밀도와 재현율
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            print(f"정밀도 (Precision): {precision:.4f}")
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            print(f"재현율 (Recall): {recall:.4f}")


if __name__ == "__main__":
    main()