"""
동형암화 이상치 탐지 서비스
CSV 파일을 받아 동형암호 기반 이상치 탐지를 수행하는 서비스 클래스
"""

import numpy as np
from datetime import datetime

# 동형암호 관련 클래스들 임포트
from csv_processor import CSVProcessor
from crypto_processor import CryptographicProcessor
from data_encryptor import DataEncryptor
from linear_interpolator import LinearInterpolator
from logistic_regression import LogisticRegression


class HomomorphicAnomalyDetectionService:
    """동형암호 기반 이상치 탐지 서비스"""
    
    def __init__(self, config=None):
        """
        서비스 초기화
        
        Args:
            config: 설정 딕셔너리 (선택사항)
        """
        # 기본 설정
        self.config = config or {
            'initial_weights_size': 300,
            'weight_scale': 0.0001,
            'initial_bias': 0.0,
            'threshold': 0.5,
            'learning_rate': 0.001,
            'num_steps': 10,
            'missing_value': -999,
            'timestamp_column': 'SensorTime',
            'label_column': None
        }
        
        # 동형암호 관련 클래스들 초기화
        self.csv_processor = None
        self.crypto_processor = None
        self.encryptor = None
        self.interpolator = None
        self.logistic_regression = None
        
        print("HomomorphicAnomalyDetectionService 초기화 완료")
    
    def _initialize_components(self):
        """동형암호 컴포넌트들 초기화"""
        if self.csv_processor is None:
            print("동형암호 컴포넌트 초기화 중...")
            self.csv_processor = CSVProcessor()
            self.crypto_processor = CryptographicProcessor()
            self.encryptor = DataEncryptor(self.crypto_processor)
            self.interpolator = LinearInterpolator(self.crypto_processor)
            self.logistic_regression = LogisticRegression(self.crypto_processor)
            print("동형암호 컴포넌트 초기화 완료")
    
    def process_csv_file(self, file_path):
        """
        CSV 파일을 처리하여 이상치 탐지 수행
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            list: 이상치 탐지 결과 리스트
            
        Raises:
            Exception: 처리 중 오류 발생 시
        """
        print(f"=== CSV 파일 처리 시작: {file_path} ===")
        
        try:
            # 1. 컴포넌트 초기화
            self._initialize_components()
            
            # 2. CSV 데이터 로드
            print("CSV 데이터 로드 중...")
            actual_data_size, sensor_data, nan_masks, y_labels, timestamps = self._load_csv_data(file_path)
            
            # 3. 데이터 암호화
            print("센서 데이터 암호화 중...")
            encrypted_data, plaintext_nan_masks = self._encrypt_data(sensor_data, nan_masks)
            
            # 4. 선형 보간
            print("동형암호 환경에서 선형 보간 수행 중...")
            interpolated_data = self._interpolate_data(actual_data_size, encrypted_data, plaintext_nan_masks)
            
            # 5. 로지스틱 회귀 학습 및 예측
            print("동형암호 로지스틱 회귀 학습 및 예측 중...")
            anomaly_results = self._perform_anomaly_detection(
                interpolated_data, timestamps, y_labels
            )
            
            # 6. 결과 평가 (로그용)
            self._evaluate_results(anomaly_results, y_labels)
            
            print(f"처리 완료: {len(anomaly_results)}개 결과")
            return anomaly_results
            
        except Exception as e:
            print(f"CSV 처리 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _load_csv_data(self, file_path):
        """CSV 데이터 로드"""
        actual_data_size, sensor_data, nan_masks, y_labels, timestamps = self.csv_processor.load_csv_to_sensor_data(
            file_path,
            label_column=self.config['label_column'],
            timestamp_column=self.config['timestamp_column'],
            missing_value=self.config['missing_value']
        )
        
        print(f"센서 데이터 형태: {sensor_data.shape}")
        print(f"NaN 마스크 형태: {nan_masks.shape}")
        print(f"타임스탬프 개수: {len(timestamps)}")
        
        # 라벨이 없으면 랜덤 생성 (실제 환경에서는 제거 또는 수정)
        if y_labels is None:
            y_labels = np.random.choice([0, 1], size=sensor_data.shape[1], p=[0.7, 0.3])
            print("라벨이 없어 랜덤 라벨 생성")
        
        return actual_data_size, sensor_data, nan_masks, y_labels, timestamps
    
    def _encrypt_data(self, sensor_data, nan_masks):
        """데이터 암호화"""
        encrypted_data, plaintext_nan_masks = self.encryptor.encrypt_sensor_data(
            sensor_data, nan_masks
        )
        print(f"암호화된 센서 데이터 개수: {len(encrypted_data)}")
        return encrypted_data, plaintext_nan_masks
    
    def _interpolate_data(self, actual_data_size, encrypted_data, plaintext_nan_masks):
        """선형 보간 수행"""
        interpolated_data = self.interpolator.interpolate_all_sensors(
            actual_data_size, encrypted_data, plaintext_nan_masks
        )
        print(f"보간된 센서 데이터 개수: {len(interpolated_data)}")
        return interpolated_data
    
    def _perform_anomaly_detection(self, interpolated_data, timestamps, y_labels):
        """이상치 탐지 수행"""
        # 로지스틱 회귀 파라미터 생성
        initial_weights = np.random.randn(self.config['initial_weights_size']) * \
                         self.config['weight_scale']
        
        anomaly_results = self.logistic_regression.train_and_predict(
            interpolated_data, 
            initial_weights, 
            self.config['initial_bias'], 
            timestamps,
            y_labels, 
            learning_rate=self.config['learning_rate'], 
            num_steps=self.config['num_steps'], 
            threshold=self.config['threshold']
        )
        
        return anomaly_results
    
    def _evaluate_results(self, anomaly_results, y_labels):
        """결과 평가 및 출력 (로그용)"""
        print("\n=== 최종 이상치 탐지 결과 ===")
        
        # 처음 20개 결과 출력
        display_count = min(20, len(anomaly_results))
        for i in range(display_count):
            timestamp, is_anomaly = anomaly_results[i]
            if i < len(y_labels):
                actual = "ANOMALY" if y_labels[i] == 1 else "NORMAL"
                predicted = "ANOMALY" if is_anomaly else "NORMAL"
                match = "✓" if (y_labels[i] == 1) == is_anomaly else "✗"
                print(f"{i:3d}: {timestamp} - Actual: {actual:7s}, Predicted: {predicted:7s} {match}")
        
        # 정확도 계산
        if len(anomaly_results) > 0:
            self._calculate_metrics(anomaly_results, y_labels)
    
    def _calculate_metrics(self, anomaly_results, y_labels):
        """성능 지표 계산"""
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
        
    
    def update_config(self, new_config):
        """설정 업데이트"""
        self.config.update(new_config)
        print("설정이 업데이트되었습니다.")
    
    def get_config(self):
        """현재 설정 반환"""
        return self.config.copy()