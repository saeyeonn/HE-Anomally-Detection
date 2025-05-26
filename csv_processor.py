import pandas as pd
import numpy as np


class CSVProcessor:
    """CSV 파일 로딩 및 전처리를 담당하는 클래스"""
    
    def __init__(self):
        self.SENSOR_COUNT = 5
        
    def load_csv_to_sensor_data(self, csv_path, label_column=None, timestamp_column='SensorTime', missing_value=-999):
        """
        CSV 파일을 로드해서 센서 데이터 형태로 변환
        
        Args:
            csv_path: CSV 파일 경로
            label_column: 라벨 컬럼명 (제외할 컬럼)
            timestamp_column: 타임스탬프 컬럼명 (기본값: 'SensorTime')
            missing_value: 결측값을 나타내는 값 (기본값: -999)
        
        Returns:
            sensor_data: (n_sensors, n_samples) 형태의 센서 데이터
            nan_masks: (n_sensors, n_samples) 형태의 NaN 마스크
            y_labels: 라벨 배열 (있는 경우)
            timestamps: 타임스탬프 리스트
        """
        print(f"Loading CSV: {csv_path}")
        
        # 1. CSV 파일 로드
        df = pd.read_csv(csv_path)
        print(f"Original shape: {df.shape} (rows x columns)")
        print(f"Columns: {list(df.columns)}")
        
        # 2. 첫 번째 컬럼이 인덱스인 경우 제외
        if df.columns[0] == '이름 상자' or df.iloc[0, 0] in [0, '0'] or str(df.columns[0]).lower() in ['index', 'unnamed']:
            df = df.iloc[:, 1:]  # 첫 번째 컬럼 제거
            print(f"Removed index column. New shape: {df.shape}")
        
        # 3. 센서 컬럼 선택 (Sensor0, Sensor1, Sensor2, Sensor3, Sensor4)
        sensor_columns = [col for col in df.columns if col.startswith('Sensor') and col != timestamp_column]
        print(f"Sensor columns ({len(sensor_columns)}): {sensor_columns}")
        
        # 4. sensor_data 생성 (센서 x 샘플) 형태
        sensor_data = df[sensor_columns].values.T  # 전치해서 (센서, 샘플) 형태로
        print(f"Sensor data shape: {sensor_data.shape} (sensors x samples)")
        
        # 5. -999를 NaN으로 처리
        sensor_data_with_nan = sensor_data.copy().astype(float)
        sensor_data_with_nan[sensor_data == missing_value] = np.nan
        
        # 6. nan_masks 생성 (-999 위치를 1로, 정상 데이터는 0으로)
        nan_masks = (sensor_data == missing_value).astype(int)
        print(f"Missing values (-999) count: {np.sum(nan_masks)} / {nan_masks.size}")
                
        # 7. y_labels 추출 (있는 경우)
        y_labels = None
        if label_column and label_column in df.columns:
            y_labels = df[label_column].values
            print(f"Labels shape: {y_labels.shape}")
        else:
            print("No labels found")
        
        # 8. timestamps 추출
        timestamps = None
        if timestamp_column and timestamp_column in df.columns:
            timestamps = df[timestamp_column].tolist()
            print(f"Timestamps extracted from column: {timestamp_column}")
        else:
            # 타임스탬프가 없으면 인덱스 기반으로 생성
            timestamps = [f"sample_{i:06d}" for i in range(len(df))]
            print("Generated sequential timestamps")
        
        print(f"Final: {sensor_data.shape[0]} sensors x {sensor_data.shape[1]} samples")
        print(f"Data range: [{np.min(sensor_data):.2f}, {np.max(sensor_data):.2f}]")
        
        return sensor_data, nan_masks, y_labels, timestamps