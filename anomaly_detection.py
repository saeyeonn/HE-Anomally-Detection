import piheaan as heaan
from piheaan.math import approx 
import os
import pandas as pd 
import numpy as np
import math
import time


def main():
    # # 데이터 생성 
    # sensor_data = np.random.randn(5, 300)
    # print("sensor_data....")
    # print(sensor_data)
    # nan_masks = np.random.choice([0, 1], size=(5, 300), p=[0.7, 0.3])
    # print("nan_masks...")
    # print(nan_masks)
    
    # # 타임스탬프 배열 생성 - DATA_SIZE와 맞춤 (300개)
    # timestamps = [f"2024-01-01T{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}" for i in range(300)]
    # print("timestamps...")
    # print(timestamps[:10])  # 처음 10개만 출력
    
    # # y_labels를 300개로 확장 (랜덤하게 0과 1 생성)
    # y_labels = np.random.choice([0, 1], size=300, p=[0.7, 0.3])  # 30% anomaly
    # print("y_labels...")
    # print(y_labels[:20])  # 처음 20개만 출력
    
    # 로지스틱 회귀 파라미터 
    initial_weights = np.random.randn(300)
    initial_bias = 0.1
    
    # 처리기 초기화
    processor = PiHEAANSensorProcessor()
    
    # 센서 데이터 처리
    threshold = 0.5
    
    # CSV에서 데이터 로드
    csv_path = "./dataset/df_final_timestamp.csv"  
    
    # CSV 로드 (라벨과 타임스탬프 컬럼명 지정 가능)
    sensor_data, nan_masks, y_labels, timestamps = processor.load_csv_to_sensor_data(
        csv_path,
        label_column=None,  # 라벨 컬럼이 없으므로 None
        timestamp_column='SensorTime',  # CSV의 실제 타임스탬프 컬럼명
        missing_value=-999
    )
    
    print("sensor_data....")
    print(sensor_data)
    print("nan_masks...")
    print(nan_masks)
    print("timestamps...")
    print(timestamps[:10])  # 처음 10개만 출력
    
    if y_labels is not None:
        print("y_labels...")
        print(y_labels[:20])  # 처음 20개만 출력
    else:
        # 라벨이 없으면 랜덤 생성 (기존과 동일)
        y_labels = np.random.choice([0, 1], size=sensor_data.shape[1], p=[0.7, 0.3])
        print("Generated random y_labels...")
        print(y_labels[:20])
    
    results = processor.process_sensor_data_with_training(
        sensor_data, nan_masks, initial_weights, initial_bias,
        y_labels, timestamps, learning_rate=0.1, num_steps=3, threshold=0.5
    )
    
    # 결과 출력 - 처음 20개만
    print("\n=== 최종 이상치 탐지 결과 (처음 20개) ===")
    for i, (timestamp, is_anomaly) in enumerate(results[:20]):
        if i < len(y_labels):
            actual = "ANOMALY" if y_labels[i] == 1 else "NORMAL"
            predicted = "ANOMALY" if is_anomaly else "NORMAL"
            match = "✓" if (y_labels[i] == 1) == is_anomaly else "✗"
            print(f"{i:3d}: {timestamp} - Actual: {actual:7s}, Predicted: {predicted:7s} {match}")
    
    # 정확도 계산
    if len(results) > 0:
        correct = 0
        total = min(len(results), len(y_labels))
        for i in range(total):
            predicted_anomaly = results[i][1]
            actual_anomaly = y_labels[i] == 1
            if predicted_anomaly == actual_anomaly:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\n전체 정확도: {accuracy:.2f}% ({correct}/{total})")
        
    
########################################################

        
class PiHEAANSensorProcessor:
    def __init__(self, key_file_path="./keys"):
        
        # 파라미터 설정
        self.params = heaan.ParameterPreset.FGb
        self.key_file_path = key_file_path
        
        # 컨텍스트 생성
        self.context = heaan.make_context(self.params)
        
        # 키 디렉토리 생성
        os.makedirs(self.key_file_path, mode=0o777, exist_ok=True)
        
        # 에러 및 기타 설정
        self.log_slots = heaan.get_log_full_slots(self.context)
        self.slots = 1 << self.log_slots

        self.SENSOR_COUNT = 5
        self.TIMESTAMP_COUNT = 300
        self.DATA_SIZE = 300
        
        print("\n\n")
        print(f"Initialized with log_slots: {self.log_slots}")
        
        # 비밀키 생성 및 저장
        self.sk = heaan.SecretKey(self.context)
        self.sk.save(f"{self.key_file_path}/secretkey.bin")
        
        # 커먼 키 생성
        key_generator = heaan.KeyGenerator(self.context, self.sk)
        key_generator.gen_common_keys()
        key_generator.save(f"{self.key_file_path}/")
        
        key_generator.gen_rot_keys_for_bootstrap(self.log_slots)
        key_generator.save(f"{self.key_file_path}/")
        
        # 키 로딩
        self.sk = heaan.SecretKey(self.context, f"{self.key_file_path}/secretkey.bin")
        self.pk = heaan.KeyPack(self.context, f"{self.key_file_path}/")
        self.pk.load_enc_key()
        self.pk.load_mult_key()
        self.pk.load_left_rot_key(768) 
        # for i in range(1, self.slots):
        #     # 파일이 있는지 확인하고 로드
        #     self.pk.load_left_rot_key(i)
        
        # 암호화기/복호화기/평가기 생성
        self.enc = heaan.Encryptor(self.context)
        self.dec = heaan.Decryptor(self.context)
        self.eval = heaan.HomEvaluator(self.context, self.pk)
        
        
        # 부트스트래퍼 설정 추가
        # print("Generating bootstrap rotation keys...")
        # key_generator.gen_bootstrap_keys()  # 부트스트래핑용 키 생성
        # key_generator.save(f"{self.key_file_path}/")
        # self.pk.load_rot_keys() 
        # heaan.make_bootstrappable(context)

        self.bootstrapper = heaan.Bootstrapper(self.eval, self.log_slots)
        
        # 부트스트래핑 준비 확인
        if self.bootstrapper.is_bootstrap_ready(self.log_slots):
            print("Bootstrap is ready")
        else:
            print("Making bootstrap constants...")
            self.bootstrapper.makeBootConstants(self.log_slots)
            print("Bootstrap constants created")
            
    
    def _bootstrap(self, ctxt):
        print("Bootstrapping...")
        result = heaan.Ciphertext(self.context)
        self.bootstrapper.bootstrap(ctxt, result)
        return result
    
    ##########################################################
    
    def load_csv_to_sensor_data(self, csv_path, label_column=None, timestamp_column='SensorTime', missing_value=-999):
        """
        CSV 파일을 로드해서 기존 코드 형태로 변환
        
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
        
        # 7. -999를 0으로 치환 (암호화용)
        sensor_data[sensor_data == missing_value] = 0.0
        
        # 8. y_labels 추출 (있는 경우)
        y_labels = None
        if label_column and label_column in df.columns:
            y_labels = df[label_column].values
            print(f"Labels shape: {y_labels.shape}")
        else:
            print("No labels found")
        
        # 9. timestamps 추출
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
    
    
    ######################################################


    def process_sensor_data_with_training(self, sensor_data, nan_masks, initial_weights, initial_bias, 
                                    y_labels, timestamps, learning_rate=0.01, num_steps=5, threshold=0.5):
        """학습을 포함한 전체 센서 데이터 처리 파이프라인"""
        print("=== 학습을 포함한 센서 데이터 처리 시작 ===")
        
        # 1. 데이터 암호화
        print("\n1. 센서별 데이터 암호화...")
        encrypted_data, plaintext_nan_masks = self.encrypt_sensor_data(sensor_data, nan_masks)
        
        # 2. 선형 보간
        print("\n2. 센서별 선형 보간...")
        interpolated_data = self.interpolate_all_sensors(encrypted_data, plaintext_nan_masks)
        
        # 3. 로지스틱 회귀 학습 및 예측
        print("\n3. 로지스틱 회귀 학습 및 예측...")
        anomaly_results = self.logistic_regression_all_data(
            interpolated_data, initial_weights, initial_bias, timestamps, 
            y_labels, learning_rate, num_steps, threshold
        )
        
        return anomaly_results
    
    ##########################################################
    
    def encrypt_sensor_data(self, sensor_data, nan_masks):
        # 각 센서별로 약 1567개 타임스탬프 데이터를 하나의 Ciphertext에 저장
        encrypted_data = []
        plaintext_nan_masks = []
        
        ctxt_scale = self.create_constant_vector(10000.0)
        print(f"Created scaling factor ciphertext: {ctxt_scale}")
        
        for sensor_id in range(self.SENSOR_COUNT):
            # 현재 센서의 데이터를 메시지에 저장
            msg_data = heaan.Message(self.log_slots)
            
            # 데이터를 슬롯에 저장
            for i in range(min(self.DATA_SIZE, 2 ** self.log_slots)):
                # NaN 위치는 1로 설정
                msg_data[i] = sensor_data[sensor_id, i] if nan_masks[sensor_id, i] == 0 else 1.0
            
            # 암호화
            ctxt_data = heaan.Ciphertext(self.context)
            self.enc.encrypt(msg_data, self.sk, ctxt_data)
            self.eval.mult(ctxt_data, ctxt_scale, ctxt_data)
            
            encrypted_data.append(ctxt_data)
            # NaN 마스크는 평문으로 저장
            plaintext_nan_masks.append(nan_masks[sensor_id].copy())

        msg_data = heaan.Message(self.log_slots)
        
        # debug
        # print("encrypted_data")
        # print(encrypted_data)
        
        # self.dec.decrypt(encrypted_data[0], self.sk, msg_data)
        # print(msg_data)
        
        # print("plaintext_nan_masks")
        # print(plaintext_nan_masks)
        
        return encrypted_data, plaintext_nan_masks
    
    
    ######################################################
    
    def interpolate_all_sensors(self, encrypted_data, plaintext_nan_masks):
        interpolated_data = []
        
        for sensor_id in range(self.SENSOR_COUNT):
            print("\n")
            print(f"Interpolating sensor {sensor_id}...")
            interpolated = self._interpolate_single_sensor(
                encrypted_data[sensor_id], 
                plaintext_nan_masks[sensor_id]
            )
            interpolated_data.append(interpolated)
            
            
            msg_data = heaan.Message(self.log_slots)
    
            # debug
            # print("decrypted_data")
            
            # self.dec.decrypt(interpolated, self.sk, msg_data)
            # for i in range(10):
            #     print(msg_data[i]) 
            
            # print("plaintext_nan_masks")
            # print(plaintext_nan_masks)
        
        return interpolated_data
    
    
    def _interpolate_single_sensor(self, enc_data, enc_mask):
        result = heaan.Ciphertext(self.context)
        result = enc_data
        
        # 평문 마스크로 NaN 위치 확인 및 보간
        for i in range(self.DATA_SIZE):
            if enc_mask[i] == 1:  # NaN 위치
                # 왼쪽과 오른쪽 유효값 찾기
                left_idx = self._find_left_valid(enc_mask, i)
                right_idx = self._find_right_valid(enc_mask, i)
                
                if left_idx >= 0 and right_idx >= 0:
                    # 선형 보간
                    interpolated_value = self._linear_interpolate_homomorphic(
                        result, left_idx, right_idx, i
                    )
                    result = self._update_value_at_position(result, interpolated_value, i)
                elif left_idx >= 0:
                    # 왼쪽 값으로 채움
                    left_value = self._extract_value_at_position(result, left_idx)
                    result = self._update_value_at_position(result, left_value, i)
                elif right_idx >= 0:
                    # 오른쪽 값으로 채움
                    right_value = self._extract_value_at_position(result, right_idx)
                    result = self._update_value_at_position(result, right_value, i)
        
        return result
    
    def _find_left_valid(self, enc_mask, current_idx):
        # 왼쪽의 유효한 값 찾기
        for i in range(current_idx - 1, -1, -1):
            if enc_mask[i] == 0:
                return i
        return -1
    
    def _find_right_valid(self, enc_mask, current_idx):
        # 오른쪽의 유효한 값 찾기
        for i in range(current_idx + 1, self.DATA_SIZE):
            if enc_mask[i] == 0:
                return i
        return -1
    
    def _extract_value_at_position(self, enc_data, position):
        
        self.bootstrapper.bootstrap(enc_data, enc_data);
            
        # 해당 위치를 첫 번째 슬롯으로 로테이트
        rotated = heaan.Ciphertext(self.context)
        if position > 0:
            self.eval.left_rotate(enc_data, position, rotated)
        else:
            rotated = enc_data
        
        # 첫 번째 슬롯만 1인 마스크 생성 및 적용
        mask_msg = heaan.Message(self.log_slots)
        mask_msg[0] = 1.0
        for i in range(1, 2**self.log_slots):
            mask_msg[i] = 0.0
        
        mask_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(mask_msg, self.sk, mask_ctxt)
        
        # 마스크 적용
        extracted = heaan.Ciphertext(self.context)
        self.eval.mult(rotated, mask_ctxt, extracted)
        
        return extracted
    
    
    def _update_value_at_position(self, enc_data, new_value, position):
        # 해당 위치를 0으로 만드는 마스크
        mask_msg = heaan.Message(self.log_slots)
        for i in range(2**self.log_slots):
            mask_msg[i] = 0.0 if i == position else 1.0
        
        mask_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(mask_msg, self.sk, mask_ctxt)
        
        # 기존 데이터에서 해당 위치 제거
        cleared = heaan.Ciphertext(self.context)
        self.eval.mult(enc_data, mask_ctxt, cleared)
        
        # 새 값을 해당 위치로 로테이트
        rotated_value = heaan.Ciphertext(self.context)
        if position > 0:
            self.eval.right_rotate(new_value, position, rotated_value)
        else:
            rotated_value = new_value
        
        # 결합
        result = heaan.Ciphertext(self.context)
        self.eval.add(cleared, rotated_value, result)
        
        return result
    
    
    def _linear_interpolate_homomorphic(self, enc_data, left_idx, right_idx, target_idx):
        # 좌우 값 추출
        left_value = self._extract_value_at_position(enc_data, left_idx)
        right_value = self._extract_value_at_position(enc_data, right_idx)
        
        # 선형 보간 계산: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        dx = right_idx - left_idx
        dt = target_idx - left_idx
        ratio = dt / dx
        
        # (y2 - y1)
        diff = heaan.Ciphertext(self.context)
        self.eval.sub(right_value, left_value, diff)
        
        # ratio * (y2 - y1)
        scaled_diff = heaan.Ciphertext(self.context)
        self.eval.mult(diff, ratio, scaled_diff)
        
        # y1 + ratio * (y2 - y1)
        result = heaan.Ciphertext(self.context)
        self.eval.add(left_value, scaled_diff, result)
        
        return result
    
    
    #########################################################
    
    def logistic_regression_all_data(self, interpolated_data, weights, bias, timestamps, y_labels, learning_rate=0.01, num_steps=2, threshold=0.5):
        """로지스틱 회귀 전체 과정"""
        print("=== 로지스틱 회귀 시작 ===")
        
        # y_labels를 암호화
        y_msg = heaan.Message(self.log_slots)
        for i in range(min(len(y_labels), 2**self.log_slots)):
            y_msg[i] = complex(float(y_labels[i]), 0.0)
            # y_msg[i] = y_labels[i]
        
        # 나머지 슬롯을 0으로 초기화
        for i in range(len(y_labels), 2**self.log_slots):
            y_msg[i] = complex(0.0, 0.0)
            
        y_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(y_msg, self.sk, y_ctxt)
        
        current_weights = weights
        current_bias = bias
        
        # 여러 Step 수행
        for step in range(num_steps):
            print(f"\n{'='*20} Step {step+1} {'='*20}")
            
            # 순전파 및 오차 계산
            sigmoid_result, error = self._forward_step(
                interpolated_data, current_weights, current_bias, y_ctxt
            )
            
            # 역전파 및 가중치 업데이트
            updated_weights = self._backward_step(
                interpolated_data, error, learning_rate
            )
            
            current_weights = updated_weights
            
            # 중간 결과 출력
            self._print_step_results(step+1, sigmoid_result, error)
        
        # 최종 예측 및 결과 생성
        final_predictions, final_error = self._forward_step(
            interpolated_data, current_weights, current_bias, y_ctxt
        )
        
        anomaly_results = self._generate_final_results(
            final_predictions, timestamps, threshold
        )
        
        return anomaly_results
    
    
    def _forward_step(self, interpolated_data, weights, bias, y_ctxt):
        """순전파: 예측값 계산 및 오차 계산"""
        print("Forward step: 예측값 계산 중...")
        
        # Step 1: 각 센서별 가중치 적용 및 합계
        result_ctxt = heaan.Ciphertext(self.context)
        self._initialize_zero_ciphertext(result_ctxt)
        
        for sensor_id in range(self.SENSOR_COUNT):
            print(f"  Processing sensor {sensor_id}...")
            
            # 가중치 벡터 생성
            weights_ctxt = self.create_weight_vector(weights)
            
            # 센서 데이터와 가중치 곱셈
            weighted_data = heaan.Ciphertext(self.context)
            self.eval.mult(interpolated_data[sensor_id], weights_ctxt, weighted_data)
            
            # 전체 합계에 누적
            temp_result = heaan.Ciphertext(self.context)
            self.eval.add(result_ctxt, weighted_data, temp_result)
            result_ctxt = temp_result
        
        # 편향 추가
        bias_ctxt = self.create_bias_vector(bias)
        linear_output = heaan.Ciphertext(self.context)
        self.eval.add(result_ctxt, bias_ctxt, linear_output)
        
        # 시그모이드 적용
        sigmoid_result = self._sigmoid_approximation(linear_output)
        
        # 오차 계산: error = y_true - y_pred
        error = heaan.Ciphertext(self.context)
        self.eval.sub(y_ctxt, sigmoid_result, error)
        
        return sigmoid_result, error
    
   
    def _sigmoid_approximation(self, x):
        """시그모이드 함수 근사 (체비셰프 다항식)"""
        # sigmoid(x) ≈ 0.5 + 0.25*x - 0.0625*x^3
        print("Applying sigmoid approximation...")
        
        # x^2 계산
        x_squared = heaan.Ciphertext(self.context)
        self.eval.square(x, x_squared)
        
        # x^3 계산
        x_cubed = heaan.Ciphertext(self.context)
        self.eval.mult(x_squared, x, x_cubed)
        
        # 0.25 * x
        term1 = heaan.Ciphertext(self.context)
        self.eval.mult(x, 0.25, term1)
        
        # -0.0625 * x^3
        term2 = heaan.Ciphertext(self.context)
        self.eval.mult(x_cubed, -0.0625, term2)
        
        # 0.5 상수 추가
        const_msg = heaan.Message(self.log_slots)
        const_msg[0] = 0.5
        const_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(const_msg, self.sk, const_ctxt)
        
        # 모든 항 더하기: 0.5 + 0.25*x - 0.0625*x^3
        temp_result = heaan.Ciphertext(self.context)
        self.eval.add(term1, term2, temp_result)
        
        final_result = heaan.Ciphertext(self.context)
        self.eval.add(temp_result, const_ctxt, final_result)
        
        # 디버깅: 시그모이드 결과 확인
        debug_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(final_result, self.sk, debug_msg)
        print(f"Sigmoid result: {debug_msg[0]}")
        
        return final_result
    
    def _sum_all_elements(self, ctxt):
        """Ciphertext의 모든 슬롯 합계 계산 (로테이션 사용)"""
        result = heaan.Ciphertext(self.context)
        result = ctxt
        
        # 로그-스케일 합산
        step = 1
        while step < self.DATA_SIZE:
            rotated = heaan.Ciphertext(self.context)
            self.eval.left_rotate(result, step, rotated)
            temp_result = heaan.Ciphertext(self.context)
            self.eval.add(result, rotated, temp_result)
            result = temp_result
            step *= 2
    
        return result


    def _initialize_zero_ciphertext(self, ctxt):
        """Ciphertext를 0으로 초기화"""
        zero_msg = heaan.Message(self.log_slots)
        for i in range(2**self.log_slots):
            zero_msg[i] = 0.0
        
        self.enc.encrypt(zero_msg, self.sk, ctxt)
        
    
    def create_weight_vector(self, weights):
        """
        가중치 배열을 암호화된 벡터로 생성
        
        Args:
            weights: list or numpy array - 가중치 값들
            
        Returns:
            heaan.Ciphertext - 암호화된 가중치 벡터
        """
        print(f"Creating weight vector with {len(weights)} weights...")
        
        # Message 객체 생성
        weights_msg = heaan.Message(self.log_slots)
        
        # 가중치를 슬롯에 설정
        for i in range(len(weights)):
            if i < 2 ** self.log_slots:  # 슬롯 범위 확인
                weights_msg[i] = weights[i]
            else:
                print(f"Warning: Weight index {i} exceeds slot capacity {2**self.log_slots}")
                break
        
        # 나머지 슬롯은 0으로 초기화
        for i in range(len(weights), 2**self.log_slots):
            weights_msg[i] = 0.0
        
        # 암호화
        weights_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(weights_msg, self.sk, weights_ctxt)
        
        # 디버깅: 생성된 가중치 확인
        debug_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(weights_ctxt, self.sk, debug_msg)
        
        print(f"Weight vector created successfully")
        print(f"First 10 weights: {[debug_msg[i] for i in range(min(10, len(weights)))]}")
        
        if len(weights) > 10:
            print(f"Last 5 weights: {[debug_msg[i] for i in range(len(weights)-5, len(weights))]}")
        
        return weights_ctxt


    def create_bias_vector(self, bias_value):
        """
        편향 값을 모든 슬롯에 설정한 벡터 생성
        
        Args:
            bias_value: float - 편향 값
            
        Returns:
            heaan.Ciphertext - 암호화된 편향 벡터 [bias, bias, bias, ...]
        """
        print(f"Creating bias vector with value {bias_value}...")
        
        # Message 객체 생성
        bias_msg = heaan.Message(self.log_slots)
        
        # 모든 슬롯에 bias 값 설정
        for i in range(2**self.log_slots):
            bias_msg[i] = bias_value
        
        # 암호화
        bias_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(bias_msg, self.sk, bias_ctxt)
        
        # 디버깅: 생성된 편향 확인
        debug_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(bias_ctxt, self.sk, debug_msg)
        
        print(f"Bias vector created: all slots = {debug_msg[0]}")
        print(f"Verification - first 5 values: {[debug_msg[i] for i in range(5)]}")
        
        return bias_ctxt
    
    def create_single_bias_value(self, bias_value):
        """
        편향 값을 첫 번째 슬롯에만 설정한 벡터 생성
        
        Args:
            bias_value: float - 편향 값
            
        Returns:
            heaan.Ciphertext - 암호화된 편향 벡터 [bias, 0, 0, ...]
        """
        print(f"Creating single bias value {bias_value}...")
        
        # Message 객체 생성
        bias_msg = heaan.Message(self.log_slots)
        
        # 첫 번째 슬롯에만 bias 값 설정
        bias_msg[0] = bias_value
        
        # 나머지 슬롯은 0으로 설정
        for i in range(1, 2**self.log_slots):
            bias_msg[i] = 0.0
        
        # 암호화
        bias_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(bias_msg, self.sk, bias_ctxt)
        
        # 디버깅: 생성된 값 확인
        debug_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(bias_ctxt, self.sk, debug_msg)
        
        print(f"Single bias value created: slot[0] = {debug_msg[0]}")
        print(f"Other slots: {[debug_msg[i] for i in range(1, 5)]} (should be 0)")
        
        return bias_ctxt


    def create_repeated_weights(self, weights, repeat_times):
        """
        가중치 패턴을 여러 번 반복한 벡터 생성
        
        Args:
            weights: list - 반복할 가중치 패턴
            repeat_times: int - 반복 횟수
            
        Returns:
            heaan.Ciphertext - [w1,w2,w3, w1,w2,w3, w1,w2,w3, ...] 형태
        """
        print(f"Creating repeated weights: {len(weights)} weights × {repeat_times} times...")
        
        # Message 객체 생성
        repeated_msg = heaan.Message(self.log_slots)
        
        weight_len = len(weights)
        slot_idx = 0
        
        # 지정된 횟수만큼 가중치 패턴 반복
        for repeat in range(repeat_times):
            for i in range(weight_len):
                if slot_idx < 2**self.log_slots:
                    repeated_msg[slot_idx] = weights[i]
                    slot_idx += 1
                else:
                    print(f"Warning: Reached slot capacity {2**self.log_slots}")
                    break
            
            if slot_idx >= 2**self.log_slots:
                break
        
        # 나머지 슬롯은 0으로 설정
        for i in range(slot_idx, 2**self.log_slots):
            repeated_msg[i] = 0.0
        
        # 암호화
        repeated_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(repeated_msg, self.sk, repeated_ctxt)
        
        # 디버깅: 반복 패턴 확인
        debug_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(repeated_ctxt, self.sk, debug_msg)
        
        print(f"Repeated weights created: total {slot_idx} values")
        print(f"First pattern: {[debug_msg[i] for i in range(min(weight_len, 10))]}")
        
        if repeat_times > 1 and slot_idx > weight_len:
            print(f"Second pattern: {[debug_msg[i] for i in range(weight_len, min(2*weight_len, slot_idx))]}")
        
        return repeated_ctxt
    
    
    def create_zero_vector(self):
        """
        모든 슬롯이 0인 벡터 생성
        
        Returns:
            heaan.Ciphertext - [0, 0, 0, ...]
        """
        print("Creating zero vector...")
        
        # Message 객체 생성
        zero_msg = heaan.Message(self.log_slots)
        
        # 모든 슬롯을 0으로 설정
        for i in range(2**self.log_slots):
            zero_msg[i] = 0.0
        
        # 암호화
        zero_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(zero_msg, self.sk, zero_ctxt)
        
        print("Zero vector created")
        
        return zero_ctxt


    def create_constant_vector(self, constant_value):
        """
        모든 슬롯이 동일한 상수값인 벡터 생성
        
        Args:
            constant_value: float - 설정할 상수 값
            
        Returns:
            heaan.Ciphertext - [constant, constant, constant, ...]
        """
        print(f"Creating constant vector with value {constant_value}...")
        
        # Message 객체 생성
        const_msg = heaan.Message(self.log_slots)
        
        # 모든 슬롯에 상수 값 설정
        for i in range(2**self.log_slots):
            const_msg[i] = constant_value
        
        # 암호화
        const_ctxt = heaan.Ciphertext(self.context)
        self.enc.encrypt(const_msg, self.sk, const_ctxt)
        
        # 디버깅: 값 확인
        debug_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(const_ctxt, self.sk, debug_msg)
        
        print(f"Constant vector created: all slots = {debug_msg[0]}")
        
        return const_ctxt
    
    def _print_step_results(self, step_num, sigmoid_result, error):
        """각 Step의 중간 결과 출력"""
        print(f"\n--- Step {step_num} 결과 ---")
        
        # 시그모이드 결과 확인
        sigmoid_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(sigmoid_result, self.sk, sigmoid_msg)
        print(f"Sigmoid predictions (first 5): {[sigmoid_msg[i] for i in range(5)]}")
        
        # 오차 확인
        error_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(error, self.sk, error_msg)
        print(f"Errors (first 5): {[error_msg[i] for i in range(5)]}")
        
        # 평균 오차 계산
        avg_error = sum(error_msg[i] for i in range(self.DATA_SIZE)) / self.DATA_SIZE
        print(f"Average error: {avg_error:.4f}")


    def _generate_final_results(self, predictions, timestamps, threshold):
        """최종 예측 결과를 timestamp별 anomaly 결과로 변환"""
        print("Generating final anomaly detection results...")
        
        # 예측값 복호화
        pred_msg = heaan.Message(self.log_slots)
        self.dec.decrypt(predictions, self.sk, pred_msg)
        
        anomaly_results = []
        
        for i in range(self.DATA_SIZE):
            if i < len(timestamps):
                timestamp = timestamps[i]
            else:
                timestamp = f"timestamp_{i}"
            
            prediction = pred_msg[i].real if hasattr(pred_msg[i], 'real') else pred_msg[i]
            is_anomaly = prediction > threshold
            
            anomaly_results.append([timestamp, is_anomaly])
            
            if i < 10:  # 처음 10개만 출력
                print(f"  {timestamp}: prediction={prediction:.4f}, anomaly={is_anomaly}")
        
        return anomaly_results

        

if __name__ == "__main__":
    main()