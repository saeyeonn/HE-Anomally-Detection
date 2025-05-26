import piheaan as heaan
from piheaan.math import approx 
import os
import pandas as pd 
import numpy as np
import math
import time
    
        
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
    
    
    def encrypt_sensor_data(self, sensor_data, nan_masks):
        # 각 센서별로 약 1567개 타임스탬프 데이터를 하나의 Ciphertext에 저장
        encrypted_data = []
        plaintext_nan_masks = []
        
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
    
    
    #########################################################3
    
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