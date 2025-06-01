import piheaan as heaan


class LinearInterpolator:
    """선형 보간을 담당하는 클래스"""
    
    def __init__(self, crypto_processor):
        self.crypto_processor = crypto_processor
        self.context = crypto_processor.context
        self.enc = crypto_processor.enc
        self.dec = crypto_processor.dec
        self.eval = crypto_processor.eval
        self.sk = crypto_processor.sk
        self.log_slots = crypto_processor.log_slots
        self.SENSOR_COUNT = crypto_processor.SENSOR_COUNT
        self.DATA_SIZE = crypto_processor.DATA_SIZE
        self.bootstrapper = crypto_processor.bootstrapper
    
    def interpolate_all_sensors(self, actual_data_size, encrypted_data, plaintext_nan_masks):
        """
        모든 센서 데이터에 대해 선형 보간 수행
        
        Args:
            encrypted_data: 암호화된 센서 데이터 리스트
            plaintext_nan_masks: 평문 NaN 마스크 리스트
            
        Returns:
            interpolated_data: 보간된 암호화 데이터 리스트
        """
        interpolated_data = []
        
        for sensor_id in range(self.SENSOR_COUNT):
            print(f"Interpolating sensor {sensor_id}...")
            interpolated = self._interpolate_single_sensor(
                actual_data_size,
                encrypted_data[sensor_id], 
                plaintext_nan_masks[sensor_id]
            )
            interpolated_data.append(interpolated)
        
        return interpolated_data
    
    def _interpolate_single_sensor(self, actual_data_size, enc_data, enc_mask):
        """
        단일 센서 데이터에 대한 선형 보간
        
        Args:
            enc_data: 암호화된 센서 데이터
            enc_mask: 평문 NaN 마스크
            
        Returns:
            result: 보간된 암호화 데이터
        """
        result = enc_data
                
        # 평문 마스크로 NaN 위치 확인 및 보간
        for i in range(actual_data_size):
            if enc_mask[i] == 1:  # NaN 위치
                # 왼쪽과 오른쪽 유효값 찾기
                left_idx = self._find_left_valid(enc_mask, i)
                right_idx = self._find_right_valid(actual_data_size, enc_mask, i)
                
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
        """왼쪽의 유효한 값 찾기"""
        for i in range(current_idx - 1, -1, -1):
            if enc_mask[i] == 0:
                return i
        return -1
    
    def _find_right_valid(self, actual_data_size, enc_mask, current_idx):
        """오른쪽의 유효한 값 찾기"""
        for i in range(current_idx + 1, actual_data_size):
            if enc_mask[i] == 0:
                return i
        return -1
    
    def _extract_value_at_position(self, enc_data, position):
        """특정 위치의 값을 추출"""
        self.bootstrapper.bootstrap(enc_data, enc_data)
            
        # 해당 위치를 첫 번째 슬롯으로 로테이트
        rotated = self.crypto_processor.create_ciphertext()
        if position > 0:
            self.eval.left_rotate(enc_data, position, rotated)
        else:
            rotated = enc_data
        
        # 첫 번째 슬롯만 1인 마스크 생성 및 적용
        mask_msg = self.crypto_processor.create_message()
        mask_msg[0] = 1.0
        for i in range(1, 2**self.log_slots):
            mask_msg[i] = 0.0
        
        mask_ctxt = self.crypto_processor.encrypt_message(mask_msg)
        
        # 마스크 적용
        extracted = self.crypto_processor.create_ciphertext()
        self.eval.mult(rotated, mask_ctxt, extracted)
        
        return extracted
    
    def _update_value_at_position(self, enc_data, new_value, position):
        """특정 위치의 값을 업데이트"""
        # 해당 위치를 0으로 만드는 마스크
        mask_msg = self.crypto_processor.create_message()
        for i in range(2**self.log_slots):
            mask_msg[i] = 0.0 if i == position else 1.0
        
        mask_ctxt = self.crypto_processor.encrypt_message(mask_msg)
        
        # 기존 데이터에서 해당 위치 제거
        cleared = self.crypto_processor.create_ciphertext()
        self.eval.mult(enc_data, mask_ctxt, cleared)
        
        # 새 값을 해당 위치로 로테이트
        rotated_value = self.crypto_processor.create_ciphertext()
        if position > 0:
            self.eval.right_rotate(new_value, position, rotated_value)
        else:
            rotated_value = new_value
        
        # 결합
        result = self.crypto_processor.create_ciphertext()
        self.eval.add(cleared, rotated_value, result)
        
        return result
    
    def _linear_interpolate_homomorphic(self, enc_data, left_idx, right_idx, target_idx):
        """동형 암호 기반 선형 보간"""
        # 좌우 값 추출
        left_value = self._extract_value_at_position(enc_data, left_idx)
        right_value = self._extract_value_at_position(enc_data, right_idx)
        
        # 선형 보간 계산: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        dx = right_idx - left_idx
        dt = target_idx - left_idx
        ratio = dt / dx
        
        # (y2 - y1)
        diff = self.crypto_processor.create_ciphertext()
        self.eval.sub(right_value, left_value, diff)
        
        # ratio * (y2 - y1)
        scaled_diff = self.crypto_processor.create_ciphertext()
        self.eval.mult(diff, ratio, scaled_diff)
        
        # y1 + ratio * (y2 - y1)
        result = self.crypto_processor.create_ciphertext()
        self.eval.add(left_value, scaled_diff, result)
        
        return result