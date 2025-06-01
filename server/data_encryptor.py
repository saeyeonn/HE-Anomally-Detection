import piheaan as heaan
import numpy as np


class DataEncryptor:
    """센서 데이터 암호화를 담당하는 클래스"""
    
    def __init__(self, crypto_processor):
        self.crypto_processor = crypto_processor
        self.context = crypto_processor.context
        self.enc = crypto_processor.enc
        self.eval = crypto_processor.eval
        self.sk = crypto_processor.sk
        self.log_slots = crypto_processor.log_slots
        self.SENSOR_COUNT = crypto_processor.SENSOR_COUNT
        self.DATA_SIZE = crypto_processor.DATA_SIZE
    
    def encrypt_sensor_data(self, sensor_data, nan_masks):
        """
        센서 데이터를 암호화
        
        Args:
            sensor_data: (n_sensors, n_samples) 형태의 센서 데이터
            nan_masks: (n_sensors, n_samples) 형태의 NaN 마스크
            
        Returns:
            encrypted_data: 암호화된 센서 데이터 리스트
            plaintext_nan_masks: 평문 NaN 마스크 리스트
        """
        encrypted_data = []
        plaintext_nan_masks = []
        actual_data_size = min(sensor_data.shape[1], self.DATA_SIZE)
        ctxt_scale = self.create_constant_vector(0.01)
        print(f"Created scaling factor ciphertext: {ctxt_scale}")
        
        for sensor_id in range(self.SENSOR_COUNT):
            # 현재 센서의 데이터를 메시지에 저장
            msg_data = self.crypto_processor.create_message()
            
            # 데이터를 슬롯에 저장
            for i in range(actual_data_size):
                # NaN 위치는 1로 설정
                msg_data[i] = sensor_data[sensor_id, i] if nan_masks[sensor_id, i] == 0 else 1.0
            
            # 암호화
            ctxt_data = self.crypto_processor.create_ciphertext()
            self.enc.encrypt(msg_data, self.sk, ctxt_data)
            self.eval.mult(ctxt_data, ctxt_scale, ctxt_data)
            
            encrypted_data.append(ctxt_data)
            # NaN 마스크는 평문으로 저장
            plaintext_nan_masks.append(nan_masks[sensor_id].copy())

        return encrypted_data, plaintext_nan_masks
    
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
        const_msg = heaan.Message(self.log_slots, 0.5)
        
        # 모든 슬롯에 상수 값 설정
        for i in range(2**self.log_slots):
            const_msg[i] = constant_value
        
        # 암호화
        const_ctxt = self.crypto_processor.create_ciphertext()
        self.enc.encrypt(const_msg, self.sk, const_ctxt)
        
        return const_ctxt
    
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
        weights_msg = self.crypto_processor.create_message()
        
        # 가중치를 슬롯에 설정
        for i in range(len(weights)):
            if i < 2 ** self.log_slots:
                weights_msg[i] = weights[i]
            else:
                print(f"Warning: Weight index {i} exceeds slot capacity {2**self.log_slots}")
                break
        
        # 나머지 슬롯은 0으로 초기화
        for i in range(len(weights), 2**self.log_slots):
            weights_msg[i] = 0.0
        
        # 암호화
        weights_ctxt = self.crypto_processor.encrypt_message(weights_msg)
        
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
        bias_msg = self.crypto_processor.create_message()
        
        # 모든 슬롯에 bias 값 설정
        for i in range(2**self.log_slots):
            bias_msg[i] = bias_value
        
        # 암호화
        bias_ctxt = self.crypto_processor.encrypt_message(bias_msg)
        
        return bias_ctxt
    
    def create_zero_vector(self):
        """
        모든 슬롯이 0인 벡터 생성
        
        Returns:
            heaan.Ciphertext - [0, 0, 0, ...]
        """
        print("Creating zero vector...")
        
        # Message 객체 생성
        zero_msg = self.crypto_processor.create_message()
        
        # 모든 슬롯을 0으로 설정
        for i in range(2**self.log_slots):
            zero_msg[i] = 0.0
        
        # 암호화
        zero_ctxt = self.crypto_processor.encrypt_message(zero_msg)
        
        return zero_ctxt