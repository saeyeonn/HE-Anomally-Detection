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
    
    