import piheaan as heaan
import os


class CryptographicProcessor:
    """HEAAN 암호화 컨텍스트 및 키 관리를 담당하는 클래스"""
    
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
        
        print(f"Initialized with log_slots: {self.log_slots}")
        
        # 키 생성 및 설정
        self._setup_keys()
        
        # 암호화기/복호화기/평가기 생성
        self.enc = heaan.Encryptor(self.context)
        self.dec = heaan.Decryptor(self.context)
        self.eval = heaan.HomEvaluator(self.context, self.pk)
        
        # 부트스트래퍼 설정
        self._setup_bootstrapper()
    
    def _setup_keys(self):
        """키 생성 및 로딩"""
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
    
    def _setup_bootstrapper(self):
        """부트스트래퍼 설정"""
        self.bootstrapper = heaan.Bootstrapper(self.eval, self.log_slots)
        
        # 부트스트래핑 준비 확인
        if self.bootstrapper.is_bootstrap_ready(self.log_slots):
            print("Bootstrap is ready")
        else:
            print("Making bootstrap constants...")
            self.bootstrapper.makeBootConstants(self.log_slots)
            print("Bootstrap constants created")
    
    def bootstrap(self, ctxt):
        """부트스트래핑 수행"""
        print("Bootstrapping...")
        result = heaan.Ciphertext(self.context)
        self.bootstrapper.bootstrap(ctxt, result)
        return result
    
    def create_message(self, values=None):
        """Message 객체 생성"""
        msg = heaan.Message(self.log_slots)
        if values is not None:
            for i, value in enumerate(values):
                if i < 2**self.log_slots:
                    msg[i] = value
                else:
                    break
        return msg
    
    def create_ciphertext(self):
        """Ciphertext 객체 생성"""
        return heaan.Ciphertext(self.context)
    
    def encrypt_message(self, msg):
        """Message를 암호화"""
        ctxt = self.create_ciphertext()
        self.enc.encrypt(msg, self.sk, ctxt)
        return ctxt
    
    def decrypt_ciphertext(self, ctxt):
        """Ciphertext를 복호화"""
        msg = heaan.Message(self.log_slots)
        self.dec.decrypt(ctxt, self.sk, msg)
        return msg