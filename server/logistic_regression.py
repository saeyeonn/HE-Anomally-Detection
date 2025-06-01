import piheaan as heaan
import numpy as np


class LogisticRegression:
    """로지스틱 회귀 학습 및 예측을 담당하는 클래스"""
    
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
    
    def train_and_predict(self, interpolated_data, weights, bias, timestamps, y_labels, 
                         learning_rate=0.01, num_steps=2, threshold=0.5):
        """
        로지스틱 회귀 학습 및 예측 수행
        
        Args:
            interpolated_data: 보간된 암호화 센서 데이터
            weights: 초기 가중치
            bias: 초기 편향
            timestamps: 타임스탬프 리스트
            y_labels: 실제 라벨
            learning_rate: 학습률
            num_steps: 학습 스텝 수
            threshold: 이상치 판별 임계값
            
        Returns:
            anomaly_results: 이상치 탐지 결과 리스트
        """
        print("=== 로지스틱 회귀 시작 ===")
        
        # y_labels를 암호화
        y_ctxt = self._encrypt_labels(y_labels)
        
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
                interpolated_data, error, learning_rate, current_weights
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
    
    def _encrypt_labels(self, y_labels):
        """라벨을 암호화"""
        y_msg = self.crypto_processor.create_message()
        for i in range(min(len(y_labels), 2**self.log_slots)):
            y_msg[i] = complex(float(y_labels[i]), 0.0)
            
        # 나머지 슬롯을 0으로 초기화
        for i in range(len(y_labels), 2**self.log_slots):
            y_msg[i] = complex(0.0, 0.0)
            
        y_ctxt = self.crypto_processor.encrypt_message(y_msg)
        return y_ctxt
    
    def _forward_step(self, interpolated_data, weights, bias, y_ctxt):
        """순전파 단계"""
        print("Forward step: 예측값 계산 중...")
        
        # 선형 조합 계산: w1*sensor1 + w2*sensor2 + ... + w5*sensor5 + bias
        result_ctxt = self._create_constant_vector(bias)  # bias로 초기화
        
        for sensor_id in range(self.SENSOR_COUNT):
            print(f"  Processing sensor {sensor_id} with weight {weights[sensor_id]}...")
            
            # 개별 센서 데이터에 해당 가중치 곱하기
            weighted_sensor = self.crypto_processor.create_ciphertext()
            self.eval.mult(interpolated_data[sensor_id], weights[sensor_id], weighted_sensor)
            
            # 결과에 누적
            temp_result = self.crypto_processor.create_ciphertext()
            self.eval.add(result_ctxt, weighted_sensor, temp_result)
            result_ctxt = temp_result
        
        # 시그모이드 적용
        sigmoid_result = self._sigmoid_approximation(result_ctxt)
        
        # 오차 계산
        error = self.crypto_processor.create_ciphertext()
        self.eval.sub(y_ctxt, sigmoid_result, error)
        
        return sigmoid_result, error
    
    def _backward_step(self, interpolated_data, error, learning_rate, current_weights):
        """역전파 단계"""
        updated_weights = []
        
        for sensor_id in range(self.SENSOR_COUNT):
            # 그래디언트 계산
            gradient = self.crypto_processor.create_ciphertext()
            self.eval.mult(interpolated_data[sensor_id], error, gradient)
            
            gradient_sum = self._sum_all_elements(gradient)
            learning_factor = learning_rate / self.DATA_SIZE
            scaled_gradient = self.crypto_processor.create_ciphertext()
            self.eval.mult(gradient_sum, learning_factor, scaled_gradient)
            
            # 그래디언트를 복호화
            gradient_msg = self.crypto_processor.decrypt_ciphertext(scaled_gradient)
            gradient_value = gradient_msg[0].real if hasattr(gradient_msg[0], 'real') else gradient_msg[0]
            print(f"  Sensor {sensor_id}: gradient = {gradient_value}")
            
            # 가중치 업데이트 (현재 가중치 + 학습률 * 그래디언트)
            new_weight = current_weights[sensor_id] - learning_rate * gradient_value
            updated_weights.append(new_weight)
        
        return updated_weights
    
    def _sigmoid_approximation(self, x):
        """시그모이드 함수 근사 (체비셰프 다항식)"""
        # sigmoid(x) ≈ 0.5 + 0.25*x - 0.0625*x^3
        print("Applying sigmoid approximation...")
        
        # x^2 계산
        x_squared = self.crypto_processor.create_ciphertext()
        self.eval.square(x, x_squared)
        
        # x^3 계산
        x_cubed = self.crypto_processor.create_ciphertext()
        self.eval.mult(x_squared, x, x_cubed)
        
        # 0.25 * x
        term1 = self.crypto_processor.create_ciphertext()
        self.eval.mult(x, 0.25, term1)
        
        # -0.0625 * x^3
        term2 = self.crypto_processor.create_ciphertext()
        self.eval.mult(x_cubed, -0.0625, term2)
        
        # 0.5 상수 추가
        const_msg = self.crypto_processor.create_message()
        const_msg[0] = 0.5
        const_ctxt = self.crypto_processor.encrypt_message(const_msg)
        
        # 모든 항 더하기: 0.5 + 0.25*x - 0.0625*x^3
        temp_result = self.crypto_processor.create_ciphertext()
        self.eval.add(term1, term2, temp_result)
        
        final_result = self.crypto_processor.create_ciphertext()
        self.eval.add(temp_result, const_ctxt, final_result)
        
        # 디버깅: 시그모이드 결과 확인
        debug_msg = self.crypto_processor.decrypt_ciphertext(final_result)
        print(f"Sigmoid result: {debug_msg[0]}")
        
        return final_result
    
    def _sum_all_elements(self, ctxt):
        """Ciphertext의 모든 슬롯 합계 계산 (로테이션 사용)"""
        result = ctxt
        
        # 로그-스케일 합산
        step = 1
        while step < self.DATA_SIZE:
            rotated = self.crypto_processor.create_ciphertext()
            self.eval.left_rotate(result, step, rotated)
            temp_result = self.crypto_processor.create_ciphertext()
            self.eval.add(result, rotated, temp_result)
            result = temp_result
            step *= 2
    
        return result
    
    def _create_constant_vector(self, constant_value):
        """상수 벡터 생성"""
        const_msg = self.crypto_processor.create_message()
        for i in range(2**self.log_slots):
            const_msg[i] = constant_value
        
        const_ctxt = self.crypto_processor.encrypt_message(const_msg)
        return const_ctxt
    
    def _print_step_results(self, step_num, sigmoid_result, error):
        """각 Step의 중간 결과 출력"""
        print(f"\n--- Step {step_num} 결과 ---")
        
        # 시그모이드 결과 확인
        sigmoid_msg = self.crypto_processor.decrypt_ciphertext(sigmoid_result)
        print(f"Sigmoid predictions (first 5): {[sigmoid_msg[i] for i in range(5)]}")
        
        # 오차 확인
        error_msg = self.crypto_processor.decrypt_ciphertext(error)
        print(f"Errors (first 5): {[error_msg[i] for i in range(5)]}")
        
        # 평균 오차 계산
        avg_error = sum(error_msg[i] for i in range(self.DATA_SIZE)) / self.DATA_SIZE
        print(f"Average error: {avg_error:.4f}")
    
    def _generate_final_results(self, predictions, timestamps, threshold):
        """최종 예측 결과를 timestamp별 anomaly 결과로 변환"""
        print("Generating final anomaly detection results...")
        
        # 예측값 복호화
        pred_msg = self.crypto_processor.decrypt_ciphertext(predictions)
        
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