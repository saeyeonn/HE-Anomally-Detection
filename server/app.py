from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime

print("1. 기본 Flask 임포트 완료")

# 분리된 서비스 및 유틸리티 임포트
try:
    from homomorphic_service import HomomorphicAnomalyDetectionService
    print("2. HomomorphicAnomalyDetectionService 임포트 성공")
    SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"2. HomomorphicAnomalyDetectionService 임포트 실패: {e}")
    SERVICE_AVAILABLE = False

try:
    from api_utils import APIUtils
    print("3. APIUtils 임포트 성공")
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"3. APIUtils 임포트 실패: {e}")
    UTILS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 서비스 인스턴스 (전역, 싱글톤 패턴)
anomaly_service = None

def get_service():
    """서비스 인스턴스 반환"""
    global anomaly_service
    if anomaly_service is None and SERVICE_AVAILABLE:
        anomaly_service = HomomorphicAnomalyDetectionService()
    return anomaly_service

def simple_convert_to_api_format(anomaly_results, limit=5):
    """간단한 API 형식 변환 (APIUtils 없을 때 사용)"""
    api_results = []
    count = 0
    
    for timestamp, is_anomaly in anomaly_results:
        # timestamp_로 시작하는 것은 제외
        if not str(timestamp).startswith('timestamp_'):
            api_results.append({
                "timestamp": timestamp,
                "anomalyResult": bool(is_anomaly)
            })
            count += 1
            if count >= limit:
                break
                
    return api_results

def process_csv(file_path):
    """CSV 처리 함수"""
    if not SERVICE_AVAILABLE:
        # 서비스가 없으면 더미 데이터 반환
        print("동형암호 서비스를 사용할 수 없어 더미 데이터를 반환합니다.")
        dummy_results = [
            ("2025-05-12T12:00:00", True),
            ("2025-05-12T12:01:00", False), 
            ("2025-05-12T12:02:00", True)
        ]
        return simple_convert_to_api_format(dummy_results, limit=5)
    
    try:
        print(f"동형암호 서비스로 CSV 처리 시작: {file_path}")
        
        # 동형암호 서비스 사용
        service = get_service()
        anomaly_results = service.process_csv_file(file_path)
        
        # API 형식으로 변환 (상위 5개만)
        if UTILS_AVAILABLE:
            flask_results = APIUtils.convert_to_api_format(anomaly_results, limit=5)
        else:
            flask_results = simple_convert_to_api_format(anomaly_results, limit=5)
        
        return flask_results
        
    except Exception as e:
        print(f"CSV 처리 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

@app.route("/upload", methods=["POST"])
def upload_csv():
    print("=== CSV 업로드 요청 받음 ===")
    
    if 'csvFile' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files['csvFile']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    
    # 파일 저장 (타임스탬프 추가로 중복 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    print(f"파일 저장: {file_path}")
    
    try:
        # CSV 처리
        results = process_csv(file_path)
        print(f"처리 완료: {len(results)}개 결과")
        
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"처리 중 오류: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 임시 파일 정리
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"임시 파일 삭제: {file_path}")
        except Exception as cleanup_error:
            print(f"파일 삭제 실패: {cleanup_error}")

@app.route("/", methods=["GET"])
def home():
    """홈 페이지"""
    return jsonify({
        "message": "동형암호 기반 이상치 탐지 서버",
        "status": "running",
        "services": {
            "homomorphic_service": SERVICE_AVAILABLE,
            "api_utils": UTILS_AVAILABLE
        },
        "endpoints": {
            "POST /upload": "CSV 파일 업로드 및 이상치 탐지",
            "GET /health": "서버 상태 확인"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "message": "서버가 정상 작동 중입니다",
        "timestamp": datetime.now().isoformat(),
        "upload_folder": UPLOAD_FOLDER
    })

if __name__ == "__main__":
    print("=" * 50)
    print("동형암호 기반 이상치 탐지 Flask 서버 시작")
    print(f"서비스 상태 - 동형암호: {SERVICE_AVAILABLE}, 유틸리티: {UTILS_AVAILABLE}")
    print("주소: http://localhost:5050")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=5050, debug=True)