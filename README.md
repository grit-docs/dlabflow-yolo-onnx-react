# D-Lab Flow 실시간 객체 탐지 데모

브라우저에서 ONNX Runtime을 사용하여 완전히 실행되는 실시간 객체 탐지 웹 애플리케이션입니다. 
이 애플리케이션은 사용자 장치의 카메라를 사용하여 실시간으로 객체를 감지하고 바운딩 박스와 클래스 예측을 표시합니다.

## 주요 기능

- YOLOv5s 모델을 사용한 실시간 객체 탐지
- ONNX Runtime으로 브라우저 기반 추론
- 후면 카메라 지원을 통한 모바일 친화적 디자인
- 컬러풀한 바운딩 박스와 클래스 라벨
- 신뢰도 점수 표시 및 조절 기능
- 간단한 단일 화면 인터페이스

## 데모

![D-Lab Flow 객체 탐지 데모 스크린샷](/public/screenshot_1.jpg)

**[라이브 데모 사이트 확인하기](https://dlabflow-yolov5s-onnx-react.vercel.app/)**

브라우저에서 바로 실시간 객체 감지 기능을 경험해보세요. 모바일 장치에서도 접속 가능합니다.

## 필수 조건

- Node.js (v16 이상)
- npm 패키지 관리자
- 카메라가 있는 장치
- 웹 브라우저 (Chrome, Firefox, Safari, Edge)

## 설치

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/grit-docs/dlabflow-yolov5s-onnx-react.git
   cd dlabflow-yolov5s-onnx-react
   ```

2. 종속성을 설치합니다:
   ```
   npm install
   ```

3. YOLOv5 ONNX 모델을 가져옵니다

   🔹 **D-Lab Flow에서 학습한 모델을 ONNX로 변환**
   - YOLOv5 저장소를 클론합니다:
     ```bash
     git clone https://github.com/ultralytics/yolov5
     cd yolov5
     pip install -r requirements.txt
     ```
   - D-Lab Flow에서 학습한 모델의 가중치 파일(.pt)을 다운로드 받아 YOLOv5 폴더에 저장합니다
   - 변환 스크립트를 실행하여 .pt 파일을 ONNX로 변환합니다:
     ```bash
     python export.py --weights 다운로드받은모델.pt --include onnx
     ```
   - 변환된 .onnx 파일을 model.onnx로 이름 변경 후 public/models/ 디렉토리에 복사합니다
   - ✅ 이 방법을 통해 D-Lab Flow에서 커스텀 학습시킨 모델을 웹 애플리케이션에서 사용할 수 있습니다
   - ✅ YOLOv5 기본 모델(yolov5s.pt)도 동일하게 변환하여 사용할 수 있습니다

## 사용법

1. 개발 서버를 시작합니다:
   ```
   npm run dev
   ```

2. 브라우저를 열고 터미널에 표시된 URL로 이동합니다 (보통 http://localhost:5173)

3. 메시지가 표시되면 카메라 접근 권한을 허용합니다

4. "카메라 시작" 버튼을 클릭하여 객체 탐지를 시작합니다

5. 카메라를 객체에 향하게 하면 실시간으로 탐지됩니다

6. 상단의 신뢰도 슬라이더를 조정하여 감지 민감도를 변경할 수 있습니다

## 프로덕션 빌드

프로덕션 빌드를 생성하려면:

```
npm run build
```

빌드 파일은 `dist` 디렉토리에 위치하며 정적 호스팅 서비스에 배포할 수 있습니다.

## 기술 세부 사항

- **프론트엔드**: 빠른 개발을 위한 React와 Vite (React 19.1.0 사용)
- **객체 탐지**: ONNX 형식으로 변환된 YOLOv5 모델
- **추론 엔진**: ONNX Runtime Web (onnxruntime-web 1.17.0)
- **카메라 접근**: React Webcam (react-webcam 7.2.0)
- **시각화**: 바운딩 박스 그리기를 위한 HTML Canvas

## 작동 방식

1. 애플리케이션은 ONNX 형식의 사전 훈련된 YOLOv5 모델을 로드합니다
2. React Webcam을 사용하여 카메라에서 비디오 프레임을 캡처합니다
3. 각 프레임은 모델의 입력 요구 사항에 맞게 전처리됩니다 (640x640 크기로 조정)
4. ONNX Runtime은 전처리된 프레임에 대해 추론을 수행합니다
5. 감지 결과(바운딩 박스, 클래스 ID, 신뢰도 점수)가 추출됩니다
6. Non-Maximum Suppression(NMS)을 적용하여 중복 감지를 필터링합니다
7. 결과는 컬러 바운딩 박스와 라벨이 있는 캔버스 오버레이에 시각화됩니다
8. 사용자는 UI에서 실시간 신뢰도 임계값을 조정할 수 있습니다
