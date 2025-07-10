# D-Lab Flow 실시간 객체 탐지 데모

브라우저에서 ONNX Runtime을 사용하여 완전히 실행되는 실시간 객체 탐지 웹 애플리케이션입니다. 
이 애플리케이션은 사용자 장치의 카메라를 사용하여 실시간으로 객체를 탐지하고 바운딩 박스와 클래스 예측을 표시합니다.

## 주요 기능

- YOLOv5s 모델을 사용한 실시간 객체 탐지
- ONNX Runtime으로 브라우저 기반 추론
- 후면 카메라 지원을 통한 모바일 친화적 디자인
- 컬러풀한 바운딩 박스와 클래스 라벨
- 신뢰도 점수 표시 및 조절 기능
- 간단한 단일 화면 인터페이스

## 데모

<div align="center">
  <img src="/public/objectDetectionDemo.png" alt="D-Lab Flow 객체 탐지 데모 스크린샷" width="50%">
</div>

**[라이브 데모 사이트 확인하기](https://dlabflow-yolov5s-onnx-react.vercel.app/)**

브라우저에서 바로 실시간 객체 탐지 기능을 경험해보세요. 모바일 장치에서도 접속 가능합니다.

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
   - 변환된 `.onnx` 파일을 `public/models/` 디렉토리에 `model.onnx`라는 이름으로 복사합니다.
   - **(선택 사항)** 클래스 이름을 직접 지정하려면, 원하는 클래스 이름들을 담은 `classes.json` 파일을 생성하여 `public/models/` 디렉토리에 복사하세요. 이렇게 하면 탐지된 객체의 클래스 이름을 더 명확하게 표시할 수 있습니다.
     - 만약 `classes.json` 파일이 없다면, 애플리케이션이 ONNX 모델의 메타데이터에서 클래스 이름을 **자동으로 추출**하려고 시도합니다. (모델 export 시 `names` 메타데이터가 포함되어 있어야 함)
   - ✅ 이제 애플리케이션이 다양한 YOLO 출력 형식을 **자동으로 감지**하므로, 대부분의 YOLO 변형 모델을 별도 코드 수정 없이 사용할 수 있습니다.

## 사용법

1. 개발 서버를 시작합니다:
   ```
   npm run dev
   ```

2. 브라우저를 열고 터미널에 표시된 URL로 이동합니다 (보통 http://localhost:5173)

3. 메시지가 표시되면 카메라 접근 권한을 허용합니다

4. "카메라 시작" 버튼을 클릭하여 객체 탐지를 시작합니다

5. 카메라를 객체에 향하게 하면 실시간으로 탐지됩니다

6. 상단의 신뢰도 슬라이더를 조정하여 탐지 민감도를 변경할 수 있습니다

## 프로덕션 빌드

프로덕션 빌드를 생성하려면:

```
npm run build
```

빌드 파일은 `dist` 디렉토리에 위치하며 정적 호스팅 서비스에 배포할 수 있습니다.

## 기술 세부 사항

- **프론트엔드**: React (`v19.1.0`) 및 Vite (`v6.3.5`)
- **객체 탐지**: ONNX 형식으로 변환된 YOLOv3, YOLOv5, YOLOv8, YOLOv11 모델 (다양한 출력 형식 자동 지원)
- **추론 엔진**: ONNX Runtime Web (`v1.22.0`)
- **카메라 접근**: React Webcam (`v7.2.0`)
- **시각화**: HTML Canvas를 이용한 동적 바운딩 박스 렌더링

## 작동 방식

1. 애플리케이션이 시작되면 ONNX 모델을 로드합니다. 클래스 정보는 `classes.json` 파일에서 우선적으로 읽어오며, 이 파일이 없을 경우 모델 메타데이터에서 자동으로 추출합니다.
2. 사용자 기기(데스크톱/모바일)를 감지하고 카메라(전면/후면)와 미러링 설정을 최적화합니다.
3. `react-webcam`으로 카메라 프레임을 실시간으로 캡처합니다.
4. UI 렌더링과 분리된 별도의 비동기 루프에서 추론을 실행합니다.
5. 캡처된 프레임은 모델 입력 크기(640x640)에 맞게 조정된 후 ONNX Runtime으로 전달됩니다.
6. **첫 추론 시 모델의 출력 텐서 형태를 분석하여, 이후의 모든 탐지 결과를 올바르게 해석하도록 모델 구조 정보를 저장합니다.**
7. 탐지 결과(바운딩 박스, 클래스 ID, 신뢰도 점수)는 설정된 신뢰도 임계값에 따라 필터링됩니다.
8. Non-Maximum Suppression(NMS) 알고리즘을 적용하여 중복된 바운딩 박스를 제거합니다.
9. **탐지된 객체 정보를 잠시 기억하여, 한두 프레임에서 객체가 사라져도 바운딩 박스가 바로 사라지지 않고 부드럽게 유지됩니다.**
10. 최종 결과는 카메라 영상 위에 HTML Canvas를 사용하여 시각화되며, 카메라 미러링 상태에 맞춰 좌표가 정확히 표시됩니다.
