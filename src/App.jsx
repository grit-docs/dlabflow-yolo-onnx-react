/**
 * D-Lab Flow 실시간 객체 탐지 데모
 *
 * 이 애플리케이션은 ONNX Runtime을 사용하여 YOLOv5 모델로 실시간 객체 탐지를 수행합니다.
 * 브라우저에서 완전히 동작하며 사용자의 카메라를 통해 객체를 탐지하고 바운딩 박스로 표시합니다.
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import * as ort from 'onnxruntime-web'
import './App.css'

// 전역 변수로 정의 (최소값 0.4로 설정)
let confidenceThreshold = 0.4;

function App() {
  // Refs
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);

  // State variables
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [classes, setClasses] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [detections, setDetections] = useState([]);
  const [debugMode] = useState(true);


  useEffect(() => {
    async function loadModel() {
      try {
        setIsModelLoading(true);
        const classesResponse = await fetch('/models/classes.json');
        const classesData = await classesResponse.json();
        setClasses(classesData);
        const modelPath = '/models/model.onnx';

        // ONNX Runtime 최적화 옵션 추가
        const session = await ort.InferenceSession.create(modelPath, {
          executionProviders: ['webgl', 'cpu'], // WebGL 우선, CPU 대체
          graphOptimizationLevel: 'all', // 그래프 최적화
          enableCpuMemArena: false, // 메모리 아레나 비활성화 (모바일에서 더 나음)
          enableMemPattern: false, // 메모리 패턴 비활성화
        });

        sessionRef.current = session;
        setIsModelLoaded(true);
        setIsModelLoading(false);
      } catch (error) {
        console.error('Error loading model:', error);
        setErrorMessage('모델 로딩 오류. public/models 디렉토리에 model.onnx 파일이 있는지 확인하세요.');
        setIsModelLoading(false);
      }
    }
    loadModel();
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    };
  }, []);


  const startCamera = async () => {
    try {
      setIsDetecting(true);
    } catch (error) {
      console.error('Camera access failed:', error);
      setErrorMessage(`카메라에 접근할 수 없습니다: ${error.message || '권한을 확인하세요.'}`);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setIsDetecting(false);
    setDetections([]);
  };

  const detectObjects = useCallback(async () => {
    if (!webcamRef.current || !canvasRef.current || !sessionRef.current || !isDetecting) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    if (!video || video.readyState !== 4 || video.paused) {
      if (isDetecting) animationRef.current = requestAnimationFrame(detectObjects);
      return;
    }

    try {
      const ctx = canvas.getContext('2d');
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      canvas.width = videoWidth;
      canvas.height = videoHeight;

      setIsProcessing(true);
      try {
        // 추론용 캔버스 생성 (모바일 성능 개선을 위해 스케일 조정)
        const scaleCanvas = document.createElement('canvas');
        const scaleFactor = 1.0; // 모바일 성능을 위해 원본 크기로 유지
        scaleCanvas.width = Math.floor(videoWidth * scaleFactor);
        scaleCanvas.height = Math.floor(videoHeight * scaleFactor);
        const scaleCtx = scaleCanvas.getContext('2d');
        scaleCtx.drawImage(video, 0, 0, scaleCanvas.width, scaleCanvas.height);

        const imageData = scaleCtx.getImageData(0, 0, scaleCanvas.width, scaleCanvas.height);
        const [input, imgWidth, imgHeight] = await prepareInput(imageData);
        const feeds = { images: input };
        const results = await sessionRef.current.run(feeds);
        const output = results[Object.keys(results)[0]];
        const currentDetections = processDetections(output.data, output.dims, imgWidth, imgHeight, videoWidth, videoHeight);

        // 탐지된 객체를 타임스탬프 없이 바로 설정
        setDetections(currentDetections);

        // 현재 탐지된 객체들만 바로 그리기
        drawBoundingBoxes(currentDetections, ctx, canvas.width, canvas.height);
      } catch (error) {
        console.error('Detection processing error:', error);
      } finally {
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Canvas operation error:', error);
    }

    if (isDetecting && !isProcessing) {
      animationRef.current = setTimeout(() => {
        detectObjects();
      }, 150); // 초당 약 6-7회, 적절한 균형점
    }

  }, [isDetecting, classes]);

  // Process detections from model output
  const processDetections = (data, dims, imgWidth, imgHeight, canvasWidth, canvasHeight) => {
    const numDetections = dims[1];

    // 모델 입력 크기 (YOLOv5는 일반적으로 640x640 입력을 사용)
    const modelWidth = 640;
    const modelHeight = 640;

    // 캔버스 크기에 맞게 스케일 조정 (모바일 디바이스에서 중요)
    const scaleX = canvasWidth / modelWidth;
    const scaleY = canvasHeight / modelHeight;

    if (debugMode) {
      console.log(`Detection scaling: ${scaleX}x${scaleY}, Canvas: ${canvasWidth}x${canvasHeight}`);
      console.log(`추론 시작: 총 ${numDetections}개의 객체 후보 처리 중...`);
    }

    const rawDetections = [];
    // 클래스별 탐지 카운터 초기화
    const classDetections = {};

    for (let i = 0; i < numDetections; i++) {
      // Get detection data (each detection has 6 values)
      const base = i * 85; // 🔄 6 → 85로 변경

      const objectness = 1 / (1 + Math.exp(-data[base + 4]));
      let maxClassProb = 0;
      let classIndex = 0;
      for (let j = 5; j < 85; j++) {
        const prob = 1 / (1 + Math.exp(-data[base + j]));
        if (prob > maxClassProb) {
          maxClassProb = prob;
          classIndex = j - 5;
        }
      }
      const confidence = objectness * maxClassProb;

      const className = classes[classIndex] || `클래스 ${classIndex}`;

      // Filter by confidence threshold
      if (confidence > confidenceThreshold) {
        // Extract bounding box coordinates
        const x = data[base] * scaleX;
        const y = data[base + 1] * scaleY;
        const width = data[base + 2] * scaleX;
        const height = data[base + 3] * scaleY;

        // Track detections by class
        if (!classDetections[className]) {
          classDetections[className] = {
            count: 0,
            confidenceSum: 0,
            highestConfidence: 0
          };
        }
        classDetections[className].count++;
        classDetections[className].confidenceSum += confidence;
        classDetections[className].highestConfidence = Math.max(classDetections[className].highestConfidence, confidence);

        // Log individual detection with high confidence (only in debug mode)
        if (debugMode) {
          console.log(`탐지된 객체: ${className}, 인식률: ${(confidence * 100).toFixed(2)}%, 위치: [x=${Math.round(x)}, y=${Math.round(y)}]`);
        }

        rawDetections.push({
          bbox: [x - width/2, y - height/2, width, height],
          class: className,
          confidence: confidence,
          classIndex: classIndex
        });
      }
    }

    // Apply Non-Maximum Suppression to filter out overlapping detections
    const detections = applyNonMaxSuppression(rawDetections, 0.3); // 0.3 is the IoU threshold (lower value = more aggressive filtering)

    // 추론 결과 요약 로그 출력 (디버그 모드에서만)
    if (debugMode) {
      if (Object.keys(classDetections).length > 0) {
        console.log('======== 추론 결과 요약 ========');
        console.log(`총 탐지된 객체 수: ${rawDetections.length}개 (NMS 적용 전), ${detections.length}개 (NMS 적용 후)`);

        // 클래스별 탐지 결과 및 인식률 출력
        Object.keys(classDetections).forEach(className => {
          const stats = classDetections[className];
          const avgConfidence = stats.confidenceSum / stats.count;
          console.log(`클래스: ${className}, 탐지 수: ${stats.count}개, 평균 인식률: ${(avgConfidence * 100).toFixed(2)}%, 최고 인식률: ${(stats.highestConfidence * 100).toFixed(2)}%`);
        });
        console.log('===============================');
      } else {
        console.log('탐지된 객체가 없습니다.');
      }
    }

    return detections;
  };

  // Calculate Intersection over Union (IoU) between two bounding boxes
  const calculateIoU = (box1, box2) => {
    // box format: [x, y, width, height] where x,y is the top-left corner
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    // Calculate coordinates of the intersection rectangle
    const xLeft = Math.max(x1, x2);
    const yTop = Math.max(y1, y2);
    const xRight = Math.min(x1 + w1, x2 + w2);
    const yBottom = Math.min(y1 + h1, y2 + h2);

    // Check if there is an intersection
    if (xRight < xLeft || yBottom < yTop) {
      return 0;
    }

    // Calculate intersection area
    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);

    // Calculate union area
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    const unionArea = box1Area + box2Area - intersectionArea;

    // Calculate IoU
    return intersectionArea / unionArea;
  };

  // Apply Non-Maximum Suppression to filter out overlapping detections
  const applyNonMaxSuppression = (detections, iouThreshold) => {
    if (detections.length === 0) {
      return [];
    }

    // Sort detections by confidence score (highest first)
    const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);

    const selectedDetections = [];
    const remainingDetections = [...sortedDetections];

    // Process all detections
    while (remainingDetections.length > 0) {
      // Select the detection with highest confidence
      const currentDetection = remainingDetections.shift();
      selectedDetections.push(currentDetection);

      // Filter out detections that have high overlap with the current detection
      // and are of the same class
      let i = 0;
      while (i < remainingDetections.length) {
        // Only apply NMS to detections of the same class
        if (remainingDetections[i].classIndex === currentDetection.classIndex) {
          const iou = calculateIoU(currentDetection.bbox, remainingDetections[i].bbox);

          // If IoU is above threshold, remove this detection
          if (iou > iouThreshold) {
            remainingDetections.splice(i, 1);
          } else {
            i++;
          }
        } else {
          i++;
        }
      }
    }

    return selectedDetections;
  };

  // Prepare input tensor for the model
  const prepareInput = async (imageData) => {
    // YOLOv5 expects input in format [batch, channels, height, width]
    const imgWidth = imageData.width;
    const imgHeight = imageData.height;

    // YOLOv5 models typically expect 640x640 input
    const modelWidth = 640;
    const modelHeight = 640;

    // Create a temporary canvas for the source image data
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = imgWidth;
    sourceCanvas.height = imgHeight;
    const sourceCtx = sourceCanvas.getContext('2d');
    sourceCtx.putImageData(imageData, 0, 0);

    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = modelWidth;
    tempCanvas.height = modelHeight;
    const tempCtx = tempCanvas.getContext('2d');

    // Draw and resize the image
    tempCtx.drawImage(
        sourceCanvas,
        0, 0, imgWidth, imgHeight,
        0, 0, modelWidth, modelHeight
    );

    // Get the resized image data
    const resizedImageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
    const pixels = resizedImageData.data;

    // Prepare the input tensor (NCHW format)
    // YOLOv5 expects RGB values normalized to [0, 1]
    const inputTensor = new Float32Array(modelWidth * modelHeight * 3);
    let inputIndex = 0;

    // Convert from RGBA to RGB and normalize
    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < modelHeight; h++) {
        for (let w = 0; w < modelWidth; w++) {
          const pixelIndex = (h * modelWidth + w) * 4; // RGBA has 4 channels
          inputTensor[inputIndex++] = pixels[pixelIndex + c] / 255.0;
        }
      }
    }

    // Create the ONNX tensor
    const tensor = new ort.Tensor('float32', inputTensor, [1, 3, modelHeight, modelWidth]);

    return [tensor, imgWidth, imgHeight];
  };

  // Draw bounding boxes on the canvas
  const drawBoundingBoxes = (detections, ctx, canvasWidth, canvasHeight) => {
    try {
      // 캔버스 상태 저장
      ctx.save();

      // 기본 캔버스 설정 (모바일 호환성 위해 간소화)
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'medium'; // high에서 medium으로 조정

      // 캔버스 클리어 - 이제 비디오를 그리지 않고 탐지 결과만 표시
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      // 모든 탐지된 객체 사용 (필터링 없음)
      const recentDetections = detections;

      // 비디오 프레임은 더 이상 캔버스에 그리지 않음 (Webcam 컴포넌트가 배경으로 표시됨)

      if (debugMode && webcamRef.current && webcamRef.current.video) {
        const video = webcamRef.current.video;
        if (video.readyState === 4) {
          // Log video dimensions only in debug mode
          console.log(`Video dimensions: ${video.videoWidth}x${video.videoHeight}`);
        }
      }

      if (debugMode) {
        console.log(`Drawing ${recentDetections.length} bounding boxes - Canvas dimensions: ${canvasWidth}x${canvasHeight}`);
      }

      // 최종 화면에 표시되는 탐지 결과 요약 (디버그 모드에서만)
      if (debugMode && recentDetections.length > 0) {
        console.log('======== 화면 표시 탐지 결과 ========');
        // 클래스별 통계 계산을 위한 객체
        const finalClassStats = {};

        recentDetections.forEach(detection => {
          const className = detection.class;
          if (!finalClassStats[className]) {
            finalClassStats[className] = {
              count: 0,
              confidenceSum: 0
            };
          }
          finalClassStats[className].count++;
          finalClassStats[className].confidenceSum += detection.confidence;
        });

        // 클래스별 통계 출력
        Object.keys(finalClassStats).forEach(className => {
          const stats = finalClassStats[className];
          const avgConfidence = stats.confidenceSum / stats.count;
          console.log(`클래스: ${className}, 개수: ${stats.count}개, 평균 인식률: ${(avgConfidence * 100).toFixed(2)}%`);
        });
        console.log('==================================');
      }

      // Set drawing styles
      ctx.lineWidth = 1; // Reduced line width as requested
      ctx.font = 'bold 18px Arial'; // Made font bold for better visibility
      ctx.textBaseline = 'top';

      // 캔버스 오버레이 배경을 더 투명하게 조정
      // 실제 카메라 영상이 더 잘 보이도록 함
      ctx.fillStyle = 'rgba(0, 0, 0, 0.0)';
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // Draw each filtered detection
      recentDetections.forEach((detection, i) => {
        const [drawX, drawY, width, height] = detection.bbox;
        const className = detection.class;
        const confidence = detection.confidence;
        const classIndex = detection.classIndex;

        // Log the detection information with more details in Korean (only in debug mode)
        if (debugMode) {
          console.log(`최종 탐지 #${i+1}: 클래스=${className}, 인식률=${(confidence * 100).toFixed(2)}%, 위치=[x=${Math.round(drawX)}, y=${Math.round(drawY)}, 너비=${Math.round(width)}, 높이=${Math.round(height)}]`);
        }

        // Generate random color based on class index
        const hue = (classIndex * 137) % 360; // Use prime number for better distribution

        // 더 얇은 테두리로 바운딩 박스 그리기
        ctx.strokeStyle = `hsl(${hue}, 100%, 50%)`;
        ctx.lineWidth = 2; // 줄어든 선 두께로 바운딩 박스를 더 깔끔하게 표시

        // 더 투명한 내부 색상 사용
        ctx.fillStyle = `hsla(${hue}, 100%, 40%, 0.15)`; // 낮은 불투명도로 내부 색상 옅게 표시

        // Draw bounding box with more visible style
        ctx.beginPath();
        ctx.rect(drawX, drawY, width, height);
        ctx.fill();
        ctx.stroke();

        // 라벨 배경도 약간 투명하게 조정
        const label = `${className} ${Math.round(confidence * 100)}%`;
        const textMetrics = ctx.measureText(label);
        const textHeight = 28; // 라벨 높이 유지
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, 0.7)`; // 약간 투명하게 조정

        // Calculate label position
        const labelX = drawX;
        const labelY = drawY - textHeight;

        ctx.fillRect(
            labelX,
            labelY,
            textMetrics.width + 10,
            textHeight
        );

        // 모바일에서 더 잘 보이는 텍스트 스타일
        // 텍스트에 그림자 효과 추가
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 3;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;

        // 흰색 굵은 텍스트
        ctx.fillStyle = 'white';
        ctx.font = 'bold 18px Arial'; // 굵은 글씨체
        ctx.fillText(
            label,
            labelX + 5,
            labelY + 5
        );

        // 그림자 효과 제거
        ctx.shadowColor = 'transparent';

        // 텍스트 외곽선 (선택적)
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.strokeText(
            label,
            labelX + 5,
            labelY + 5
        );
      });

      // Restore the canvas state
      ctx.restore();

    } catch (error) {
      console.error("Error in drawBoundingBoxes:", error);
    }
  };

  // Inline UI components
  const Card = ({ className, children }) => (
      <div className={`card ${className || ''}`}>
        {children}
      </div>
  );

  const Button = ({ onClick, disabled, className, children }) => (
      <button
          onClick={onClick}
          disabled={disabled}
          className={`button ${className || ''}`}
      >
        {children}
      </button>
  );

  const Badge = ({ className, children }) => (
      <span className={`badge ${className || ''}`}>
      {children}
    </span>
  );

  // Icons (simplified versions)
  const Icon = ({ name, className }) => {
    const icons = {
      loader: <div className="icon-loader"></div>,
      camera: <div className="icon-camera"></div>,
      power: <div className="icon-power"></div>,
      zap: <div className="icon-zap"></div>,
      logo: <img src="/logo_icon.svg" alt="바운딩 박스" className="icon-logo" style={{ width: '50px', height: '50px' }} />,
      logo2: <img src="/logo_icon.svg" alt="바운딩 박스" className="icon-logo" />
    };
    return <div className={`icon ${className || ''}`}>{icons[name]}</div>;
  };

  return (
      <div className="app-container">
        {/* Header */}
        <div className="header">
          <div className="header-content">
            <div className="logo-container">
              <div>
                <Icon name="logo" />
              </div>
              <div>
                <h1>D-Lab Flow</h1>
                <p className="subtitle">실시간 객체 탐지 데모</p>
              </div>
            </div>

            {/* Status Indicators */}
            <div className="status-indicators">
              <div className="confidence-slider-container">
                <label htmlFor="confidence-slider">신뢰도: {Math.round(confidenceThreshold * 100)}%</label>
                <input
                  id="confidence-slider"
                  type="range"
                  min="40"
                  max="90"
                  value={confidenceThreshold * 100}
                  onChange={(e) => {
                    confidenceThreshold = Number(e.target.value) / 100;
                    // 강제 리렌더링을 위한 더미 상태 업데이트
                    setIsProcessing(prev => !prev);
                    setIsProcessing(prev => !prev);
                  }}
                  className="confidence-slider"
                  disabled={!isModelLoaded || isModelLoading}
                />
              </div>
              {isProcessing && (
                  <Badge className="processing-badge">
                    <Icon name="zap" />
                    처리 중
                  </Badge>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {isModelLoading ? (
              <Card className="loading-card">
                <div className="loading-spinner">
                  <Icon name="loader" />
                </div>
                <h2>AI 모델 로딩 중...</h2>
              </Card>
          ) : errorMessage ? (
              <Card className="error-card">
                <div className="error-icon">
                  <Icon name="power" />
                </div>
                <h2>오류 발생</h2>
                <p>{errorMessage}</p>
                <Button
                    onClick={() => window.location.reload()}
                    className="start-button"
                    style={{ marginTop: '1rem', background: 'var(--danger-color)' }}
                >
                  다시 시도
                </Button>
              </Card>
          ) : !isDetecting ? (
              <Card className="start-card">
                <div>
                  <Icon name="logo2" />
                </div>
                <h2>D-Lab Flow 실시간 객체 탐지 데모</h2>
                <p>카메라 시작 버튼을 눌러주세요</p>
                <Button
                    onClick={startCamera}
                    disabled={!isModelLoaded}
                    className="start-button"
                >
                  <Icon name="camera" />
                  카메라 시작
                </Button>
              </Card>
          ) : (
              <div className="camera-view">
                {/* Camera View */}
                <div className="camera-container">
                  <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{
                        facingMode: "environment", // 후면 카메라
                        width: { ideal: 960, min: 640 }, // 적절한 해상도로 조정
                        height: { ideal: 540, min: 480 },
                        frameRate: { ideal: 24, max: 30 }, // 프레임 레이트 소폭 감소
                        aspectRatio: 16/9 // 와이드스크린 비율 유지
                      }}
                      style={{
                        position: 'absolute',
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        zIndex: 1, // 캔버스보다 낮은 z-index로 배경으로 표시
                        left: 0,
                        top: 0,
                        borderRadius: 'inherit' // 부모 요소의 둥근 모서리 상속
                      }}
                      onUserMedia={(stream) => {
                        if (debugMode) {
                          console.log("Camera stream obtained successfully");
                          console.log("Stream settings:", stream.getVideoTracks()[0].getSettings());
                        }
                        streamRef.current = stream;

                        // 캔버스 정보 확인 (디버깅용)
                        if (debugMode) {
                          const canvas = canvasRef.current;
                          if (canvas) {
                            console.log("Canvas ready:", canvas.width, canvas.height);
                          }
                        }

                        // 카메라 스트림 획득 후 즉시 탐지 시작 (requestAnimationFrame 사용)
                        requestAnimationFrame(detectObjects);
                      }}
                      onUserMediaError={(error) => {
                        console.error("Camera access error:", error);
                        let errorMsg = '카메라 접근 오류';

                        if (error.name === 'NotAllowedError') {
                          errorMsg = '카메라 권한이 거부되었습니다. 브라우저 설정에서 카메라 권한을 허용해주세요.';
                        } else if (error.name === 'NotFoundError') {
                          errorMsg = '카메라를 찾을 수 없습니다.';
                        } else if (error.name === 'NotSupportedError') {
                          errorMsg = 'HTTPS 연결이 필요합니다.';
                        }

                        setErrorMessage(errorMsg);
                        setIsDetecting(false);
                      }}
                  />
                  <canvas
                      ref={canvasRef}
                      className="detection-canvas"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        zIndex: 2, // 웹캠보다 높은 z-index로 탐지 결과만 표시
                        objectFit: 'cover',
                        borderRadius: 'inherit', // 부모 요소의 둥근 모서리 상속
                        touchAction: 'none', // 터치 이벤트 방지
                        pointerEvents: 'none' // 마우스 이벤트 통과시켜 아래 요소와 상호작용 가능하게 함
                      }}
                  />

                  {/* Controls Overlay */}
                  <div className="camera-controls">
                    <Button
                        onClick={stopCamera}
                        className="stop-button"
                        style={{
                          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)', // 더 강한 그림자
                          background: 'rgba(239, 68, 68, 0.25)', // 더 진한 배경색
                          border: '2px solid rgba(239, 68, 68, 0.5)', // 더 눈에 띄는 테두리
                          color: '#ef4444', // 더 밝은 텍스트 색상
                          fontWeight: 'bold'
                        }}
                    >
                      <Icon name="power" />
                      카메라 중지
                    </Button>
                  </div>
                </div>

                {/* Detection Info - Always show the card, even when empty */}
                <Card className="detections-card">
                  <h3>
                    {detections.length > 0
                        ? `탐지된 객체 (${detections.length})`
                      : '객체를 카메라에 비춰보세요'}
                  </h3>
                  {detections.length > 0 ? (
                      <div className="detection-badges">
                        {detections.map((detection, index) => (
                            <Badge
                                key={index}
                                className="detection-badge"
                                style={{
                                  // 클래스에 따라 다른 색상 적용
                                  backgroundColor: `rgba(${(index * 50) % 255}, ${(index * 120) % 255}, ${(index * 180) % 255}, 0.15)`,
                                  borderColor: `rgba(${(index * 50) % 255}, ${(index * 120) % 255}, ${(index * 180) % 255}, 0.4)`
                                }}
                            >
                              {detection.class} ({Math.round(detection.confidence * 100)}%)
                            </Badge>
                        ))}
                      </div>
                  ) : isDetecting ? (
                      <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                        <p style={{ color: 'var(--text-secondary)' }}>
                          탐지된 객체가 없습니다
                        </p>
                        <p style={{ color: 'var(--primary-color)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                          현재 신뢰도 기준: {Math.round(confidenceThreshold * 100)}% 이상
                        </p>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                          카메라에 객체를 비추거나 신뢰도 설정을 조정해보세요
                        </p>
                      </div>
                  ) : (
                      <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                        카메라에 객체를 비추면 여기에 탐지 결과가 표시됩니다
                      </p>
                  )}
                </Card>
              </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="footer">
          <p>💡 D-Lab Flow에서 생성한 인공지능 모델을 사용한 실시간 객체 탐지 데모</p>
        </div>
      </div>
  );
}

export default App
