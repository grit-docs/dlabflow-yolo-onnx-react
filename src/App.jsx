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
  const scaleCanvasRef = useRef(document.createElement('canvas'));

  const detectionMemoryRef = useRef([]); // 최신 탐지 결과 + 유지할 이전 결과 포함
  const MAX_MISSED_FRAMES = 3; // 객체가 안 보여도 몇 프레임 유지

  // State variables
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [classes, setClasses] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [detections, setDetections] = useState([]);
  const [debugMode] = useState(true); // 디버깅 모드 활성화
  
  // 모델 출력 구조 정보를 저장하는 ref 추가
  const modelInfoRef = useRef({
    outputShape: null,
    detectionLength: 85, // 기본값 (YOLOv5s)
    numClasses: 80,
    isYOLOv5su: false,
    coordinateFormat: 'center', // 'center' or 'corner'
    numDetections: null,
    isTransposed: false
  });

  // 모델 로드
  useEffect(() => {
    console.log('🔥 NEW CODE LOADED - 새로운 코드가 로드되었습니다!'); // 강제 확인용
    async function loadModel() {
      try {
        setIsModelLoading(true);
        const res = await fetch('/models/classes.json');
        setClasses(await res.json());
        const session = await ort.InferenceSession.create('/models/model.onnx', {
          executionProviders: ['webgpu', 'webgl', 'cpu'],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: false,
          enableMemPattern: false,
        });
        sessionRef.current = session;
        
        // 모델의 출력 구조 정보 확인
        const outputNames = session.outputNames;
        console.log('🚀 Model output names:', outputNames);
        
        // 테스트 추론을 통해 출력 구조 확인
        await detectModelStructure(session);
        
        setIsModelLoaded(true);
      } catch (e) {
        console.error('Error loading model:', e);
        setErrorMessage('모델 로딩 오류. public/models 디렉토리에 model.onnx 파일이 있는지 확인하세요.');
      } finally {
        setIsModelLoading(false);
      }
    }
    loadModel();
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  // 모델 출력 구조를 감지하는 함수 추가
  const detectModelStructure = async (session) => {
    try {
      console.log('🔍 모델 구조 감지 시작...');
      // 더미 입력으로 테스트 추론 수행
      const dummyInput = new Float32Array(640 * 640 * 3).fill(0.5);
      const tensor = new ort.Tensor('float32', dummyInput, [1, 3, 640, 640]);
      const results = await session.run({ images: tensor });
      
      const outputKey = Object.keys(results)[0];
      const output = results[outputKey];
      const dims = output.dims;
      
      console.log('🎯 Model output dimensions:', dims);
      console.log('📊 Sample output data (first 20 values):', Array.from(output.data.slice(0, 20)));
      
      // 출력 구조 분석
      if (dims.length === 3) {
        const [batch, dim1, dim2] = dims;
        modelInfoRef.current.outputShape = dims;
        
        // YOLOv5su는 종종 transpose된 형태로 출력됨: [1, 84, 8400] vs [1, 8400, 84]
        let numDetections, detectionLength;
        let isTransposed = false;
        
        // 일반적으로 detection 개수가 feature 개수보다 훨씬 많음 (8400 vs 84/85)
        if (dim2 > dim1) {
          // transpose된 형태: [1, 84/85/6, 8400]
          numDetections = dim2;
          detectionLength = dim1;
          isTransposed = true;
        } else {
          // 일반적인 형태: [1, 8400, 84/85/6]
          numDetections = dim1;
          detectionLength = dim2;
          isTransposed = false;
        }
        
        modelInfoRef.current.detectionLength = detectionLength;
        modelInfoRef.current.numDetections = numDetections;
        modelInfoRef.current.isTransposed = isTransposed;
        
        console.log(`🔄 Transpose 감지: ${isTransposed ? 'YES' : 'NO'}`);
        console.log(`📏 Detection length: ${detectionLength}, Num detections: ${numDetections}`);
        
        // 출력 형태별 처리
        if (detectionLength === 6) {
          // YOLOv5su corner format: [x1, y1, x2, y2, confidence, class_id]
          modelInfoRef.current.isYOLOv5su = true;
          modelInfoRef.current.numClasses = classes.length || 80;
          modelInfoRef.current.coordinateFormat = 'corner';
          console.log('✅ YOLOv5su 형태 모델 감지됨 (6개 값 출력 - corner format)');
        } else if (detectionLength === 85) {
          // YOLOv5s 형태: [cx, cy, w, h, objectness, class1, ..., class80]
          modelInfoRef.current.isYOLOv5su = false;
          modelInfoRef.current.numClasses = 80;
          modelInfoRef.current.coordinateFormat = 'center';
          console.log('✅ YOLOv5s 형태 모델 감지됨 (85개 값 출력 - center format)');
        } else if (detectionLength === 84) {
          // YOLOv5su center format: [cx, cy, w, h, class1, ..., class80] (objectness 없음)
          modelInfoRef.current.isYOLOv5su = true;
          modelInfoRef.current.numClasses = 80;
          modelInfoRef.current.coordinateFormat = 'center';
          console.log('✅ YOLOv5su 형태 모델 감지됨 (84개 값 출력 - center format, no objectness)');
        } else {
          // 다른 구조의 경우
          const possibleNumClasses = detectionLength - 5;
          modelInfoRef.current.numClasses = Math.max(possibleNumClasses, 1);
          modelInfoRef.current.coordinateFormat = detectionLength <= 10 ? 'corner' : 'center';
          console.log(`✅ 사용자 정의 모델 감지됨 (${detectionLength}개 값, ${modelInfoRef.current.numClasses}개 클래스)`);
        }
      }
      
      console.log('📋 Model structure detected:', modelInfoRef.current);
    } catch (error) {
      console.warn('⚠️ 모델 구조 감지 실패, 기본값 사용:', error);
      // 기본값 유지
    }
  };

  // 렌더링 루프
  const drawLoop = useCallback(() => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;
    if (!canvas || !video) return;
    // 캔버스 크기를 비디오 해상도에 맞춤
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // 그리기 전 클리어
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBoundingBoxes(detectionMemoryRef.current, ctx, canvas.width, canvas.height);
    animationRef.current = requestAnimationFrame(drawLoop);
  }, []);

  // 추론 루프
  const inferenceLoop = useCallback(async () => {
    if (!isDetecting || !webcamRef.current || !sessionRef.current) return;
    const video = webcamRef.current.video;
    if (video.readyState === 4) {
      const scaleCanvas = scaleCanvasRef.current;
      const factor = 0.5;
      scaleCanvas.width = Math.floor(video.videoWidth * factor);
      scaleCanvas.height = Math.floor(video.videoHeight * factor);
      const sctx = scaleCanvas.getContext('2d');
      sctx.drawImage(video, 0, 0, scaleCanvas.width, scaleCanvas.height);
      const imageData = sctx.getImageData(0, 0, scaleCanvas.width, scaleCanvas.height);
      const [input, iw, ih] = await prepareInput(imageData);
      const results = await sessionRef.current.run({ images: input });
      const out = results[Object.keys(results)[0]];
      
      // 모델 출력 정보 로그 추가 및 실시간 transpose 감지
      if (debugMode) {
        console.log('=== 추론 결과 ===');
        console.log('Output dimensions:', out.dims);
        console.log('Sample raw data (first 12 values):', Array.from(out.data.slice(0, 12)));
        
        // 🎯 실시간 모델 구조 자동 감지 (YOLOv5s/YOLOv5su 모든 형태 지원)
        if (out.dims.length === 3) {
          const [batch, dim1, dim2] = out.dims;
          let numDetections, detectionLength, isTransposed;
          
          // transpose 여부 결정 (detection 개수가 feature 개수보다 훨씬 많음)
          if (dim2 > dim1) {
            // transpose된 형태: [1, N, 8400]
            isTransposed = true;
            numDetections = dim2;
            detectionLength = dim1;
          } else {
            // 일반적인 형태: [1, 8400, N]
            isTransposed = false;
            numDetections = dim1;
            detectionLength = dim2;
          }
          
          // 모델 정보 업데이트
          modelInfoRef.current.isTransposed = isTransposed;
          modelInfoRef.current.numDetections = numDetections;
          modelInfoRef.current.detectionLength = detectionLength;
          
          // 🔍 모델 타입 자동 감지
          let modelType = "Unknown";
          let coordinateFormat = "center";
          let isYOLOv5su = false;
          let numClasses = 80;
          
          if (detectionLength === 85) {
            // YOLOv5s: [cx, cy, w, h, objectness, class1, ..., class80]
            modelType = "YOLOv5s";
            coordinateFormat = "center";
            isYOLOv5su = false;
            numClasses = 80;
          } else if (detectionLength === 84) {
            // YOLOv5su Center: [cx, cy, w, h, class1, ..., class80] (objectness 없음)
            modelType = "YOLOv5su (Center)";
            coordinateFormat = "center";
            isYOLOv5su = true;
            numClasses = 80;
          } else if (detectionLength === 6) {
            // YOLOv5su Corner: [x1, y1, x2, y2, confidence, class_id]
            modelType = "YOLOv5su (Corner)";
            coordinateFormat = "corner";
            isYOLOv5su = true;
            numClasses = classes.length || 80;
          } else if (detectionLength > 6) {
            // 사용자 정의 모델 (추정)
            const possibleClasses = detectionLength - 5; // objectness + 4개 좌표
            if (possibleClasses > 0) {
              modelType = `Custom YOLO (${possibleClasses} classes)`;
              coordinateFormat = "center";
              isYOLOv5su = false;
              numClasses = possibleClasses;
            } else {
              modelType = `Custom YOLO (${detectionLength} features)`;
              coordinateFormat = detectionLength <= 10 ? "corner" : "center";
              isYOLOv5su = true;
              numClasses = Math.max(1, detectionLength - 4);
            }
          }
          
          // 설정 적용
          modelInfoRef.current.isYOLOv5su = isYOLOv5su;
          modelInfoRef.current.coordinateFormat = coordinateFormat;
          modelInfoRef.current.numClasses = numClasses;
          
          console.log(`🎯 자동 감지 완료: ${modelType}`);
          console.log(`📐 형태: ${isTransposed ? 'Transpose' : 'Normal'} [${batch}, ${dim1}, ${dim2}]`);
          console.log(`📊 Features: ${detectionLength}, Detections: ${numDetections}, Classes: ${numClasses}`);
          console.log(`📍 좌표 형식: ${coordinateFormat}, YOLOv5su: ${isYOLOv5su}`);
        }
        
        console.log('Model info (updated):', modelInfoRef.current);
      }
      
      const newDetections = processDetections(
          out.data,
          out.dims,
          iw,
          ih,
          video.videoWidth,
          video.videoHeight
      );
      if (newDetections.length > 0) {
        detectionMemoryRef.current = newDetections.map(d => ({ ...d, missed: 0 }));
      } else {
        detectionMemoryRef.current = detectionMemoryRef.current
            .map(d => ({ ...d, missed: d.missed + 1 }))
            .filter(d => d.missed <= MAX_MISSED_FRAMES);
      }
      setDetections(detectionMemoryRef.current);
    }
    setTimeout(inferenceLoop, 10);
  }, [isDetecting]);

  // isDetecting 변경시 루프 시작/중지
  useEffect(() => {
    if (isDetecting) {
      requestAnimationFrame(drawLoop);
      inferenceLoop();
    } else {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    }
  }, [isDetecting, drawLoop, inferenceLoop]);

  // 카메라 시작
  const startCamera = () => {
    console.log('📹 카메라 시작 - 새로운 디버깅 코드 활성화됨!'); // 강제 확인용
    setIsDetecting(true);
  };

  // 카메라 중지
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsDetecting(false);
    setDetections([]);
  };

    // Process detections from model output - 수정된 버전
  const processDetections = (data, dims, imgWidth, imgHeight, canvasWidth, canvasHeight) => {
    const detectionLength = modelInfoRef.current.detectionLength;
    const numDetections = modelInfoRef.current.numDetections || dims[1];
    const isYOLOv5su = modelInfoRef.current.isYOLOv5su;
    const coordinateFormat = modelInfoRef.current.coordinateFormat;
    const isTransposed = modelInfoRef.current.isTransposed || false;

    // 모델 입력 크기 (YOLOv5는 일반적으로 640x640 입력을 사용)
    const modelWidth = 640;
    const modelHeight = 640;

    if (debugMode) {
      console.log(`=== Detection Processing ===`);
      console.log(`Canvas: ${canvasWidth}x${canvasHeight}, Model: ${modelWidth}x${modelHeight}`);
      console.log(`Model type: ${isYOLOv5su ? 'YOLOv5su' : 'YOLOv5s'}, Coordinate format: ${coordinateFormat}`);
      console.log(`Detection length: ${detectionLength}, Num detections: ${numDetections}`);
      console.log(`Output dimensions: [${dims.join(', ')}], Transposed: ${isTransposed}`);
    }

    const rawDetections = [];
    const classDetections = {};

    // transpose된 데이터에서 값을 가져오는 헬퍼 함수
    const getValue = (detectionIndex, valueIndex) => {
      if (isTransposed) {
        // [1, 84, 8400] 형태에서: data[valueIndex * numDetections + detectionIndex]
        return data[valueIndex * numDetections + detectionIndex];
      } else {
        // [1, 8400, 84] 형태에서: data[detectionIndex * detectionLength + valueIndex]
        return data[detectionIndex * detectionLength + valueIndex];
      }
    };

    for (let i = 0; i < numDetections; i++) {
      let bbox, confidence, classIndex, className;

      if (isYOLOv5su) {
        if (coordinateFormat === 'corner' && detectionLength === 6) {
          // YOLOv5su corner format: [x1, y1, x2, y2, confidence, class_id]
          let x1 = getValue(i, 0);
          let y1 = getValue(i, 1);
          let x2 = getValue(i, 2);
          let y2 = getValue(i, 3);
          confidence = getValue(i, 4);
          classIndex = Math.round(getValue(i, 5));
          
          // confidence 값이 sigmoid를 거쳐야 하는지 확인
          if (confidence > 1) {
            confidence = 1 / (1 + Math.exp(-confidence));
          }
          
          if (debugMode && i < 3) {
            console.log(`Raw detection ${i}: [${x1.toFixed(4)}, ${y1.toFixed(4)}, ${x2.toFixed(4)}, ${y2.toFixed(4)}, ${confidence.toFixed(4)}, ${classIndex}]`);
          }
          
          // 좌표가 0-1 범위인지 확인 (정규화된 좌표)
          if (x1 <= 1 && y1 <= 1 && x2 <= 1 && y2 <= 1 && x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0) {
            // 정규화된 좌표를 캔버스 크기로 변환
            x1 *= canvasWidth;
            y1 *= canvasHeight;
            x2 *= canvasWidth;
            y2 *= canvasHeight;
          } else {
            // 이미 픽셀 좌표인 경우 모델 크기에서 캔버스 크기로 스케일링
            const scaleX = canvasWidth / modelWidth;
            const scaleY = canvasHeight / modelHeight;
            x1 *= scaleX;
            y1 *= scaleY;
            x2 *= scaleX;
            y2 *= scaleY;
          }
          
          // x1,y1,x2,y2를 x,y,width,height 형태로 변환
          const x = Math.min(x1, x2);
          const y = Math.min(y1, y2);
          const width = Math.abs(x2 - x1);
          const height = Math.abs(y2 - y1);
          
          // 유효성 검사 - 너무 작거나 화면 밖의 박스 제거
          if (width < 5 || height < 5 || x < -canvasWidth || y < -canvasHeight || 
              x > canvasWidth * 2 || y > canvasHeight * 2) {
            if (debugMode && i < 3) {
              console.log(`Invalid bbox rejected: [${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}]`);
            }
            continue;
          }
          
          bbox = [x, y, width, height];
          className = classes[classIndex] || `클래스 ${classIndex}`;
          
        } else if (coordinateFormat === 'center') {
          // YOLOv5su center format: [cx, cy, w, h, class1, ..., classN] (84개)
          let cx = getValue(i, 0);
          let cy = getValue(i, 1);
          let width = getValue(i, 2);
          let height = getValue(i, 3);
          
          // 클래스 확률 계산 (objectness 없음)
          let maxClassProb = 0;
          classIndex = 0;
          const numClasses = modelInfoRef.current.numClasses;
          
          for (let j = 4; j < Math.min(4 + numClasses, detectionLength); j++) {
            const prob = getValue(i, j);
            // 이미 sigmoid가 적용된 값일 가능성이 높음
            const finalProb = prob > 1 ? (1 / (1 + Math.exp(-prob))) : prob;
            if (finalProb > maxClassProb) {
              maxClassProb = finalProb;
              classIndex = j - 4;
            }
          }
          
          confidence = maxClassProb;
          className = classes[classIndex] || `클래스 ${classIndex}`;
          
          if (debugMode && i < 3) {
            console.log(`Raw center detection ${i}: [${cx.toFixed(4)}, ${cy.toFixed(4)}, ${width.toFixed(4)}, ${height.toFixed(4)}], conf=${confidence.toFixed(4)}, class=${classIndex}`);
          }
          
          // 좌표 변환 (정규화 여부 확인)
          if (cx <= 1 && cy <= 1 && width <= 1 && height <= 1 && cx >= 0 && cy >= 0) {
            // 정규화된 좌표
            cx *= canvasWidth;
            cy *= canvasHeight;
            width *= canvasWidth;
            height *= canvasHeight;
          } else {
            // 픽셀 좌표
            const scaleX = canvasWidth / modelWidth;
            const scaleY = canvasHeight / modelHeight;
            cx *= scaleX;
            cy *= scaleY;
            width *= scaleX;
            height *= scaleY;
          }
          
          const x = cx - width/2;
          const y = cy - height/2;
          
          // 유효성 검사
          if (width < 5 || height < 5 || x < -canvasWidth || y < -canvasHeight || 
              x > canvasWidth * 2 || y > canvasHeight * 2) {
            if (debugMode && i < 3) {
              console.log(`Invalid center bbox rejected: [${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}]`);
            }
            continue;
          }
          
          bbox = [x, y, width, height];
          
          if (debugMode && i < 3) {
            console.log(`Processed center detection ${i}: bbox=[${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}], conf=${confidence.toFixed(3)}, class=${classIndex}(${className})`);
          }
        }
        
      } else {
        // YOLOv5s 형태: [cx, cy, w, h, objectness, class1, ..., class80]
        const objectness = 1 / (1 + Math.exp(-getValue(i, 4)));
        let maxClassProb = 0;
        classIndex = 0;
        
        const numClasses = modelInfoRef.current.numClasses;
        for (let j = 5; j < 5 + numClasses; j++) {
          const prob = 1 / (1 + Math.exp(-getValue(i, j)));
          if (prob > maxClassProb) {
            maxClassProb = prob;
            classIndex = j - 5;
          }
        }
        
        confidence = objectness * maxClassProb;
        className = classes[classIndex] || `클래스 ${classIndex}`;

        // Extract bounding box coordinates (center format을 corner format으로 변환)
        const scaleX = canvasWidth / modelWidth;
        const scaleY = canvasHeight / modelHeight;
        const cx = getValue(i, 0) * scaleX;
        const cy = getValue(i, 1) * scaleY;
        const width = getValue(i, 2) * scaleX;
        const height = getValue(i, 3) * scaleY;
        
        bbox = [cx - width/2, cy - height/2, width, height];
      }

      // Filter by confidence threshold
      if (confidence > confidenceThreshold) {
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
        if (debugMode && rawDetections.length < 5) {
          console.log(`탐지된 객체: ${className}, 인식률: ${(confidence * 100).toFixed(2)}%, 위치: [x=${Math.round(bbox[0])}, y=${Math.round(bbox[1])}, w=${Math.round(bbox[2])}, h=${Math.round(bbox[3])}]`);
        }

        rawDetections.push({
          bbox: bbox,
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

      if (detections.length > 0) {
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      }

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
        ctx.lineWidth = 1;

        // Draw bounding box with more visible style
        ctx.beginPath();
        ctx.rect(drawX, drawY, width, height);
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
                  }}
                  className="confidence-slider"
                  disabled={!isModelLoaded || isModelLoading}
                />
              </div>
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
