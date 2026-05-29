const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const switchCameraButton = document.getElementById('switchCameraButton');
const snapshotButton = document.getElementById('snapshotButton');
const fullscreenButton = document.getElementById('fullscreenButton');
const themeButton = document.getElementById('themeButton');
const qualitySelect = document.getElementById('qualitySelect');
const gridToggle = document.getElementById('gridToggle');
const clearHistoryButton = document.getElementById('clearHistoryButton');

const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const cameraHint = document.getElementById('cameraHint');
const localVideo = document.getElementById('localVideo');
const captureCanvas = document.getElementById('captureCanvas');
const processedFrame = document.getElementById('processedFrame');
const placeholder = document.getElementById('placeholder');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const cameraWrap = document.getElementById('cameraWrap');
const resultWrap = document.getElementById('resultWrap');
const resultPanel = document.getElementById('resultPanel');
const gestureHistory = document.getElementById('gestureHistory');
const actionHint = document.getElementById('actionHint');
const sessionTime = document.getElementById('sessionTime');
const resolution = document.getElementById('resolution');

let stream = null;
let processing = false;
let running = false;
let timerId = null;
let sessionTimerId = null;
let sessionStartedAt = null;
let lastFrameAt = 0;
let facingMode = 'user';
let gestureCandidate = '';
let candidateSince = 0;
let stableGesture = '';
let historyWasCleared = false;

const qualityMap = {
  low: { maxWidth: 480, interval: 20, jpeg: 0.66, label: 'экономное' },
  balanced: { maxWidth: 720, interval: 20, jpeg: 0.78, label: 'обычное' },
  high: { maxWidth: 960, interval: 20, jpeg: 0.86, label: 'чёткое' },
};

function currentQuality() {
  const value = qualitySelect?.value || 'balanced';
  return qualityMap[value] || qualityMap.balanced;
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function setStatus(text, type = 'ok') {
  if (statusText) statusText.textContent = text;
  if (!statusDot) return;
  statusDot.classList.toggle('bad', type === 'bad');
  statusDot.classList.toggle('wait', type === 'wait');
}

function setHint(message) {
  if (cameraHint) cameraHint.innerHTML = message;
}

function formatTime(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function updateSessionTimer() {
  if (!sessionStartedAt || !sessionTime) return;
  sessionTime.textContent = formatTime(performance.now() - sessionStartedAt);
}

function startSessionTimer() {
  sessionStartedAt = performance.now();
  updateSessionTimer();
  if (sessionTimerId) clearInterval(sessionTimerId);
  sessionTimerId = setInterval(updateSessionTimer, 500);
}

function stopSessionTimer() {
  if (sessionTimerId) {
    clearInterval(sessionTimerId);
    sessionTimerId = null;
  }
}

function getActionText(gesture, handsCount) {
  if (!handsCount) return 'Покажите руку в кадре.';
  return gesture || '—';
}

function updateState(state) {
  const hands = state?.hands_count ?? 0;
  const fingers = state?.fingers_count ?? 0;
  const gesture = state?.gesture ?? '—';

  setText('hands', hands);
  setText('fingers', fingers);
  setText('gesture', gesture);
  setText('fps', Number(state?.fps ?? 0).toFixed(1));
  setText('frames', state?.frames ?? 0);

  if (actionHint) actionHint.textContent = getActionText(gesture, hands);
  maybeAddGestureToHistory(gesture, hands, fingers);
}

function maybeAddGestureToHistory(gesture, hands, fingers) {
  if (!gesture || !hands || gesture === 'Рука не обнаружена') return;

  const now = performance.now();
  if (gesture !== gestureCandidate) {
    gestureCandidate = gesture;
    candidateSince = now;
    return;
  }

  const isStable = now - candidateSince > 450;
  if (!isStable || gesture === stableGesture) return;

  stableGesture = gesture;
  addHistoryItem(gesture, fingers);
}

function addHistoryItem(gesture, fingers) {
  if (!gestureHistory) return;

  if (!historyWasCleared) {
    gestureHistory.innerHTML = '';
    historyWasCleared = true;
  }

  const item = document.createElement('li');
  const time = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  item.innerHTML = `<span>${time}</span><strong>${gesture}</strong><em>${fingers} пальцев</em>`;
  gestureHistory.prepend(item);

  while (gestureHistory.children.length > 8) {
    gestureHistory.lastElementChild.remove();
  }
}

function clearHistory() {
  stableGesture = '';
  gestureCandidate = '';
  historyWasCleared = false;
  if (gestureHistory) {
    gestureHistory.innerHTML = '<li class="muted-row">История очищена.</li>';
  }
}

function isLocalAddress() {
  const host = window.location.hostname;
  return host === 'localhost' || host === '127.0.0.1' || host === '::1';
}

function hasSecureCameraContext() {
  return window.isSecureContext || isLocalAddress();
}

function getCameraApi() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    return constraints => navigator.mediaDevices.getUserMedia(constraints);
  }

  const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
  if (!legacy) return null;

  return constraints => new Promise((resolve, reject) => legacy.call(navigator, constraints, resolve, reject));
}

function explainCameraError(error) {
  const name = error?.name || '';

  if (!hasSecureCameraContext()) {
    return 'Откройте через <b>http://127.0.0.1:5000</b>, <b>localhost</b> или HTTPS.';
  }

  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Разрешите доступ к камере в браузере.';
  }

  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'Камера не найдена или занята.';
  }

  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Камера занята другой программой.';
  }

  if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') {
    return 'Попробуйте качество «Экономное».';
  }

  if (name === 'SecurityError') {
    return 'Используйте localhost или HTTPS.';
  }

  return error?.message || 'Камера не запустилась.';
}

async function requestCamera(mode = 'user') {
  const getUserMedia = getCameraApi();
  if (!getUserMedia) {
    throw new Error('Браузер не поддерживает камеру.');
  }

  if (!hasSecureCameraContext()) {
    const error = new Error('Insecure context');
    error.name = 'SecurityError';
    throw error;
  }

  const attempts = [
    { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: mode } }, audio: false },
    { video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: { ideal: mode } }, audio: false },
    { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: { ideal: mode } }, audio: false },
    { video: { facingMode: { ideal: mode } }, audio: false },
    { video: true, audio: false },
  ];

  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') break;
      if (error?.name === 'SecurityError') break;
    }
  }

  throw lastError || new Error('Камеру открыть не удалось.');
}

function startProcessingLoop() {
  if (timerId) clearInterval(timerId);
  sendFrame();
  timerId = setInterval(sendFrame, currentQuality().interval);
}

async function startCamera() {
  if (!startButton || !stopButton) return;

  try {
    setStatus('Открываем камеру...', 'wait');
    startButton.disabled = true;

    stream = await requestCamera(facingMode);
    localVideo.srcObject = stream;
    await localVideo.play();

    running = true;
    stopButton.disabled = false;
    localVideo.style.display = 'block';
    if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
    if (resolution) resolution.textContent = `${localVideo.videoWidth || '—'}×${localVideo.videoHeight || '—'}`;

    startSessionTimer();
    setHint(`Камера включена. Качество: <b>${currentQuality().label}</b>.`);
    setStatus('Камера работает');
    startProcessingLoop();
  } catch (error) {
    startButton.disabled = false;
    stopButton.disabled = true;
    setStatus('Камера не запущена', 'bad');
    setHint(explainCameraError(error));
  }
}

function stopCamera() {
  running = false;
  processing = false;

  if (timerId) {
    clearInterval(timerId);
    timerId = null;
  }

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  if (localVideo) {
    localVideo.pause();
    localVideo.srcObject = null;
    localVideo.style.display = 'none';
  }

  if (cameraPlaceholder) cameraPlaceholder.style.display = 'grid';
  if (resolution) resolution.textContent = '—';

  stopSessionTimer();
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus('Камера остановлена', 'wait');
  setHint('Камера остановлена.');
}

async function switchCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  const label = facingMode === 'user' ? 'передняя камера' : 'задняя камера';

  if (!running) {
    setHint(`Выбрана ${label}.`);
    return;
  }

  stopCamera();
  setHint(`Камера: ${label}...`);
  await startCamera();
}

async function sendFrame() {
    if (!running || processing || !localVideo.videoWidth || !localVideo.videoHeight) return;

    const settings = currentQuality();
    const now = performance.now();
    if (now - lastFrameAt < settings.interval - 10) return;
    lastFrameAt = now;
    processing = true;

    const scale = Math.min(1, settings.maxWidth / localVideo.videoWidth);
    captureCanvas.width = Math.round(localVideo.videoWidth * scale);
    captureCanvas.height = Math.round(localVideo.videoHeight * scale);

    const ctx = captureCanvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(localVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob(async (frameBlob) => {
        if (!frameBlob) {
            processing = false;
            return;
        }

        try {
            const formData = new FormData();
            formData.append('frame', frameBlob, 'frame.jpg');

            const response = await fetch('/process_frame', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errText = await response.text();
                setStatus('Ошибка обработки', 'bad');
                setHint(errText || 'Кадр не обработан.');
                return;
            }

            const imageBlob = await response.blob();
            const url = URL.createObjectURL(imageBlob);

            processedFrame.src = url;
            processedFrame.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';

            setStatus('Распознавание работает');

        } catch (error) {
            console.error(error);
            setStatus('Нет связи с сервером', 'bad');
            setHint('Сервер недоступен.');
        } finally {
            processing = false;
        }
    }, 'image/jpeg', settings.jpeg);
}

function downloadDataUrl(dataUrl, filename) {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function makeSnapshot() {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

  if (processedFrame?.src?.startsWith('data:image')) {
    downloadDataUrl(processedFrame.src, `gesture-result-${stamp}.jpg`);
    setHint('Снимок сохранён.');
    return;
  }

  if (running && localVideo.videoWidth && localVideo.videoHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = localVideo.videoWidth;
    canvas.height = localVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
    downloadDataUrl(canvas.toDataURL('image/jpeg', 0.9), `gesture-camera-${stamp}.jpg`);
    setHint('Снимок сохранён.');
    return;
  }

  setHint('Сначала включите камеру.');
}

async function openFullscreen() {
  const target = resultPanel || resultWrap;
  if (!target) return;

  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await target.requestFullscreen();
    }
  } catch (error) {
    setHint('Полноэкранный режим не открылся.');
  }
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  localStorage.setItem('gesture-theme', theme);
  if (themeButton) themeButton.textContent = theme === 'dark' ? 'Светлая тема' : 'Тёмная тема';
}

function toggleTheme() {
  const current = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(current);
}

function toggleGrid() {
  const enabled = Boolean(gridToggle?.checked);
  cameraWrap?.classList.toggle('show-grid', enabled);
  resultWrap?.classList.toggle('show-grid', enabled);
}

function handleQualityChange() {
  if (running) startProcessingLoop();
  setHint(`Качество: <b>${currentQuality().label}</b>.`);
}

async function checkHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    if (data.ok) {
      setStatus('Сервер готов');
      if (!hasSecureCameraContext()) {
        setStatus('Нужен localhost или HTTPS', 'bad');
        setHint('Откройте <b>http://127.0.0.1:5000</b> или HTTPS.');
      }
    } else {
      setStatus('Сервер не готов', 'bad');
      setHint(data.error || 'Проверьте модель.');
    }
  } catch (error) {
    setStatus('Сервер недоступен', 'bad');
    setHint('Запустите <b>python app.py</b>.');
  }
}

const savedTheme = localStorage.getItem('gesture-theme') || 'light';
applyTheme(savedTheme);

if (startButton) startButton.addEventListener('click', startCamera);
if (stopButton) stopButton.addEventListener('click', stopCamera);
if (switchCameraButton) switchCameraButton.addEventListener('click', switchCamera);
if (snapshotButton) snapshotButton.addEventListener('click', makeSnapshot);
if (fullscreenButton) fullscreenButton.addEventListener('click', openFullscreen);
if (themeButton) themeButton.addEventListener('click', toggleTheme);
if (qualitySelect) qualitySelect.addEventListener('change', handleQualityChange);
if (gridToggle) gridToggle.addEventListener('change', toggleGrid);
if (clearHistoryButton) clearHistoryButton.addEventListener('click', clearHistory);

checkHealth();
