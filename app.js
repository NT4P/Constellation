import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const video = document.querySelector("#video");
const canvas = document.querySelector("#visualiser");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const statusElement = document.querySelector("#status");
const faceCountElement = document.querySelector("#faceCount");

let faceLandmarker;
let cameraStream;
let animationFrame;
let latestFaces = [];
let faceClouds = [];
let sequenceStart = 0;

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);

camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 1);

function createParticleTexture() {
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = 64;
  textureCanvas.height = 64;

  const context = textureCanvas.getContext("2d");
  const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);

  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.25, "rgba(255,255,255,0.9)");
  gradient.addColorStop(0.7, "rgba(255,255,255,0.25)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");

  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);

  return new THREE.CanvasTexture(textureCanvas);
}

const particleTexture = createParticleTexture();

async function initialiseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/" +
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 4,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

function expandLandmarks(landmarks, particlesPerPoint = 7) {
  const expanded = [];

  landmarks.forEach((landmark, index) => {
    const regionWeight = index % 7 === 0 ? 1.7 : 1;

    for (let i = 0; i < particlesPerPoint * regionWeight; i++) {
      expanded.push({
        x: landmark.x - 0.5 + (Math.random() - 0.5) * 0.025,
        y: -(landmark.y - 0.5) + (Math.random() - 0.5) * 0.025,
        z: landmark.z + (Math.random() - 0.5) * 0.025
      });
    }
  });

  return expanded;
}

function createFaceCloud(landmarks, position) {
  const points = expandLandmarks(landmarks);

  const positions = [];
  const sizes = [];

  points.forEach((point) => {
    positions.push(point.x, point.y, point.z);
    sizes.push(0.025 + Math.random() * 0.04);
  });

  const geometry = new THREE.BufferGeometry();

  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

  geometry.setAttribute(
    "size",
    new THREE.Float32BufferAttribute(sizes, 1)
  );

  const material = new THREE.PointsMaterial({
    color: 0xf2f2f2,
    size: 0.035,
    map: particleTexture,
    transparent: true,
    opacity: 0.8,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  const cloud = new THREE.Points(geometry, material);

  cloud.position.set(...position);

  cloud.userData = {
    origin: new THREE.Vector3(...position),
    rotationSpeed: {
      x: (Math.random() - 0.5) * 0.001,
      y: (Math.random() - 0.5) * 0.0015,
      z: (Math.random() - 0.5) * 0.0008
    },
    phase: Math.random() * Math.PI * 2
  };

  scene.add(cloud);
  faceClouds.push(cloud);
}

function clearFaceClouds() {
  faceClouds.forEach((cloud) => {
    cloud.geometry.dispose();
    cloud.material.dispose();
    scene.remove(cloud);
  });

  faceClouds = [];
}

function updateClouds(time) {
  if (!sequenceStart) return;

  const elapsed = time - sequenceStart;
  const joiningProgress = Math.max(
    0,
    Math.min((elapsed - 9000) / 7000, 1)
  );

  const eased =
    joiningProgress * joiningProgress * (3 - 2 * joiningProgress);

  faceClouds.forEach((cloud) => {
    const data = cloud.userData;

    if (joiningProgress === 0) {
      cloud.rotation.x += data.rotationSpeed.x;
      cloud.rotation.y += data.rotationSpeed.y;
      cloud.rotation.z += data.rotationSpeed.z;

      cloud.position.x =
        data.origin.x +
        Math.sin(time * 0.001 + data.phase) * 0.01;

      cloud.position.y =
        data.origin.y +
        Math.cos(time * 0.0008 + data.phase) * 0.01;
    } else {
      cloud.position.lerpVectors(
        data.origin,
        new THREE.Vector3(0, 0, 0),
        eased
      );

      cloud.rotation.x *= 0.985;
      cloud.rotation.y *= 0.985;
      cloud.rotation.z *= 0.985;
    }
  });
}

function render(time) {
  updateClouds(time);
  renderer.render(scene, camera);
  animationFrame = requestAnimationFrame(render);
}

async function startCamera() {
  try {
    statusElement.textContent = "Loading face model...";

    if (!faceLandmarker) {
      await initialiseLandmarker();
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    video.srcObject = cameraStream;
    await video.play();

    startButton.disabled = true;
    stopButton.disabled = false;
    sequenceStart = performance.now();

    statusElement.textContent = "Camera active";
    animationFrame = requestAnimationFrame(render);
    detectFaces();
  } catch (error) {
    console.error(error);
    statusElement.textContent =
      "Camera could not start. Check browser permissions.";
  }
}

function stopCamera() {
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }

  cameraStream = null;
  latestFaces = [];
  clearFaceClouds();

  startButton.disabled = false;
  stopButton.disabled = true;
  faceCountElement.textContent = "0";
  statusElement.textContent = "Camera stopped";
}

function detectFaces() {
  if (!faceLandmarker || !cameraStream) return;

  const result = faceLandmarker.detectForVideo(
    video,
    performance.now()
  );

  latestFaces = result.faceLandmarks || [];
  faceCountElement.textContent = latestFaces.length;

  clearFaceClouds();

  const positions = [
    [-1.25, 0.2, 0],
    [1.25, 0.2, 0],
    [0, -0.8, 0],
    [0, 1.1, -0.1]
  ];

  latestFaces.forEach((face, index) => {
    createFaceCloud(
      face,
      positions[index] || [0, 0, 0]
    );
  });

  statusElement.textContent =
    latestFaces.length > 0
      ? `${latestFaces.length} face(s) detected`
      : "No face detected";

  requestAnimationFrame(detectFaces);
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

video.style.display = "none";
stopButton.disabled = true;
