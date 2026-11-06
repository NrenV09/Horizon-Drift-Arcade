import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import * as CANNON from 'cannon-es';
import { createNoise2D } from 'simplex-noise';
import {
  CAR_CATALOG,
  CAR_ORDER,
  CarId,
  CarSpec,
  computeAnalogSteerFromPad,
  computeEffectiveTouchDriveInput,
  computeSafeNitroImpulse,
  computeSurfacePhysicsProfile,
  computeTerrainHeight,
  createInitialChaseCameraState,
  getCarSpec,
  getHighwayElevationAtAngle,
  getHighwayPointAtAngle,
  getRoadDistanceInfo,
  getScenerySpec,
  resolveMapColliderImpact,
  ROAD_HALF_WIDTH,
  SCENERY_CATALOG,
  SCENERY_ORDER,
  SceneryId,
  ScenerySpec,
  smoothstep,
  stepAutomotiveHorizontalPhysics,
  stepChaseCameraRelative,
  stepVerticalGroundPhysics,
  SurfacePhysicsProfile,
  TimeOfDayMode,
} from './rules';

// Detect software WebGL (SwiftShader / llvmpipe) from renderer string
const isSoftwareGl = (() => {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
})();

const WORLD_SIZE = 3000;
const OASIS_X = -210;
const OASIS_Z = 40;
const OASIS_RADIUS = 152;
const OASIS_WATER_Y = -1.6;

function createSeededRandom(seed = 4202026) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const rng = createSeededRandom(91827364);
const noise2D = createNoise2D(rng);

function getTerrainY(x: number, z: number): number {
  return computeTerrainHeight(x, z, noise2D);
}

function getWheelGroundPose(carX: number, carZ: number, carYaw: number, spec: CarSpec) {
  const fwdX = -Math.sin(carYaw);
  const fwdZ = -Math.cos(carYaw);
  const rightX = Math.cos(carYaw);
  const rightZ = -Math.sin(carYaw);
  const halfWb = spec.wheelbase * 0.5;
  const halfTw = spec.trackWidth * 0.5;

  const flX = carX + fwdX * halfWb - rightX * halfTw;
  const flZ = carZ + fwdZ * halfWb - rightZ * halfTw;
  const frX = carX + fwdX * halfWb + rightX * halfTw;
  const frZ = carZ + fwdZ * halfWb + rightZ * halfTw;
  const rlX = carX - fwdX * halfWb - rightX * halfTw;
  const rlZ = carZ - fwdZ * halfWb - rightZ * halfTw;
  const rrX = carX - fwdX * halfWb + rightX * halfTw;
  const rrZ = carZ - fwdZ * halfWb + rightZ * halfTw;

  const hFL = getTerrainY(flX, flZ);
  const hFR = getTerrainY(frX, frZ);
  const hRL = getTerrainY(rlX, rlZ);
  const hRR = getTerrainY(rrX, rrZ);

  const avgGroundY = (hFL + hFR + hRL + hRR) * 0.25;
  const frontAvg = (hFL + hFR) * 0.5;
  const rearAvg = (hRL + hRR) * 0.5;
  const rightAvg = (hFR + hRR) * 0.5;
  const leftAvg = (hFL + hRL) * 0.5;

  return {
    hFL,
    hFR,
    hRL,
    hRR,
    flX,
    flZ,
    frX,
    frZ,
    rlX,
    rlZ,
    rrX,
    rrZ,
    avgGroundY,
    terrainPitch: Math.atan2(frontAvg - rearAvg, spec.wheelbase),
    terrainRoll: Math.atan2(leftAvg - rightAvg, spec.trackWidth),
  };
}

// Web Audio Engine & Tire Screech Synthesizer
class SoundEngine {
  ctx: AudioContext | null = null;
  muted = false;
  engineOsc1: OscillatorNode | null = null;
  engineOsc2: OscillatorNode | null = null;
  engineGain: GainNode | null = null;
  engineFilter: BiquadFilterNode | null = null;
  tireNoiseSource: AudioBufferSourceNode | null = null;
  tireGain: GainNode | null = null;
  tireFilter: BiquadFilterNode | null = null;

  init() {
    if (this.ctx || isSoftwareGl) return;
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      this.ctx = new AudioCtx();
      this.engineOsc1 = this.ctx.createOscillator();
      this.engineOsc2 = this.ctx.createOscillator();
      this.engineOsc1.type = 'sawtooth';
      this.engineOsc2.type = 'triangle';
      this.engineFilter = this.ctx.createBiquadFilter();
      this.engineFilter.type = 'lowpass';
      this.engineFilter.frequency.value = 450;
      this.engineGain = this.ctx.createGain();
      this.engineGain.gain.value = 0.0001;
      this.engineOsc1.connect(this.engineFilter);
      this.engineOsc2.connect(this.engineFilter);
      this.engineFilter.connect(this.engineGain);
      this.engineGain.connect(this.ctx.destination);
      this.engineOsc1.start();
      this.engineOsc2.start();

      const bufferSize = this.ctx.sampleRate * 2;
      const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
      }
      this.tireNoiseSource = this.ctx.createBufferSource();
      this.tireNoiseSource.buffer = noiseBuffer;
      this.tireNoiseSource.loop = true;
      this.tireFilter = this.ctx.createBiquadFilter();
      this.tireFilter.type = 'bandpass';
      this.tireFilter.frequency.value = 1750;
      this.tireFilter.Q.value = 3.2;
      this.tireGain = this.ctx.createGain();
      this.tireGain.gain.value = 0.0001;
      this.tireNoiseSource.connect(this.tireFilter);
      this.tireFilter.connect(this.tireGain);
      this.tireGain.connect(this.ctx.destination);
      this.tireNoiseSource.start();
    } catch {
      // Ignore audio restricted context errors
    }
  }

  isResuming = false;
  curFreq = 60;
  curFilterFreq = 450;
  curEngineGain = 0.0001;
  curTireGain = 0.0001;

  update(
    rpmNorm: number,
    throttle: number,
    isDrifting: boolean,
    isPlaying: boolean,
    kmh = 0,
    baseHz = 54,
    dt = 0.016,
  ) {
    if (!this.ctx || !this.engineGain || !this.engineOsc1 || !this.engineOsc2) return;
    if (this.muted || !isPlaying) {
      this.curEngineGain = THREE.MathUtils.damp(this.curEngineGain, 0.0001, 14, dt);
      this.curTireGain = THREE.MathUtils.damp(this.curTireGain, 0.0001, 14, dt);
      this.engineGain.gain.value = this.curEngineGain;
      if (this.tireGain) this.tireGain.gain.value = this.curTireGain;
      return;
    }
    if (this.ctx.state === 'suspended' && !this.isResuming) {
      this.isResuming = true;
      this.ctx
        .resume()
        .catch(() => {})
        .finally(() => {
          this.isResuming = false;
        });
    }
    const targetFreq = baseHz + rpmNorm * 165;
    this.curFreq = THREE.MathUtils.damp(this.curFreq, targetFreq, 20, dt);
    this.engineOsc1.frequency.value = this.curFreq;
    this.engineOsc2.frequency.value = this.curFreq * 1.5;

    if (this.engineFilter) {
      const targetFilter = 440 + rpmNorm * 980;
      this.curFilterFreq = THREE.MathUtils.damp(this.curFilterFreq, targetFilter, 18, dt);
      this.engineFilter.frequency.value = this.curFilterFreq;
    }

    const targetGain = 0.038 + Math.abs(throttle) * 0.045 + (isDrifting ? 0.02 : 0);
    this.curEngineGain = THREE.MathUtils.damp(this.curEngineGain, targetGain, 16, dt);
    this.engineGain.gain.value = this.curEngineGain;

    if (this.tireGain) {
      const screech = isDrifting && kmh > 14;
      const targetTire = screech ? Math.min(0.055, 0.02 + (kmh / 200) * 0.035) : 0.0001;
      this.curTireGain = THREE.MathUtils.damp(this.curTireGain, targetTire, screech ? 22 : 12, dt);
      this.tireGain.gain.value = this.curTireGain;
    }
  }

  playChime(freq = 660, duration = 0.18) {
    if (!this.ctx || this.muted || isSoftwareGl) return;
    try {
      if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(freq * 1.5, this.ctx.currentTime + duration);
      gain.gain.setValueAtTime(0.09, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch {
      // Ignore
    }
  }
}

const soundEngine = new SoundEngine();
window.addEventListener('pointerdown', () => soundEngine.init(), { once: true });
window.addEventListener('keydown', () => soundEngine.init(), { once: true });

// Renderer & Scene Initialization
const container = document.getElementById('canvas-container')!;
const renderer = new THREE.WebGLRenderer({
  antialias: !isSoftwareGl,
  powerPreference: 'high-performance',
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.54;
renderer.shadowMap.enabled = !isSoftwareGl;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x6e3a4c);
const sceneFog = new THREE.FogExp2(0x6e3a4c, 0.00048);
scene.fog = sceneFog;

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.5, 2400);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.5;
pmrem.dispose();

// Preetham Sky & Celestial Dome
const sky = new Sky();
sky.scale.setScalar(5000);
scene.add(sky);
const sunDir = new THREE.Vector3();
const skyUniforms = sky.material.uniforms;
skyUniforms.turbidity.value = 3.6;
skyUniforms.rayleigh.value = 1.85;
skyUniforms.mieCoefficient.value = 0.0018;
skyUniforms.mieDirectionalG.value = 0.93;

const sunDiscGeo = new THREE.SphereGeometry(40, 24, 24);
const sunDiscMat = new THREE.MeshBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.98,
  blending: THREE.AdditiveBlending,
  fog: false,
  depthWrite: false,
  toneMapped: false,
});
const sunDiscMesh = new THREE.Mesh(sunDiscGeo, sunDiscMat);
sunDiscMesh.frustumCulled = false;
scene.add(sunDiscMesh);

const sunCoronaGeo = new THREE.SphereGeometry(78, 24, 24);
const sunCoronaMat = new THREE.MeshBasicMaterial({
  color: 0xff9e64,
  transparent: true,
  opacity: 0.45,
  blending: THREE.AdditiveBlending,
  fog: false,
  depthWrite: false,
  toneMapped: false,
});
const sunCoronaMesh = new THREE.Mesh(sunCoronaGeo, sunCoronaMat);
sunDiscMesh.add(sunCoronaMesh);

const starCount = 550;
const starPositions = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {
  const u = rng();
  const v = rng();
  const theta = u * Math.PI * 2;
  const phi = Math.acos(0.12 + v * 0.88);
  const r = 1850;
  starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
  starPositions[i * 3 + 1] = r * Math.cos(phi);
  starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
}
const starGeo = new THREE.BufferGeometry();
starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
const starMat = new THREE.PointsMaterial({
  color: 0xffffff,
  size: 2.6,
  sizeAttenuation: false,
  transparent: true,
  opacity: 0,
  blending: THREE.AdditiveBlending,
  fog: false,
  depthWrite: false,
  toneMapped: false,
});
const starField = new THREE.Points(starGeo, starMat);
starField.frustumCulled = false;
scene.add(starField);

const orbDir = new THREE.Vector3();
const initPhi = THREE.MathUtils.degToRad(85.5);
const initTheta = THREE.MathUtils.degToRad(168);
sunDir.setFromSphericalCoords(1, initPhi, initTheta);
orbDir.copy(sunDir);
skyUniforms.sunPosition.value.copy(sunDir);
sunDiscMesh.position.copy(orbDir).multiplyScalar(1750);

const hemiLight = new THREE.HemisphereLight(0xffb076, 0x2e2238, 0.85);
hemiLight.position.set(0, 150, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xff964f, 2.1);
dirLight.position.copy(sunDir).multiplyScalar(160);
dirLight.castShadow = !isSoftwareGl;
if (!isSoftwareGl) {
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.left = -75;
  dirLight.shadow.camera.right = 75;
  dirLight.shadow.camera.top = 75;
  dirLight.shadow.camera.bottom = -75;
  dirLight.shadow.camera.near = 10;
  dirLight.shadow.camera.far = 380;
  dirLight.shadow.bias = -0.0005;
  dirLight.shadow.normalBias = 0.02;
}
scene.add(dirLight);
scene.add(dirLight.target);

const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(
    Math.max(1, Math.floor(window.innerWidth * 0.5)),
    Math.max(1, Math.floor(window.innerHeight * 0.5)),
  ),
  0.16,
  0.25,
  2.6,
);
bloomPass.enabled = !isSoftwareGl;
composer.addPass(bloomPass);
const outputPass = new OutputPass();
composer.addPass(outputPass);

// Procedural Textures
function createRoadTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#181c26';
  ctx.fillRect(0, 0, 512, 512);

  const gradL = ctx.createLinearGradient(60, 0, 220, 0);
  gradL.addColorStop(0, 'rgba(10, 12, 18, 0)');
  gradL.addColorStop(0.5, 'rgba(10, 12, 18, 0.35)');
  gradL.addColorStop(1, 'rgba(10, 12, 18, 0)');
  ctx.fillStyle = gradL;
  ctx.fillRect(60, 0, 160, 512);

  const gradR = ctx.createLinearGradient(292, 0, 452, 0);
  gradR.addColorStop(0, 'rgba(10, 12, 18, 0)');
  gradR.addColorStop(0.5, 'rgba(10, 12, 18, 0.35)');
  gradR.addColorStop(1, 'rgba(10, 12, 18, 0)');
  ctx.fillStyle = gradR;
  ctx.fillRect(292, 0, 160, 512);

  for (let i = 0; i < 7500; i++) {
    const rx = rng() * 512;
    const ry = rng() * 512;
    const c = 18 + Math.floor(rng() * 22);
    ctx.fillStyle = `rgb(${c}, ${c + 2}, ${c + 6})`;
    ctx.fillRect(rx, ry, 2, 2);
  }

  ctx.fillStyle = '#5c4033';
  ctx.fillRect(0, 0, 12, 512);
  ctx.fillRect(500, 0, 12, 512);
  for (let y = 0; y < 512; y += 32) {
    ctx.fillStyle = (y / 32) % 2 === 0 ? '#ff2a85' : '#f1f5f9';
    ctx.fillRect(10, y, 16, 32);
    ctx.fillRect(486, y, 16, 32);
  }
  ctx.fillStyle = '#e2e8f0';
  ctx.fillRect(32, 0, 7, 512);
  ctx.fillRect(473, 0, 7, 512);
  for (let y = 0; y < 512; y += 64) {
    ctx.fillStyle = '#ff8a2b';
    ctx.fillRect(248, y + 6, 5, 44);
    ctx.fillRect(259, y + 6, 5, 44);
    ctx.fillStyle = '#00f0ff';
    ctx.fillRect(254, y + 54, 4, 6);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

function createTerrainBumpTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 4500; i++) {
    const v = 85 + Math.floor(rng() * 85);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(rng() * 256, rng() * 256, 4, 2);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(140, 140);
  return tex;
}

function createContactShadowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d')!;
  const g = ctx.createRadialGradient(64, 64, 12, 64, 64, 60);
  g.addColorStop(0, 'rgba(0, 0, 0, 0.78)');
  g.addColorStop(0.65, 'rgba(0, 0, 0, 0.36)');
  g.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}

const physicsWorld = new CANNON.World();
physicsWorld.gravity.set(0, -20, 0);

// Terrain Geometry & Dynamic Scenery Vertex Coloring
const TERRAIN_SEGMENTS = 240;
const terrainGeom = new THREE.PlaneGeometry(WORLD_SIZE, WORLD_SIZE, TERRAIN_SEGMENTS, TERRAIN_SEGMENTS);
terrainGeom.rotateX(-Math.PI / 2);

const posAttr = terrainGeom.attributes.position;
const terrainColors = new Float32Array(posAttr.count * 3);
const terrainVertexMeta: Array<{ x: number; z: number; y: number; dist: number; roadElev: number }> = [];

for (let i = 0; i < posAttr.count; i++) {
  const x = posAttr.getX(i);
  const z = posAttr.getZ(i);
  const y = getTerrainY(x, z);
  const { distance, roadElevation } = getRoadDistanceInfo(x, z);
  let finalY = y;
  if (distance < 19.5) {
    finalY = y - 0.65 * (1 - smoothstep(14, 19.5, distance));
  }
  posAttr.setY(i, finalY);
  terrainVertexMeta.push({ x, z, y, dist: distance, roadElev: roadElevation });
}
terrainGeom.setAttribute('color', new THREE.BufferAttribute(terrainColors, 3));
terrainGeom.computeVertexNormals();

const _tmpColor = new THREE.Color();
const _cLakeBed = new THREE.Color();
const _cValleyLow = new THREE.Color();
const _cValleyMid = new THREE.Color();
const _cCanyonSlope = new THREE.Color();
const _cMesaTop = new THREE.Color();
const _cPeakCap = new THREE.Color();
const _cShoulder = new THREE.Color();

function applyTerrainPalette(spec: ScenerySpec) {
  const pal = spec.terrainPalette;
  _cLakeBed.setHex(pal.lakeBedHex);
  _cValleyLow.setHex(pal.valleyLowHex);
  _cValleyMid.setHex(pal.valleyMidHex);
  _cCanyonSlope.setHex(pal.canyonSlopeHex);
  _cMesaTop.setHex(pal.mesaTopHex);
  _cPeakCap.setHex(pal.peakCapHex);
  _cShoulder.setHex(pal.shoulderBlendHex);

  const colorAttr = terrainGeom.attributes.color as THREE.BufferAttribute;
  for (let i = 0; i < terrainVertexMeta.length; i++) {
    const { x, z, y, dist, roadElev } = terrainVertexMeta[i];
    const relH = y - roadElev;
    if (Math.hypot(x - OASIS_X, z - OASIS_Z) < 178 && y < 1.2) {
      _tmpColor.copy(_cLakeBed);
    } else if (relH < 4.5) {
      const t = Math.max(0, Math.min(1, (relH + 2) / 6.5));
      _tmpColor.copy(_cValleyLow).lerp(_cValleyMid, t * 0.7);
    } else if (relH < 24) {
      const t = (relH - 4.5) / 19.5;
      _tmpColor.copy(_cValleyMid).lerp(_cCanyonSlope, t);
      const strata = Math.sin(y * 1.35) * 0.5 + 0.5;
      _tmpColor.lerp(_cPeakCap, strata * 0.25);
    } else if (y > 64) {
      const t = Math.min(1, (y - 64) / 22);
      _tmpColor.copy(_cMesaTop).lerp(_cPeakCap, t);
    } else {
      const strata = Math.sin(y * 1.6) * 0.5 + 0.5;
      _tmpColor.copy(_cMesaTop).lerp(_cPeakCap, strata * 0.38);
    }
    if (dist < 22) {
      const t = Math.max(0, 1 - dist / 22);
      _tmpColor.lerp(_cShoulder, t * 0.75);
    }
    terrainColors[i * 3] = _tmpColor.r;
    terrainColors[i * 3 + 1] = _tmpColor.g;
    terrainColors[i * 3 + 2] = _tmpColor.b;
  }
  colorAttr.needsUpdate = true;
}

const bumpTex = createTerrainBumpTexture();
const terrainMat = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 0.84,
  metalness: 0.04,
  bumpMap: bumpTex,
  bumpScale: 0.35,
});
const terrainMesh = new THREE.Mesh(terrainGeom, terrainMat);
terrainMesh.receiveShadow = !isSoftwareGl;
scene.add(terrainMesh);

// Oasis / Fjord / Cyber Bay Water Surface
const waterGeo = new THREE.CircleGeometry(OASIS_RADIUS, 64);
waterGeo.rotateX(-Math.PI / 2);
const waterMat = new THREE.MeshStandardMaterial({
  color: 0x0ea5e9,
  roughness: 0.12,
  metalness: 0.45,
  bumpMap: bumpTex,
  bumpScale: 0.08,
  transparent: true,
  opacity: 0.9,
});
const waterMesh = new THREE.Mesh(waterGeo, waterMat);
waterMesh.position.set(OASIS_X, OASIS_WATER_Y, OASIS_Z);
waterMesh.receiveShadow = !isSoftwareGl;
scene.add(waterMesh);

// Closed-Loop Asphalt Highway Ribbon
const ROAD_STEPS = 600;
const roadPositions: number[] = [];
const roadUvs: number[] = [];
const roadIndices: number[] = [];
const roadRepeat = 64;
const roadOuterWidth = 16.2;

for (let i = 0; i <= ROAD_STEPS; i++) {
  const pt = getHighwayPointAtAngle((i / ROAD_STEPS) * Math.PI * 2);
  const elev = getTerrainY(pt.x, pt.z);
  const lxOuter = pt.x - pt.nx * roadOuterWidth;
  const lzOuter = pt.z - pt.nz * roadOuterWidth;
  const lyOuter = elev - 0.35;
  const lxInner = pt.x - pt.nx * ROAD_HALF_WIDTH;
  const lzInner = pt.z - pt.nz * ROAD_HALF_WIDTH;
  const lyInner = elev + 0.16;
  const rxInner = pt.x + pt.nx * ROAD_HALF_WIDTH;
  const rzInner = pt.z + pt.nz * ROAD_HALF_WIDTH;
  const ryInner = elev + 0.16;
  const rxOuter = pt.x + pt.nx * roadOuterWidth;
  const rzOuter = pt.z + pt.nz * roadOuterWidth;
  const ryOuter = elev - 0.35;

  roadPositions.push(
    lxOuter, lyOuter, lzOuter,
    lxInner, lyInner, lzInner,
    rxInner, ryInner, rzInner,
    rxOuter, ryOuter, rzOuter,
  );
  const v = (i / ROAD_STEPS) * roadRepeat;
  roadUvs.push(0, v, 0.03, v, 0.97, v, 1, v);
  if (i < ROAD_STEPS) {
    const base = i * 4;
    const next = (i + 1) * 4;
    for (let k = 0; k < 3; k++) {
      const a = base + k;
      const b = base + k + 1;
      const c = next + k;
      const d = next + k + 1;
      roadIndices.push(a, c, b, b, c, d);
    }
  }
}

const roadGeo = new THREE.BufferGeometry();
roadGeo.setAttribute('position', new THREE.Float32BufferAttribute(roadPositions, 3));
roadGeo.setAttribute('uv', new THREE.Float32BufferAttribute(roadUvs, 2));
roadGeo.setIndex(roadIndices);
roadGeo.computeVertexNormals();
const roadMesh = new THREE.Mesh(
  roadGeo,
  new THREE.MeshStandardMaterial({
    map: createRoadTexture(),
    roughness: 0.58,
    metalness: 0.14,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  }),
);
roadMesh.receiveShadow = !isSoftwareGl;
scene.add(roadMesh);

// Landmark 1: Festival Illuminated Ferris Wheel
const _dummy = new THREE.Object3D();
const _instColor = new THREE.Color();
const ferrisGroup = new THREE.Group();
const ferrisX = -115;
const ferrisZ = -175;
ferrisGroup.position.set(ferrisX, getTerrainY(ferrisX, ferrisZ), ferrisZ);
ferrisGroup.rotation.y = 0.42;
scene.add(ferrisGroup);

const steelMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.8, roughness: 0.25 });
const neonPinkMat = new THREE.MeshStandardMaterial({
  color: 0xff2a85,
  emissive: 0xff2a85,
  emissiveIntensity: 3.2,
  roughness: 0.2,
});
const neonCyanMat = new THREE.MeshStandardMaterial({
  color: 0x00f0ff,
  emissive: 0x00f0ff,
  emissiveIntensity: 3.2,
  roughness: 0.2,
});

const legGeo = new THREE.CylinderGeometry(0.45, 0.7, 44, 8);
for (const zOff of [-4.2, 4.2]) {
  for (const dir of [-1, 1]) {
    const leg = new THREE.Mesh(legGeo, steelMat);
    leg.position.set(dir * 9.5, 20, zOff);
    leg.rotation.z = dir * 0.42;
    ferrisGroup.add(leg);
  }
}

const wheelRimGroup = new THREE.Group();
wheelRimGroup.position.set(0, 39, 0);
ferrisGroup.add(wheelRimGroup);

const outerTorus = new THREE.TorusGeometry(32, 0.48, 10, 48);
const innerTorus = new THREE.TorusGeometry(24, 0.32, 8, 48);
for (const zOff of [-2.4, 2.4]) {
  const m1 = new THREE.Mesh(outerTorus, neonPinkMat);
  m1.position.z = zOff;
  const m2 = new THREE.Mesh(innerTorus, neonCyanMat);
  m2.position.z = zOff;
  wheelRimGroup.add(m1, m2);
}

const gondolas: THREE.Group[] = [];
const GONDOLA_COUNT = 16;
const gondolaGeo = new THREE.CylinderGeometry(1.6, 1.3, 2.4, 8);
const gondolaMats = [
  new THREE.MeshStandardMaterial({ color: 0xff2a85, emissive: 0xff2a85, emissiveIntensity: 2.5 }),
  new THREE.MeshStandardMaterial({ color: 0x00f0ff, emissive: 0x00f0ff, emissiveIntensity: 2.5 }),
  new THREE.MeshStandardMaterial({ color: 0xff8a2b, emissive: 0xff8a2b, emissiveIntensity: 2.5 }),
  new THREE.MeshStandardMaterial({ color: 0xa3e635, emissive: 0xa3e635, emissiveIntensity: 2.5 }),
];
const spokeGeo = new THREE.CylinderGeometry(0.14, 0.14, 64, 6);
for (let i = 0; i < GONDOLA_COUNT; i++) {
  const ang = (i / GONDOLA_COUNT) * Math.PI * 2;
  if (i < GONDOLA_COUNT / 2) {
    const spoke = new THREE.Mesh(spokeGeo, steelMat);
    spoke.rotation.z = ang;
    wheelRimGroup.add(spoke);
  }
  const gPivot = new THREE.Group();
  gPivot.position.set(Math.cos(ang) * 32, Math.sin(ang) * 32, 0);
  const pod = new THREE.Mesh(gondolaGeo, gondolaMats[i % gondolaMats.length]);
  pod.position.y = -1.4;
  gPivot.add(pod);
  wheelRimGroup.add(gPivot);
  gondolas.push(gPivot);
}

// Landmark 2: Horizon Main Stage & Sky Searchlights
const stageGroup = new THREE.Group();
const stageX = -46;
const stageZ = -82;
stageGroup.position.set(stageX, getTerrainY(stageX, stageZ), stageZ);
stageGroup.rotation.y = 0.55;
scene.add(stageGroup);

const stageDeck = new THREE.Mesh(
  new THREE.BoxGeometry(28, 1.6, 12),
  new THREE.MeshStandardMaterial({ color: 0x161924, roughness: 0.4, metalness: 0.7 }),
);
stageDeck.position.y = 0.8;
const stageArch = new THREE.Mesh(new THREE.TorusGeometry(14, 0.65, 8, 24, Math.PI), neonPinkMat);
stageArch.position.set(0, 1.5, -3);
const stageScreen = new THREE.Mesh(
  new THREE.BoxGeometry(20, 8.5, 0.6),
  new THREE.MeshStandardMaterial({ color: 0x0b0f1d, emissive: 0x1e1b4b, emissiveIntensity: 0.65, roughness: 0.3 }),
);
stageScreen.position.set(0, 6.2, -4.2);
stageGroup.add(stageDeck, stageArch, stageScreen);

for (let i = 0; i < 7; i++) {
  const h = 2.2 + ((i * 3) % 5) * 1.1;
  const eqBar = new THREE.Mesh(new THREE.BoxGeometry(1.6, h, 0.72), i % 2 === 0 ? neonPinkMat : neonCyanMat);
  eqBar.position.set((i - 3) * 2.4, 2.4 + h * 0.5, -4.2);
  stageGroup.add(eqBar);
}

const searchlights: THREE.Mesh[] = [];
const beamGeo = new THREE.ConeGeometry(9.5, 165, 16, 1, true);
beamGeo.translate(0, 82.5, 0);
const beamCyanMat = new THREE.MeshBasicMaterial({
  color: 0x00f0ff,
  transparent: true,
  opacity: 0.14,
  blending: THREE.AdditiveBlending,
  side: THREE.DoubleSide,
  depthWrite: false,
  fog: false,
});
const beamPinkMat = new THREE.MeshBasicMaterial({
  color: 0xff2a85,
  transparent: true,
  opacity: 0.14,
  blending: THREE.AdditiveBlending,
  side: THREE.DoubleSide,
  depthWrite: false,
  fog: false,
});
for (let i = 0; i < 4; i++) {
  const beam = new THREE.Mesh(beamGeo, i % 2 === 0 ? beamCyanMat : beamPinkMat);
  beam.position.set(-10.5 + i * 7, 2, -2);
  stageGroup.add(beam);
  searchlights.push(beam);
}

// ============================================================================
// SPATIAL COLLIDER GRID FOR REAL MAP PHYSICS (Rigid Obstacles & Smashables)
// ============================================================================
const COLLIDER_CELL_SIZE = 28;
interface RigidMapCollider {
  x: number;
  z: number;
  radius: number;
  sparkColorHex: number;
}
interface SmashableMapProp {
  id: number;
  meshKind: 'bollard' | 'flag' | 'sapling';
  instanceIdx: number;
  homeX: number;
  homeY: number;
  homeZ: number;
  homeYaw: number;
  scale: number;
  radius: number;
  debrisColorHex: number;
  smashed: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  rx: number;
  ry: number;
  rz: number;
  avx: number;
  avy: number;
  avz: number;
  smashedAge: number;
}

const rigidGrid = new Map<number, RigidMapCollider[]>();
const smashableGrid = new Map<number, SmashableMapProp[]>();
const activeFlyingProps: SmashableMapProp[] = [];
let nextSmashableId = 1;

function getGridCellKey(x: number, z: number): number {
  const cx = Math.floor(x / COLLIDER_CELL_SIZE) + 500;
  const cz = Math.floor(z / COLLIDER_CELL_SIZE) + 500;
  return (cx << 12) | (cz & 0xfff);
}

function registerRigidCollider(x: number, z: number, radius: number, sparkColorHex = 0xffaa3b) {
  const key = getGridCellKey(x, z);
  let bucket = rigidGrid.get(key);
  if (!bucket) {
    bucket = [];
    rigidGrid.set(key, bucket);
  }
  bucket.push({ x, z, radius, sparkColorHex });
}

function registerSmashableProp(
  meshKind: 'bollard' | 'flag' | 'sapling',
  instanceIdx: number,
  x: number,
  y: number,
  z: number,
  yaw: number,
  scale: number,
  radius: number,
  debrisColorHex: number,
) {
  const prop: SmashableMapProp = {
    id: nextSmashableId++,
    meshKind,
    instanceIdx,
    homeX: x,
    homeY: y,
    homeZ: z,
    homeYaw: yaw,
    scale,
    radius,
    debrisColorHex,
    smashed: false,
    x,
    y,
    z,
    vx: 0,
    vy: 0,
    vz: 0,
    rx: 0,
    ry: yaw,
    rz: 0,
    avx: 0,
    avy: 0,
    avz: 0,
    smashedAge: 0,
  };
  const key = getGridCellKey(x, z);
  let bucket = smashableGrid.get(key);
  if (!bucket) {
    bucket = [];
    smashableGrid.set(key, bucket);
  }
  bucket.push(prop);
}

// Register Ferris Wheel & Festival Stage bases as rigid colliders
registerRigidCollider(ferrisX, ferrisZ, 9.2, 0x00f0ff);
registerRigidCollider(stageX, stageZ, 10.5, 0xff2a85);

// Landmark 3: Cable-Stayed Highway Bridge Towers
const bridgeAngles = [0.68, 0.92];
const pylonMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.35, metalness: 0.65 });
const cableMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.3, metalness: 0.85 });
const pylonGeo = new THREE.BoxGeometry(2.4, 52, 2.4);
pylonGeo.translate(0, 26, 0);
const crossBeamGeo = new THREE.BoxGeometry(ROAD_HALF_WIDTH * 2.45, 2.2, 2.6);

bridgeAngles.forEach((theta) => {
  const pt = getHighwayPointAtAngle(theta);
  const y = getTerrainY(pt.x, pt.z);
  const yaw = Math.atan2(pt.tx, pt.tz);
  const bGroup = new THREE.Group();
  bGroup.position.set(pt.x, y, pt.z);
  bGroup.rotation.y = yaw;
  const span = 15.6;
  const pLeft = new THREE.Mesh(pylonGeo, pylonMat);
  pLeft.position.x = -span;
  const pRight = new THREE.Mesh(pylonGeo, pylonMat);
  pRight.position.x = span;
  // Register both bridge pylons in world space
  registerRigidCollider(pt.x - pt.nx * span, pt.z - pt.nz * span, 1.85, 0xe2e8f0);
  registerRigidCollider(pt.x + pt.nx * span, pt.z + pt.nz * span, 1.85, 0xe2e8f0);
  const beamTop = new THREE.Mesh(crossBeamGeo, pylonMat);
  beamTop.position.y = 44;
  const beamMid = new THREE.Mesh(crossBeamGeo, neonPinkMat);
  beamMid.position.y = 26;
  bGroup.add(pLeft, pRight, beamTop, beamMid);
  const cableGeo = new THREE.CylinderGeometry(0.08, 0.08, 46, 6);
  for (const xSide of [-span, span]) {
    for (const zDir of [-1, 1]) {
      for (let c = 1; c <= 3; c++) {
        const cable = new THREE.Mesh(cableGeo, cableMat);
        cable.position.set(xSide, 22, zDir * c * 7.5);
        cable.rotation.x = zDir * (0.35 + c * 0.12);
        bGroup.add(cable);
      }
    }
  }
  scene.add(bGroup);
});

// Landmark 4: Hot Air Balloons
const balloons: Array<{ mesh: THREE.Group; baseY: number; phase: number }> = [];
const balloonEnvGeo = new THREE.SphereGeometry(7.2, 16, 14);
balloonEnvGeo.scale(1, 1.26, 1);
const balloonSkirtGeo = new THREE.ConeGeometry(4.2, 4.5, 12);
balloonSkirtGeo.rotateX(Math.PI);
balloonSkirtGeo.translate(0, -6.8, 0);
const basketGeo = new THREE.BoxGeometry(2.1, 1.8, 2.1);
basketGeo.translate(0, -11.5, 0);
const burnerMat = new THREE.MeshBasicMaterial({ color: 0xffaa3b });
const burnerGeo = new THREE.SphereGeometry(0.9, 8, 8);
burnerGeo.translate(0, -9.2, 0);
const balloonColors = [0xff2a85, 0xff8a2b, 0x00f0ff, 0xa3e635, 0xfacc15, 0xa855f7];

for (let i = 0; i < 14; i++) {
  const pt = getHighwayPointAtAngle((i / 14) * Math.PI * 2);
  const side = i % 2 === 0 ? 1 : -1;
  const bx = pt.x + pt.nx * side * (48 + (i % 3) * 32);
  const bz = pt.z + pt.nz * side * (48 + (i % 3) * 32);
  const by = getTerrainY(bx, bz) + 38 + (i % 4) * 16;
  const bGroup = new THREE.Group();
  bGroup.position.set(bx, by, bz);
  const envMat = new THREE.MeshStandardMaterial({
    color: balloonColors[i % balloonColors.length],
    roughness: 0.45,
    emissive: balloonColors[i % balloonColors.length],
    emissiveIntensity: 0.35,
  });
  const skirtMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
  const basketMat = new THREE.MeshStandardMaterial({ color: 0x784b2a, roughness: 0.9 });
  bGroup.add(
    new THREE.Mesh(balloonEnvGeo, envMat),
    new THREE.Mesh(balloonSkirtGeo, skirtMat),
    new THREE.Mesh(basketGeo, basketMat),
    new THREE.Mesh(burnerGeo, burnerMat),
  );
  scene.add(bGroup);
  balloons.push({ mesh: bGroup, baseY: by, phase: i * 0.9 });
}

// Natural Stone / Glacier / Cyber Archways Spanning the Highway
const archMat = new THREE.MeshStandardMaterial({ color: 0xa8462a, roughness: 0.88, flatShading: true });
[1.85, 2.75, 3.85, 4.95].forEach((theta) => {
  const pt = getHighwayPointAtAngle(theta);
  const y = getTerrainY(pt.x, pt.z);
  const yaw = Math.atan2(pt.tx, pt.tz);
  const archGroup = new THREE.Group();
  archGroup.position.set(pt.x, y - 1.5, pt.z);
  archGroup.rotation.y = yaw;
  const archMesh = new THREE.Mesh(new THREE.TorusGeometry(22.5, 4.6, 7, 18, Math.PI), archMat);
  archMesh.scale.set(1.15, 0.95, 1.8);
  archGroup.add(archMesh);
  scene.add(archGroup);
  // Register archway rock base pillars on both sides of the highway
  const archSpan = 22.5 * 1.15;
  registerRigidCollider(pt.x - pt.nx * archSpan, pt.z - pt.nz * archSpan, 3.8, 0xd97736);
  registerRigidCollider(pt.x + pt.nx * archSpan, pt.z + pt.nz * archSpan, 3.8, 0xd97736);
});

// Instanced Trees (1,500 Composite 4-Part Trees)
const TREE_COUNT = 1500;
const trunkGeo = new THREE.CylinderGeometry(0.24, 0.46, 3.8, 7);
trunkGeo.translate(0, 1.9, 0);
const trunkMat = new THREE.MeshStandardMaterial({ color: 0x42291d, roughness: 0.95 });
const treeTrunkInst = new THREE.InstancedMesh(trunkGeo, trunkMat, TREE_COUNT);
treeTrunkInst.castShadow = !isSoftwareGl;

const coneGeo1 = new THREE.ConeGeometry(2.6, 3.6, 7);
coneGeo1.translate(0, 4, 0);
const coneGeo2 = new THREE.ConeGeometry(2, 3.2, 7);
coneGeo2.translate(0, 6, 0);
const coneGeo3 = new THREE.ConeGeometry(1.3, 2.6, 7);
coneGeo3.translate(0, 7.8, 0);
const foliageMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, flatShading: true });
const treeCone1 = new THREE.InstancedMesh(coneGeo1, foliageMat, TREE_COUNT);
const treeCone2 = new THREE.InstancedMesh(coneGeo2, foliageMat, TREE_COUNT);
const treeCone3 = new THREE.InstancedMesh(coneGeo3, foliageMat, TREE_COUNT);
treeCone1.castShadow = !isSoftwareGl;
treeCone2.castShadow = !isSoftwareGl;

let placedTrees = 0;
for (let attempts = 0; placedTrees < TREE_COUNT && attempts < 12000; attempts++) {
  let tx: number;
  let tz: number;
  if (placedTrees < TREE_COUNT * 0.75) {
    const pt = getHighwayPointAtAngle(rng() * Math.PI * 2);
    const side = rng() > 0.5 ? 1 : -1;
    const distOff = 24 + rng() * 145;
    tx = pt.x + pt.nx * side * distOff + (rng() - 0.5) * 18;
    tz = pt.z + pt.nz * side * distOff + (rng() - 0.5) * 18;
  } else {
    tx = (rng() - 0.5) * 2600;
    tz = (rng() - 0.5) * 2600;
  }
  const { distance } = getRoadDistanceInfo(tx, tz);
  if (distance < 23 || Math.hypot(tx - OASIS_X, tz - OASIS_Z) < 158) continue;
  const ty = getTerrainY(tx, tz);
  const scale = 0.85 + rng() * 0.85;
  const treeYaw = rng() * Math.PI * 2;
  _dummy.position.set(tx, ty - 0.15, tz);
  _dummy.rotation.set(0, treeYaw, 0);
  _dummy.scale.setScalar(scale);
  _dummy.updateMatrix();
  treeTrunkInst.setMatrixAt(placedTrees, _dummy.matrix);
  treeCone1.setMatrixAt(placedTrees, _dummy.matrix);
  treeCone2.setMatrixAt(placedTrees, _dummy.matrix);
  treeCone3.setMatrixAt(placedTrees, _dummy.matrix);
  // Small saplings (< 1.18 scale) are destructible Forza Horizon "LANDSCAPING" smashables; large trees are rigid trunks
  if (scale < 1.18) {
    registerSmashableProp('sapling', placedTrees, tx, ty - 0.15, tz, treeYaw, scale, 0.65 * scale, 0x3f7a2e);
  } else {
    registerRigidCollider(tx, tz, 0.56 * scale, 0x8f5e26);
  }
  placedTrees++;
}
treeTrunkInst.instanceMatrix.needsUpdate = true;
treeCone1.instanceMatrix.needsUpdate = true;
treeCone2.instanceMatrix.needsUpdate = true;
treeCone3.instanceMatrix.needsUpdate = true;
treeTrunkInst.computeBoundingSphere();
treeCone1.computeBoundingSphere();
treeCone2.computeBoundingSphere();
treeCone3.computeBoundingSphere();
scene.add(treeTrunkInst, treeCone1, treeCone2, treeCone3);

// Instanced Rocks & Boulders (950 Instances)
const ROCK_COUNT = 950;
const rockGeo = new THREE.DodecahedronGeometry(2.1, 0);
rockGeo.translate(0, 1, 0);
const rockInst = new THREE.InstancedMesh(
  rockGeo,
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.86, metalness: 0.06, flatShading: true }),
  ROCK_COUNT,
);
rockInst.castShadow = !isSoftwareGl;
rockInst.receiveShadow = !isSoftwareGl;

let placedRocks = 0;
for (let attempts = 0; placedRocks < ROCK_COUNT && attempts < 9000; attempts++) {
  let rx: number;
  let rz: number;
  if (placedRocks < ROCK_COUNT * 0.78) {
    const pt = getHighwayPointAtAngle(rng() * Math.PI * 2);
    const side = rng() > 0.5 ? 1 : -1;
    const distOff = 22.5 + rng() * 130;
    rx = pt.x + pt.nx * side * distOff;
    rz = pt.z + pt.nz * side * distOff;
  } else {
    rx = (rng() - 0.5) * 2700;
    rz = (rng() - 0.5) * 2700;
  }
  const { distance } = getRoadDistanceInfo(rx, rz);
  if (distance < 22 || Math.hypot(rx - OASIS_X, rz - OASIS_Z) < 148) continue;
  const ry = getTerrainY(rx, rz);
  _dummy.position.set(rx, ry - 0.35, rz);
  _dummy.rotation.set(rng() * 0.4, rng() * Math.PI * 2, rng() * 0.4);
  const spire = placedRocks % 9 === 0 ? 3.4 : 1;
  const sx = (0.8 + rng() * 2.2) * (spire > 1 ? 1.8 : 1);
  const sy = (0.6 + rng() * 1.6) * spire;
  const sz = (0.8 + rng() * 2.2) * (spire > 1 ? 1.8 : 1);
  _dummy.scale.set(sx, sy, sz);
  _dummy.updateMatrix();
  rockInst.setMatrixAt(placedRocks, _dummy.matrix);
  registerRigidCollider(rx, rz, Math.min(4.8, (sx + sz) * 0.68), 0xffaa3b);
  placedRocks++;
}
rockInst.instanceMatrix.needsUpdate = true;
rockInst.computeBoundingSphere();
scene.add(rockInst);

// Highway Bollards & Festival Flags (Destructible Forza Horizon Smashables!)
const BOLLARD_PAIRS = 120;
const bollardGeo = new THREE.BoxGeometry(0.28, 1.25, 0.28);
bollardGeo.translate(0, 0.62, 0);
const bollardMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  emissive: 0xff8a2b,
  emissiveIntensity: 2,
  roughness: 0.25,
});
const bollardInst = new THREE.InstancedMesh(bollardGeo, bollardMat, BOLLARD_PAIRS * 2);
const flagGeo = new THREE.BoxGeometry(0.12, 4.8, 1.15);
flagGeo.translate(0, 2.8, 0);
const flagInst = new THREE.InstancedMesh(
  flagGeo,
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 }),
  BOLLARD_PAIRS * 2,
);
const flagPalette = [0xff2a85, 0x00f0ff, 0xff8a2b, 0xfacc15];

for (let i = 0; i < BOLLARD_PAIRS; i++) {
  const pt = getHighwayPointAtAngle((i / BOLLARD_PAIRS) * Math.PI * 2);
  const offset = 14.15;
  const yaw = Math.atan2(pt.tx, pt.tz);
  const lx = pt.x - pt.nx * offset;
  const lz = pt.z - pt.nz * offset;
  const ly = getTerrainY(lx, lz);
  _dummy.position.set(lx, ly, lz);
  _dummy.rotation.set(0, yaw, 0);
  _dummy.scale.setScalar(1);
  _dummy.updateMatrix();
  bollardInst.setMatrixAt(i * 2, _dummy.matrix);
  registerSmashableProp('bollard', i * 2, lx, ly, lz, yaw, 1, 0.55, 0xff8a2b);

  const flx = pt.x - pt.nx * 15.95;
  const flz = pt.z - pt.nz * 15.95;
  _dummy.position.set(flx, ly, flz);
  _dummy.updateMatrix();
  flagInst.setMatrixAt(i * 2, _dummy.matrix);
  const fColL = flagPalette[i % flagPalette.length];
  _instColor.setHex(fColL);
  flagInst.setColorAt(i * 2, _instColor);
  registerSmashableProp('flag', i * 2, flx, ly, flz, yaw, 1, 0.65, fColL);

  const rx = pt.x + pt.nx * offset;
  const rz = pt.z + pt.nz * offset;
  const ry = getTerrainY(rx, rz);
  _dummy.position.set(rx, ry, rz);
  _dummy.updateMatrix();
  bollardInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  registerSmashableProp('bollard', i * 2 + 1, rx, ry, rz, yaw, 1, 0.55, 0xff8a2b);

  const frx = pt.x + pt.nx * 15.95;
  const frz = pt.z + pt.nz * 15.95;
  _dummy.position.set(frx, ry, frz);
  _dummy.updateMatrix();
  flagInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  const fColR = flagPalette[(i + 1) % flagPalette.length];
  _instColor.setHex(fColR);
  flagInst.setColorAt(i * 2 + 1, _instColor);
  registerSmashableProp('flag', i * 2 + 1, frx, ry, frz, yaw, 1, 0.65, fColR);
}
bollardInst.instanceMatrix.needsUpdate = true;
flagInst.instanceMatrix.needsUpdate = true;
if (flagInst.instanceColor) flagInst.instanceColor.needsUpdate = true;
bollardInst.computeBoundingSphere();
flagInst.computeBoundingSphere();
scene.add(bollardInst, flagInst);

// 16 Neon Checkpoint Gates
const GATE_COUNT = 16;
const gates: Array<{ x: number; z: number; cleared: boolean; cooldown: number }> = [];
const gatePillarGeo = new THREE.CylinderGeometry(0.45, 0.56, 12, 10);
gatePillarGeo.translate(0, 6, 0);
const gateHeaderGeo = new THREE.BoxGeometry(ROAD_HALF_WIDTH * 2.2, 1.4, 1.15);
gateHeaderGeo.translate(0, 11.6, 0);
const gateNeonBarGeo = new THREE.BoxGeometry(ROAD_HALF_WIDTH * 2.1, 0.42, 1.28);
gateNeonBarGeo.translate(0, 11.6, 0);
const gateFrameMat = new THREE.MeshStandardMaterial({ color: 0x161922, roughness: 0.35, metalness: 0.75 });
const gatePinkNeonMat = new THREE.MeshStandardMaterial({
  color: 0xff2a85,
  emissive: 0xff2a85,
  emissiveIntensity: 3.4,
  roughness: 0.2,
});
const gateCyanNeonMat = new THREE.MeshStandardMaterial({
  color: 0x00f0ff,
  emissive: 0x00f0ff,
  emissiveIntensity: 3.4,
  roughness: 0.2,
});
const gatePillarInst = new THREE.InstancedMesh(gatePillarGeo, gateFrameMat, GATE_COUNT * 2);
const gateHeaderInst = new THREE.InstancedMesh(gateHeaderGeo, gateFrameMat, GATE_COUNT);
const gateTopBarInst = new THREE.InstancedMesh(gateNeonBarGeo, gatePinkNeonMat, GATE_COUNT);
const gateColSleeveGeo = new THREE.CylinderGeometry(0.5, 0.5, 4.8, 10);
gateColSleeveGeo.translate(0, 4.6, 0);
const gateSleeveInst = new THREE.InstancedMesh(gateColSleeveGeo, gateCyanNeonMat, GATE_COUNT * 2);

for (let i = 0; i < GATE_COUNT; i++) {
  const pt = getHighwayPointAtAngle((i / GATE_COUNT) * Math.PI * 2 + 0.12);
  const y = getTerrainY(pt.x, pt.z);
  const yaw = Math.atan2(pt.tx, pt.tz);
  const lx = pt.x - pt.nx * 15;
  const lz = pt.z - pt.nz * 15;
  _dummy.position.set(lx, y, lz);
  _dummy.rotation.set(0, yaw, 0);
  _dummy.scale.setScalar(1);
  _dummy.updateMatrix();
  gatePillarInst.setMatrixAt(i * 2, _dummy.matrix);
  gateSleeveInst.setMatrixAt(i * 2, _dummy.matrix);
  registerRigidCollider(lx, lz, 0.72, 0x00f0ff);

  const rx = pt.x + pt.nx * 15;
  const rz = pt.z + pt.nz * 15;
  _dummy.position.set(rx, y, rz);
  _dummy.updateMatrix();
  gatePillarInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  gateSleeveInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  registerRigidCollider(rx, rz, 0.72, 0x00f0ff);

  _dummy.position.set(pt.x, y, pt.z);
  _dummy.rotation.set(0, yaw, 0);
  _dummy.updateMatrix();
  gateHeaderInst.setMatrixAt(i, _dummy.matrix);
  gateTopBarInst.setMatrixAt(i, _dummy.matrix);
  gates.push({ x: pt.x, z: pt.z, cleared: false, cooldown: 0 });
}
gatePillarInst.instanceMatrix.needsUpdate = true;
gateHeaderInst.instanceMatrix.needsUpdate = true;
gateTopBarInst.instanceMatrix.needsUpdate = true;
gateSleeveInst.instanceMatrix.needsUpdate = true;
gatePillarInst.computeBoundingSphere();
gateHeaderInst.computeBoundingSphere();
gateTopBarInst.computeBoundingSphere();
gateSleeveInst.computeBoundingSphere();
scene.add(gatePillarInst, gateHeaderInst, gateTopBarInst, gateSleeveInst);

// ============================================================================
// 3 PLAYABLE COMPOSITE 3D VEHICLES (Apex V10 GT, Kaze RX-79 Drift, Titan 4x4)
// ============================================================================
const carRootGroup = new THREE.Group();
scene.add(carRootGroup);

const carPaintMat = new THREE.MeshPhysicalMaterial({
  color: 0xff5511,
  metalness: 0.88,
  roughness: 0.15,
  clearcoat: 1,
  clearcoatRoughness: 0.06,
  reflectivity: 0.95,
});
const canopyGlassMat = new THREE.MeshPhysicalMaterial({
  color: 0x090c14,
  metalness: 0.25,
  roughness: 0.05,
  transmission: isSoftwareGl ? 0 : 0.65,
  transparent: true,
  opacity: isSoftwareGl ? 0.88 : 0.92,
});
const carbonTrimMat = new THREE.MeshStandardMaterial({ color: 0x16181f, roughness: 0.42, metalness: 0.72 });
const chromeExhaustMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.92, roughness: 0.22 });
const headlightMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  emissive: 0xfffac0,
  emissiveIntensity: 4.8,
  roughness: 0.15,
});
const taillightMat = new THREE.MeshStandardMaterial({
  color: 0xff2a45,
  emissive: 0xff2a45,
  emissiveIntensity: 2.8,
  roughness: 0.2,
});
const cyanAccentMat = new THREE.MeshStandardMaterial({
  color: 0x00f0ff,
  emissive: 0x00f0ff,
  emissiveIntensity: 2.4,
  roughness: 0.2,
});

// 1) CAR 1: Apex V10 GT (Mid-Engine Supercar - 12 parts)
const apexBodyGroup = new THREE.Group();
{
  const lowerHull = new THREE.Mesh(new THREE.BoxGeometry(1.98, 0.4, 4.48), carPaintMat);
  lowerHull.position.set(0, 0.12, 0);
  lowerHull.castShadow = !isSoftwareGl;
  const canopy = new THREE.Mesh(new THREE.BoxGeometry(1.48, 0.43, 2.02), canopyGlassMat);
  canopy.position.set(0, 0.48, 0.15);
  canopy.castShadow = !isSoftwareGl;
  const frunkHood = new THREE.Mesh(new THREE.BoxGeometry(1.88, 0.14, 1.48), carPaintMat);
  frunkHood.position.set(0, 0.25, -1.38);
  frunkHood.rotation.x = 0.09;
  const roofScoop = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.14, 0.85), carbonTrimMat);
  roofScoop.position.set(0, 0.73, -0.05);
  const frontFenders = new THREE.Mesh(new THREE.BoxGeometry(2.06, 0.32, 0.95), carPaintMat);
  frontFenders.position.set(0, 0.15, -1.35);
  const rearHaunches = new THREE.Mesh(new THREE.BoxGeometry(2.08, 0.36, 1.08), carPaintMat);
  rearHaunches.position.set(0, 0.18, 1.35);
  const sideIntakes = new THREE.Mesh(new THREE.BoxGeometry(2.04, 0.22, 1.3), carbonTrimMat);
  sideIntakes.position.set(0, 0.14, 0.35);
  const splitterFloor = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.07, 4.74), carbonTrimMat);
  splitterFloor.position.set(0, -0.14, 0);
  const rearWing = new THREE.Mesh(new THREE.BoxGeometry(1.98, 0.06, 0.38), carbonTrimMat);
  rearWing.position.set(0, 0.72, 2.08);
  rearWing.rotation.x = 0.14;
  const strutGeo = new THREE.BoxGeometry(0.06, 0.32, 0.24);
  const strutL = new THREE.Mesh(strutGeo, carbonTrimMat);
  strutL.position.set(-0.58, 0.56, 2.04);
  const strutR = new THREE.Mesh(strutGeo, carbonTrimMat);
  strutR.position.set(0.58, 0.56, 2.04);

  const hlGeo = new THREE.BoxGeometry(0.46, 0.09, 0.16);
  const hlL = new THREE.Mesh(hlGeo, headlightMat);
  hlL.position.set(-0.68, 0.18, -2.26);
  hlL.rotation.y = 0.15;
  const hlR = new THREE.Mesh(hlGeo, headlightMat);
  hlR.position.set(0.68, 0.18, -2.26);
  hlR.rotation.y = -0.15;
  const tailBar = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.11, 0.1), taillightMat);
  tailBar.position.set(0, 0.3, 2.31);

  apexBodyGroup.add(
    lowerHull, canopy, frunkHood, roofScoop, frontFenders,
    rearHaunches, sideIntakes, splitterFloor, rearWing, strutL, strutR,
    hlL, hlR, tailBar,
  );
}

// 2) CAR 2: Kaze RX-79 Drift (Widebody JDM Street Tuner - 12 parts)
const kazeBodyGroup = new THREE.Group();
kazeBodyGroup.visible = false;
{
  const coupeBody = new THREE.Mesh(new THREE.BoxGeometry(1.92, 0.38, 4.42), carPaintMat);
  coupeBody.position.set(0, 0.12, 0);
  coupeBody.castShadow = !isSoftwareGl;
  const wideArchF = new THREE.Mesh(new THREE.BoxGeometry(2.12, 0.28, 1.05), carPaintMat);
  wideArchF.position.set(0, 0.1, -1.28);
  const wideArchR = new THREE.Mesh(new THREE.BoxGeometry(2.16, 0.32, 1.15), carPaintMat);
  wideArchR.position.set(0, 0.14, 1.28);
  const fastbackCabin = new THREE.Mesh(new THREE.BoxGeometry(1.46, 0.44, 1.92), canopyGlassMat);
  fastbackCabin.position.set(0, 0.5, 0.22);
  fastbackCabin.rotation.x = -0.06;
  const ventedHood = new THREE.Mesh(new THREE.BoxGeometry(1.62, 0.08, 1.42), carbonTrimMat);
  ventedHood.position.set(0, 0.32, -1.32);
  const frontLip = new THREE.Mesh(new THREE.BoxGeometry(2.14, 0.08, 0.55), carbonTrimMat);
  frontLip.position.set(0, -0.1, -2.12);
  const sideSkirts = new THREE.Mesh(new THREE.BoxGeometry(2.14, 0.09, 2.65), carbonTrimMat);
  sideSkirts.position.set(0, -0.09, 0);
  const gtWingBlade = new THREE.Mesh(new THREE.BoxGeometry(2.08, 0.06, 0.42), carbonTrimMat);
  gtWingBlade.position.set(0, 0.86, 2.08);
  gtWingBlade.rotation.x = 0.2;
  const endplateGeo = new THREE.BoxGeometry(0.04, 0.24, 0.48);
  const endL = new THREE.Mesh(endplateGeo, carPaintMat);
  endL.position.set(-1.04, 0.86, 2.08);
  const endR = new THREE.Mesh(endplateGeo, carPaintMat);
  endR.position.set(1.04, 0.86, 2.08);
  const swanStrutGeo = new THREE.BoxGeometry(0.05, 0.48, 0.2);
  const swanL = new THREE.Mesh(swanStrutGeo, carbonTrimMat);
  swanL.position.set(-0.48, 0.62, 2.02);
  const swanR = new THREE.Mesh(swanStrutGeo, carbonTrimMat);
  swanR.position.set(0.48, 0.62, 2.02);
  const jdmHeadlightL = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.1, 0.14), headlightMat);
  jdmHeadlightL.position.set(-0.64, 0.2, -2.23);
  const jdmHeadlightR = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.1, 0.14), headlightMat);
  jdmHeadlightR.position.set(0.64, 0.2, -2.23);
  const underglowStrip = new THREE.Mesh(new THREE.BoxGeometry(1.95, 0.04, 3.6), cyanAccentMat);
  underglowStrip.position.set(0, -0.15, 0);
  const jdmTaillight = new THREE.Mesh(new THREE.BoxGeometry(1.82, 0.14, 0.1), taillightMat);
  jdmTaillight.position.set(0, 0.28, 2.24);

  kazeBodyGroup.add(
    coupeBody, wideArchF, wideArchR, fastbackCabin, ventedHood,
    frontLip, sideSkirts, gtWingBlade, endL, endR, swanL, swanR,
    jdmHeadlightL, jdmHeadlightR, underglowStrip, jdmTaillight,
  );
}

// 3) CAR 3: Titan Baja 4x4 (Lifted Desert Trophy Truck - 12 parts)
const titanBodyGroup = new THREE.Group();
titanBodyGroup.visible = false;
{
  const truckChassis = new THREE.Mesh(new THREE.BoxGeometry(2.14, 0.56, 4.72), carPaintMat);
  truckChassis.position.set(0, 0.28, 0);
  truckChassis.castShadow = !isSoftwareGl;
  const crewCab = new THREE.Mesh(new THREE.BoxGeometry(1.86, 0.58, 1.95), canopyGlassMat);
  crewCab.position.set(0, 0.82, -0.15);
  crewCab.castShadow = !isSoftwareGl;
  const cabRoof = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.1, 1.98), carPaintMat);
  cabRoof.position.set(0, 1.12, -0.15);
  const roofLightBar = new THREE.Mesh(new THREE.BoxGeometry(1.56, 0.12, 0.16), headlightMat);
  roofLightBar.position.set(0, 1.22, -1.08);
  const flareF = new THREE.Mesh(new THREE.BoxGeometry(2.36, 0.36, 1.18), carbonTrimMat);
  flareF.position.set(0, 0.22, -1.45);
  const flareR = new THREE.Mesh(new THREE.BoxGeometry(2.36, 0.36, 1.18), carbonTrimMat);
  flareR.position.set(0, 0.22, 1.45);
  const bullBar = new THREE.Mesh(new THREE.BoxGeometry(1.96, 0.42, 0.34), carbonTrimMat);
  bullBar.position.set(0, 0.22, -2.44);
  const skidPlate = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.28, 0.35), chromeExhaustMat);
  skidPlate.position.set(0, -0.02, -2.38);
  skidPlate.rotation.x = 0.4;
  // Roll cage bars in truck bed
  const rollCageGeo = new THREE.CylinderGeometry(0.06, 0.06, 1.65, 8);
  const barL = new THREE.Mesh(rollCageGeo, carbonTrimMat);
  barL.position.set(-0.78, 0.78, 1.42);
  barL.rotation.x = -0.62;
  const barR = new THREE.Mesh(rollCageGeo, carbonTrimMat);
  barR.position.set(0.78, 0.78, 1.42);
  barR.rotation.x = -0.62;
  // Angled spare tire mounted in rear bed
  const spareTire = new THREE.Mesh(
    new THREE.CylinderGeometry(0.5, 0.5, 0.36, 18),
    new THREE.MeshStandardMaterial({ color: 0x14161d, roughness: 0.85 }),
  );
  spareTire.rotation.z = Math.PI / 2;
  spareTire.rotation.y = Math.PI / 2;
  spareTire.position.set(0, 0.62, 1.55);
  spareTire.rotation.x = -0.45;
  const truckHlL = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.22, 0.14), headlightMat);
  truckHlL.position.set(-0.76, 0.36, -2.38);
  const truckHlR = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.22, 0.14), headlightMat);
  truckHlR.position.set(0.76, 0.36, -2.38);
  const truckTail = new THREE.Mesh(new THREE.BoxGeometry(2.04, 0.2, 0.12), taillightMat);
  truckTail.position.set(0, 0.38, 2.38);

  titanBodyGroup.add(
    truckChassis, crewCab, cabRoof, roofLightBar, flareF, flareR,
    bullBar, skidPlate, barL, barR, spareTire, truckHlL, truckHlR, truckTail,
  );
}

carRootGroup.add(apexBodyGroup, kazeBodyGroup, titanBodyGroup);

// Shared Headlight Spotlights, Exhaust Pipes, Nitro Plumes & Contact Shadow
const spotL = new THREE.SpotLight(0xfffac6, 8.5, 95, Math.PI / 6, 0.5, 1.15);
spotL.position.set(-0.68, 0.22, -2.26);
const spotTargetL = new THREE.Object3D();
spotTargetL.position.set(-0.68, -0.35, -34);
carRootGroup.add(spotL, spotTargetL);
spotL.target = spotTargetL;

const spotR = new THREE.SpotLight(0xfffac6, 8.5, 95, Math.PI / 6, 0.5, 1.15);
spotR.position.set(0.68, 0.22, -2.26);
const spotTargetR = new THREE.Object3D();
spotTargetR.position.set(0.68, -0.35, -34);
carRootGroup.add(spotR, spotTargetR);
spotR.target = spotTargetR;

const exhaustPipeGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.28, 12);
exhaustPipeGeo.rotateX(Math.PI / 2);
const exhaustL = new THREE.Mesh(exhaustPipeGeo, chromeExhaustMat);
exhaustL.position.set(-0.34, -0.04, 2.36);
const exhaustR = new THREE.Mesh(exhaustPipeGeo, chromeExhaustMat);
exhaustR.position.set(0.34, -0.04, 2.36);
carRootGroup.add(exhaustL, exhaustR);

const nitroFlameGeo = new THREE.ConeGeometry(0.14, 1.35, 10);
nitroFlameGeo.rotateX(Math.PI / 2);
nitroFlameGeo.translate(0, 0, 0.72);
const nitroFlameMat = new THREE.MeshBasicMaterial({
  color: 0x00f0ff,
  transparent: true,
  opacity: 0.92,
  blending: THREE.AdditiveBlending,
});
const nitroFlameL = new THREE.Mesh(nitroFlameGeo, nitroFlameMat);
nitroFlameL.position.set(-0.34, -0.04, 2.42);
nitroFlameL.visible = false;
const nitroFlameR = new THREE.Mesh(nitroFlameGeo, nitroFlameMat);
nitroFlameR.position.set(0.34, -0.04, 2.42);
nitroFlameR.visible = false;
carRootGroup.add(nitroFlameL, nitroFlameR);

const contactShadowGeo = new THREE.PlaneGeometry(3.4, 5.8);
contactShadowGeo.rotateX(-Math.PI / 2);
const contactShadowMat = new THREE.MeshBasicMaterial({
  map: createContactShadowTexture(),
  transparent: true,
  opacity: 0.7,
  depthWrite: false,
  polygonOffset: true,
  polygonOffsetFactor: -1,
  polygonOffsetUnits: -1,
});
const contactShadowMesh = new THREE.Mesh(contactShadowGeo, contactShadowMat);
contactShadowMesh.position.set(0, -0.34, 0);
carRootGroup.add(contactShadowMesh);

// 4 Independent Composite Wheels
const wheelRoots: THREE.Group[] = [];
const wheelSpins: THREE.Group[] = [];
const tireGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.34, 24);
tireGeo.rotateZ(Math.PI / 2);
const tireMat = new THREE.MeshStandardMaterial({ color: 0x141619, roughness: 0.86, metalness: 0.1 });
const rimGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.36, 16);
rimGeo.rotateZ(Math.PI / 2);
const rimMat = new THREE.MeshStandardMaterial({ color: 0xdbeafe, metalness: 0.92, roughness: 0.18 });
const spokeBarGeo = new THREE.BoxGeometry(0.37, 0.06, 0.52);
const hubGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.38, 12);
hubGeo.rotateZ(Math.PI / 2);
const hubMat = new THREE.MeshStandardMaterial({ color: 0xff2a85, roughness: 0.3, metalness: 0.6 });
const caliperGeo = new THREE.BoxGeometry(0.12, 0.18, 0.13);
const caliperMat = new THREE.MeshStandardMaterial({ color: 0xff1e38, roughness: 0.28, metalness: 0.55 });

for (let i = 0; i < 4; i++) {
  const wRoot = new THREE.Group();
  const wSpin = new THREE.Group();
  const tire = new THREE.Mesh(tireGeo, tireMat);
  tire.castShadow = !isSoftwareGl;
  const rim = new THREE.Mesh(rimGeo, rimMat);
  const sp1 = new THREE.Mesh(spokeBarGeo, rimMat);
  const sp2 = new THREE.Mesh(spokeBarGeo, rimMat);
  sp2.rotation.x = Math.PI / 3;
  const sp3 = new THREE.Mesh(spokeBarGeo, rimMat);
  sp3.rotation.x = -Math.PI / 3;
  const hub = new THREE.Mesh(hubGeo, hubMat);
  wSpin.add(tire, rim, sp1, sp2, sp3, hub);
  const caliper = new THREE.Mesh(caliperGeo, caliperMat);
  caliper.position.set(0, 0.17, 0);
  wRoot.add(wSpin, caliper);
  scene.add(wRoot);
  wheelRoots.push(wRoot);
  wheelSpins.push(wSpin);
}

// 6 AI Rival Cars on the Highway Loop
const rivals: Array<{
  mesh: THREE.Group;
  theta: number;
  angularSpeed: number;
  baseLaneOffset: number;
  laneOffset: number;
  lateralVel: number;
  nearMissCooldown: number;
  hitCooldown: number;
}> = [];
const rivalColors = [0x00f0ff, 0xfacc15, 0xa3e635, 0xec4899, 0x38bdf8, 0xf97316];

for (let i = 0; i < 6; i++) {
  const rGroup = new THREE.Group();
  const rPaint = new THREE.MeshPhysicalMaterial({
    color: rivalColors[i],
    metalness: 0.85,
    roughness: 0.18,
    clearcoat: 1,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.92, 0.42, 4.35), rPaint);
  body.position.y = 0.48;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.44, 0.4, 1.95), canopyGlassMat);
  cabin.position.set(0, 0.82, 0.1);
  const tail = new THREE.Mesh(new THREE.BoxGeometry(1.78, 0.1, 0.1), taillightMat);
  tail.position.set(0, 0.62, 2.2);
  const head = new THREE.Mesh(new THREE.BoxGeometry(1.65, 0.09, 0.1), headlightMat);
  head.position.set(0, 0.55, -2.2);
  const shad = new THREE.Mesh(contactShadowGeo, contactShadowMat);
  shad.position.y = 0.04;
  rGroup.add(body, cabin, tail, head, shad);
  scene.add(rGroup);
  const baseLane = i % 2 === 0 ? 5.6 : -5.6;
  rivals.push({
    mesh: rGroup,
    theta: ((i + 0.5) / 6) * Math.PI * 2,
    angularSpeed: (i % 2 === 0 ? 1 : -1) * (0.048 + (i % 3) * 0.008),
    baseLaneOffset: baseLane,
    laneOffset: baseLane,
    lateralVel: 0,
    nearMissCooldown: 0,
    hitCooldown: 0,
  });
}

// Pooled Particle & Skidmark Ring Buffers
const chassisBody = new CANNON.Body({ mass: 1450 });
physicsWorld.addBody(chassisBody);

let currentCarId: CarId = 'apex';
let currentCarSpec: CarSpec = CAR_CATALOG.apex;
let currentSceneryId: SceneryId = 'canyon';
let currentScenerySpec: ScenerySpec = SCENERY_CATALOG.canyon;

const carState = {
  x: 0,
  y: 0.56,
  z: 0,
  vx: 0,
  vy: 0,
  vz: 0,
  yaw: 0,
  yawRate: 0,
  visualYawOffset: 0,
  pitch: 0,
  roll: 0,
  wheelSpin: 0,
  suspensionBounce: 0,
  suspensionVel: 0,
  isGrounded: true,
  slopeRate: 0,
};

const MAX_PARTICLES = 180;
const particleInst = new THREE.InstancedMesh(
  new THREE.DodecahedronGeometry(0.32, 0),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.78 }),
  MAX_PARTICLES,
);
particleInst.frustumCulled = false;
scene.add(particleInst);

const particles = Array.from({ length: MAX_PARTICLES }, () => ({
  active: false,
  x: 0,
  y: -100,
  z: 0,
  vx: 0,
  vy: 0,
  vz: 0,
  scale: 0.001,
  growRate: 1.5,
  age: 0,
  maxAge: 0.6,
  colorHex: 0xdbeafe,
}));
let particleCursor = 0;
let particleColorsDirty = false;

function spawnParticle(
  x: number,
  y: number,
  z: number,
  vx: number,
  vy: number,
  vz: number,
  scale: number,
  growRate: number,
  maxAge: number,
  colorHex: number,
) {
  const idx = particleCursor;
  const p = particles[idx];
  particleCursor = (particleCursor + 1) % MAX_PARTICLES;
  p.active = true;
  p.x = x;
  p.y = y;
  p.z = z;
  p.vx = vx;
  p.vy = vy;
  p.vz = vz;
  p.scale = scale;
  p.growRate = growRate;
  p.age = 0;
  p.maxAge = maxAge;
  p.colorHex = colorHex;
  _instColor.setHex(colorHex);
  particleInst.setColorAt(idx, _instColor);
  particleColorsDirty = true;
}

const MAX_SKIDS = 220;
const skidGeo = new THREE.PlaneGeometry(0.34, 1.05);
skidGeo.rotateX(-Math.PI / 2);
const skidInst = new THREE.InstancedMesh(
  skidGeo,
  new THREE.MeshBasicMaterial({
    color: 0x0e1015,
    transparent: true,
    opacity: 0.68,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  }),
  MAX_SKIDS,
);
skidInst.frustumCulled = false;
scene.add(skidInst);

const skids = Array.from({ length: MAX_SKIDS }, () => ({
  active: false,
  x: 0,
  y: -500,
  z: 0,
  yaw: 0,
  age: 0,
  maxAge: 7,
}));
let skidCursor = 0;
let skidTimer = 0;

function spawnSkid(x: number, y: number, z: number, yaw: number) {
  const s = skids[skidCursor];
  skidCursor = (skidCursor + 1) % MAX_SKIDS;
  s.active = true;
  s.x = x;
  s.y = y + 0.19;
  s.z = z;
  s.yaw = yaw;
  s.age = 0;
}

// Game State, Input & DOM References
let gameState: 'title' | 'playing' | 'paused' = 'title';
let cameraMode: 'chase' | 'hood' = 'chase';
let speedUnit: 'KMH' | 'MPH' = 'KMH';

const keys = {
  forward: false,
  backward: false,
  left: false,
  right: false,
  handbrake: false,
  nitro: false,
};
const touchInput = {
  gas: false,
  brake: false,
  left: false,
  right: false,
  steerAnalog: 0,
  drift: false,
  nitro: false,
};
let autoGasEnabled = false;

let nitroCharge = 1;
let currentSteerAngle = 0;
let airTime = 0;
let gatesClearedCount = 0;
let sessionTopSpeedKmh = 0;
let bankedFlashTimer = 0;
let cameraTrauma = 0;
let cameraRoll = 0;

const skillState = {
  activeScore: 0,
  multiplier: 1,
  bankedScore: 0,
  bestChainScore: 0,
  cleanTimer: 0,
  lastSkillLabel: 'DRIFT OR BOOST TO START CHAIN',
};

const PAINT_PRESETS = ['#ff5511', '#00e5ff', '#a3e635', '#facc15', '#ec4899', '#e2e8f0'];
let currentPaintHex = '#ff5511';

const paintPickerFrame = document.getElementById('paint-picker-frame');
const paintPickerInput = document.getElementById('car-paint-picker') as HTMLInputElement | null;

function applyCarPaint(hex: string) {
  currentPaintHex = hex;
  carPaintMat.color.set(hex);
  if (paintPickerFrame) paintPickerFrame.style.backgroundColor = hex;
  if (paintPickerInput) paintPickerInput.value = hex;
  document.querySelectorAll<HTMLElement>('.swatch-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.paint?.toLowerCase() === hex.toLowerCase());
  });
}

if (paintPickerInput) {
  paintPickerInput.addEventListener('input', (e) => applyCarPaint((e.target as HTMLInputElement).value));
  paintPickerInput.addEventListener('change', (e) => applyCarPaint((e.target as HTMLInputElement).value));
}
document.querySelectorAll<HTMLElement>('.swatch-btn').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    const p = btn.dataset.paint;
    if (p) applyCarPaint(p);
  });
});

// UI DOM Elements
const titleScreenEl = document.getElementById('title-screen')!;
const statusModalEl = document.getElementById('status-modal')!;
const hudSpeedEl = document.getElementById('hud-speed')!;
const hudGearEl = document.getElementById('hud-gear')!;
const hudGearBoxEl = document.getElementById('hud-gear-box')!;
const hudRpmBarEl = document.getElementById('hud-rpm-bar')!;
const hudRpmArcEl = document.getElementById('hud-rpm-arc');
const hudRpmTextEl = document.getElementById('hud-rpm-text')!;
const hudNitroBarEl = document.getElementById('hud-nitro-bar')!;
const hudNitroArcEl = document.getElementById('hud-nitro-arc');
const hudUnitToggleEl = document.getElementById('hud-unit-toggle')!;
const hudBankedXpEl = document.getElementById('hud-banked-xp')!;
const hudGatesCountEl = document.getElementById('hud-gates-count')!;
const hudBestChainEl = document.getElementById('hud-best-chain')!;
const hudCarWorldLabelEl = document.getElementById('hud-car-world-label');
const menuPiClassEl = document.getElementById('menu-pi-class');
const menuPiNumEl = document.getElementById('menu-pi-num');
const specPiClassEl = document.getElementById('spec-pi-class');
const specPiNumEl = document.getElementById('spec-pi-num');
const hudPiClassEl = document.getElementById('hud-pi-class');
const hudPiNumEl = document.getElementById('hud-pi-num');
const skillChainCardEl = document.getElementById('skill-chain-card')!;
const hudSkillPointsEl = document.getElementById('hud-skill-points')!;
const hudSkillMultEl = document.getElementById('hud-skill-mult')!;
const hudSkillTimerEl = document.getElementById('hud-skill-timer')!;
const hudSkillLabelEl = document.getElementById('hud-skill-label')!;
const modalStatXpEl = document.getElementById('modal-stat-xp')!;
const modalStatSpeedEl = document.getElementById('modal-stat-speed')!;
const modalStatGatesEl = document.getElementById('modal-stat-gates')!;
const bottomDockWrapperEl = document.getElementById('bottom-dock-wrapper');
const dockCollapseBtnEl = document.getElementById('dock-collapse-btn');
const dockCarBtnEl = document.getElementById('dock-car-btn');
const dockSceneryBtnEl = document.getElementById('dock-scenery-btn');
const dockCamBtnEl = document.getElementById('dock-cam-btn');
const dockTimeBtnEl = document.getElementById('dock-time-btn');
const btnTimeTopEl = document.getElementById('btn-time-top');
const touchControlsLayerEl = document.getElementById('touch-controls-layer');
const touchAutoGasBtnEl = document.getElementById('touch-autogas-btn');
const touchCarBtnEl = document.getElementById('touch-car-btn');
const touchWorldBtnEl = document.getElementById('touch-world-btn');
const touchCamBtnEl = document.getElementById('touch-cam-btn');
const touchTimeBtnEl = document.getElementById('touch-time-btn');
const touchSteerReadoutEl = document.getElementById('touch-steer-readout');
const touchNitroFillEl = document.getElementById('touch-nitro-fill');
const touchGasBtnEl = document.getElementById('touch-gas');
const touchGasSubEl = document.getElementById('touch-gas-sub');
const btnTouchTopEl = document.getElementById('btn-touch-top');
const radarCtx = (document.getElementById('radar-canvas') as HTMLCanvasElement).getContext('2d')!;

// Main Menu Stat Bars
const statBarSpeedEl = document.getElementById('stat-bar-speed');
const statValSpeedEl = document.getElementById('stat-val-speed');
const statBarHandlingEl = document.getElementById('stat-bar-handling');
const statValHandlingEl = document.getElementById('stat-val-handling');
const statBarDriftEl = document.getElementById('stat-bar-drift');
const statValDriftEl = document.getElementById('stat-val-drift');
const statBarOffroadEl = document.getElementById('stat-bar-offroad');
const statValOffroadEl = document.getElementById('stat-val-offroad');
const menuActiveCarSummaryEl = document.getElementById('menu-active-car-summary');
const menuActiveScenerySummaryEl = document.getElementById('menu-active-scenery-summary');

function toggleDockCollapse() {
  if (!bottomDockWrapperEl) return;
  const collapsed = bottomDockWrapperEl.classList.toggle('collapsed');
  if (dockCollapseBtnEl) {
    dockCollapseBtnEl.textContent = collapsed ? '▴ Show Dock [H]' : '▾ Hide Dock [H]';
  }
}
if (dockCollapseBtnEl) dockCollapseBtnEl.addEventListener('click', toggleDockCollapse);

// Time of Day & Scenery Lighting Harmonizer
const TIME_MODES: TimeOfDayMode[] = ['sunset', 'night', 'day'];
let currentTimeOfDay: TimeOfDayMode = 'sunset';

function applyTimeOfDay(mode: TimeOfDayMode) {
  currentTimeOfDay = mode;
  document.body.dataset.time = mode;
  const skyHex = currentScenerySpec.skyTintHex[mode];

  if (mode === 'day') {
    sky.visible = true;
    starMat.opacity = 0;
    skyUniforms.turbidity.value = 2.2;
    skyUniforms.rayleigh.value = 1.15;
    skyUniforms.mieCoefficient.value = 0.0018;
    skyUniforms.mieDirectionalG.value = 0.78;
    const phi = THREE.MathUtils.degToRad(68);
    const theta = THREE.MathUtils.degToRad(168);
    sunDir.setFromSphericalCoords(1, phi, theta);
    orbDir.copy(sunDir);
    skyUniforms.sunPosition.value.copy(sunDir);
    sunDiscMesh.position.copy(orbDir).multiplyScalar(1750);
    sunDiscMat.color.setHex(0xffffff);
    sunDiscMat.opacity = 0.98;
    sunCoronaMat.color.setHex(0xffdf82);
    sunCoronaMat.opacity = 0.42;
    scene.background = new THREE.Color(skyHex);
    sceneFog.color.setHex(skyHex);
    sceneFog.density = 0.00038 * currentScenerySpec.fogDensityMultiplier;
    renderer.toneMappingExposure = 0.52;
    scene.environmentIntensity = 0.56;
    hemiLight.color.setHex(0x93c5fd);
    hemiLight.groundColor.setHex(currentScenerySpec.terrainPalette.valleyLowHex);
    hemiLight.intensity = 0.95;
    dirLight.color.setHex(0xfff7ed);
    dirLight.intensity = 2.4;
    spotL.intensity = 0;
    spotR.intensity = 0;
    headlightMat.emissiveIntensity = 1.5;
    gatePinkNeonMat.emissiveIntensity = 2.5;
    gateCyanNeonMat.emissiveIntensity = 2.5;
    bollardMat.emissiveIntensity = 1.3;
  } else if (mode === 'night') {
    sky.visible = true;
    starMat.opacity = 0.95;
    skyUniforms.turbidity.value = 1.2;
    skyUniforms.rayleigh.value = 0.35;
    skyUniforms.mieCoefficient.value = 0.0004;
    skyUniforms.mieDirectionalG.value = 0.65;
    const phi = THREE.MathUtils.degToRad(96);
    const theta = THREE.MathUtils.degToRad(168);
    sunDir.setFromSphericalCoords(1, phi, theta);
    skyUniforms.sunPosition.value.copy(sunDir);
    const moonPhi = THREE.MathUtils.degToRad(74);
    orbDir.setFromSphericalCoords(1, moonPhi, theta);
    sunDiscMesh.position.copy(orbDir).multiplyScalar(1750);
    sunDiscMat.color.setHex(0xe2f1ff);
    sunDiscMat.opacity = 0.96;
    sunCoronaMat.color.setHex(0x60a5fa);
    sunCoronaMat.opacity = 0.4;
    scene.background = new THREE.Color(skyHex);
    sceneFog.color.setHex(skyHex);
    sceneFog.density = 0.00044 * currentScenerySpec.fogDensityMultiplier;
    renderer.toneMappingExposure = 0.72;
    scene.environmentIntensity = 0.34;
    hemiLight.color.setHex(0x365078);
    hemiLight.groundColor.setHex(0x0d1322);
    hemiLight.intensity = 0.72;
    dirLight.color.setHex(0x88b0ff);
    dirLight.intensity = 0.95;
    spotL.intensity = 28;
    spotR.intensity = 28;
    spotL.distance = 135;
    spotR.distance = 135;
    headlightMat.emissiveIntensity = 6.5;
    gatePinkNeonMat.emissiveIntensity = 4.5;
    gateCyanNeonMat.emissiveIntensity = 4.5;
    bollardMat.emissiveIntensity = 3.2;
  } else {
    sky.visible = true;
    starMat.opacity = 0;
    skyUniforms.turbidity.value = 4.2;
    skyUniforms.rayleigh.value = 1.85;
    skyUniforms.mieCoefficient.value = 0.0024;
    skyUniforms.mieDirectionalG.value = 0.76;
    const phi = THREE.MathUtils.degToRad(85.2);
    const theta = THREE.MathUtils.degToRad(168);
    sunDir.setFromSphericalCoords(1, phi, theta);
    orbDir.copy(sunDir);
    skyUniforms.sunPosition.value.copy(sunDir);
    sunDiscMesh.position.copy(orbDir).multiplyScalar(1750);
    sunDiscMat.color.setHex(0xffffff);
    sunDiscMat.opacity = 0.98;
    sunCoronaMat.color.setHex(0xff9456);
    sunCoronaMat.opacity = 0.48;
    scene.background = new THREE.Color(skyHex);
    sceneFog.color.setHex(skyHex);
    sceneFog.density = 0.00048 * currentScenerySpec.fogDensityMultiplier;
    renderer.toneMappingExposure = 0.54;
    scene.environmentIntensity = 0.5;
    hemiLight.color.setHex(0xffb076);
    hemiLight.groundColor.setHex(0x2e2238);
    hemiLight.intensity = 0.85;
    dirLight.color.setHex(0xff964f);
    dirLight.intensity = 2.1;
    spotL.intensity = 8.5;
    spotR.intensity = 8.5;
    spotL.distance = 95;
    spotR.distance = 95;
    headlightMat.emissiveIntensity = 3.8;
    gatePinkNeonMat.emissiveIntensity = 3.4;
    gateCyanNeonMat.emissiveIntensity = 3.4;
    bollardMat.emissiveIntensity = 2.0;
  }

  const label = mode.toUpperCase();
  const prettyMode = mode.charAt(0).toUpperCase() + mode.slice(1);
  if (btnTimeTopEl) btnTimeTopEl.textContent = `${label} [L]`;
  if (dockTimeBtnEl) dockTimeBtnEl.textContent = `Sky: ${prettyMode} [L]`;
  if (touchTimeBtnEl) touchTimeBtnEl.textContent = `Sky: ${prettyMode}`;
  document.querySelectorAll<HTMLElement>('[data-time-select]').forEach((btn) => {
    const active = btn.dataset.timeSelect === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
}

function cycleTimeOfDay() {
  const next = TIME_MODES[(TIME_MODES.indexOf(currentTimeOfDay) + 1) % TIME_MODES.length];
  applyTimeOfDay(next);
}

// ============================================================================
// CAR & SCENERY SELECTION LOGIC (MAIN MENU + IN-GAME)
// ============================================================================
function updateHudCarWorldBanner() {
  if (hudCarWorldLabelEl) {
    hudCarWorldLabelEl.textContent = `${currentCarSpec.name.toUpperCase()} · ${currentScenerySpec.name.toUpperCase()}`;
  }
}

function selectCar(carId: CarId, applyDefaultPaintColor = true) {
  currentCarId = carId;
  currentCarSpec = getCarSpec(carId);
  document.body.dataset.car = carId;

  // Toggle 3D vehicle body mesh groups
  apexBodyGroup.visible = carId === 'apex';
  kazeBodyGroup.visible = carId === 'kaze';
  titanBodyGroup.visible = carId === 'titan';

  // Adjust wheel scale for Trophy Truck vs Street/Supercar
  const wheelScale = carId === 'titan' ? 1.28 : carId === 'kaze' ? 0.96 : 1.0;
  for (let i = 0; i < 4; i++) {
    wheelRoots[i].scale.setScalar(wheelScale);
  }

  if (applyDefaultPaintColor) {
    applyCarPaint(currentCarSpec.defaultPaint);
  }

  // Update Main Menu car selection cards
  document.querySelectorAll<HTMLElement>('[data-car-select]').forEach((btn) => {
    const isSelected = btn.dataset.carSelect === carId;
    btn.classList.toggle('active', isSelected);
    btn.setAttribute('aria-pressed', String(isSelected));
  });

  // Update Forza Horizon Performance Index (PI) Badges
  const piClass = carId === 'kaze' ? 'A' : 'S1';
  const piNum = carId === 'apex' ? '900' : carId === 'kaze' ? '800' : '880';
  [menuPiClassEl, specPiClassEl, hudPiClassEl].forEach((el) => {
    if (el) {
      el.textContent = piClass;
      el.dataset.class = piClass;
    }
  });
  [menuPiNumEl, specPiNumEl, hudPiNumEl].forEach((el) => {
    if (el) el.textContent = piNum;
  });

  // Update Main Menu Forza 0.0-10.0 performance telemetry bars
  if (statBarSpeedEl) statBarSpeedEl.style.width = `${currentCarSpec.stats.speed}%`;
  if (statValSpeedEl) statValSpeedEl.textContent = (currentCarSpec.stats.speed / 10).toFixed(1);
  if (statBarHandlingEl) statBarHandlingEl.style.width = `${currentCarSpec.stats.handling}%`;
  if (statValHandlingEl) statValHandlingEl.textContent = (currentCarSpec.stats.handling / 10).toFixed(1);
  if (statBarDriftEl) statBarDriftEl.style.width = `${currentCarSpec.stats.drift}%`;
  if (statValDriftEl) statValDriftEl.textContent = (currentCarSpec.stats.drift / 10).toFixed(1);
  if (statBarOffroadEl) statBarOffroadEl.style.width = `${currentCarSpec.stats.offroad}%`;
  if (statValOffroadEl) statValOffroadEl.textContent = (currentCarSpec.stats.offroad / 10).toFixed(1);
  if (menuActiveCarSummaryEl) {
    menuActiveCarSummaryEl.textContent = currentCarSpec.name.toUpperCase();
  }
  const shortName = carId === 'apex' ? 'Apex V10' : carId === 'kaze' ? 'Kaze RX-79' : 'Titan 4x4';
  if (dockCarBtnEl) {
    dockCarBtnEl.textContent = `Car: ${shortName} [V]`;
  }
  if (touchCarBtnEl) {
    touchCarBtnEl.textContent = `Car: ${carId === 'apex' ? 'Apex' : carId === 'kaze' ? 'Kaze' : 'Titan'}`;
  }
  updateHudCarWorldBanner();

  // Snap ride height smoothly to new car spec
  const pose = getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec);
  carState.y = pose.avgGroundY + currentCarSpec.rideHeightOffset;
  carState.isGrounded = true;
  carState.slopeRate = 0;
}

function cycleCar() {
  const nextIdx = (CAR_ORDER.indexOf(currentCarId) + 1) % CAR_ORDER.length;
  selectCar(CAR_ORDER[nextIdx], true);
  soundEngine.playChime(780, 0.14);
}

function selectScenery(sceneryId: SceneryId, syncDefaultTimeOfDay = true) {
  currentSceneryId = sceneryId;
  currentScenerySpec = getScenerySpec(sceneryId);
  document.body.dataset.scenery = sceneryId;

  // 1) Recolor 240x240 terrain vertices
  applyTerrainPalette(currentScenerySpec);

  // 2) Update water surface material
  waterMat.color.setHex(currentScenerySpec.waterColorHex);
  waterMat.opacity = currentScenerySpec.waterOpacity;
  waterMat.roughness = currentScenerySpec.waterRoughness;

  // 3) Update instanced trees (trunks & 3-tier foliage cones)
  trunkMat.color.setHex(currentScenerySpec.treeTrunkHex);
  const treePal = currentScenerySpec.treePaletteHex;
  for (let i = 0; i < placedTrees; i++) {
    _instColor.setHex(treePal[i % treePal.length]);
    treeCone1.setColorAt(i, _instColor);
    treeCone2.setColorAt(i, _instColor.clone().multiplyScalar(1.12));
    treeCone3.setColorAt(i, _instColor.clone().multiplyScalar(1.24));
  }
  if (treeCone1.instanceColor) treeCone1.instanceColor.needsUpdate = true;
  if (treeCone2.instanceColor) treeCone2.instanceColor.needsUpdate = true;
  if (treeCone3.instanceColor) treeCone3.instanceColor.needsUpdate = true;

  // 4) Update instanced rocks
  const rockPal = currentScenerySpec.rockPaletteHex;
  for (let i = 0; i < placedRocks; i++) {
    _instColor.setHex(rockPal[i % rockPal.length]);
    rockInst.setColorAt(i, _instColor);
  }
  if (rockInst.instanceColor) rockInst.instanceColor.needsUpdate = true;

  // 5) Update natural arches
  archMat.color.setHex(currentScenerySpec.archColorHex);
  archMat.emissive.setHex(sceneryId === 'cyber' ? 0x2e1065 : 0x000000);
  archMat.emissiveIntensity = sceneryId === 'cyber' ? 0.8 : 0;

  // 6) Sync sky & fog
  if (syncDefaultTimeOfDay) {
    applyTimeOfDay(currentScenerySpec.defaultTimeOfDay);
  } else {
    applyTimeOfDay(currentTimeOfDay);
  }

  // Update Main Menu scenery selection cards
  document.querySelectorAll<HTMLElement>('[data-scenery-select]').forEach((btn) => {
    const isSelected = btn.dataset.scenerySelect === sceneryId;
    btn.classList.toggle('active', isSelected);
    btn.setAttribute('aria-pressed', String(isSelected));
  });
  if (menuActiveScenerySummaryEl) {
    menuActiveScenerySummaryEl.textContent = currentScenerySpec.name.toUpperCase();
  }
  const shortWorld = sceneryId === 'canyon' ? 'Canyon' : sceneryId === 'alpine' ? 'Alpine' : 'Cyber Bay';
  if (dockSceneryBtnEl) {
    dockSceneryBtnEl.textContent = `World: ${shortWorld} [B]`;
  }
  if (touchWorldBtnEl) {
    touchWorldBtnEl.textContent = `World: ${shortWorld}`;
  }
  updateHudCarWorldBanner();
}

function cycleScenery() {
  const nextIdx = (SCENERY_ORDER.indexOf(currentSceneryId) + 1) % SCENERY_ORDER.length;
  selectScenery(SCENERY_ORDER[nextIdx], true);
  soundEngine.playChime(680, 0.16);
}

// Bind Main Menu Car, Scenery & Time-of-Day Buttons
document.querySelectorAll<HTMLElement>('[data-car-select]').forEach((btn) => {
  btn.addEventListener('click', () => {
    soundEngine.init();
    const id = btn.dataset.carSelect as CarId;
    if (id) {
      selectCar(id, true);
      soundEngine.playChime(760, 0.12);
    }
  });
});

document.querySelectorAll<HTMLElement>('[data-scenery-select]').forEach((btn) => {
  btn.addEventListener('click', () => {
    soundEngine.init();
    const id = btn.dataset.scenerySelect as SceneryId;
    if (id) {
      selectScenery(id, true);
      soundEngine.playChime(660, 0.14);
    }
  });
});

document.querySelectorAll<HTMLElement>('[data-time-select]').forEach((btn) => {
  btn.addEventListener('click', () => {
    soundEngine.init();
    const mode = btn.dataset.timeSelect as TimeOfDayMode;
    if (mode) {
      applyTimeOfDay(mode);
      soundEngine.playChime(840, 0.1);
    }
  });
});

if (btnTimeTopEl) btnTimeTopEl.addEventListener('click', cycleTimeOfDay);
if (dockTimeBtnEl) dockTimeBtnEl.addEventListener('click', cycleTimeOfDay);
if (dockCarBtnEl) dockCarBtnEl.addEventListener('click', cycleCar);
if (dockSceneryBtnEl) dockSceneryBtnEl.addEventListener('click', cycleScenery);
if (touchTimeBtnEl) {
  touchTimeBtnEl.addEventListener('click', (e) => {
    e.preventDefault();
    cycleTimeOfDay();
  });
}
if (touchCarBtnEl) {
  touchCarBtnEl.addEventListener('click', (e) => {
    e.preventDefault();
    cycleCar();
  });
}
if (touchWorldBtnEl) {
  touchWorldBtnEl.addEventListener('click', (e) => {
    e.preventDefault();
    cycleScenery();
  });
}

function setAutoGas(enabled: boolean) {
  autoGasEnabled = !!enabled;
  if (touchAutoGasBtnEl) {
    touchAutoGasBtnEl.textContent = `Auto-Gas: ${autoGasEnabled ? 'ON' : 'Off'}`;
    touchAutoGasBtnEl.classList.toggle('autogas-active', autoGasEnabled);
    touchAutoGasBtnEl.setAttribute('aria-pressed', String(autoGasEnabled));
  }
  if (touchGasSubEl) {
    touchGasSubEl.textContent = autoGasEnabled ? 'Auto ON' : 'Thrust';
  }
  if (touchGasBtnEl) {
    touchGasBtnEl.classList.toggle('active', autoGasEnabled || touchInput.gas);
  }
}

if (touchAutoGasBtnEl) {
  touchAutoGasBtnEl.addEventListener('click', (e) => {
    e.preventDefault();
    soundEngine.init();
    setAutoGas(!autoGasEnabled);
    soundEngine.playChime(autoGasEnabled ? 820 : 540, 0.1);
  });
}

// Initialize URL Parameters & Default Selection
const searchParams = new URLSearchParams(window.location.search);
const initCarParam = searchParams.get('car') as CarId | null;
const initSceneryParam = searchParams.get('scenery') as SceneryId | null;
const initTimeParam = searchParams.get('time') as TimeOfDayMode | null;

selectScenery(initSceneryParam && SCENERY_ORDER.includes(initSceneryParam) ? initSceneryParam : 'canyon', true);
selectCar(initCarParam && CAR_ORDER.includes(initCarParam) ? initCarParam : 'apex', true);
if (initTimeParam && TIME_MODES.includes(initTimeParam)) {
  applyTimeOfDay(initTimeParam);
}

// Touch Controls & Input Hygiene
let isTouchEnabled = false;
function setTouchMode(enabled: boolean) {
  isTouchEnabled = !!enabled;
  document.body.dataset.touch = isTouchEnabled ? 'true' : 'false';
  if (touchControlsLayerEl) {
    touchControlsLayerEl.classList.toggle('visible', isTouchEnabled && gameState === 'playing');
  }
  if (btnTouchTopEl) {
    btnTouchTopEl.textContent = 'TOUCH [T]';
    btnTouchTopEl.style.borderBottomColor = isTouchEnabled ? '#00f0ff' : '';
    btnTouchTopEl.style.color = isTouchEnabled ? '#00f0ff' : '';
  }
}
if (
  window.matchMedia('(pointer: coarse), (max-width: 900px)').matches ||
  searchParams.get('touch') === '1'
) {
  setTouchMode(true);
} else {
  setTouchMode(false);
}
window.addEventListener(
  'touchstart',
  () => {
    if (!isTouchEnabled) setTouchMode(true);
  },
  { once: true, passive: true },
);

function toggleTouchControls() {
  setTouchMode(!isTouchEnabled);
}
if (btnTouchTopEl) btnTouchTopEl.addEventListener('click', toggleTouchControls);

function clearAllHeldInputs() {
  keys.forward = false;
  keys.backward = false;
  keys.left = false;
  keys.right = false;
  keys.handbrake = false;
  keys.nitro = false;
  touchInput.gas = false;
  touchInput.brake = false;
  touchInput.drift = false;
  touchInput.nitro = false;
  document.querySelectorAll('.touch-action-btn').forEach((el) => {
    el.classList.remove('pressed');
    if (el.id !== 'touch-gas' || !autoGasEnabled) {
      el.classList.remove('active');
    }
  });
  resetAnalogSteerPad();
}
window.addEventListener('blur', clearAllHeldInputs);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearAllHeldInputs();
});

function setGameState(nextState: 'title' | 'playing' | 'paused') {
  gameState = nextState;
  document.body.dataset.state = nextState;
  if (touchControlsLayerEl) {
    touchControlsLayerEl.classList.toggle('visible', isTouchEnabled && nextState === 'playing');
  }
  if (nextState === 'title') {
    clearAllHeldInputs();
    titleScreenEl.removeAttribute('hidden');
    statusModalEl.setAttribute('hidden', '');
    statusModalEl.classList.remove('visible');
  } else if (nextState === 'playing') {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
    titleScreenEl.setAttribute('hidden', '');
    statusModalEl.setAttribute('hidden', '');
    statusModalEl.classList.remove('visible');
  } else if (nextState === 'paused') {
    clearAllHeldInputs();
    modalStatXpEl.textContent = `${skillState.bankedScore.toLocaleString()} XP`;
    modalStatSpeedEl.textContent = `${Math.round(sessionTopSpeedKmh)} KMH`;
    modalStatGatesEl.textContent = `${gatesClearedCount} / ${GATE_COUNT}`;
    statusModalEl.removeAttribute('hidden');
    statusModalEl.classList.add('visible');
  }
}

function resetCarOnHighway() {
  const { theta } = getRoadDistanceInfo(carState.x, carState.z);
  const pt = getHighwayPointAtAngle(theta);
  const y = getTerrainY(pt.x, pt.z);
  const yaw = Math.atan2(-pt.tx, -pt.tz);
  carState.x = pt.x;
  carState.z = pt.z;
  carState.y = y + currentCarSpec.rideHeightOffset;
  carState.vx = 0;
  carState.vy = 0;
  carState.vz = 0;
  carState.yaw = yaw;
  carState.yawRate = 0;
  carState.visualYawOffset = 0;
  carState.pitch = 0;
  carState.roll = 0;
  carState.suspensionBounce = 0;
  carState.suspensionVel = 0;
  carState.isGrounded = true;
  carState.slopeRate = 0;
  airTime = 0;
  cameraTrauma = 0;
  chassisBody.position.set(carState.x, carState.y, carState.z);
}

document.getElementById('start-drive-btn')?.addEventListener('click', () => {
  soundEngine.init();
  setGameState('playing');
});
document.getElementById('title-paint-btn')?.addEventListener('click', () => {
  const nextPaint = PAINT_PRESETS[(PAINT_PRESETS.indexOf(currentPaintHex) + 1) % PAINT_PRESETS.length];
  applyCarPaint(nextPaint);
});
document.getElementById('modal-resume-btn')?.addEventListener('click', () => {
  setGameState('playing');
});
document.getElementById('modal-reset-btn')?.addEventListener('click', () => {
  resetCarOnHighway();
  setGameState('playing');
});
document.getElementById('modal-title-btn')?.addEventListener('click', () => {
  setGameState('title');
});
document.getElementById('btn-menu-top')?.addEventListener('click', () => {
  setGameState('title');
});
document.querySelector('.radar-shell')?.addEventListener('click', () => {
  setGameState('title');
});

function toggleCameraMode() {
  cameraMode = cameraMode === 'chase' ? 'hood' : 'chase';
  const label = cameraMode === 'chase' ? 'Chase' : 'Hood';
  if (dockCamBtnEl) {
    dockCamBtnEl.textContent = `Cam: ${label} [C]`;
  }
  if (touchCamBtnEl) {
    touchCamBtnEl.textContent = `Cam: ${label}`;
  }
  const topCamEl = document.getElementById('btn-camera-top');
  if (topCamEl) {
    topCamEl.textContent = `CAM [C]`;
  }
}
if (dockCamBtnEl) dockCamBtnEl.addEventListener('click', toggleCameraMode);
document.getElementById('btn-camera-top')?.addEventListener('click', toggleCameraMode);
document.getElementById('dock-reset-btn')?.addEventListener('click', resetCarOnHighway);
document.getElementById('touch-reset-btn')?.addEventListener('click', (e) => {
  e.preventDefault();
  resetCarOnHighway();
});
document.getElementById('touch-cam-btn')?.addEventListener('click', (e) => {
  e.preventDefault();
  toggleCameraMode();
});
hudUnitToggleEl.addEventListener('click', () => {
  speedUnit = speedUnit === 'KMH' ? 'MPH' : 'KMH';
  hudUnitToggleEl.textContent = speedUnit;
});
document.getElementById('btn-mute-top')?.addEventListener('click', () => {
  soundEngine.muted = !soundEngine.muted;
  const el = document.getElementById('btn-mute-top');
  if (el) el.style.opacity = soundEngine.muted ? '0.45' : '1';
});
document.getElementById('btn-pause-top')?.addEventListener('click', () => {
  setGameState(gameState === 'playing' ? 'paused' : 'playing');
});

window.addEventListener('keydown', (e) => {
  soundEngine.init();
  if (gameState === 'title' && (e.code === 'Enter' || e.code === 'Space')) {
    e.preventDefault();
    setGameState('playing');
    return;
  }
  if (e.repeat && e.code !== 'Space') return;
  switch (e.code) {
    case 'KeyW':
    case 'ArrowUp':
      keys.forward = true;
      break;
    case 'KeyS':
    case 'ArrowDown':
      keys.backward = true;
      break;
    case 'KeyA':
    case 'ArrowLeft':
      keys.left = true;
      break;
    case 'KeyD':
    case 'ArrowRight':
      keys.right = true;
      break;
    case 'Space':
      keys.handbrake = true;
      e.preventDefault();
      break;
    case 'ShiftLeft':
    case 'ShiftRight':
      keys.nitro = true;
      break;
    case 'KeyV':
      cycleCar();
      break;
    case 'KeyB':
      cycleScenery();
      break;
    case 'KeyL':
      cycleTimeOfDay();
      break;
    case 'KeyC':
      toggleCameraMode();
      break;
    case 'KeyR':
      resetCarOnHighway();
      break;
    case 'KeyH':
      toggleDockCollapse();
      break;
    case 'KeyT':
      toggleTouchControls();
      break;
    case 'KeyM':
      soundEngine.muted = !soundEngine.muted;
      break;
    case 'KeyP':
    case 'Escape':
      if (gameState === 'playing' || gameState === 'paused') {
        setGameState(gameState === 'playing' ? 'paused' : 'playing');
      }
      break;
  }
});

window.addEventListener('keyup', (e) => {
  switch (e.code) {
    case 'KeyW':
    case 'ArrowUp':
      keys.forward = false;
      break;
    case 'KeyS':
    case 'ArrowDown':
      keys.backward = false;
      break;
    case 'KeyA':
    case 'ArrowLeft':
      keys.left = false;
      break;
    case 'KeyD':
    case 'ArrowRight':
      keys.right = false;
      break;
    case 'Space':
      keys.handbrake = false;
      break;
    case 'ShiftLeft':
    case 'ShiftRight':
      keys.nitro = false;
      break;
  }
});

// Touch Steer Pad & Multi-Touch Arcade Pedal Cluster
const touchSteerPadEl = document.getElementById('touch-steer-pad');
const touchSteerThumbEl = document.getElementById('touch-steer-thumb');
const touchLeftEl = document.getElementById('touch-left');
const touchRightEl = document.getElementById('touch-right');
let activeSteerPointerId: number | null = null;

function triggerHapticPulse(ms = 10) {
  try {
    if (typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function') {
      navigator.vibrate(ms);
    }
  } catch {
    // Ignore on unsupported devices
  }
}

function handleSteerPadMove(clientX: number) {
  if (!touchSteerPadEl || !touchSteerThumbEl) return;
  const rect = touchSteerPadEl.getBoundingClientRect();
  const res = computeAnalogSteerFromPad(clientX, rect.left, rect.width, 62);
  touchInput.steerAnalog = res.steerInput;
  touchInput.left = res.zone === 'left';
  touchInput.right = res.zone === 'right';
  touchSteerThumbEl.style.transform = `translateX(${res.thumbOffsetPx}px) rotate(${res.steerDegrees}deg)`;
  touchSteerPadEl.setAttribute('aria-valuenow', String(res.ariaValueNow));
  if (touchSteerReadoutEl) {
    touchSteerReadoutEl.textContent =
      res.zone === 'center'
        ? 'CENTER'
        : `${res.zone === 'left' ? 'L' : 'R'} ${Math.abs(res.steerDegrees)}°`;
  }
  touchLeftEl?.classList.toggle('pressed', touchInput.left);
  touchLeftEl?.classList.toggle('active', touchInput.left);
  touchRightEl?.classList.toggle('pressed', touchInput.right);
  touchRightEl?.classList.toggle('active', touchInput.right);
}

function resetAnalogSteerPad() {
  activeSteerPointerId = null;
  touchInput.steerAnalog = 0;
  touchInput.left = false;
  touchInput.right = false;
  if (touchSteerPadEl) {
    touchSteerPadEl.classList.remove('active');
    touchSteerPadEl.setAttribute('aria-valuenow', '0');
  }
  if (touchSteerThumbEl) {
    touchSteerThumbEl.style.transform = 'translateX(0px) rotate(0deg)';
  }
  if (touchSteerReadoutEl) {
    touchSteerReadoutEl.textContent = 'CENTER';
  }
  touchLeftEl?.classList.remove('pressed', 'active');
  touchRightEl?.classList.remove('pressed', 'active');
}

if (touchSteerPadEl) {
  touchSteerPadEl.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    soundEngine.init();
    if (!isTouchEnabled) setTouchMode(true);
    activeSteerPointerId = e.pointerId;
    touchSteerPadEl.classList.add('active');
    triggerHapticPulse(8);
    try {
      touchSteerPadEl.setPointerCapture(e.pointerId);
    } catch {
      // Ignore
    }
    handleSteerPadMove(e.clientX);
  });
  touchSteerPadEl.addEventListener('pointermove', (e) => {
    if (activeSteerPointerId === e.pointerId) {
      e.preventDefault();
      handleSteerPadMove(e.clientX);
    }
  });
  const onEnd = (e: PointerEvent) => {
    if (activeSteerPointerId === null || e.pointerId === activeSteerPointerId) {
      resetAnalogSteerPad();
    }
  };
  touchSteerPadEl.addEventListener('pointerup', onEnd);
  touchSteerPadEl.addEventListener('pointercancel', onEnd);
  touchSteerPadEl.addEventListener('lostpointercapture', resetAnalogSteerPad);
}

type TouchPedalKey = 'gas' | 'brake' | 'drift' | 'nitro';
const activePedalPointers = new Map<number, TouchPedalKey>();
const pedalElementMap: Record<TouchPedalKey, HTMLElement | null> = {
  gas: document.getElementById('touch-gas'),
  brake: document.getElementById('touch-brake'),
  drift: document.getElementById('touch-drift'),
  nitro: document.getElementById('touch-nitro'),
};

function syncTouchPedalStates() {
  const held: Record<TouchPedalKey, boolean> = {
    gas: false,
    brake: false,
    drift: false,
    nitro: false,
  };
  for (const key of activePedalPointers.values()) {
    held[key] = true;
  }
  (Object.keys(held) as TouchPedalKey[]).forEach((key) => {
    touchInput[key] = held[key];
    const el = pedalElementMap[key];
    if (el) {
      el.classList.toggle('pressed', held[key]);
      el.classList.toggle('active', held[key] || (key === 'gas' && autoGasEnabled));
    }
  });
}

function resolvePedalKeyFromElement(target: Element | null): TouchPedalKey | null {
  const btn = target?.closest('.touch-action-btn');
  if (!btn) return null;
  if (btn.id === 'touch-gas') return 'gas';
  if (btn.id === 'touch-brake') return 'brake';
  if (btn.id === 'touch-drift') return 'drift';
  if (btn.id === 'touch-nitro') return 'nitro';
  return null;
}

function bindTouchPedal(id: string, key: TouchPedalKey) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    soundEngine.init();
    if (!isTouchEnabled) setTouchMode(true);
    triggerHapticPulse(12);
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      // Ignore
    }
    activePedalPointers.set(e.pointerId, key);
    syncTouchPedalStates();
  });
  el.addEventListener('pointermove', (e) => {
    if (!activePedalPointers.has(e.pointerId)) return;
    e.preventDefault();
    // Allow thumb sliding across adjacent pedals in the cluster
    const hitEl = document.elementFromPoint(e.clientX, e.clientY);
    const hoveredKey = resolvePedalKeyFromElement(hitEl);
    if (hoveredKey && hoveredKey !== activePedalPointers.get(e.pointerId)) {
      activePedalPointers.set(e.pointerId, hoveredKey);
      triggerHapticPulse(8);
      syncTouchPedalStates();
    }
  });
  const releasePointer = (e: PointerEvent) => {
    e.preventDefault();
    activePedalPointers.delete(e.pointerId);
    syncTouchPedalStates();
  };
  el.addEventListener('pointerup', releasePointer);
  el.addEventListener('pointercancel', releasePointer);
  el.addEventListener('lostpointercapture', (e) => {
    activePedalPointers.delete(e.pointerId);
    syncTouchPedalStates();
  });
}

bindTouchPedal('touch-gas', 'gas');
bindTouchPedal('touch-brake', 'brake');
bindTouchPedal('touch-drift', 'drift');
bindTouchPedal('touch-nitro', 'nitro');

if (searchParams.get('autostart') === '1') {
  setGameState('playing');
}

// Precompute Radar Highway Loop Coordinates (Zero Per-Frame Trig Overhead)
const RADAR_HIGHWAY_POINTS = Array.from({ length: 61 }, (_, i) => {
  const pt = getHighwayPointAtAngle((i / 60) * Math.PI * 2);
  return { x: pt.x, z: pt.z };
});

// Highway Radar Minimap (Forza Horizon 5 Circular GPS Styling)
function drawRadar(carX: number, carZ: number, carYaw: number) {
  radarCtx.clearRect(0, 0, 168, 168);
  radarCtx.save();
  radarCtx.translate(84, 84);
  radarCtx.rotate(carYaw);
  const scale = 0.16;
  const lx = (OASIS_X - carX) * scale;
  const lz = (OASIS_Z - carZ) * scale;
  radarCtx.fillStyle =
    currentSceneryId === 'cyber'
      ? 'rgba(0, 240, 255, 0.35)'
      : currentSceneryId === 'alpine'
        ? 'rgba(125, 211, 252, 0.38)'
        : 'rgba(14, 165, 233, 0.32)';
  radarCtx.beginPath();
  radarCtx.arc(lx, lz, OASIS_RADIUS * scale, 0, Math.PI * 2);
  radarCtx.fill();

  // Road dark casing
  radarCtx.lineCap = 'round';
  radarCtx.lineJoin = 'round';
  radarCtx.lineWidth = 8.5;
  radarCtx.strokeStyle = 'rgba(8, 10, 16, 0.9)';
  radarCtx.beginPath();
  for (let i = 0; i < RADAR_HIGHWAY_POINTS.length; i++) {
    const pt = RADAR_HIGHWAY_POINTS[i];
    const rx = (pt.x - carX) * scale;
    const rz = (pt.z - carZ) * scale;
    if (i === 0) radarCtx.moveTo(rx, rz);
    else radarCtx.lineTo(rx, rz);
  }
  radarCtx.stroke();

  // Bright Horizon white asphalt loop
  radarCtx.lineWidth = 4.2;
  radarCtx.strokeStyle = 'rgba(255, 255, 255, 0.88)';
  radarCtx.stroke();

  // Checkpoint gates
  for (let i = 0; i < gates.length; i++) {
    const g = gates[i];
    const gx = (g.x - carX) * scale;
    const gz = (g.z - carZ) * scale;
    radarCtx.fillStyle = g.cleared ? '#00f0ff' : '#ff0055';
    radarCtx.beginPath();
    radarCtx.arc(gx, gz, 3.6, 0, Math.PI * 2);
    radarCtx.fill();
  }

  // Rival Drivatars
  radarCtx.fillStyle = '#facc15';
  for (let i = 0; i < rivals.length; i++) {
    const pos = rivals[i].mesh.position;
    const rx = (pos.x - carX) * scale;
    const rz = (pos.z - carZ) * scale;
    radarCtx.beginPath();
    radarCtx.arc(rx, rz, 2.8, 0, Math.PI * 2);
    radarCtx.fill();
  }
  radarCtx.restore();

  // Forza Horizon Player Chevron (White halo + Horizon Magenta core)
  radarCtx.fillStyle = '#ffffff';
  radarCtx.beginPath();
  radarCtx.moveTo(84, 73);
  radarCtx.lineTo(91, 91);
  radarCtx.lineTo(84, 87);
  radarCtx.lineTo(77, 91);
  radarCtx.closePath();
  radarCtx.fill();

  radarCtx.fillStyle = '#ff0055';
  radarCtx.beginPath();
  radarCtx.moveTo(84, 76);
  radarCtx.lineTo(89, 89);
  radarCtx.lineTo(84, 86);
  radarCtx.lineTo(79, 89);
  radarCtx.closePath();
  radarCtx.fill();
}

// Render initial GPS minimap frame immediately
drawRadar(carState.x, carState.z, carState.yaw);

// Reusable Math Scratch Objects (Zero-GC Loop)
const clock = new THREE.Clock();
let hudUpdateAccumulator = 0;
const _carPos = new THREE.Vector3();
const _carQuat = new THREE.Quaternion();
const _fwdVec = new THREE.Vector3();
const _upVec = new THREE.Vector3();
const _bodyEuler = new THREE.Euler();
const _wheelEuler = new THREE.Euler();
let chaseCamState = createInitialChaseCameraState(0, -1);

// Cached HUD strings to prevent redundant DOM layout/style thrashing
let lastRpmScaleStr = '';
let lastNitroScaleStr = '';
let lastSkillTimerScaleStr = '';
let lastSpeedStr = '';
let lastGearStr = '';
let lastRpmTextStr = '';
let lastBankedXpStr = '';
let lastGatesStr = '';
let lastBestChainStr = '';

// Adaptive frame-time monitor to prevent GPU fill-rate lag
let smoothedFrameDt = 0.016;
let slowFrameCounter = 0;
let pixelRatioScaledDown = false;

function syncCarAndWheelTransforms(
  pose: ReturnType<typeof getWheelGroundPose>,
  steerAngle: number,
  isGrounded: boolean,
) {
  _carPos.set(carState.x, carState.y + carState.suspensionBounce, carState.z);
  const visualYaw = carState.yaw + carState.visualYawOffset;
  _bodyEuler.set(carState.pitch, visualYaw, carState.roll, 'YXZ');
  _carQuat.setFromEuler(_bodyEuler);
  carRootGroup.position.copy(_carPos);
  carRootGroup.quaternion.copy(_carQuat);

  const wRadius = currentCarSpec.wheelRadius;
  const wheelCorners = [
    { x: pose.flX, y: pose.hFL + wRadius, z: pose.flZ, isFront: true },
    { x: pose.frX, y: pose.hFR + wRadius, z: pose.frZ, isFront: true },
    { x: pose.rlX, y: pose.hRL + wRadius, z: pose.rlZ, isFront: false },
    { x: pose.rrX, y: pose.hRR + wRadius, z: pose.rrZ, isFront: false },
  ];

  for (let i = 0; i < 4; i++) {
    const wc = wheelCorners[i];
    const wy = isGrounded ? wc.y : carState.y - 0.14;
    wheelRoots[i].position.set(wc.x, wy, wc.z);
    _wheelEuler.set(carState.pitch, visualYaw + (wc.isFront ? steerAngle : 0), carState.roll, 'YXZ');
    wheelRoots[i].quaternion.setFromEuler(_wheelEuler);
    wheelSpins[i].rotation.x = carState.wheelSpin;
  }
}

let lastGroundPose = getWheelGroundPose(0, 0, 0, currentCarSpec);
carState.y = lastGroundPose.avgGroundY + currentCarSpec.rideHeightOffset;
syncCarAndWheelTransforms(lastGroundPose, 0, true);

function afterDraw(cb: () => void) {
  const gl = renderer.getContext() as WebGL2RenderingContext;
  if (!gl || !gl.fenceSync) {
    window.setTimeout(cb, 80);
    return;
  }
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  gl.flush();
  const poll = () => {
    const status = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
    if (status === gl.UNSIGNALED) {
      window.setTimeout(poll, 5);
    } else {
      if (sync) gl.deleteSync(sync);
      window.setTimeout(cb, 80);
    }
  };
  window.setTimeout(poll, 5);
}

function tick() {
  const rawDt = clock.getDelta();
  let dt = rawDt;
  if (dt > 0.05) dt = 0.05;
  const elapsed = clock.getElapsedTime();

  // Adaptive resolution scaler if GPU frame rate drops below 37 FPS
  if (!isSoftwareGl && !pixelRatioScaledDown && rawDt > 0.001 && rawDt < 0.25) {
    smoothedFrameDt = smoothedFrameDt * 0.9 + rawDt * 0.1;
    if (smoothedFrameDt > 0.027) {
      slowFrameCounter++;
      if (slowFrameCounter > 24 && renderer.getPixelRatio() > 1) {
        pixelRatioScaledDown = true;
        renderer.setPixelRatio(1);
        composer.setSize(window.innerWidth, window.innerHeight);
      }
    } else {
      slowFrameCounter = Math.max(0, slowFrameCounter - 1);
    }
  }

  // Animate world landmarks
  wheelRimGroup.rotation.z = elapsed * 0.22;
  for (let i = 0; i < gondolas.length; i++) {
    gondolas[i].rotation.z = -wheelRimGroup.rotation.z;
  }
  for (let i = 0; i < searchlights.length; i++) {
    searchlights[i].rotation.z = Math.sin(elapsed * 0.9 + i * 1.4) * 0.38;
    searchlights[i].rotation.x = Math.cos(elapsed * 0.7 + i * 1.1) * 0.22;
  }
  for (let i = 0; i < balloons.length; i++) {
    const b = balloons[i];
    b.mesh.position.y = b.baseY + Math.sin(elapsed * 0.8 + b.phase) * 2.4;
    b.mesh.rotation.y = elapsed * 0.12 + b.phase;
  }
  bumpTex.offset.x = (elapsed * 0.015) % 1;
  bumpTex.offset.y = (elapsed * 0.01) % 1;

  // Step AI Rivals along the highway loop with physical lateral knockback recovery
  let nearMissJustTriggered = false;
  let tradingPaintJustTriggered = false;
  let wreckageJustTriggered = false;
  let landscapingJustTriggered = false;

  for (let i = 0; i < rivals.length; i++) {
    const r = rivals[i];
    r.theta = (r.theta + r.angularSpeed * dt) % (Math.PI * 2);
    if (r.theta < 0) r.theta += Math.PI * 2;
    if (r.nearMissCooldown > 0) r.nearMissCooldown = Math.max(0, r.nearMissCooldown - dt);
    if (r.hitCooldown > 0) r.hitCooldown = Math.max(0, r.hitCooldown - dt);
    if (Math.abs(r.lateralVel) > 0.01) {
      r.laneOffset = THREE.MathUtils.clamp(r.laneOffset + r.lateralVel * dt, -11.5, 11.5);
      r.lateralVel *= Math.exp(-4.5 * dt);
    } else {
      r.laneOffset = THREE.MathUtils.damp(r.laneOffset, r.baseLaneOffset, 1.8, dt);
    }
    const pt = getHighwayPointAtAngle(r.theta);
    const rx = pt.x + pt.nx * r.laneOffset;
    const rz = pt.z + pt.nz * r.laneOffset;
    const ry = getHighwayElevationAtAngle(r.theta) + 0.16;
    const dirSign = r.angularSpeed >= 0 ? 1 : -1;
    const ryaw = Math.atan2(-pt.tx * dirSign, -pt.tz * dirSign) - r.lateralVel * 0.035;
    r.mesh.position.set(rx, ry, rz);
    r.mesh.rotation.set(0, ryaw, 0);
  }

  // Step Airborne Smashed Props (Bollards, Festival Flags, Sapling Trees)
  if (activeFlyingProps.length > 0) {
    let bollardDirty = false;
    let flagDirty = false;
    let treeDirty = false;

    for (let i = activeFlyingProps.length - 1; i >= 0; i--) {
      const p = activeFlyingProps[i];
      p.smashedAge += dt;

      if (p.smashedAge < 3.6) {
        p.vy -= 26 * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        p.rx += p.avx * dt;
        p.ry += p.avy * dt;
        p.rz += p.avz * dt;

        const gY = getTerrainY(p.x, p.z) + 0.15;
        if (p.y < gY) {
          p.y = gY;
          p.vy = -p.vy * 0.34;
          p.vx *= 0.68;
          p.vz *= 0.68;
          p.avx *= 0.72;
          p.avz *= 0.72;
        }

        _dummy.position.set(p.x, p.y, p.z);
        _dummy.rotation.set(p.rx, p.ry, p.rz);
        _dummy.scale.setScalar(p.scale);
        _dummy.updateMatrix();
        if (p.meshKind === 'bollard') {
          bollardInst.setMatrixAt(p.instanceIdx, _dummy.matrix);
          bollardDirty = true;
        } else if (p.meshKind === 'flag') {
          flagInst.setMatrixAt(p.instanceIdx, _dummy.matrix);
          flagDirty = true;
        } else {
          treeTrunkInst.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeCone1.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeCone2.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeCone3.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeDirty = true;
        }
      } else if (
        p.smashedAge > 22 &&
        (carState.x - p.homeX) ** 2 + (carState.z - p.homeZ) ** 2 > 45 ** 2
      ) {
        // Respawn prop cleanly at its home coordinates once player drives away
        p.smashed = false;
        p.x = p.homeX;
        p.y = p.homeY;
        p.z = p.homeZ;
        p.rx = 0;
        p.ry = p.homeYaw;
        p.rz = 0;
        _dummy.position.set(p.homeX, p.homeY, p.homeZ);
        _dummy.rotation.set(0, p.homeYaw, 0);
        _dummy.scale.setScalar(p.scale);
        _dummy.updateMatrix();
        if (p.meshKind === 'bollard') {
          bollardInst.setMatrixAt(p.instanceIdx, _dummy.matrix);
          bollardDirty = true;
        } else if (p.meshKind === 'flag') {
          flagInst.setMatrixAt(p.instanceIdx, _dummy.matrix);
          flagDirty = true;
        } else {
          treeTrunkInst.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeCone1.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeCone2.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeCone3.setMatrixAt(p.instanceIdx, _dummy.matrix);
          treeDirty = true;
        }
        activeFlyingProps.splice(i, 1);
      }
    }

    if (bollardDirty) bollardInst.instanceMatrix.needsUpdate = true;
    if (flagDirty) flagInst.instanceMatrix.needsUpdate = true;
    if (treeDirty) {
      treeTrunkInst.instanceMatrix.needsUpdate = true;
      treeCone1.instanceMatrix.needsUpdate = true;
      treeCone2.instanceMatrix.needsUpdate = true;
      treeCone3.instanceMatrix.needsUpdate = true;
    }
  }

  // Title Screen / Pause Showcase Orbit Camera (Frames Car in Right Viewport Half)
  if (gameState !== 'playing') {
    lastGroundPose = getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec);
    syncCarAndWheelTransforms(lastGroundPose, 0.18, true);
    const orbitAng = elapsed * 0.25 + 0.75;
    const radius = 7.4;
    const camX = _carPos.x + Math.sin(orbitAng) * radius;
    const camY = _carPos.y + 1.85 + Math.sin(orbitAng * 0.5) * 0.25;
    const camZ = _carPos.z + Math.cos(orbitAng) * radius;
    camera.position.set(camX, camY, camZ);
    const camRightX = Math.cos(orbitAng);
    const camRightZ = -Math.sin(orbitAng);
    camera.lookAt(_carPos.x - camRightX * 2.15, _carPos.y + 0.68, _carPos.z - camRightZ * 2.15);
    sunDiscMesh.position.copy(camera.position).addScaledVector(orbDir, 1750);
    starField.position.copy(camera.position);
    soundEngine.update(0.14, 0, false, false, 0, currentCarSpec.engineBaseHz, dt);
    if (bloomPass.enabled) composer.render();
    else renderer.render(scene, camera);
    if (isSoftwareGl) afterDraw(tick);
    else requestAnimationFrame(tick);
    return;
  }

  // Active Gameplay Vehicle Physics & Controls
  const driveCmd = computeEffectiveTouchDriveInput({
    keyForward: keys.forward,
    keyBackward: keys.backward,
    keyLeft: keys.left,
    keyRight: keys.right,
    keyHandbrake: keys.handbrake,
    keyNitro: keys.nitro,
    touchGas: touchInput.gas,
    touchBrake: touchInput.brake,
    touchLeft: touchInput.left,
    touchRight: touchInput.right,
    touchSteerAnalog: touchInput.steerAnalog,
    touchDrift: touchInput.drift,
    touchNitro: touchInput.nitro,
    autoGasEnabled,
    nitroCharge,
  });
  const { throttle, steerInput, handbrake, isBoosting } = driveCmd;

  // Sub-step vehicle kinematics at high speed or longer frame intervals for rock-solid 200-350 km/h stability
  const subSteps = dt > 0.02 ? 2 : 1;
  const stepDt = dt / subSteps;

  let kmh = 0;
  let vLong = 0;
  let isDrifting = false;
  let newFwdX = -Math.sin(carState.yaw);
  let newFwdZ = -Math.cos(carState.yaw);
  let newRightX = Math.cos(carState.yaw);
  let newRightZ = -Math.sin(carState.yaw);
  let pose = lastGroundPose;
  let surfaceProfile: SurfacePhysicsProfile = {
    surface: 'asphalt',
    gripMu: 1.08,
    rollingResistance: 0.36,
    waterDepth: 0,
    waterDragCoeff: 0,
    surfaceRoughnessAmp: 0,
  };

  for (let step = 0; step < subSteps; step++) {
    const speedMs = Math.hypot(carState.vx, carState.vz);
    kmh = speedMs * 3.6;
    if (kmh > sessionTopSpeedKmh) sessionTopSpeedKmh = kmh;

    const targetSteer = steerInput * THREE.MathUtils.lerp(0.46, 0.2, Math.min(1, kmh / 230));
    currentSteerAngle = THREE.MathUtils.damp(currentSteerAngle, targetSteer, 13, stepDt);

    const prevGroundY = lastGroundPose.avgGroundY + currentCarSpec.rideHeightOffset;
    const wasGrounded = carState.isGrounded;

    // 1. Evaluate Surface Friction, Rolling Resistance & Oasis Water Drag
    const { distance: roadDist } = getRoadDistanceInfo(carState.x, carState.z);
    const isOffroad = roadDist > ROAD_HALF_WIDTH + 3;
    surfaceProfile = computeSurfacePhysicsProfile({
      groundY: lastGroundPose.avgGroundY,
      roadDistance: roadDist,
      carId: currentCarId,
      sceneryId: currentSceneryId,
    });

    // 2. Step Real Automotive 2D Dynamics (Slope Gravity, Weight Transfer, Pacejka Lateral Grip)
    const hStep = stepAutomotiveHorizontalPhysics({
      vx: carState.vx,
      vz: carState.vz,
      yaw: carState.yaw,
      yawRate: carState.yawRate,
      visualYawOffset: carState.visualYawOffset,
      steerAngle: currentSteerAngle,
      throttle,
      handbrake,
      isGrounded: wasGrounded,
      terrainPitch: lastGroundPose.terrainPitch,
      terrainRoll: lastGroundPose.terrainRoll,
      carSpec: currentCarSpec,
      surfaceProfile,
      dt: stepDt,
    });

    carState.vx = hStep.vx;
    carState.vz = hStep.vz;
    carState.yaw = hStep.yaw;
    carState.yawRate = hStep.yawRate;
    carState.visualYawOffset = hStep.visualYawOffset;
    vLong = hStep.vLong;
    isDrifting = hStep.isDrifting;

    newFwdX = -Math.sin(carState.yaw);
    newFwdZ = -Math.cos(carState.yaw);
    newRightX = Math.cos(carState.yaw);
    newRightZ = -Math.sin(carState.yaw);

    if (isBoosting) {
      const impulse = computeSafeNitroImpulse({
        forwardX: newFwdX,
        forwardZ: newFwdZ,
        kmh,
        dt: stepDt,
        isGrounded: wasGrounded,
        maxNitroKmh: currentCarSpec.maxNitroKmh,
      });
      carState.vx += impulse.dvx;
      carState.vz += impulse.dvz;
    }

    carState.x += carState.vx * stepDt;
    carState.z += carState.vz * stepDt;
    const bound = WORLD_SIZE * 0.46;
    carState.x = THREE.MathUtils.clamp(carState.x, -1380, bound);
    carState.z = THREE.MathUtils.clamp(carState.z, -1380, bound);

    // 3. Resolve Map Obstacle Physics via Spatial Hash Grid (3x3 Neighbor Cells)
    const carRadius = currentCarId === 'titan' ? 1.52 : 1.36;
    const baseCellX = Math.floor(carState.x / COLLIDER_CELL_SIZE) + 500;
    const baseCellZ = Math.floor(carState.z / COLLIDER_CELL_SIZE) + 500;

    for (let ox = -1; ox <= 1; ox++) {
      for (let oz = -1; oz <= 1; oz++) {
        const cellKey = ((baseCellX + ox) << 12) | ((baseCellZ + oz) & 0xfff);

        // 3a. Destructible Smashable Props (Bollards, Festival Flags, Sapling Trees)
        const smashBucket = smashableGrid.get(cellKey);
        if (smashBucket) {
          for (let j = 0; j < smashBucket.length; j++) {
            const prop = smashBucket[j];
            if (prop.smashed) continue;
            const res = resolveMapColliderImpact({
              carX: carState.x,
              carZ: carState.z,
              carVx: carState.vx,
              carVz: carState.vz,
              carYaw: carState.yaw,
              carRadius,
              colX: prop.homeX,
              colZ: prop.homeZ,
              colRadius: prop.radius,
              kind: 'smashable',
            });
            if (res.collided) {
              prop.smashed = true;
              prop.smashedAge = 0;
              prop.x = prop.homeX;
              prop.y = prop.homeY + 0.25;
              prop.z = prop.homeZ;
              prop.vx = res.propLaunchVx + (Math.random() - 0.5) * 4;
              prop.vy = res.propLaunchVy + Math.random() * 2.5;
              prop.vz = res.propLaunchVz + (Math.random() - 0.5) * 4;
              prop.avx = (Math.random() - 0.5) * 14;
              prop.avy = (Math.random() - 0.5) * 10;
              prop.avz = (Math.random() - 0.5) * 14;
              activeFlyingProps.push(prop);

              carState.vx = res.carVx;
              carState.vz = res.carVz;
              cameraTrauma = Math.min(0.28, cameraTrauma + 0.08);
              soundEngine.playChime(prop.meshKind === 'sapling' ? 520 : 640, 0.09);

              if (prop.meshKind === 'sapling') {
                landscapingJustTriggered = true;
              } else {
                wreckageJustTriggered = true;
              }

              for (let sp = 0; sp < 5; sp++) {
                spawnParticle(
                  prop.homeX,
                  prop.homeY + 1.1,
                  prop.homeZ,
                  res.propLaunchVx * 0.4 + (Math.random() - 0.5) * 8,
                  2.5 + Math.random() * 5,
                  res.propLaunchVz * 0.4 + (Math.random() - 0.5) * 8,
                  0.26,
                  0.6,
                  0.48,
                  prop.debrisColorHex,
                );
              }
            }
          }
        }

        // 3b. Rigid Map Colliders (Rocks, Boulders, Thick Trees, Bridge Pylons, Gate Pillars)
        const rigidBucket = rigidGrid.get(cellKey);
        if (rigidBucket) {
          for (let j = 0; j < rigidBucket.length; j++) {
            const col = rigidBucket[j];
            const res = resolveMapColliderImpact({
              carX: carState.x,
              carZ: carState.z,
              carVx: carState.vx,
              carVz: carState.vz,
              carYaw: carState.yaw,
              carRadius,
              colX: col.x,
              colZ: col.z,
              colRadius: col.radius,
              kind: 'rigid',
            });
            if (res.collided) {
              carState.x = res.carX;
              carState.z = res.carZ;
              carState.vx = res.carVx;
              carState.vz = res.carVz;
              carState.yaw += res.carYawDelta;
              carState.yawRate += res.carYawDelta * 8;
              if (res.impactSpeedKmh > 18) {
                const traumaAdd = Math.min(0.45, (res.impactSpeedKmh / 180) * 0.38);
                cameraTrauma = Math.min(0.5, cameraTrauma + traumaAdd);
                soundEngine.playChime(320, 0.14);
                const sparkCount = Math.min(8, 2 + Math.floor(res.impactSpeedKmh / 35));
                for (let sp = 0; sp < sparkCount; sp++) {
                  spawnParticle(
                    carState.x - res.normalX * carRadius,
                    carState.y + 0.35,
                    carState.z - res.normalZ * carRadius,
                    res.normalX * (4 + Math.random() * 8) + (Math.random() - 0.5) * 7,
                    2 + Math.random() * 5,
                    res.normalZ * (4 + Math.random() * 8) + (Math.random() - 0.5) * 7,
                    0.22,
                    0.4,
                    0.35,
                    col.sparkColorHex,
                  );
                }
              }
            }
          }
        }
      }
    }

    // 4. AI Rival Collisions (2-Body Momentum Exchange) & Near-Misses
    for (let i = 0; i < rivals.length; i++) {
      const r = rivals[i];
      const rPos = r.mesh.position;
      const dist = Math.hypot(carState.x - rPos.x, carState.z - rPos.z);
      if (dist < 3.6 && dist > 0.01) {
        const res = resolveMapColliderImpact({
          carX: carState.x,
          carZ: carState.z,
          carVx: carState.vx,
          carVz: carState.vz,
          carYaw: carState.yaw,
          carRadius: 1.45,
          colX: rPos.x,
          colZ: rPos.z,
          colRadius: 2.15,
          kind: 'rival',
        });
        carState.x = res.carX;
        carState.z = res.carZ;
        carState.vx = res.carVx;
        carState.vz = res.carVz;
        carState.yaw += res.carYawDelta * 0.6;
        // Transfer lateral momentum to the rival car so it gets shoved across the lane
        const pt = getHighwayPointAtAngle(r.theta);
        const rivalPushLat = res.propLaunchVx * pt.nx + res.propLaunchVz * pt.nz;
        r.lateralVel = THREE.MathUtils.clamp(r.lateralVel + rivalPushLat * 0.55, -14, 14);
        cameraTrauma = Math.min(0.38, cameraTrauma + 0.18);
        if (r.hitCooldown === 0 && kmh > 45) {
          r.hitCooldown = 3.0;
          tradingPaintJustTriggered = true;
          soundEngine.playChime(440, 0.14);
          for (let sp = 0; sp < 6; sp++) {
            spawnParticle(
              (carState.x + rPos.x) * 0.5,
              carState.y + 0.45,
              (carState.z + rPos.z) * 0.5,
              (Math.random() - 0.5) * 10,
              2.5 + Math.random() * 5,
              (Math.random() - 0.5) * 10,
              0.2,
              0.4,
              0.32,
              0xffaa3b,
            );
          }
        }
      } else if (dist < 7.8 && kmh > 95 && r.nearMissCooldown === 0) {
        r.nearMissCooldown = 4.5;
        nearMissJustTriggered = true;
        soundEngine.playChime(920, 0.15);
      }
    }

    // 5. 4-Wheel Ground Pose & Vertical Ballistic / Downforce Physics
    pose = getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec);
    lastGroundPose = pose;
    const targetY = pose.avgGroundY + currentCarSpec.rideHeightOffset;

    const vStep = stepVerticalGroundPhysics({
      wasGrounded,
      currentY: carState.y,
      currentVy: carState.vy,
      prevGroundY,
      targetGroundY: targetY,
      prevSlopeRate: carState.slopeRate,
      kmh,
      isOffroad,
      dt: stepDt,
    });

    carState.isGrounded = vStep.isGrounded;
    carState.y = vStep.y;
    carState.vy = vStep.vy;
    carState.slopeRate = vStep.slopeRate;

    if (vStep.isGrounded) {
      if (vStep.justLanded && airTime > 0.22) {
        carState.suspensionVel = -Math.min(2.4, airTime * 3.0);
        cameraTrauma = Math.min(0.38, cameraTrauma + airTime * 0.28);
      }
      // Off-road surface micro-bumps excite the suspension spring-damper
      if (surfaceProfile.surfaceRoughnessAmp > 0 && kmh > 25) {
        const bumpWave =
          Math.sin(carState.x * 0.85 + carState.z * 0.65) *
          surfaceProfile.surfaceRoughnessAmp *
          Math.min(1, kmh / 140);
        carState.suspensionVel += bumpWave * 18 * stepDt;
      }
      airTime = 0;
    } else {
      airTime += stepDt;
    }

    carState.suspensionVel +=
      (-140 * carState.suspensionBounce - 18 * carState.suspensionVel) * stepDt;
    carState.suspensionBounce += carState.suspensionVel * stepDt;

    // 6. Dynamic Longitudinal Squat/Dive & Centripetal Body Roll
    const squat =
      THREE.MathUtils.clamp(hStep.accelLong * 0.0012, -0.032, 0.028) + (isBoosting ? 0.015 : 0);
    const bodyRoll = THREE.MathUtils.clamp(-hStep.accelLat * 0.0018, -0.095, 0.095);
    carState.pitch = THREE.MathUtils.damp(carState.pitch, pose.terrainPitch + squat, 14, stepDt);
    carState.roll = THREE.MathUtils.damp(carState.roll, pose.terrainRoll + bodyRoll, 14, stepDt);
    carState.wheelSpin += (vLong / currentCarSpec.wheelRadius) * stepDt;
  }

  const isGrounded = carState.isGrounded;
  taillightMat.emissiveIntensity = (throttle < 0 && vLong > 1.5) || handbrake ? 5.2 : 2.4;

  nitroFlameL.visible = isBoosting;
  nitroFlameR.visible = isBoosting;
  if (isBoosting) {
    nitroCharge = Math.max(0, nitroCharge - dt * 0.26);
    const flicker = 0.85 + Math.sin(elapsed * 45) * 0.25;
    nitroFlameL.scale.set(1, 1, flicker);
    nitroFlameR.scale.set(1, 1, flicker);

    const exX = carState.x - newFwdX * 2.4;
    const exY = carState.y + 0.05;
    const exZ = carState.z - newFwdZ * 2.4;
    spawnParticle(
      exX - newRightX * 0.34,
      exY,
      exZ - newRightZ * 0.34,
      -newFwdX * 16 + (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 1.2,
      -newFwdZ * 16 + (Math.random() - 0.5) * 2,
      0.24,
      -0.3,
      0.14,
      Math.random() > 0.4 ? 0x00f0ff : 0xff8a2b,
    );
    spawnParticle(
      exX + newRightX * 0.34,
      exY,
      exZ + newRightZ * 0.34,
      -newFwdX * 16 + (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 1.2,
      -newFwdZ * 16 + (Math.random() - 0.5) * 2,
      0.24,
      -0.3,
      0.14,
      Math.random() > 0.4 ? 0xff2a85 : 0x00f0ff,
    );
  } else {
    nitroCharge = Math.min(1, nitroCharge + dt * 0.15);
  }

  chassisBody.position.set(carState.x, carState.y, carState.z);
  chassisBody.velocity.set(carState.vx, carState.vy, carState.vz);
  syncCarAndWheelTransforms(pose, currentSteerAngle, isGrounded);

  // Surface-Specific Tire Plumes (Asphalt Drift Smoke, Oasis Water Spray, Canyon Dirt, Alpine Snow)
  if (isGrounded && (isDrifting || (surfaceProfile.surface !== 'asphalt' && kmh > 38))) {
    const rearWheels = [
      { x: pose.rlX, y: pose.hRL + 0.18, z: pose.rlZ },
      { x: pose.rrX, y: pose.hRR + 0.18, z: pose.rrZ },
    ];
    const plumeColor =
      surfaceProfile.surface === 'water'
        ? 0x7dd3fc
        : surfaceProfile.surface === 'dirt'
          ? currentSceneryId === 'cyber'
            ? 0xa855f7
            : 0xd97736
          : surfaceProfile.surface === 'snow'
            ? 0xf1f5f9
            : 0xe2e8f0;
    const upKick = surfaceProfile.surface === 'water' ? 2.4 : 0.9;
    for (let i = 0; i < 2; i++) {
      const rw = rearWheels[i];
      spawnParticle(
        rw.x + (Math.random() - 0.5) * 0.25,
        rw.y,
        rw.z + (Math.random() - 0.5) * 0.25,
        (Math.random() - 0.5) * 2.4,
        upKick + Math.random() * 1.2,
        (Math.random() - 0.5) * 2.4,
        0.28,
        1.65,
        0.52,
        plumeColor,
      );
    }
    if (isDrifting) {
      skidTimer += dt;
      if (skidTimer >= 0.045) {
        skidTimer = 0;
        const vYaw = carState.yaw + carState.visualYawOffset;
        spawnSkid(pose.rlX, pose.hRL, pose.rlZ, vYaw);
        spawnSkid(pose.rrX, pose.hRR, pose.rrZ, vYaw);
      }
    }
  }

  // Update Particle & Skid Buffers (only upload GPU buffers when dirty)
  let anyParticleUpdated = false;
  for (let i = 0; i < MAX_PARTICLES; i++) {
    const p = particles[i];
    if (!p.active) continue;
    anyParticleUpdated = true;
    p.age += dt;
    if (p.age >= p.maxAge) {
      p.active = false;
      _dummy.position.set(0, -500, 0);
      _dummy.scale.setScalar(0.001);
      _dummy.updateMatrix();
      particleInst.setMatrixAt(i, _dummy.matrix);
      continue;
    }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;
    const lifeLeft = 1 - p.age / p.maxAge;
    const s = Math.max(0.02, (p.scale + p.growRate * p.age) * lifeLeft);
    _dummy.position.set(p.x, p.y, p.z);
    _dummy.scale.setScalar(s);
    _dummy.updateMatrix();
    particleInst.setMatrixAt(i, _dummy.matrix);
  }
  if (anyParticleUpdated) {
    particleInst.instanceMatrix.needsUpdate = true;
  }
  if (particleColorsDirty && particleInst.instanceColor) {
    particleInst.instanceColor.needsUpdate = true;
    particleColorsDirty = false;
  }

  let anySkidUpdated = false;
  for (let i = 0; i < MAX_SKIDS; i++) {
    const s = skids[i];
    if (!s.active) continue;
    anySkidUpdated = true;
    s.age += dt;
    if (s.age >= s.maxAge) {
      s.active = false;
      _dummy.position.set(0, -600, 0);
      _dummy.scale.setScalar(0.001);
      _dummy.updateMatrix();
      skidInst.setMatrixAt(i, _dummy.matrix);
      continue;
    }
    const fade = Math.max(0.01, 1 - s.age / s.maxAge);
    _dummy.position.set(s.x, s.y, s.z);
    _dummy.rotation.set(0, s.yaw, 0);
    _dummy.scale.set(1, 1, fade);
    _dummy.updateMatrix();
    skidInst.setMatrixAt(i, _dummy.matrix);
  }
  if (anySkidUpdated) {
    skidInst.instanceMatrix.needsUpdate = true;
  }

  // Checkpoint Gates & Horizon Skill Chain
  let gateJustCleared = false;
  for (let i = 0; i < gates.length; i++) {
    const g = gates[i];
    if (g.cooldown > 0) g.cooldown = Math.max(0, g.cooldown - dt);
    const dSq = (carState.x - g.x) ** 2 + (carState.z - g.z) ** 2;
    if (dSq < 15.5 ** 2 && g.cooldown === 0) {
      g.cooldown = 8;
      if (!g.cleared) {
        g.cleared = true;
        gatesClearedCount++;
      }
      gateJustCleared = true;
      soundEngine.playChime(740, 0.18);
    }
  }

  let addPts = 0;
  let addMult = 0;
  let skillHit = false;
  if (gateJustCleared) {
    addPts += 500;
    addMult += 0.5;
    skillHit = true;
    skillState.lastSkillLabel = 'HORIZON CHECKPOINT +500';
  }
  if (wreckageJustTriggered) {
    addPts += 250;
    addMult += 0.25;
    skillHit = true;
    skillState.lastSkillLabel = 'WRECKAGE +250';
  }
  if (landscapingJustTriggered) {
    addPts += 300;
    addMult += 0.3;
    skillHit = true;
    skillState.lastSkillLabel = 'LANDSCAPING +300';
  }
  if (tradingPaintJustTriggered) {
    addPts += 350;
    addMult += 0.35;
    skillHit = true;
    skillState.lastSkillLabel = 'TRADING PAINT +350';
  }
  if (nearMissJustTriggered) {
    addPts += 400;
    addMult += 0.4;
    skillHit = true;
    skillState.lastSkillLabel = 'DAREDEVIL NEAR MISS +400';
  }
  if (!isGrounded && airTime > 0.22) {
    addPts += 420 * currentCarSpec.airScoreMultiplier * dt;
    addMult += 0.28 * currentCarSpec.airScoreMultiplier * dt;
    skillHit = true;
    skillState.lastSkillLabel = airTime > 0.8 ? 'ULTIMATE AIR · KANGAROO' : 'KANGAROO AIR';
  }
  if (isDrifting && kmh > 32) {
    addPts += 340 * currentCarSpec.driftScoreMultiplier * dt;
    addMult += 0.24 * currentCarSpec.driftScoreMultiplier * dt;
    skillHit = true;
    skillState.lastSkillLabel = kmh > 120 ? 'ULTIMATE E-BRAKE DRIFT' : 'E-BRAKE DRIFT';
  }
  if (kmh > 125 || (isBoosting && kmh > 65)) {
    addPts += 240 * dt;
    addMult += 0.18 * dt;
    skillHit = true;
    skillState.lastSkillLabel = isBoosting ? 'ULTIMATE BURNOUT · NITRO' : 'SPEED DEMON';
  }

  if (skillHit) {
    skillState.activeScore += addPts;
    skillState.multiplier = Math.min(10, skillState.multiplier + addMult);
    skillState.cleanTimer = 3;
  } else if (skillState.activeScore > 0) {
    skillState.cleanTimer = Math.max(0, skillState.cleanTimer - dt);
    if (skillState.cleanTimer === 0) {
      const banked = Math.round(skillState.activeScore * skillState.multiplier);
      skillState.bankedScore += banked;
      skillState.bestChainScore = Math.max(skillState.bestChainScore, banked);
      skillState.activeScore = 0;
      skillState.multiplier = 1;
      skillState.lastSkillLabel = `BANKED +${banked.toLocaleString()} XP`;
      bankedFlashTimer = 1.25;
      soundEngine.playChime(880, 0.24);
    }
  }
  if (bankedFlashTimer > 0) bankedFlashTimer = Math.max(0, bankedFlashTimer - dt);

  // Camera Follow & FOV (Car-Anchored Relative Damping — Zero 200+ km/h Rubber-Banding)
  const speedRatio = Math.min(1, kmh / 245);
  const targetFov = 60 + speedRatio * 20 + (isBoosting ? 4 : 0);
  const nextFov = THREE.MathUtils.damp(camera.fov, targetFov, 7, dt);
  if (Math.abs(nextFov - camera.fov) > 0.02) {
    camera.fov = nextFov;
    camera.updateProjectionMatrix();
  }

  cameraTrauma = Math.max(0, cameraTrauma - dt * 2.8);
  const shakeOffset = cameraTrauma * cameraTrauma * Math.sin(elapsed * 42) * 0.14;

  if (cameraMode === 'chase') {
    const camOut = stepChaseCameraRelative(chaseCamState, {
      carX: _carPos.x,
      carY: _carPos.y,
      carZ: _carPos.z,
      carFwdX: newFwdX,
      carFwdZ: newFwdZ,
      visualYawOffset: carState.visualYawOffset,
      kmh,
      throttle,
      isBoosting,
      isTitan: currentCarId === 'titan',
      dt,
      getGroundHeight: getTerrainY,
    });
    chaseCamState = camOut.state;
    camera.position.set(camOut.camX, camOut.camY, camOut.camZ);
    camera.lookAt(camOut.lookAtX, camOut.lookAtY + shakeOffset, camOut.lookAtZ);
    const targetRoll = -currentSteerAngle * Math.min(1, kmh / 95) * 0.08;
    cameraRoll = THREE.MathUtils.damp(cameraRoll, targetRoll, 8, dt);
    camera.rotation.z += cameraRoll;
  } else {
    _fwdVec.set(0, 0, -1).applyQuaternion(_carQuat);
    _upVec.set(0, 1, 0).applyQuaternion(_carQuat);
    const hoodY = currentCarId === 'titan' ? 1.18 : 0.84;
    camera.position.set(
      _carPos.x + _fwdVec.x * 0.45 + _upVec.x * hoodY,
      _carPos.y + _fwdVec.y * 0.45 + _upVec.y * hoodY + shakeOffset * 0.35,
      _carPos.z + _fwdVec.z * 0.45 + _upVec.z * hoodY,
    );
    camera.lookAt(_carPos.x + _fwdVec.x * 24, _carPos.y + _fwdVec.y * 24 + 0.4, _carPos.z + _fwdVec.z * 24);
  }

  if (!isSoftwareGl) {
    // Snap shadow target to 0.5m grid increments and update world matrix to eliminate high-speed shadow shimmer
    const snapX = Math.round(_carPos.x * 2) * 0.5;
    const snapY = Math.round(_carPos.y * 2) * 0.5;
    const snapZ = Math.round(_carPos.z * 2) * 0.5;
    dirLight.target.position.set(snapX, snapY, snapZ);
    dirLight.target.updateMatrixWorld();
    dirLight.position.set(snapX, snapY, snapZ).addScaledVector(sunDir, 160);
  }

  // Digital Telemetry & HUD Update
  const gearThresholds = [0, 42, 82, 128, 178, 232, 310];
  let gear = '1';
  let gearIdx = 0;
  if (vLong < -1.8 && throttle < 0) {
    gear = 'R';
  } else {
    for (let i = 0; i < 6; i++) {
      if (kmh >= gearThresholds[i]) {
        gearIdx = i;
        gear = String(i + 1);
      }
    }
  }
  const lowKmh = gearThresholds[gearIdx] ?? 0;
  const highKmh = gearThresholds[gearIdx + 1] ?? 310;
  const gearProg = Math.max(0, Math.min(1, (kmh - lowKmh) / Math.max(1, highKmh - lowKmh)));
  const rpmNorm = Math.min(0.98, Math.max(0.14, 0.22 + gearProg * 0.68 + (throttle > 0 ? 0.08 : 0)));

  const rpmScaleStr = `scaleX(${rpmNorm.toFixed(3)})`;
  if (rpmScaleStr !== lastRpmScaleStr) {
    lastRpmScaleStr = rpmScaleStr;
    hudRpmBarEl.style.transform = rpmScaleStr;
    if (hudRpmArcEl) {
      // 240° sweep length of r=82 circle is 343.48px
      hudRpmArcEl.style.strokeDashoffset = (343.48 * (1 - rpmNorm)).toFixed(1);
    }
  }

  const nitroScaleStr = `scaleX(${nitroCharge.toFixed(3)})`;
  if (nitroScaleStr !== lastNitroScaleStr) {
    lastNitroScaleStr = nitroScaleStr;
    hudNitroBarEl.style.transform = nitroScaleStr;
    if (hudNitroArcEl) {
      // 95° sweep length of r=93 circle is 154.2px
      hudNitroArcEl.style.strokeDashoffset = (154.2 * (1 - nitroCharge)).toFixed(1);
    }
    if (touchNitroFillEl) {
      touchNitroFillEl.style.transform = nitroScaleStr;
    }
  }

  const skillTimerScaleStr = `scaleX(${(skillState.cleanTimer / 3).toFixed(3)})`;
  if (skillTimerScaleStr !== lastSkillTimerScaleStr) {
    lastSkillTimerScaleStr = skillTimerScaleStr;
    hudSkillTimerEl.style.transform = skillTimerScaleStr;
  }

  soundEngine.update(rpmNorm, throttle, isDrifting, true, kmh, currentCarSpec.engineBaseHz, dt);

  hudUpdateAccumulator += dt;
  if (hudUpdateAccumulator >= 0.065) {
    hudUpdateAccumulator = 0;
    const dispSpeedStr = String(Math.round(speedUnit === 'KMH' ? kmh : kmh * 0.621371));
    if (dispSpeedStr !== lastSpeedStr) {
      lastSpeedStr = dispSpeedStr;
      hudSpeedEl.textContent = dispSpeedStr;
    }
    if (gear !== lastGearStr) {
      lastGearStr = gear;
      hudGearEl.textContent = gear;
    }
    hudGearBoxEl.classList.toggle('redline', rpmNorm > 0.86);
    const rpmTextStr = `${Math.round(1100 + rpmNorm * 7900).toLocaleString()} RPM`;
    if (rpmTextStr !== lastRpmTextStr) {
      lastRpmTextStr = rpmTextStr;
      hudRpmTextEl.textContent = rpmTextStr;
    }

    if (skillState.activeScore > 0) {
      skillChainCardEl.classList.add('active');
      skillChainCardEl.classList.remove('banked-flash');
      hudSkillPointsEl.textContent = Math.round(skillState.activeScore).toLocaleString();
      hudSkillMultEl.textContent = `× ${skillState.multiplier.toFixed(1)}`;
      hudSkillLabelEl.textContent = skillState.lastSkillLabel;
    } else if (bankedFlashTimer > 0) {
      skillChainCardEl.classList.remove('active');
      skillChainCardEl.classList.add('banked-flash');
      hudSkillPointsEl.textContent = 'BANKED';
      hudSkillMultEl.textContent = '✓';
      hudSkillLabelEl.textContent = skillState.lastSkillLabel;
    } else {
      skillChainCardEl.classList.remove('active', 'banked-flash');
    }

    const bankedStr = skillState.bankedScore.toLocaleString();
    if (bankedStr !== lastBankedXpStr) {
      lastBankedXpStr = bankedStr;
      hudBankedXpEl.textContent = bankedStr;
    }
    const gatesStr = `${gatesClearedCount} / ${GATE_COUNT}`;
    if (gatesStr !== lastGatesStr) {
      lastGatesStr = gatesStr;
      hudGatesCountEl.textContent = gatesStr;
    }
    const bestChainStr = skillState.bestChainScore.toLocaleString();
    if (bestChainStr !== lastBestChainStr) {
      lastBestChainStr = bestChainStr;
      hudBestChainEl.textContent = bestChainStr;
    }
    drawRadar(carState.x, carState.z, carState.yaw);
  }

  sunDiscMesh.position.copy(camera.position).addScaledVector(orbDir, 1750);
  starField.position.copy(camera.position);
  if (bloomPass.enabled) composer.render();
  else renderer.render(scene, camera);
  if (isSoftwareGl) afterDraw(tick);
  else requestAnimationFrame(tick);
}

tick();

window.addEventListener('resize', () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.resolution.set(Math.max(1, Math.floor(w * 0.5)), Math.max(1, Math.floor(h * 0.5)));
});
