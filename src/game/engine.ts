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
  computeSafeNitroImpulse,
  computeTerrainHeight,
  getCarSpec,
  getHighwayPointAtAngle,
  getRoadDistanceInfo,
  getScenerySpec,
  ROAD_HALF_WIDTH,
  SCENERY_CATALOG,
  SCENERY_ORDER,
  SceneryId,
  ScenerySpec,
  smoothstep,
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

  update(rpmNorm: number, throttle: number, isDrifting: boolean, isPlaying: boolean, kmh = 0, baseHz = 54) {
    if (!this.ctx || !this.engineGain || !this.engineOsc1 || !this.engineOsc2) return;
    if (this.muted || !isPlaying) {
      this.engineGain.gain.setTargetAtTime(0.0001, this.ctx.currentTime, 0.08);
      if (this.tireGain) this.tireGain.gain.setTargetAtTime(0.0001, this.ctx.currentTime, 0.08);
      return;
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    const freq = baseHz + rpmNorm * 165;
    this.engineOsc1.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.04);
    this.engineOsc2.frequency.setTargetAtTime(freq * 1.5, this.ctx.currentTime, 0.04);
    if (this.engineFilter) {
      this.engineFilter.frequency.setTargetAtTime(440 + rpmNorm * 980, this.ctx.currentTime, 0.05);
    }
    const gainVal = 0.038 + Math.abs(throttle) * 0.045 + (isDrifting ? 0.02 : 0);
    this.engineGain.gain.setTargetAtTime(gainVal, this.ctx.currentTime, 0.06);
    if (this.tireGain) {
      const screech = isDrifting && kmh > 14;
      const tireVal = screech ? Math.min(0.055, 0.02 + (kmh / 200) * 0.035) : 0.0001;
      this.tireGain.gain.setTargetAtTime(tireVal, this.ctx.currentTime, screech ? 0.04 : 0.1);
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
  new THREE.Vector2(window.innerWidth, window.innerHeight),
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
  _dummy.position.set(tx, ty - 0.15, tz);
  _dummy.rotation.set(0, rng() * Math.PI * 2, 0);
  _dummy.scale.setScalar(scale);
  _dummy.updateMatrix();
  treeTrunkInst.setMatrixAt(placedTrees, _dummy.matrix);
  treeCone1.setMatrixAt(placedTrees, _dummy.matrix);
  treeCone2.setMatrixAt(placedTrees, _dummy.matrix);
  treeCone3.setMatrixAt(placedTrees, _dummy.matrix);
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
  _dummy.scale.set(
    (0.8 + rng() * 2.2) * (spire > 1 ? 1.8 : 1),
    (0.6 + rng() * 1.6) * spire,
    (0.8 + rng() * 2.2) * (spire > 1 ? 1.8 : 1),
  );
  _dummy.updateMatrix();
  rockInst.setMatrixAt(placedRocks, _dummy.matrix);
  placedRocks++;
}
rockInst.instanceMatrix.needsUpdate = true;
rockInst.computeBoundingSphere();
scene.add(rockInst);

// Highway Bollards & Festival Flags
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
  _dummy.position.set(pt.x - pt.nx * 15.95, ly, pt.z - pt.nz * 15.95);
  _dummy.updateMatrix();
  flagInst.setMatrixAt(i * 2, _dummy.matrix);
  _instColor.setHex(flagPalette[i % flagPalette.length]);
  flagInst.setColorAt(i * 2, _instColor);

  const rx = pt.x + pt.nx * offset;
  const rz = pt.z + pt.nz * offset;
  const ry = getTerrainY(rx, rz);
  _dummy.position.set(rx, ry, rz);
  _dummy.updateMatrix();
  bollardInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  _dummy.position.set(pt.x + pt.nx * 15.95, ry, pt.z + pt.nz * 15.95);
  _dummy.updateMatrix();
  flagInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  _instColor.setHex(flagPalette[(i + 1) % flagPalette.length]);
  flagInst.setColorAt(i * 2 + 1, _instColor);
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

  const rx = pt.x + pt.nx * 15;
  const rz = pt.z + pt.nz * 15;
  _dummy.position.set(rx, y, rz);
  _dummy.updateMatrix();
  gatePillarInst.setMatrixAt(i * 2 + 1, _dummy.matrix);
  gateSleeveInst.setMatrixAt(i * 2 + 1, _dummy.matrix);

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
  laneOffset: number;
  nearMissCooldown: number;
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
  rivals.push({
    mesh: rGroup,
    theta: ((i + 0.5) / 6) * Math.PI * 2,
    angularSpeed: (i % 2 === 0 ? 1 : -1) * (0.048 + (i % 3) * 0.008),
    laneOffset: i % 2 === 0 ? 5.6 : -5.6,
    nearMissCooldown: 0,
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
  visualYawOffset: 0,
  pitch: 0,
  roll: 0,
  wheelSpin: 0,
  suspensionBounce: 0,
  suspensionVel: 0,
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
  const p = particles[particleCursor];
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
const hudRpmTextEl = document.getElementById('hud-rpm-text')!;
const hudNitroBarEl = document.getElementById('hud-nitro-bar')!;
const hudUnitToggleEl = document.getElementById('hud-unit-toggle')!;
const hudBankedXpEl = document.getElementById('hud-banked-xp')!;
const hudGatesCountEl = document.getElementById('hud-gates-count')!;
const hudBestChainEl = document.getElementById('hud-best-chain')!;
const hudCarWorldLabelEl = document.getElementById('hud-car-world-label');
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
const touchTimeBtnEl = document.getElementById('touch-time-btn');
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
  if (btnTimeTopEl) btnTimeTopEl.textContent = `${label} [L]`;
  if (dockTimeBtnEl) dockTimeBtnEl.textContent = `Sky: ${mode.charAt(0).toUpperCase() + mode.slice(1)} [L]`;
  if (touchTimeBtnEl) touchTimeBtnEl.textContent = label;
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
    hudCarWorldLabelEl.textContent = `${currentCarSpec.name} · ${currentScenerySpec.name}`;
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

  // Update Main Menu performance telemetry bars
  if (statBarSpeedEl) statBarSpeedEl.style.width = `${currentCarSpec.stats.speed}%`;
  if (statValSpeedEl) statValSpeedEl.textContent = `${currentCarSpec.maxNitroKmh} KMH`;
  if (statBarHandlingEl) statBarHandlingEl.style.width = `${currentCarSpec.stats.handling}%`;
  if (statValHandlingEl) statValHandlingEl.textContent = `${currentCarSpec.stats.handling}`;
  if (statBarDriftEl) statBarDriftEl.style.width = `${currentCarSpec.stats.drift}%`;
  if (statValDriftEl) statValDriftEl.textContent = `${currentCarSpec.stats.drift}`;
  if (statBarOffroadEl) statBarOffroadEl.style.width = `${currentCarSpec.stats.offroad}%`;
  if (statValOffroadEl) statValOffroadEl.textContent = `${currentCarSpec.stats.offroad}`;
  if (menuActiveCarSummaryEl) {
    menuActiveCarSummaryEl.textContent = `${currentCarSpec.name} — ${currentCarSpec.subtitle}`;
  }
  if (dockCarBtnEl) {
    const shortName = carId === 'apex' ? 'Apex V10' : carId === 'kaze' ? 'Kaze RX-79' : 'Titan 4x4';
    dockCarBtnEl.textContent = `Car: ${shortName} [V]`;
  }
  updateHudCarWorldBanner();

  // Snap ride height smoothly to new car spec
  const pose = getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec);
  carState.y = pose.avgGroundY + currentCarSpec.rideHeightOffset;
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
    menuActiveScenerySummaryEl.textContent = `${currentScenerySpec.name} — ${currentScenerySpec.subtitle}`;
  }
  if (dockSceneryBtnEl) {
    const shortWorld = sceneryId === 'canyon' ? 'Canyon' : sceneryId === 'alpine' ? 'Alpine' : 'Cyber Bay';
    dockSceneryBtnEl.textContent = `World: ${shortWorld} [B]`;
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
  if (btnTouchTopEl) {
    btnTouchTopEl.style.borderColor = isTouchEnabled ? 'rgba(0, 240, 255, 0.75)' : '';
    btnTouchTopEl.style.color = isTouchEnabled ? '#00f0ff' : '';
  }
}
if (
  window.matchMedia('(pointer: coarse), (max-width: 900px)').matches ||
  searchParams.get('touch') === '1'
) {
  setTouchMode(true);
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
  resetAnalogSteerPad();
}
window.addEventListener('blur', clearAllHeldInputs);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearAllHeldInputs();
});

function setGameState(nextState: 'title' | 'playing' | 'paused') {
  gameState = nextState;
  document.body.dataset.state = nextState;
  if (nextState === 'title') {
    clearAllHeldInputs();
    titleScreenEl.removeAttribute('hidden');
    statusModalEl.setAttribute('hidden', '');
  } else if (nextState === 'playing') {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
    titleScreenEl.setAttribute('hidden', '');
    statusModalEl.setAttribute('hidden', '');
  } else if (nextState === 'paused') {
    clearAllHeldInputs();
    modalStatXpEl.textContent = skillState.bankedScore.toLocaleString();
    modalStatSpeedEl.textContent = String(Math.round(sessionTopSpeedKmh));
    modalStatGatesEl.textContent = `${gatesClearedCount} / ${GATE_COUNT}`;
    statusModalEl.removeAttribute('hidden');
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
  carState.visualYawOffset = 0;
  carState.pitch = 0;
  carState.roll = 0;
  carState.suspensionBounce = 0;
  carState.suspensionVel = 0;
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

function toggleCameraMode() {
  cameraMode = cameraMode === 'chase' ? 'hood' : 'chase';
  if (dockCamBtnEl) {
    dockCamBtnEl.textContent = `Cam: ${cameraMode === 'chase' ? 'Chase [C]' : 'Hood [C]'}`;
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

// Touch Steer Pad & Pedal Cluster
const touchSteerPadEl = document.getElementById('touch-steer-pad');
const touchSteerThumbEl = document.getElementById('touch-steer-thumb');
const touchLeftEl = document.getElementById('touch-left');
const touchRightEl = document.getElementById('touch-right');
let activeSteerPointerId: number | null = null;

function handleSteerPadMove(clientX: number) {
  if (!touchSteerPadEl || !touchSteerThumbEl) return;
  const rect = touchSteerPadEl.getBoundingClientRect();
  const res = computeAnalogSteerFromPad(clientX, rect.left, rect.width);
  touchInput.steerAnalog = res.steerInput;
  touchInput.left = res.steerInput > 0.15;
  touchInput.right = res.steerInput < -0.15;
  touchSteerThumbEl.style.transform = `translateX(${res.thumbOffsetPx}px)`;
  touchSteerPadEl.setAttribute('aria-valuenow', String(res.ariaValueNow));
  touchLeftEl?.classList.toggle('pressed', touchInput.left);
  touchRightEl?.classList.toggle('pressed', touchInput.right);
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
  if (touchSteerThumbEl) touchSteerThumbEl.style.transform = 'translateX(0px)';
  touchLeftEl?.classList.remove('pressed');
  touchRightEl?.classList.remove('pressed');
}

if (touchSteerPadEl) {
  touchSteerPadEl.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    soundEngine.init();
    activeSteerPointerId = e.pointerId;
    touchSteerPadEl.classList.add('active');
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

function bindTouchPedal(id: string, key: 'gas' | 'brake' | 'drift' | 'nitro') {
  const el = document.getElementById(id);
  if (!el) return;
  const setPressed = (val: boolean) => {
    touchInput[key] = val;
    el.classList.toggle('pressed', val);
  };
  el.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    soundEngine.init();
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      // Ignore
    }
    setPressed(true);
  });
  el.addEventListener('pointerup', (e) => {
    e.preventDefault();
    setPressed(false);
  });
  el.addEventListener('pointercancel', () => setPressed(false));
  el.addEventListener('lostpointercapture', () => setPressed(false));
}

bindTouchPedal('touch-gas', 'gas');
bindTouchPedal('touch-brake', 'brake');
bindTouchPedal('touch-drift', 'drift');
bindTouchPedal('touch-nitro', 'nitro');

if (searchParams.get('autostart') === '1') {
  setGameState('playing');
}

// Highway Radar Minimap
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
      ? 'rgba(0, 240, 255, 0.38)'
      : currentSceneryId === 'alpine'
        ? 'rgba(125, 211, 252, 0.42)'
        : 'rgba(14, 165, 233, 0.35)';
  radarCtx.beginPath();
  radarCtx.arc(lx, lz, OASIS_RADIUS * scale, 0, Math.PI * 2);
  radarCtx.fill();

  radarCtx.lineWidth = 4.5;
  radarCtx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
  radarCtx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const pt = getHighwayPointAtAngle((i / 60) * Math.PI * 2);
    const rx = (pt.x - carX) * scale;
    const rz = (pt.z - carZ) * scale;
    if (i === 0) radarCtx.moveTo(rx, rz);
    else radarCtx.lineTo(rx, rz);
  }
  radarCtx.stroke();

  for (let i = 0; i < gates.length; i++) {
    const g = gates[i];
    const gx = (g.x - carX) * scale;
    const gz = (g.z - carZ) * scale;
    radarCtx.fillStyle = g.cleared ? '#00f0ff' : '#ff2a85';
    radarCtx.beginPath();
    radarCtx.arc(gx, gz, 3.5, 0, Math.PI * 2);
    radarCtx.fill();
  }

  radarCtx.fillStyle = '#facc15';
  for (let i = 0; i < rivals.length; i++) {
    const pos = rivals[i].mesh.position;
    const rx = (pos.x - carX) * scale;
    const rz = (pos.z - carZ) * scale;
    radarCtx.beginPath();
    radarCtx.arc(rx, rz, 2.6, 0, Math.PI * 2);
    radarCtx.fill();
  }
  radarCtx.restore();

  radarCtx.fillStyle = '#ff8a2b';
  radarCtx.beginPath();
  radarCtx.moveTo(84, 76);
  radarCtx.lineTo(89, 90);
  radarCtx.lineTo(84, 86);
  radarCtx.lineTo(79, 90);
  radarCtx.closePath();
  radarCtx.fill();
}

// Reusable Math Scratch Objects (Zero-GC Loop)
const clock = new THREE.Clock();
let hudUpdateAccumulator = 0;
const _carPos = new THREE.Vector3();
const _carQuat = new THREE.Quaternion();
const _fwdVec = new THREE.Vector3();
const _upVec = new THREE.Vector3();
const _bodyEuler = new THREE.Euler();
const _wheelEuler = new THREE.Euler();
const _desiredCamPos = new THREE.Vector3();
const _dampedCamPos = new THREE.Vector3(0, 3, 8);
const _desiredLookAt = new THREE.Vector3();
const _dampedLookAt = new THREE.Vector3(0, 1, -5);

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

const initialPose = getWheelGroundPose(0, 0, 0, currentCarSpec);
carState.y = initialPose.avgGroundY + currentCarSpec.rideHeightOffset;
syncCarAndWheelTransforms(initialPose, 0, true);

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
  let dt = clock.getDelta();
  if (dt > 0.05) dt = 0.05;
  const elapsed = clock.getElapsedTime();

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

  // Step AI Rivals along the highway loop
  let nearMissJustTriggered = false;
  for (let i = 0; i < rivals.length; i++) {
    const r = rivals[i];
    r.theta = (r.theta + r.angularSpeed * dt) % (Math.PI * 2);
    if (r.theta < 0) r.theta += Math.PI * 2;
    if (r.nearMissCooldown > 0) r.nearMissCooldown = Math.max(0, r.nearMissCooldown - dt);
    const pt = getHighwayPointAtAngle(r.theta);
    const rx = pt.x + pt.nx * r.laneOffset;
    const rz = pt.z + pt.nz * r.laneOffset;
    const ry = getTerrainY(rx, rz) + 0.16;
    const dirSign = r.angularSpeed >= 0 ? 1 : -1;
    const ryaw = Math.atan2(-pt.tx * dirSign, -pt.tz * dirSign);
    r.mesh.position.set(rx, ry, rz);
    r.mesh.rotation.set(0, ryaw, 0);
  }

  // Title Screen / Pause Showcase Orbit Camera (Frames Car in Right Viewport Half)
  if (gameState !== 'playing') {
    syncCarAndWheelTransforms(getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec), 0.18, true);
    const orbitAng = elapsed * 0.25 + 0.75;
    const radius = 7.4;
    const camX = _carPos.x + Math.sin(orbitAng) * radius;
    const camY = _carPos.y + 1.85 + Math.sin(orbitAng * 0.5) * 0.25;
    const camZ = _carPos.z + Math.cos(orbitAng) * radius;
    camera.position.set(camX, camY, camZ);
    // Camera forward is (-sin(orbitAng), 0, -cos(orbitAng)), so camera right is (cos(orbitAng), 0, -sin(orbitAng)).
    // Looking slightly to the camera's LEFT (-camRight * 2.15) places the car cleanly in the open RIGHT half of the screen.
    const camRightX = Math.cos(orbitAng);
    const camRightZ = -Math.sin(orbitAng);
    camera.lookAt(_carPos.x - camRightX * 2.15, _carPos.y + 0.68, _carPos.z - camRightZ * 2.15);
    sunDiscMesh.position.copy(camera.position).addScaledVector(orbDir, 1750);
    starField.position.copy(camera.position);
    soundEngine.update(0.14, 0, false, false, 0, currentCarSpec.engineBaseHz);
    if (bloomPass.enabled) composer.render();
    else renderer.render(scene, camera);
    if (isSoftwareGl) afterDraw(tick);
    else requestAnimationFrame(tick);
    return;
  }

  // Active Gameplay Vehicle Physics & Controls
  const fwdX = -Math.sin(carState.yaw);
  const fwdZ = -Math.cos(carState.yaw);
  const speedMs = Math.hypot(carState.vx, carState.vz);
  const forwardVel = carState.vx * fwdX + carState.vz * fwdZ;
  const kmh = speedMs * 3.6;
  if (kmh > sessionTopSpeedKmh) sessionTopSpeedKmh = kmh;

  const throttle = (keys.forward || touchInput.gas ? 1 : 0) - (keys.backward || touchInput.brake ? 1 : 0);
  const digitalSteer = (keys.left || touchInput.left ? 1 : 0) - (keys.right || touchInput.right ? 1 : 0);
  const steerInput = Math.abs(touchInput.steerAnalog) > 0.01 ? touchInput.steerAnalog : digitalSteer;
  const handbrake = keys.handbrake || touchInput.drift;
  const isBoosting = (keys.nitro || touchInput.nitro) && nitroCharge > 0.05;

  const targetSteer = steerInput * THREE.MathUtils.lerp(0.46, 0.22, Math.min(1, kmh / 220));
  currentSteerAngle = THREE.MathUtils.damp(currentSteerAngle, targetSteer, 13, dt);

  const prevGroundY =
    getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec).avgGroundY +
    currentCarSpec.rideHeightOffset;
  const isGrounded = carState.y <= prevGroundY + 0.16 && carState.vy <= 3.5;

  if (isGrounded) {
    if (airTime > 0.18) {
      carState.suspensionVel = -Math.min(2.8, airTime * 3.5);
      cameraTrauma = Math.min(0.45, cameraTrauma + airTime * 0.35);
    }
    airTime = 0;
  } else {
    airTime += dt;
  }

  const isDrifting = isGrounded && speedMs > 7.5 && (handbrake || Math.abs(carState.visualYawOffset) > 0.16);
  if (isGrounded && speedMs > 0.4) {
    const revSign = forwardVel < -0.5 ? -1 : 1;
    const lowSpeedFactor = Math.min(1, speedMs / 9);
    const turnRate = currentCarSpec.steerRate * (handbrake ? currentCarSpec.driftSteerMultiplier : 1);
    carState.yaw += currentSteerAngle * turnRate * lowSpeedFactor * revSign * dt;
    const slipTarget =
      handbrake && speedMs > 7.5
        ? currentSteerAngle * 0.68 * (currentCarId === 'kaze' ? 1.15 : 1)
        : currentSteerAngle * Math.min(1, kmh / 180) * 0.12;
    carState.visualYawOffset = THREE.MathUtils.damp(
      carState.visualYawOffset,
      slipTarget,
      handbrake ? 7.5 : 9.5,
      dt,
    );
  } else {
    carState.visualYawOffset = THREE.MathUtils.damp(carState.visualYawOffset, 0, 6, dt);
  }

  const newFwdX = -Math.sin(carState.yaw);
  const newFwdZ = -Math.cos(carState.yaw);
  const newRightX = Math.cos(carState.yaw);
  const newRightZ = -Math.sin(carState.yaw);
  let vLong = carState.vx * newFwdX + carState.vz * newFwdZ;
  let vLat = carState.vx * newRightX + carState.vz * newRightZ;

  taillightMat.emissiveIntensity = (throttle < 0 && vLong > 1.5) || handbrake ? 5.2 : 2.4;

  // Off-road surface drag (Titan 4x4 ignores off-road penalty!)
  const { distance: roadDist } = getRoadDistanceInfo(carState.x, carState.z);
  const isOffroad = roadDist > ROAD_HALF_WIDTH + 3;
  const offroadDragFactor = isOffroad && currentCarId !== 'titan' ? 0.88 : 1.0;

  if (isGrounded) {
    const maxCarMs = (currentCarSpec.maxSpeedKmh * offroadDragFactor) / 3.6;
    if (throttle > 0) {
      const headroom = Math.max(0, 1 - Math.max(0, vLong) / maxCarMs);
      const accel = vLong < 0 ? 34 : currentCarSpec.accelRate * Math.pow(headroom, 0.7);
      vLong += accel * dt;
    } else if (throttle < 0) {
      if (vLong > 1.2) vLong = Math.max(0, vLong - 38 * dt);
      else if (vLong > -68 / 3.6) vLong -= 16.5 * dt;
    } else {
      vLong *= Math.exp(-0.55 * dt);
      if (Math.abs(vLong) < 0.15) vLong = 0;
    }
    if (handbrake) vLong *= Math.exp(-0.65 * dt);
    vLat *= Math.exp(-(handbrake ? currentCarSpec.driftLateralDamping : 11.5) * dt);
  }

  carState.vx = newFwdX * vLong + newRightX * vLat;
  carState.vz = newFwdZ * vLong + newRightZ * vLat;

  nitroFlameL.visible = isBoosting;
  nitroFlameR.visible = isBoosting;
  if (isBoosting) {
    nitroCharge = Math.max(0, nitroCharge - dt * 0.26);
    const flicker = 0.85 + Math.sin(elapsed * 45) * 0.25;
    nitroFlameL.scale.set(1, 1, flicker);
    nitroFlameR.scale.set(1, 1, flicker);

    const impulse = computeSafeNitroImpulse({
      forwardX: newFwdX,
      forwardZ: newFwdZ,
      kmh,
      dt,
      isGrounded,
      maxNitroKmh: currentCarSpec.maxNitroKmh,
    });
    carState.vx += impulse.dvx;
    carState.vz += impulse.dvz;
    cameraTrauma = Math.min(0.55, cameraTrauma + dt * 1.2);

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

  carState.x += carState.vx * dt;
  carState.z += carState.vz * dt;
  const bound = WORLD_SIZE * 0.46;
  carState.x = THREE.MathUtils.clamp(carState.x, -1380, bound);
  carState.z = THREE.MathUtils.clamp(carState.z, -1380, bound);

  const pose = getWheelGroundPose(carState.x, carState.z, carState.yaw, currentCarSpec);
  const targetY = pose.avgGroundY + currentCarSpec.rideHeightOffset;
  const slopeRate = (targetY - prevGroundY) / Math.max(dt, 0.001);

  if (isGrounded) {
    if (slopeRate < -6.5 && kmh > 135 && carState.vy > 1.5) {
      carState.y += carState.vy * dt;
    } else {
      carState.y = targetY;
      carState.vy = THREE.MathUtils.clamp(slopeRate, -12, 9);
    }
  } else {
    carState.vy -= 22 * dt;
    carState.y += carState.vy * dt;
    if (carState.y <= targetY) {
      carState.y = targetY;
      carState.vy = 0;
    }
  }

  carState.suspensionVel += (-140 * carState.suspensionBounce - 16 * carState.suspensionVel) * dt;
  carState.suspensionBounce += carState.suspensionVel * dt;

  const squat = (throttle > 0 ? 0.018 : throttle < 0 ? -0.024 : 0) + (isBoosting ? 0.018 : 0);
  const bodyRoll = -currentSteerAngle * Math.min(1, kmh / 110) * (isDrifting ? 0.085 : 0.048);
  carState.pitch = THREE.MathUtils.damp(carState.pitch, pose.terrainPitch + squat, 14, dt);
  carState.roll = THREE.MathUtils.damp(carState.roll, pose.terrainRoll + bodyRoll, 14, dt);
  carState.wheelSpin += (vLong / currentCarSpec.wheelRadius) * dt;

  chassisBody.position.set(carState.x, carState.y, carState.z);
  chassisBody.velocity.set(carState.vx, carState.vy, carState.vz);
  syncCarAndWheelTransforms(pose, currentSteerAngle, isGrounded);

  // Rival Collisions & Near-Misses
  for (let i = 0; i < rivals.length; i++) {
    const r = rivals[i];
    const rPos = r.mesh.position;
    const dx = carState.x - rPos.x;
    const dz = carState.z - rPos.z;
    const dist = Math.hypot(dx, dz);
    if (dist < 3.6 && dist > 0.01) {
      const push = (3.6 - dist) * 0.8;
      carState.x += (dx / dist) * push;
      carState.z += (dz / dist) * push;
      cameraTrauma = Math.min(0.4, cameraTrauma + 0.2);
    } else if (dist < 7.8 && kmh > 95 && r.nearMissCooldown === 0) {
      r.nearMissCooldown = 4.5;
      nearMissJustTriggered = true;
      soundEngine.playChime(920, 0.15);
    }
  }

  // Drift Smoke & Tire Skidmarks
  if (isDrifting && isGrounded) {
    const rearWheels = [
      { x: pose.rlX, y: pose.hRL + 0.18, z: pose.rlZ },
      { x: pose.rrX, y: pose.hRR + 0.18, z: pose.rrZ },
    ];
    for (let i = 0; i < 2; i++) {
      const rw = rearWheels[i];
      spawnParticle(
        rw.x + (Math.random() - 0.5) * 0.25,
        rw.y,
        rw.z + (Math.random() - 0.5) * 0.25,
        (Math.random() - 0.5) * 2.2,
        0.7 + Math.random() * 0.9,
        (Math.random() - 0.5) * 2.2,
        0.28,
        1.65,
        0.55,
        0xe2e8f0,
      );
    }
    skidTimer += dt;
    if (skidTimer >= 0.045) {
      skidTimer = 0;
      const vYaw = carState.yaw + carState.visualYawOffset;
      spawnSkid(pose.rlX, pose.hRL, pose.rlZ, vYaw);
      spawnSkid(pose.rrX, pose.hRR, pose.rrZ, vYaw);
    }
  }

  // Update Particle & Skid Buffers
  for (let i = 0; i < MAX_PARTICLES; i++) {
    const p = particles[i];
    if (!p.active) continue;
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
    _instColor.setHex(p.colorHex);
    particleInst.setColorAt(i, _instColor);
  }
  particleInst.instanceMatrix.needsUpdate = true;
  if (particleInst.instanceColor) particleInst.instanceColor.needsUpdate = true;

  for (let i = 0; i < MAX_SKIDS; i++) {
    const s = skids[i];
    if (!s.active) continue;
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
  skidInst.instanceMatrix.needsUpdate = true;

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

  // Camera Follow & FOV
  const speedRatio = Math.min(1, kmh / 245);
  const targetFov = 60 + speedRatio * 20 + (isBoosting ? 4 : 0);
  camera.fov = THREE.MathUtils.damp(camera.fov, targetFov, 7, dt);
  camera.updateProjectionMatrix();

  cameraTrauma = Math.max(0, cameraTrauma - dt * 2.4);
  const shakeOffset = cameraTrauma * cameraTrauma * (Math.random() - 0.5) * 0.22;

  if (cameraMode === 'chase') {
    const camDist = 6.4 + (throttle > 0 ? 0.5 : 0) + (isBoosting ? 0.65 : 0) + (currentCarId === 'titan' ? 0.6 : 0);
    const camHeight = 2.2 + speedRatio * 0.35 + (currentCarId === 'titan' ? 0.45 : 0);
    const driftSwing = carState.visualYawOffset * 1.45;
    _desiredCamPos.set(
      _carPos.x - newFwdX * camDist + newRightX * driftSwing,
      _carPos.y + camHeight,
      _carPos.z - newFwdZ * camDist + newRightZ * driftSwing,
    );
    const groundUnderCam = getTerrainY(_desiredCamPos.x, _desiredCamPos.z);
    if (_desiredCamPos.y < groundUnderCam + 1.45) {
      _desiredCamPos.y = groundUnderCam + 1.45;
    }
    _dampedCamPos.x = THREE.MathUtils.damp(_dampedCamPos.x, _desiredCamPos.x, 13, dt);
    _dampedCamPos.y = THREE.MathUtils.damp(_dampedCamPos.y, _desiredCamPos.y, 11, dt);
    _dampedCamPos.z = THREE.MathUtils.damp(_dampedCamPos.z, _desiredCamPos.z, 13, dt);

    _desiredLookAt.set(_carPos.x + newFwdX * 8.5, _carPos.y + 0.82 + shakeOffset, _carPos.z + newFwdZ * 8.5);
    _dampedLookAt.lerp(_desiredLookAt, 1 - Math.exp(-18 * dt));
    camera.position.copy(_dampedCamPos);
    camera.lookAt(_dampedLookAt);
    const targetRoll = -currentSteerAngle * Math.min(1, kmh / 95) * 0.09;
    cameraRoll = THREE.MathUtils.damp(cameraRoll, targetRoll, 8, dt);
    camera.rotation.z += cameraRoll;
  } else {
    _fwdVec.set(0, 0, -1).applyQuaternion(_carQuat);
    _upVec.set(0, 1, 0).applyQuaternion(_carQuat);
    const hoodY = currentCarId === 'titan' ? 1.18 : 0.84;
    camera.position.set(
      _carPos.x + _fwdVec.x * 0.45 + _upVec.x * hoodY,
      _carPos.y + _fwdVec.y * 0.45 + _upVec.y * hoodY + shakeOffset * 0.4,
      _carPos.z + _fwdVec.z * 0.45 + _upVec.z * hoodY,
    );
    camera.lookAt(_carPos.x + _fwdVec.x * 24, _carPos.y + _fwdVec.y * 24 + 0.4, _carPos.z + _fwdVec.z * 24);
  }

  if (!isSoftwareGl) {
    dirLight.target.position.copy(_carPos);
    dirLight.position.copy(_carPos).addScaledVector(sunDir, 160);
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

  hudRpmBarEl.style.transform = `scaleX(${rpmNorm.toFixed(3)})`;
  hudNitroBarEl.style.transform = `scaleX(${nitroCharge.toFixed(3)})`;
  hudSkillTimerEl.style.transform = `scaleX(${(skillState.cleanTimer / 3).toFixed(3)})`;
  soundEngine.update(rpmNorm, throttle, isDrifting, true, kmh, currentCarSpec.engineBaseHz);

  hudUpdateAccumulator += dt;
  if (hudUpdateAccumulator >= 0.065) {
    hudUpdateAccumulator = 0;
    const dispSpeed = Math.round(speedUnit === 'KMH' ? kmh : kmh * 0.621371);
    hudSpeedEl.textContent = String(dispSpeed);
    hudGearEl.textContent = gear;
    hudGearBoxEl.classList.toggle('redline', rpmNorm > 0.86);
    hudRpmTextEl.textContent = `${Math.round(1100 + rpmNorm * 7900).toLocaleString()} RPM`;

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

    hudBankedXpEl.textContent = skillState.bankedScore.toLocaleString();
    hudGatesCountEl.textContent = `${gatesClearedCount} / ${GATE_COUNT}`;
    hudBestChainEl.textContent = skillState.bestChainScore.toLocaleString();
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
  bloomPass.resolution.set(w, h);
});
