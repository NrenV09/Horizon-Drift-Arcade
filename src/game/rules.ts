export interface SkillChainState {
  activeScore: number;
  multiplier: number;
  bankedScore: number;
  bestChainScore: number;
  cleanTimer: number;
  activeSkills: string[];
  lastSkillLabel: string;
  justBankedAmount: number;
}

export interface TelemetryOutput {
  kmh: number;
  mph: number;
  gear: string;
  rpmNormalized: number;
  rpm: number;
  targetFov: number;
}

export const LOOP_CENTER_X = -400;
export const LOOP_CENTER_Z = 0;
export const ROAD_HALF_WIDTH = 13;
export const ROAD_FLAT_SHOULDER = 15;
export const ROAD_SHOULDER_WIDTH = 44;

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Analytical closed-loop highway centerline radius and derivative as a function of polar angle theta.
 * At theta = 0, R(0) = 400 and R'(0) = 0, placing world origin (0, 0) directly on the straightaway facing -Z.
 */
export function getHighwayRadiusAndSlope(theta: number): { radius: number; dRadius: number } {
  const radius =
    320 +
    60 * Math.cos(2 * theta) +
    35 * Math.cos(3 * theta) -
    15 * Math.cos(4 * theta);
  const dRadius =
    -120 * Math.sin(2 * theta) -
    105 * Math.sin(3 * theta) +
    60 * Math.sin(4 * theta);
  return { radius, dRadius };
}

/**
 * Returns highway point and unit tangent vector for a given angle theta around the loop.
 */
export function getHighwayPointAtAngle(theta: number): {
  x: number;
  z: number;
  tx: number;
  tz: number;
  nx: number;
  nz: number;
} {
  const { radius, dRadius } = getHighwayRadiusAndSlope(theta);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const x = LOOP_CENTER_X + radius * cosT;
  const z = LOOP_CENTER_Z + radius * sinT;

  // Derivative with respect to theta (traveling counter-clockwise so theta=0 goes toward -Z when reversed)
  const dx = dRadius * cosT - radius * sinT;
  const dz = dRadius * sinT + radius * cosT;
  const len = Math.hypot(dx, dz) || 1;
  // Flip tangent so at theta=0 we travel toward -Z (forward in Three.js)
  const tx = -dx / len;
  const tz = -dz / len;
  const nx = -tz;
  const nz = tx;
  return { x, z, tx, tz, nx, nz };
}

export function getHighwayElevationAtAngle(theta: number): number {
  return (
    8.5 * (1 - Math.cos(theta)) +
    4.2 * Math.sin(2 * theta) -
    2.1 * Math.sin(4 * theta)
  );
}

/**
 * Returns perpendicular distance from world (x, z) to the asphalt highway loop centerline,
 * plus the polar angle theta on the loop.
 */
export function getRoadDistanceInfo(x: number, z: number): {
  distance: number;
  signedOffset: number;
  theta: number;
  roadElevation: number;
} {
  const cx = x - LOOP_CENTER_X;
  const cz = z - LOOP_CENTER_Z;
  let theta = Math.atan2(cz, cx);

  // Two fast orthogonal projection steps onto the highway curve
  for (let iter = 0; iter < 2; iter++) {
    const { radius, dRadius } = getHighwayRadiusAndSlope(theta);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const px = radius * cosT;
    const pz = radius * sinT;
    const dx = dRadius * cosT - radius * sinT;
    const dz = dRadius * sinT + radius * cosT;
    const denom = dx * dx + dz * dz;
    if (denom > 1e-4) {
      const step = ((cx - px) * dx + (cz - pz) * dz) / denom;
      theta += Math.max(-0.25, Math.min(0.25, step));
    }
  }

  const { radius, dRadius } = getHighwayRadiusAndSlope(theta);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const px = radius * cosT;
  const pz = radius * sinT;
  const dx = dRadius * cosT - radius * sinT;
  const dz = dRadius * sinT + radius * cosT;
  const len = Math.hypot(dx, dz) || 1;
  // Normal pointing outward from loop
  const nx = dz / len;
  const nz = -dx / len;

  const ex = cx - px;
  const ez = cz - pz;
  const signedOffset = ex * nx + ez * nz;
  const distance = Math.hypot(ex, ez);

  // Smooth rolling elevation along the highway loop (0 at theta=0 so origin sits at y=0)
  const roadElevation = getHighwayElevationAtAngle(theta);

  return { distance, signedOffset, theta, roadElevation };
}

/**
 * Deterministic multi-octave terrain height function.
 * Keeps a wide flat margin around the highway corridor so terrain triangles never poke through asphalt.
 */
export function computeTerrainHeight(
  x: number,
  z: number,
  noise2D?: (nx: number, nz: number) => number,
): number {
  const { distance, roadElevation } = getRoadDistanceInfo(x, z);

  // Wide flat roadbed margin guarantees zero diagonal triangle poke-through on curves
  // and fast-paths height queries on the highway without evaluating 3 octaves of simplex noise.
  const flatMargin = ROAD_HALF_WIDTH + ROAD_FLAT_SHOULDER;
  if (distance <= flatMargin) {
    return roadElevation;
  }
  const roadBlend = smoothstep(flatMargin, flatMargin + ROAD_SHOULDER_WIDTH, distance);

  const sample = noise2D
    ? (nx: number, nz: number) => noise2D(nx, nz)
    : (nx: number, nz: number) =>
        Math.sin(nx * 1.7 + Math.cos(nz * 1.3)) * 0.6 +
        Math.cos(nz * 2.1 - nx * 0.9) * 0.4;

  // Rolling Forza-style canyon hills and dramatic red-rock mesas
  const n1 = sample(x * 0.0022, z * 0.0022) * 30;
  const n2 = sample(x * 0.0065 + 11.3, z * 0.0065 - 7.1) * 10.5;
  const n3 = sample(x * 0.018 - 5.4, z * 0.018 + 3.8) * 2.6;

  // Sculpted mesa plateau effect for high terrain
  const rawNoise = n1 + n2 + n3;
  const mesaBoost = rawNoise > 12 ? Math.min(22, (rawNoise - 12) * 1.35) : 0;

  // Oasis lake basin inside the eastern valley (around x = -210, z = 40)
  const lakeDist = Math.hypot(x - -210, z - 40);
  const lakeBasin = (1 - smoothstep(45, 165, lakeDist)) * -11.5;

  // Natural perimeter canyon/ridge elevation near the edges of the 3000x3000 map
  const distFromWorldCenter = Math.hypot(x, z);
  const perimeterRidge = smoothstep(1050, 1460, distFromWorldCenter) * 88;

  const offroadHeight = roadElevation + rawNoise + mesaBoost + lakeBasin + perimeterRidge;
  return roadElevation * (1 - roadBlend) + offroadHeight * roadBlend;
}

/**
 * Computes 4-wheel ground contact heights, average ride elevation, and terrain pitch/roll angles
 * so the vehicle tracks the road surface smoothly without flat-box stair-stepping or clipping.
 */
export function computeFourWheelSurfacePose(
  carX: number,
  carZ: number,
  carYaw: number,
  wheelbase = 2.7,
  trackWidth = 1.86,
  heightFn: (x: number, z: number) => number = computeTerrainHeight,
): {
  hFL: number;
  hFR: number;
  hRL: number;
  hRR: number;
  avgGroundY: number;
  terrainPitch: number;
  terrainRoll: number;
} {
  const fwdX = -Math.sin(carYaw);
  const fwdZ = -Math.cos(carYaw);
  const rightX = Math.cos(carYaw);
  const rightZ = -Math.sin(carYaw);

  const halfWb = wheelbase * 0.5;
  const halfTw = trackWidth * 0.5;

  const flX = carX + fwdX * halfWb - rightX * halfTw;
  const flZ = carZ + fwdZ * halfWb - rightZ * halfTw;

  const frX = carX + fwdX * halfWb + rightX * halfTw;
  const frZ = carZ + fwdZ * halfWb + rightZ * halfTw;

  const rlX = carX - fwdX * halfWb - rightX * halfTw;
  const rlZ = carZ - fwdZ * halfWb - rightZ * halfTw;

  const rrX = carX - fwdX * halfWb + rightX * halfTw;
  const rrZ = carZ - fwdZ * halfWb + rightZ * halfTw;

  const hFL = heightFn(flX, flZ);
  const hFR = heightFn(frX, frZ);
  const hRL = heightFn(rlX, rlZ);
  const hRR = heightFn(rrX, rrZ);

  const avgGroundY = (hFL + hFR + hRL + hRR) * 0.25;
  const frontAvg = (hFL + hFR) * 0.5;
  const rearAvg = (hRL + hRR) * 0.5;
  const rightAvg = (hFR + hRR) * 0.5;
  const leftAvg = (hFL + hRL) * 0.5;

  const terrainPitch = Math.atan2(frontAvg - rearAvg, wheelbase);
  const terrainRoll = Math.atan2(leftAvg - rightAvg, trackWidth);

  return {
    hFL,
    hFR,
    hRL,
    hRR,
    avgGroundY,
    terrainPitch,
    terrainRoll,
  };
}

/**
 * Computes digital speedometer telemetry (KMH, MPH, automatic 6-speed gear + R, RPM, and dynamic FOV).
 */
export function computeVehicleTelemetry(
  speedMetersPerSec: number,
  forwardVelocity: number,
  throttleInput: number,
  isGrounded: boolean,
): TelemetryOutput {
  const absSpeed = Math.max(0, Math.abs(speedMetersPerSec));
  const kmh = absSpeed * 3.6;
  const mph = absSpeed * 2.23694;

  // Gear thresholds in KMH for 6-speed simcade gearbox
  const gearThresholds = [0, 42, 82, 128, 178, 232, 310];
  let gear = '1';
  let gearIdx = 0;

  if (forwardVelocity < -1.8 && throttleInput < 0) {
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
  const span = Math.max(1, highKmh - lowKmh);
  const gearProgress = Math.max(0, Math.min(1, (kmh - lowKmh) / span));

  const idleRpm = 0.14;
  const throttleBoost = !isGrounded && throttleInput > 0 ? 0.25 : throttleInput * 0.12;
  const rpmNormalized =
    gear === 'R'
      ? Math.min(0.95, idleRpm + (kmh / 55) * 0.78 + throttleBoost)
      : Math.min(0.98, Math.max(idleRpm, 0.24 + gearProgress * 0.68 + throttleBoost));

  const rpm = Math.round(1100 + rpmNormalized * 7900);

  // Dynamic FOV from 60 to 80 as speed increases
  const speedRatio = Math.max(0, Math.min(1, kmh / 260));
  const targetFov = 60 + speedRatio * 20;

  return {
    kmh,
    mph,
    gear,
    rpmNormalized,
    rpm,
    targetFov,
  };
}

/**
 * Creates initial Horizon Skill Chain state.
 */
export function createInitialSkillState(): SkillChainState {
  return {
    activeScore: 0,
    multiplier: 1.0,
    bankedScore: 0,
    bestChainScore: 0,
    cleanTimer: 0,
    activeSkills: [],
    lastSkillLabel: '',
    justBankedAmount: 0,
  };
}

/**
 * Advances the Horizon Skill Chain state machine by dt seconds.
 * Awards skill points for drifting, high speed ("Speed Demon"), air time ("Kangaroo"),
 * checkpoints, and rival near-misses, banking the score after 3 seconds of clean driving.
 */
export function stepSkillChain(
  state: SkillChainState,
  input: {
    dt: number;
    kmh: number;
    isDrifting: boolean;
    isAirborne: boolean;
    airTimeSeconds: number;
    isBoosting: boolean;
    gateJustCleared?: boolean;
    nearMissJustTriggered?: boolean;
  },
): SkillChainState {
  const next: SkillChainState = {
    ...state,
    activeSkills: [...state.activeSkills],
    justBankedAmount: 0,
  };

  let earnedPoints = 0;
  let earnedMultiplier = 0;
  let skillTriggered = false;

  const pushSkillLabel = (label: string) => {
    next.lastSkillLabel = label;
    if (!next.activeSkills.includes(label)) {
      next.activeSkills.push(label);
      if (next.activeSkills.length > 4) {
        next.activeSkills.shift();
      }
    }
  };

  if (input.gateJustCleared) {
    earnedPoints += 500;
    earnedMultiplier += 0.5;
    skillTriggered = true;
    pushSkillLabel('HORIZON CHECKPOINT +500');
  }

  if (input.nearMissJustTriggered) {
    earnedPoints += 400;
    earnedMultiplier += 0.4;
    skillTriggered = true;
    pushSkillLabel('DAREDEVIL NEAR MISS +400');
  }

  if (input.isAirborne && input.airTimeSeconds > 0.22) {
    earnedPoints += 420 * input.dt;
    earnedMultiplier += 0.28 * input.dt;
    skillTriggered = true;
    pushSkillLabel(input.airTimeSeconds > 0.85 ? 'ULTIMATE AIR · KANGAROO' : 'KANGAROO AIR');
  }

  if (input.isDrifting && input.kmh > 32) {
    earnedPoints += 340 * input.dt;
    earnedMultiplier += 0.24 * input.dt;
    skillTriggered = true;
    pushSkillLabel(input.kmh > 120 ? 'ULTIMATE E-BRAKE DRIFT' : 'E-BRAKE DRIFT');
  }

  if (input.kmh > 145) {
    earnedPoints += 220 * input.dt;
    earnedMultiplier += 0.16 * input.dt;
    skillTriggered = true;
    pushSkillLabel(input.isBoosting ? 'ULTIMATE BURNOUT · NITRO' : 'SPEED DEMON');
  }

  if (skillTriggered) {
    next.activeScore += earnedPoints;
    next.multiplier = Math.min(10.0, Number((next.multiplier + earnedMultiplier).toFixed(2)));
    next.cleanTimer = 3.0;
  } else if (next.activeScore > 0) {
    next.cleanTimer = Math.max(0, next.cleanTimer - input.dt);
    if (next.cleanTimer === 0) {
      const totalBank = Math.round(next.activeScore * next.multiplier);
      next.bankedScore += totalBank;
      next.bestChainScore = Math.max(next.bestChainScore, totalBank);
      next.justBankedAmount = totalBank;
      next.activeScore = 0;
      next.multiplier = 1.0;
      next.activeSkills = [];
      next.lastSkillLabel = 'SKILL CHAIN BANKED';
    }
  }

  return next;
}

/**
 * Computes proportional analog steering input from a horizontal touch/pointer coordinate
 * inside the virtual steering pad, including a center deadzone, thumb puck pixel offset,
 * visual wheel rotation angle in degrees, and active directional zone.
 * Returns steerInput in [-1, 1] where +1 is full left turn and -1 is full right turn.
 */
export function computeAnalogSteerFromPad(
  clientX: number,
  rectLeft: number,
  rectWidth: number,
  maxThumbTravelPx = 56,
  deadzone = 0.06,
  sensitivity = 1.0,
): {
  steerInput: number;
  thumbOffsetPx: number;
  ariaValueNow: number;
  steerDegrees: number;
  zone: 'left' | 'center' | 'right';
} {
  const safeWidth = Math.max(1, rectWidth);
  const center = rectLeft + safeWidth * 0.5;
  const halfSpan = Math.max(1, safeWidth * 0.42);
  const rawNormalized = Math.max(-1, Math.min(1, (clientX - center) / halfSpan));

  let shaped = 0;
  const absRaw = Math.abs(rawNormalized);
  if (absRaw > deadzone) {
    const sign = Math.sign(rawNormalized);
    const rescaled = (absRaw - deadzone) / (1 - deadzone);
    // Slight non-linear response curve for fine center control, scaled by sensitivity
    const curvePower = Math.max(0.75, Math.min(1.6, 1.15 / Math.max(0.5, sensitivity)));
    shaped = Math.max(-1, Math.min(1, sign * Math.pow(rescaled, curvePower)));
  }

  // In vehicle coordinates, positive steer angle turns left, negative turns right
  const steerInput = shaped === 0 ? 0 : Number((-shaped).toFixed(4)) || 0;
  const thumbOffsetPx = Math.round(rawNormalized * maxThumbTravelPx) || 0;
  const ariaValueNow = Math.round(rawNormalized * 100) || 0;
  // Visual steering wheel rotation: negative (counter-clockwise) for left, positive (clockwise) for right
  const steerDegrees = steerInput === 0 ? 0 : Math.round(-steerInput * 45) || 0;
  const zone: 'left' | 'center' | 'right' =
    steerInput > 0.12 ? 'left' : steerInput < -0.12 ? 'right' : 'center';

  return {
    steerInput,
    thumbOffsetPx,
    ariaValueNow,
    steerDegrees,
    zone,
  };
}

export interface EffectiveDriveInputParams {
  keyForward: boolean;
  keyBackward: boolean;
  keyLeft: boolean;
  keyRight: boolean;
  keyHandbrake: boolean;
  keyNitro: boolean;
  touchGas: boolean;
  touchBrake: boolean;
  touchLeft: boolean;
  touchRight: boolean;
  touchSteerAnalog: number;
  touchDrift: boolean;
  touchNitro: boolean;
  autoGasEnabled: boolean;
  nitroCharge: number;
}

/**
 * Resolves unified vehicle control commands from keyboard and touch inputs,
 * supporting Touch Auto-Gas (cruise control overridden by Brake) and Touch Nitro Auto-Throttle.
 */
export function computeEffectiveTouchDriveInput(input: EffectiveDriveInputParams): {
  throttle: number;
  steerInput: number;
  handbrake: boolean;
  isBoosting: boolean;
  effectiveGasActive: boolean;
} {
  const isBraking = input.keyBackward || input.touchBrake;
  // Touch Nitro or Auto-Gas automatically engages forward throttle when not actively braking
  const effectiveGasActive =
    input.keyForward ||
    input.touchGas ||
    (!isBraking && (input.autoGasEnabled || input.touchNitro));

  const throttle = (effectiveGasActive ? 1 : 0) - (isBraking ? 1 : 0);
  const digitalSteer =
    (input.keyLeft || input.touchLeft ? 1 : 0) -
    (input.keyRight || input.touchRight ? 1 : 0);
  const steerInput =
    Math.abs(input.touchSteerAnalog) > 0.01 ? input.touchSteerAnalog : digitalSteer;
  const handbrake = input.keyHandbrake || input.touchDrift;
  const isBoosting = (input.keyNitro || input.touchNitro) && input.nitroCharge > 0.05 && !isBraking;

  return {
    throttle,
    steerInput,
    handbrake,
    isBoosting,
    effectiveGasActive,
  };
}

/**
 * Computes calibrated ACESFilmic tone mapping exposure and UnrealBloom threshold
 * so the Preetham Golden Hour Sky shader does not overexpose or wash out the terrain.
 */
export function computeSceneExposureProfile(sunElevationDeg: number): {
  toneMappingExposure: number;
  bloomThreshold: number;
  bloomStrength: number;
} {
  const clampedElev = Math.max(0.5, Math.min(15, sunElevationDeg));
  const toneMappingExposure = Number(
    Math.max(0.38, Math.min(0.58, 0.50 - (clampedElev - 1.5) * 0.02)).toFixed(3),
  );
  return {
    toneMappingExposure,
    bloomThreshold: 2.6,
    bloomStrength: 0.16,
  };
}

export type TimeOfDayMode = 'day' | 'sunset' | 'night';

/**
 * Returns calibrated sky, sun, fog, exposure, and headlight parameters for Day, Sunset, and Night modes.
 */
export function getTimeOfDayLightingProfile(mode: TimeOfDayMode): {
  mode: TimeOfDayMode;
  label: string;
  sunElevationDeg: number;
  sunAzimuthDeg: number;
  turbidity: number;
  rayleigh: number;
  mieCoefficient: number;
  mieDirectionalG: number;
  toneMappingExposure: number;
  sunIntensity: number;
  hemiIntensity: number;
  headlightSpotIntensity: number;
  showStarsAndMoon: boolean;
} {
  if (mode === 'day') {
    return {
      mode: 'day',
      label: 'DAY',
      sunElevationDeg: 32,
      sunAzimuthDeg: 175,
      turbidity: 2.2,
      rayleigh: 1.15,
      mieCoefficient: 0.001,
      mieDirectionalG: 0.95,
      toneMappingExposure: 0.52,
      sunIntensity: 2.4,
      hemiIntensity: 0.95,
      headlightSpotIntensity: 0.0,
      showStarsAndMoon: false,
    };
  }
  if (mode === 'night') {
    return {
      mode: 'night',
      label: 'NIGHT',
      sunElevationDeg: -18,
      sunAzimuthDeg: 198,
      turbidity: 1.5,
      rayleigh: 0.4,
      mieCoefficient: 0.0005,
      mieDirectionalG: 0.96,
      toneMappingExposure: 0.72,
      sunIntensity: 0.65,
      hemiIntensity: 0.58,
      headlightSpotIntensity: 24.0,
      showStarsAndMoon: true,
    };
  }
  return {
    mode: 'sunset',
    label: 'SUNSET',
    sunElevationDeg: 4.5,
    sunAzimuthDeg: 168,
    turbidity: 3.6,
    rayleigh: 1.85,
    mieCoefficient: 0.0018,
    mieDirectionalG: 0.93,
    toneMappingExposure: 0.54,
    sunIntensity: 2.1,
    hemiIntensity: 0.85,
    headlightSpotIntensity: 8.5,
    showStarsAndMoon: false,
  };
}

/**
 * Computes a stabilized planar Nitro velocity delta and vertical downforce
 * so the vehicle never launches vertically, porpoises, or exceeds safe physics speeds.
 */
export function computeSafeNitroImpulse(input: {
  forwardX: number;
  forwardZ: number;
  kmh: number;
  dt: number;
  isGrounded: boolean;
  maxNitroKmh?: number;
}): {
  dvx: number;
  dvy: number;
  dvz: number;
} {
  const maxKmh = input.maxNitroKmh ?? 295;
  if (!input.isGrounded || input.kmh >= maxKmh) {
    return { dvx: 0, dvy: 0, dvz: 0 };
  }

  // Normalize horizontal forward direction so chassis pitch never injects vertical velocity
  const planarLen = Math.hypot(input.forwardX, input.forwardZ) || 1;
  const dirX = input.forwardX / planarLen;
  const dirZ = input.forwardZ / planarLen;

  // Smoothly taper boost as speed approaches maxNitroKmh
  const headroom = Math.max(0, Math.min(1, (maxKmh - input.kmh) / 55));
  const boostAccel = 22.0 * headroom * input.dt;

  // Gentle aerodynamic downforce keeps tires planted over crests during Nitro
  const downforceY = -2.8 * input.dt;

  return {
    dvx: dirX * boostAccel,
    dvy: downforceY,
    dvz: dirZ * boostAccel,
  };
}

export type CarId = 'apex' | 'kaze' | 'titan';

export interface CarSpec {
  id: CarId;
  name: string;
  subtitle: string;
  classTag: string;
  defaultPaint: string;
  maxSpeedKmh: number;
  maxNitroKmh: number;
  accelRate: number;
  steerRate: number;
  driftSteerMultiplier: number;
  driftLateralDamping: number;
  driftScoreMultiplier: number;
  airScoreMultiplier: number;
  rideHeightOffset: number;
  wheelRadius: number;
  wheelbase: number;
  trackWidth: number;
  engineBaseHz: number;
  stats: {
    speed: number;
    handling: number;
    drift: number;
    offroad: number;
  };
}

export const CAR_ORDER: CarId[] = ['apex', 'kaze', 'titan'];

export const CAR_CATALOG: Record<CarId, CarSpec> = {
  apex: {
    id: 'apex',
    name: 'Apex V10 GT',
    subtitle: 'Mid-Engine Clearcoat Supercar',
    classTag: 'S1 · 900 SUPERCAR',
    defaultPaint: '#ff5511',
    maxSpeedKmh: 262,
    maxNitroKmh: 298,
    accelRate: 25.0,
    steerRate: 2.15,
    driftSteerMultiplier: 1.25,
    driftLateralDamping: 3.2,
    driftScoreMultiplier: 1.0,
    airScoreMultiplier: 1.0,
    rideHeightOffset: 0.56,
    wheelRadius: 0.42,
    wheelbase: 2.7,
    trackWidth: 1.86,
    engineBaseHz: 54,
    stats: {
      speed: 96,
      handling: 90,
      drift: 82,
      offroad: 68,
    },
  },
  kaze: {
    id: 'kaze',
    name: 'Kaze RX-79 Drift',
    subtitle: 'Widebody Twin-Turbo JDM Street Spec',
    classTag: 'A · 800 DRIFT TUNER',
    defaultPaint: '#00e5ff',
    maxSpeedKmh: 246,
    maxNitroKmh: 284,
    accelRate: 26.8,
    steerRate: 2.48,
    driftSteerMultiplier: 1.44,
    driftLateralDamping: 2.35,
    driftScoreMultiplier: 1.35,
    airScoreMultiplier: 1.0,
    rideHeightOffset: 0.52,
    wheelRadius: 0.4,
    wheelbase: 2.58,
    trackWidth: 1.9,
    engineBaseHz: 66,
    stats: {
      speed: 88,
      handling: 94,
      drift: 99,
      offroad: 62,
    },
  },
  titan: {
    id: 'titan',
    name: 'Titan Baja 4x4',
    subtitle: 'Lifted V8 Desert Trophy Raid Truck',
    classTag: 'S1 · 880 EXTREME OFFROAD',
    defaultPaint: '#a3e635',
    maxSpeedKmh: 236,
    maxNitroKmh: 274,
    accelRate: 28.2,
    steerRate: 2.05,
    driftSteerMultiplier: 1.18,
    driftLateralDamping: 3.6,
    driftScoreMultiplier: 1.05,
    airScoreMultiplier: 1.45,
    rideHeightOffset: 0.78,
    wheelRadius: 0.54,
    wheelbase: 2.92,
    trackWidth: 2.08,
    engineBaseHz: 42,
    stats: {
      speed: 84,
      handling: 82,
      drift: 78,
      offroad: 99,
    },
  },
};

export function getCarSpec(id?: string | null): CarSpec {
  if (id && id in CAR_CATALOG) {
    return CAR_CATALOG[id as CarId];
  }
  return CAR_CATALOG.apex;
}

export type SceneryId = 'canyon' | 'alpine' | 'cyber';

export interface ScenerySpec {
  id: SceneryId;
  name: string;
  subtitle: string;
  regionTag: string;
  accentColor: string;
  defaultTimeOfDay: TimeOfDayMode;
  skyTintHex: Record<TimeOfDayMode, number>;
  fogDensityMultiplier: number;
  waterColorHex: number;
  waterOpacity: number;
  waterRoughness: number;
  treePaletteHex: number[];
  treeTrunkHex: number;
  rockPaletteHex: number[];
  archColorHex: number;
  terrainPalette: {
    lakeBedHex: number;
    valleyLowHex: number;
    valleyMidHex: number;
    canyonSlopeHex: number;
    mesaTopHex: number;
    peakCapHex: number;
    shoulderBlendHex: number;
  };
}

export const SCENERY_ORDER: SceneryId[] = ['canyon', 'alpine', 'cyber'];

export const SCENERY_CATALOG: Record<SceneryId, ScenerySpec> = {
  canyon: {
    id: 'canyon',
    name: 'Red Rock Canyon',
    subtitle: 'Sandstone Mesas & Turquoise Desert Oasis',
    regionTag: 'STAGE 01 · SONORAN OASIS',
    accentColor: '#ff8a2b',
    defaultTimeOfDay: 'sunset',
    skyTintHex: {
      sunset: 0x6e3a4c,
      night: 0x060914,
      day: 0x78a9d4,
    },
    fogDensityMultiplier: 1.0,
    waterColorHex: 0x0ea5e9,
    waterOpacity: 0.9,
    waterRoughness: 0.12,
    treePaletteHex: [0x2e6b34, 0x3f7a2e, 0x5c7428, 0x8f5e26, 0x245838, 0xd97734],
    treeTrunkHex: 0x42291d,
    rockPaletteHex: [0xa84c32, 0x8c3b28, 0xc26a3d, 0x6e3a4c, 0x785240],
    archColorHex: 0xa8462a,
    terrainPalette: {
      lakeBedHex: 0xd6b578,
      valleyLowHex: 0x486b38,
      valleyMidHex: 0x7c6838,
      canyonSlopeHex: 0xb85834,
      mesaTopHex: 0x8a3324,
      peakCapHex: 0xe2e8f0,
      shoulderBlendHex: 0x222628,
    },
  },
  alpine: {
    id: 'alpine',
    name: 'Alpine Glacier Pass',
    subtitle: 'Snow-Crowned Peaks & Frozen Crystal Fjord',
    regionTag: 'STAGE 02 · NORDIC SUMMIT',
    accentColor: '#00f0ff',
    defaultTimeOfDay: 'day',
    skyTintHex: {
      sunset: 0x56466e,
      night: 0x050b1a,
      day: 0x8abce6,
    },
    fogDensityMultiplier: 0.95,
    waterColorHex: 0x38bdf8,
    waterOpacity: 0.94,
    waterRoughness: 0.06,
    treePaletteHex: [0x1e4d3b, 0x285e48, 0xdbeafe, 0x163b2d, 0x93c5fd, 0x336650],
    treeTrunkHex: 0x33241b,
    rockPaletteHex: [0x475569, 0x64748b, 0x94a3b8, 0x334155, 0xcbd5e1],
    archColorHex: 0x64748b,
    terrainPalette: {
      lakeBedHex: 0x7dd3fc,
      valleyLowHex: 0xcbd5e1,
      valleyMidHex: 0x94a3b8,
      canyonSlopeHex: 0x64748b,
      mesaTopHex: 0xe2e8f0,
      peakCapHex: 0xffffff,
      shoulderBlendHex: 0x1e293b,
    },
  },
  cyber: {
    id: 'cyber',
    name: 'Neon Cyber Bay',
    subtitle: 'Obsidian Volcanic Cliffs & Sakura Synth Coast',
    regionTag: 'STAGE 03 · NEO BAYSHORE',
    accentColor: '#ff2a85',
    defaultTimeOfDay: 'night',
    skyTintHex: {
      sunset: 0x4a1942,
      night: 0x070412,
      day: 0x52689f,
    },
    fogDensityMultiplier: 0.92,
    waterColorHex: 0x00f0ff,
    waterOpacity: 0.88,
    waterRoughness: 0.08,
    treePaletteHex: [0xff2a85, 0xf472b6, 0x00f0ff, 0xa855f7, 0xfb7185, 0x38bdf8],
    treeTrunkHex: 0x1e1b2e,
    rockPaletteHex: [0x1e1b4b, 0x311042, 0x0f172a, 0x4c1d95, 0x1e293b],
    archColorHex: 0x3b0764,
    terrainPalette: {
      lakeBedHex: 0x0284c7,
      valleyLowHex: 0x111827,
      valleyMidHex: 0x1e1b4b,
      canyonSlopeHex: 0x311042,
      mesaTopHex: 0x701a75,
      peakCapHex: 0x00f0ff,
      shoulderBlendHex: 0x0f172a,
    },
  },
};

export function getScenerySpec(id?: string | null): ScenerySpec {
  if (id && id in SCENERY_CATALOG) {
    return SCENERY_CATALOG[id as SceneryId];
  }
  return SCENERY_CATALOG.canyon;
}

export interface VerticalPhysicsStepInput {
  wasGrounded: boolean;
  currentY: number;
  currentVy: number;
  prevGroundY: number;
  targetGroundY: number;
  prevSlopeRate: number;
  kmh: number;
  isOffroad: boolean;
  dt: number;
}

export interface VerticalPhysicsStepOutput {
  isGrounded: boolean;
  justLanded: boolean;
  y: number;
  vy: number;
  slopeRate: number;
}

/**
 * Steps vertical ride height and grounded/airborne transitions without false uphill
 * detachment at >= 200 km/h. On the asphalt highway corridor, aerodynamic downforce keeps
 * the car planted across rolling uphill/downhill grades. Off-road, cresting a sharp dune
 * at high speed cleanly launches the car into a ballistic jump.
 */
export function stepVerticalGroundPhysics(input: VerticalPhysicsStepInput): VerticalPhysicsStepOutput {
  const safeDt = Math.max(input.dt, 0.0005);
  const rawSlopeRate = (input.targetGroundY - input.prevGroundY) / safeDt;
  const slopeRate = Math.max(-24, Math.min(24, rawSlopeRate));

  if (input.wasGrounded) {
    // Only launch off sharp off-road hill crests when ground drops away beneath a rising car
    const crestDrop = slopeRate - input.prevSlopeRate;
    if (
      input.isOffroad &&
      input.kmh > 115 &&
      input.prevSlopeRate > 2.2 &&
      crestDrop < -13.5
    ) {
      const launchVy = Math.min(11, Math.max(2.0, input.prevSlopeRate));
      const nextVy = launchVy - 22 * safeDt;
      const nextY = input.currentY + nextVy * safeDt;
      if (nextY > input.targetGroundY + 0.04) {
        return {
          isGrounded: false,
          justLanded: false,
          y: nextY,
          vy: nextVy,
          slopeRate,
        };
      }
    }

    return {
      isGrounded: true,
      justLanded: false,
      y: input.targetGroundY,
      vy: Math.max(-14, Math.min(12, slopeRate)),
      slopeRate,
    };
  }

  // Airborne ballistic trajectory with gravity + high-speed downforce
  const nextVy = input.currentVy - 22 * safeDt;
  const nextY = input.currentY + nextVy * safeDt;
  if (nextY <= input.targetGroundY + 0.02) {
    return {
      isGrounded: true,
      justLanded: true,
      y: input.targetGroundY,
      vy: Math.max(-12, Math.min(9, slopeRate)),
      slopeRate,
    };
  }

  return {
    isGrounded: false,
    justLanded: false,
    y: nextY,
    vy: nextVy,
    slopeRate,
  };
}

export interface ChaseCameraState {
  fwdX: number;
  fwdZ: number;
  dist: number;
  height: number;
  verticalOffset: number;
  driftSwing: number;
}

export function createInitialChaseCameraState(carFwdX = 0, carFwdZ = -1): ChaseCameraState {
  return {
    fwdX: carFwdX,
    fwdZ: carFwdZ,
    dist: 6.4,
    height: 2.2,
    verticalOffset: 2.2,
    driftSwing: 0,
  };
}

function expDamp(current: number, target: number, lambda: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-lambda * Math.max(0, dt)));
}

/**
 * Computes a car-anchored chase camera pose by damping relative follow distance,
 * elevation, heading, and drift swing in the vehicle's local reference frame.
 * Eliminates world-space translation lag and frame-time rubber-banding at 200-360 km/h.
 */
export function stepChaseCameraRelative(
  state: ChaseCameraState,
  input: {
    carX: number;
    carY: number;
    carZ: number;
    carFwdX: number;
    carFwdZ: number;
    visualYawOffset: number;
    kmh: number;
    throttle: number;
    isBoosting: boolean;
    isTitan: boolean;
    dt: number;
    getGroundHeight?: (x: number, z: number) => number;
  },
): {
  state: ChaseCameraState;
  camX: number;
  camY: number;
  camZ: number;
  lookAtX: number;
  lookAtY: number;
  lookAtZ: number;
} {
  const dt = Math.max(0, Math.min(0.05, input.dt));
  const speedRatio = Math.max(0, Math.min(1, input.kmh / 245));

  // Damp camera forward unit vector smoothly toward vehicle forward vector
  let fwdX = expDamp(state.fwdX, input.carFwdX, 12, dt);
  let fwdZ = expDamp(state.fwdZ, input.carFwdZ, 12, dt);
  const fwdLen = Math.hypot(fwdX, fwdZ) || 1;
  fwdX /= fwdLen;
  fwdZ /= fwdLen;

  const rightX = -fwdZ;
  const rightZ = fwdX;

  const targetDist =
    6.4 +
    (input.throttle > 0 ? 0.45 : 0) +
    (input.isBoosting ? 0.55 : 0) +
    (input.isTitan ? 0.6 : 0);
  const targetHeight = 2.2 + speedRatio * 0.35 + (input.isTitan ? 0.45 : 0);
  const targetSwing = input.visualYawOffset * 1.35;

  const dist = expDamp(state.dist, targetDist, 10, dt);
  const height = expDamp(state.height, targetHeight, 10, dt);
  const driftSwing = expDamp(state.driftSwing, targetSwing, 12, dt);

  const camX = input.carX - fwdX * dist + rightX * driftSwing;
  const camZ = input.carZ - fwdZ * dist + rightZ * driftSwing;

  let minGroundY = -Infinity;
  if (input.getGroundHeight) {
    minGroundY = input.getGroundHeight(camX, camZ) + 1.45;
  }
  const desiredY = Math.max(input.carY + height, minGroundY);
  const targetVerticalOffset = desiredY - input.carY;
  const verticalOffset = expDamp(state.verticalOffset, targetVerticalOffset, 14, dt);
  const camY = Math.max(minGroundY, input.carY + verticalOffset);

  const lookAtX = input.carX + fwdX * 8.5 + rightX * (driftSwing * 0.3);
  const lookAtY = input.carY + 0.82;
  const lookAtZ = input.carZ + fwdZ * 8.5 + rightZ * (driftSwing * 0.3);

  return {
    state: {
      fwdX,
      fwdZ,
      dist,
      height,
      verticalOffset,
      driftSwing,
    },
    camX,
    camY,
    camZ,
    lookAtX,
    lookAtY,
    lookAtZ,
  };
}

export type SurfaceType = 'asphalt' | 'shoulder' | 'dirt' | 'snow' | 'water';

export interface SurfacePhysicsProfile {
  surface: SurfaceType;
  gripMu: number;
  rollingResistance: number;
  waterDepth: number;
  waterDragCoeff: number;
  surfaceRoughnessAmp: number;
}

export const OASIS_WATER_LEVEL = -1.6;

/**
 * Computes real surface-dependent tire friction (mu), rolling resistance,
 * hydrodynamic water drag (in Oasis Lake), and terrain micro-bump roughness.
 */
export function computeSurfacePhysicsProfile(input: {
  groundY: number;
  roadDistance: number;
  carId: CarId;
  sceneryId: SceneryId;
}): SurfacePhysicsProfile {
  const isTitan = input.carId === 'titan';
  const waterSurfaceY = OASIS_WATER_LEVEL + 0.15;

  if (input.groundY < waterSurfaceY) {
    const waterDepth = Math.max(0.05, Math.min(2.8, waterSurfaceY - input.groundY));
    return {
      surface: 'water',
      gripMu: isTitan ? 0.86 : 0.68,
      rollingResistance: isTitan ? 0.65 : 1.15,
      waterDepth,
      waterDragCoeff: Math.min(1.45, waterDepth * (isTitan ? 0.34 : 0.68)),
      surfaceRoughnessAmp: 0.014,
    };
  }

  if (input.roadDistance <= ROAD_HALF_WIDTH + 0.8) {
    return {
      surface: 'asphalt',
      gripMu: 1.08,
      rollingResistance: 0.36,
      waterDepth: 0,
      waterDragCoeff: 0,
      surfaceRoughnessAmp: 0,
    };
  }

  if (input.roadDistance <= ROAD_HALF_WIDTH + 4.2) {
    return {
      surface: 'shoulder',
      gripMu: isTitan ? 1.02 : 0.94,
      rollingResistance: 0.48,
      waterDepth: 0,
      waterDragCoeff: 0,
      surfaceRoughnessAmp: 0.01,
    };
  }

  if (input.sceneryId === 'alpine') {
    return {
      surface: 'snow',
      gripMu: isTitan ? 0.92 : 0.72,
      rollingResistance: isTitan ? 0.46 : 0.82,
      waterDepth: 0,
      waterDragCoeff: 0,
      surfaceRoughnessAmp: 0.028,
    };
  }

  return {
    surface: 'dirt',
    gripMu: isTitan ? 0.98 : 0.82,
    rollingResistance: isTitan ? 0.42 : 0.74,
    waterDepth: 0,
    waterDragCoeff: 0,
    surfaceRoughnessAmp: 0.032,
  };
}

export interface AutomotiveHorizontalPhysicsInput {
  vx: number;
  vz: number;
  yaw: number;
  yawRate: number;
  visualYawOffset: number;
  steerAngle: number;
  throttle: number;
  handbrake: boolean;
  isGrounded: boolean;
  terrainPitch: number;
  terrainRoll: number;
  carSpec: CarSpec;
  surfaceProfile: SurfacePhysicsProfile;
  dt: number;
}

export interface AutomotiveHorizontalPhysicsOutput {
  vx: number;
  vz: number;
  yaw: number;
  yawRate: number;
  visualYawOffset: number;
  vLong: number;
  vLat: number;
  accelLong: number;
  accelLat: number;
  slipAngleRad: number;
  isDrifting: boolean;
}

/**
 * Steps real 2D automotive dynamics including:
 * - Terrain slope gravity forces (uphill resistance, downhill gravity acceleration, camber pull)
 * - Quadratic aerodynamic drag + rolling resistance + Oasis water hydrodynamic drag
 * - Longitudinal weight transfer (trail-braking front bite vs throttle rear squat)
 * - Pacejka-inspired lateral tire cornering force, slip angle, and self-aligning counter-steer
 */
export function stepAutomotiveHorizontalPhysics(
  input: AutomotiveHorizontalPhysicsInput,
): AutomotiveHorizontalPhysicsOutput {
  const dt = Math.max(0.0005, Math.min(0.05, input.dt));
  const { carSpec, surfaceProfile } = input;
  const mu = surfaceProfile.gripMu;

  const prevFwdX = -Math.sin(input.yaw);
  const prevFwdZ = -Math.cos(input.yaw);
  const speedMs = Math.hypot(input.vx, input.vz);
  const kmh = speedMs * 3.6;
  const prevForwardVel = input.vx * prevFwdX + input.vz * prevFwdZ;
  const prevRightX = Math.cos(input.yaw);
  const prevRightZ = -Math.sin(input.yaw);
  const prevLatVel = input.vx * prevRightX + input.vz * prevRightZ;

  // Tire slip angle (angle between velocity vector and chassis heading)
  const slipAngleRad = speedMs > 1.5 ? Math.atan2(prevLatVel, Math.abs(prevForwardVel) + 1.2) : 0;

  // Approximate longitudinal acceleration for weight transfer
  const estAccelLong =
    input.throttle > 0 ? carSpec.accelRate * 0.7 : input.throttle < 0 ? -28 : -1.5;
  // Front axle weight shift under braking improves turn-in response (trail-braking)
  const frontLoadFactor = Math.max(0.82, Math.min(1.24, 1.0 - estAccelLong * 0.008));

  let yaw = input.yaw;
  let yawRate = input.yawRate;
  let visualYawOffset = input.visualYawOffset;

  if (input.isGrounded && speedMs > 0.35) {
    const revSign = prevForwardVel < -0.5 ? -1 : 1;
    const lowSpeedFactor = Math.min(1, speedMs / 8.5);
    const baseTurnRate =
      carSpec.steerRate *
      frontLoadFactor *
      (0.35 + 0.65 * Math.min(1.1, mu)) *
      (input.handbrake ? carSpec.driftSteerMultiplier : 1);

    const desiredYawRate = input.steerAngle * baseTurnRate * lowSpeedFactor * revSign;
    // Self-aligning counter-steer torque helps stabilize drifts when steering into the slide
    const selfAlignTorque = -slipAngleRad * (input.handbrake ? 0.55 : 1.15);
    const targetYawRate = desiredYawRate + selfAlignTorque * 0.22;

    yawRate = expDamp(yawRate, targetYawRate, input.handbrake ? 10.5 : 14.5, dt);
    yaw += yawRate * dt;

    const slipTarget =
      input.handbrake && speedMs > 7.0
        ? input.steerAngle * 0.7 * (carSpec.id === 'kaze' ? 1.16 : 1) +
          Math.max(-0.28, Math.min(0.28, -slipAngleRad * 0.35))
        : input.steerAngle * Math.min(1, kmh / 175) * (0.14 / Math.max(0.65, mu));

    visualYawOffset = expDamp(
      visualYawOffset,
      slipTarget,
      input.handbrake ? 7.5 : 10.0,
      dt,
    );
  } else {
    yawRate = expDamp(yawRate, 0, 5, dt);
    yaw += yawRate * dt * 0.35;
    visualYawOffset = expDamp(visualYawOffset, 0, 6, dt);
  }

  const newFwdX = -Math.sin(yaw);
  const newFwdZ = -Math.cos(yaw);
  const newRightX = Math.cos(yaw);
  const newRightZ = -Math.sin(yaw);

  let vLong = input.vx * newFwdX + input.vz * newFwdZ;
  let vLat = input.vx * newRightX + input.vz * newRightZ;
  const vLongBefore = vLong;

  if (input.isGrounded) {
    const surfaceSpeedCapFactor =
      surfaceProfile.surface === 'asphalt' || surfaceProfile.surface === 'shoulder'
        ? 1.0
        : carSpec.id === 'titan'
          ? 0.98
          : 0.88;
    const maxCarMs = (carSpec.maxSpeedKmh * surfaceSpeedCapFactor) / 3.6;

    // 1. Slope gravity force along terrain pitch (climbing hills slows car; descending accelerates)
    const gravityLong = -9.81 * Math.sin(input.terrainPitch) * 0.58;
    // 2. Camber gravity force along terrain roll
    const gravityLat = 9.81 * Math.sin(input.terrainRoll) * 0.38;

    // 3. Engine traction / Braking
    const tractionScale = 0.4 + 0.6 * Math.min(1.08, mu);
    if (input.throttle > 0) {
      const headroom = Math.max(0, 1 - Math.max(0, vLong) / maxCarMs);
      const engineAccel =
        vLong < 0
          ? 34 * tractionScale
          : carSpec.accelRate * tractionScale * Math.pow(headroom, 0.68);
      vLong += engineAccel * dt;
    } else if (input.throttle < 0) {
      if (vLong > 1.2) {
        vLong = Math.max(0, vLong - 39 * tractionScale * dt);
      } else if (vLong > -68 / 3.6) {
        vLong -= 16.5 * tractionScale * dt;
      }
    } else {
      // Engine braking + rolling resistance
      vLong *= Math.exp(-surfaceProfile.rollingResistance * dt);
      if (Math.abs(vLong) < 0.14 && Math.abs(gravityLong) < 0.4) vLong = 0;
    }

    // Apply slope gravity when moving or when throttle is applied / steep slope
    if (Math.abs(vLong) > 0.25 || Math.abs(input.throttle) > 0.01) {
      vLong += gravityLong * dt;
      vLat += gravityLat * dt;
    }

    // 4. Quadratic aerodynamic drag + water hydrodynamic drag
    const aeroDragDecel = 0.00018 * vLong * Math.abs(vLong);
    vLong -= aeroDragDecel * dt;
    if (surfaceProfile.waterDragCoeff > 0) {
      const waterFactor = Math.exp(-surfaceProfile.waterDragCoeff * dt);
      vLong *= waterFactor;
      vLat *= waterFactor;
    }

    // 5. Handbrake & Lateral Pacejka-inspired tire cornering damping
    if (input.handbrake) {
      vLong *= Math.exp(-0.62 * dt);
    }
    const baseLatGrip = input.handbrake
      ? carSpec.driftLateralDamping * Math.min(1, mu)
      : 12.2 * Math.pow(Math.min(1.08, mu), 1.15);
    vLat *= Math.exp(-baseLatGrip * dt);
  } else {
    // Airborne air resistance
    vLong *= Math.exp(-0.08 * dt);
    vLat *= Math.exp(-0.08 * dt);
  }

  const vx = newFwdX * vLong + newRightX * vLat;
  const vz = newFwdZ * vLong + newRightZ * vLat;
  const accelLong = (vLong - vLongBefore) / dt;
  const accelLat = vLong * yawRate;

  const isDrifting =
    input.isGrounded &&
    speedMs > 7.5 &&
    (input.handbrake || Math.abs(visualYawOffset) > 0.15 || Math.abs(slipAngleRad) > 0.22);

  return {
    vx,
    vz,
    yaw,
    yawRate,
    visualYawOffset,
    vLong,
    vLat,
    accelLong,
    accelLat,
    slipAngleRad,
    isDrifting,
  };
}

export type MapColliderKind = 'rigid' | 'smashable' | 'rival';

export interface MapColliderImpactInput {
  carX: number;
  carZ: number;
  carVx: number;
  carVz: number;
  carYaw: number;
  carRadius: number;
  colX: number;
  colZ: number;
  colRadius: number;
  kind: MapColliderKind;
}

export interface MapColliderImpactOutput {
  collided: boolean;
  carX: number;
  carZ: number;
  carVx: number;
  carVz: number;
  carYawDelta: number;
  impactSpeedKmh: number;
  normalX: number;
  normalZ: number;
  propLaunchVx: number;
  propLaunchVy: number;
  propLaunchVz: number;
}

/**
 * Resolves 2D physical collisions between the car chassis and map obstacles:
 * - 'rigid' (boulders, rock formations, thick trees, bridge pylons, gate pillars):
 *   Restitution bounce + wall-scrape tangential friction + off-center yaw deflection.
 * - 'smashable' (Forza Horizon bollards, festival flags, saplings):
 *   Launches the destructible prop airborne with 3D impulse while scrubbing minimal car momentum.
 * - 'rival' (AI Drivatar cars):
 *   Elastic/inelastic momentum transfer + lateral knockback.
 */
export function resolveMapColliderImpact(
  input: MapColliderImpactInput,
): MapColliderImpactOutput {
  const dx = input.carX - input.colX;
  const dz = input.carZ - input.colZ;
  const dist = Math.hypot(dx, dz);
  const minSep = input.carRadius + input.colRadius;

  if (dist >= minSep) {
    return {
      collided: false,
      carX: input.carX,
      carZ: input.carZ,
      carVx: input.carVx,
      carVz: input.carVz,
      carYawDelta: 0,
      impactSpeedKmh: 0,
      normalX: 0,
      normalZ: -1,
      propLaunchVx: 0,
      propLaunchVy: 0,
      propLaunchVz: 0,
    };
  }

  // Unit collision normal pointing from obstacle center to car center
  const nx = dist > 0.0001 ? dx / dist : -Math.sin(input.carYaw);
  const nz = dist > 0.0001 ? dz / dist : -Math.cos(input.carYaw);
  const overlap = minSep - dist;

  const carSpeedMs = Math.hypot(input.carVx, input.carVz);
  const carSpeedKmh = carSpeedMs * 3.6;

  if (input.kind === 'smashable') {
    // Smash through destructible prop! Launch prop forward & upward with car velocity
    const launchSpeed = Math.max(12, carSpeedMs * 1.25);
    const propLaunchVx = input.carVx * 1.15 - nx * launchSpeed * 0.45;
    const propLaunchVz = input.carVz * 1.15 - nz * launchSpeed * 0.45;
    const propLaunchVy = Math.min(18, 5.5 + carSpeedMs * 0.24);

    // Slight 2% speed scrub from smashing the prop
    return {
      collided: true,
      carX: input.carX,
      carZ: input.carZ,
      carVx: input.carVx * 0.98,
      carVz: input.carVz * 0.98,
      carYawDelta: 0,
      impactSpeedKmh: carSpeedKmh,
      normalX: nx,
      normalZ: nz,
      propLaunchVx,
      propLaunchVy,
      propLaunchVz,
    };
  }

  // Push car cleanly out of penetration
  const pushFactor = input.kind === 'rival' ? 0.85 : 1.02;
  const nextX = input.carX + nx * overlap * pushFactor;
  const nextZ = input.carZ + nz * overlap * pushFactor;

  // Decompose car velocity into normal (along n) and tangential (along t) components
  const tx = -nz;
  const tz = nx;
  const vn = input.carVx * nx + input.carVz * nz;
  const vt = input.carVx * tx + input.carVz * tz;

  const impactSpeedKmh = Math.max(0, -vn * 3.6);
  let nextVx = input.carVx;
  let nextVz = input.carVz;
  let carYawDelta = 0;

  if (vn < 0) {
    const restitution = input.kind === 'rival' ? 0.35 : 0.26;
    const wallFriction = input.kind === 'rival' ? 0.88 : 0.78;
    const bouncedVn = Math.max(1.8, -vn * restitution);
    const scrapedVt = vt * wallFriction;
    nextVx = nx * bouncedVn + tx * scrapedVt;
    nextVz = nz * bouncedVn + tz * scrapedVt;

    // Off-center glancing blow imparts realistic yaw deflection
    const fwdX = -Math.sin(input.carYaw);
    const fwdZ = -Math.cos(input.carYaw);
    const cross = fwdX * nz - fwdZ * nx;
    carYawDelta = Math.max(-0.22, Math.min(0.22, cross * Math.min(1, impactSpeedKmh / 140) * 0.16));
  }

  return {
    collided: true,
    carX: nextX,
    carZ: nextZ,
    carVx: nextVx,
    carVz: nextVz,
    carYawDelta,
    impactSpeedKmh,
    normalX: nx,
    normalZ: nz,
    propLaunchVx: -nx * Math.max(4, -vn * 0.65),
    propLaunchVy: 0,
    propLaunchVz: -nz * Math.max(4, -vn * 0.65),
  };
}


