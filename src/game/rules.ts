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
  const roadElevation =
    8.5 * (1 - Math.cos(theta)) +
    4.2 * Math.sin(2 * theta) -
    2.1 * Math.sin(4 * theta);

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
  const flatMargin = ROAD_HALF_WIDTH + ROAD_FLAT_SHOULDER;
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
 * inside the virtual steering pad, including a center deadzone and thumb puck pixel offset.
 * Returns steerInput in [-1, 1] where +1 is full left turn and -1 is full right turn.
 */
export function computeAnalogSteerFromPad(
  clientX: number,
  rectLeft: number,
  rectWidth: number,
  maxThumbTravelPx = 56,
  deadzone = 0.06,
): {
  steerInput: number;
  thumbOffsetPx: number;
  ariaValueNow: number;
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
    // Slight non-linear response curve for fine center control
    shaped = sign * Math.pow(rescaled, 1.15);
  }

  // In vehicle coordinates, positive steer angle turns left, negative turns right
  const steerInput = Number((-shaped).toFixed(4));
  const thumbOffsetPx = Math.round(rawNormalized * maxThumbTravelPx);
  const ariaValueNow = Math.round(rawNormalized * 100);

  return {
    steerInput,
    thumbOffsetPx,
    ariaValueNow,
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
