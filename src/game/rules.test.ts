import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CAR_CATALOG,
  CAR_ORDER,
  computeAnalogSteerFromPad,
  computeEffectiveTouchDriveInput,
  computeFourWheelSurfacePose,
  computeSafeNitroImpulse,
  computeSceneExposureProfile,
  computeSurfacePhysicsProfile,
  computeTerrainHeight,
  computeVehicleTelemetry,
  createInitialChaseCameraState,
  createInitialSkillState,
  getCarSpec,
  getHighwayPointAtAngle,
  getRoadDistanceInfo,
  getScenerySpec,
  getTimeOfDayLightingProfile,
  resolveMapColliderImpact,
  ROAD_HALF_WIDTH,
  SCENERY_CATALOG,
  SCENERY_ORDER,
  stepAutomotiveHorizontalPhysics,
  stepChaseCameraRelative,
  stepSkillChain,
  stepVerticalGroundPhysics,
} from './rules.ts';

test('world origin (0, 0) sits directly on the highway centerline at ground level', () => {
  const info = getRoadDistanceInfo(0, 0);
  assert.ok(info.distance < 0.01, `Expected origin to be on highway, got distance=${info.distance}`);
  assert.ok(Math.abs(info.roadElevation) < 0.01, `Expected 0 elevation at origin, got ${info.roadElevation}`);

  const heightAtOrigin = computeTerrainHeight(0, 0);
  assert.ok(Math.abs(heightAtOrigin) < 0.01, `Expected terrain height 0 at origin, got ${heightAtOrigin}`);

  const startPt = getHighwayPointAtAngle(0);
  assert.ok(Math.abs(startPt.x) < 0.01);
  assert.ok(Math.abs(startPt.z) < 0.01);
  // Tangent at theta=0 points along -Z
  assert.ok(startPt.tz < -0.99);
});

test('wide road flat shoulder keeps terrain strictly level across highway width and shoulders', () => {
  const extremeNoise = () => 500;
  for (let i = 0; i < 32; i++) {
    const theta = (i / 32) * Math.PI * 2;
    const pt = getHighwayPointAtAngle(theta);
    const centerElev = computeTerrainHeight(pt.x, pt.z);
    // Even with extreme terrain noise, points within ROAD_HALF_WIDTH + 6m have 0 noise bleed
    const leftWithNoise = computeTerrainHeight(
      pt.x - pt.nx * (ROAD_HALF_WIDTH + 6),
      pt.z - pt.nz * (ROAD_HALF_WIDTH + 6),
      extremeNoise,
    );
    const leftNoNoise = computeTerrainHeight(
      pt.x - pt.nx * (ROAD_HALF_WIDTH + 6),
      pt.z - pt.nz * (ROAD_HALF_WIDTH + 6),
    );
    assert.equal(leftWithNoise, leftNoNoise);
    assert.ok(Math.abs(leftNoNoise - centerElev) < 0.85);
  }
});

test('4-wheel surface pose solver computes exact ground height and pitch/roll without stair-stepping', () => {
  const poseOrigin = computeFourWheelSurfacePose(0, 0, 0);
  assert.ok(Math.abs(poseOrigin.avgGroundY) < 0.05);
  assert.ok(Math.abs(poseOrigin.terrainRoll) < 0.01);

  // Test synthetic upward slope along -Z
  const slopedPose = computeFourWheelSurfacePose(0, 0, 0, 2.7, 1.86, (_x, z) => -z * 0.1);
  assert.ok(slopedPose.hFL > slopedPose.hRL);
  assert.ok(slopedPose.terrainPitch > 0.08);
  assert.equal(slopedPose.terrainRoll, 0);
});

test('vehicle telemetry computes KMH, MPH, gears 1-6/R, and dynamic FOV 60..80', () => {
  const standstill = computeVehicleTelemetry(0, 0, 0, true);
  assert.equal(standstill.kmh, 0);
  assert.equal(standstill.gear, '1');
  assert.equal(standstill.targetFov, 60);

  const cruising = computeVehicleTelemetry(30, 30, 0.8, true); // 108 km/h
  assert.ok(cruising.kmh > 107 && cruising.kmh < 109);
  assert.equal(cruising.gear, '3');
  assert.ok(cruising.targetFov > 66 && cruising.targetFov < 80);

  const topSpeed = computeVehicleTelemetry(75, 75, 1.0, true); // 270 km/h
  assert.equal(topSpeed.gear, '6');
  assert.equal(topSpeed.targetFov, 80);

  const reverse = computeVehicleTelemetry(8, -8, -1.0, true);
  assert.equal(reverse.gear, 'R');
});

test('Horizon skill chain accumulates drift, speed demon, near miss, and kangaroo points and banks after 3s', () => {
  let state = createInitialSkillState();

  // Drift at 95 km/h for 1 second + near miss
  state = stepSkillChain(state, {
    dt: 1.0,
    kmh: 95,
    isDrifting: true,
    isAirborne: false,
    airTimeSeconds: 0,
    isBoosting: false,
    nearMissJustTriggered: true,
  });
  assert.ok(state.activeScore > 700);
  assert.ok(state.multiplier > 1.5);
  assert.equal(state.cleanTimer, 3.0);
  assert.ok(state.activeSkills.includes('E-BRAKE DRIFT'));
  assert.ok(state.activeSkills.includes('DAREDEVIL NEAR MISS +400'));

  // Jump a hill ("Kangaroo") + high speed ("Speed Demon")
  state = stepSkillChain(state, {
    dt: 0.5,
    kmh: 165,
    isDrifting: false,
    isAirborne: true,
    airTimeSeconds: 0.4,
    isBoosting: true,
  });
  assert.ok(state.activeSkills.includes('KANGAROO AIR'));
  assert.ok(state.activeSkills.includes('ULTIMATE BURNOUT · NITRO'));

  // Drive cleanly for 3.0 seconds -> should bank score
  state = stepSkillChain(state, {
    dt: 1.5,
    kmh: 80,
    isDrifting: false,
    isAirborne: false,
    airTimeSeconds: 0,
    isBoosting: false,
  });
  assert.ok(state.activeScore > 0);
  assert.equal(state.bankedScore, 0);

  state = stepSkillChain(state, {
    dt: 1.6,
    kmh: 80,
    isDrifting: false,
    isAirborne: false,
    airTimeSeconds: 0,
    isBoosting: false,
  });
  assert.equal(state.activeScore, 0);
  assert.ok(state.bankedScore > 1000);
  assert.equal(state.bestChainScore, state.bankedScore);
  assert.equal(state.multiplier, 1.0);
});

test('analog touch steering pad maps horizontal touch coordinates to proportional steering, wheel degrees, and zones', () => {
  const centerTouch = computeAnalogSteerFromPad(120, 20, 200);
  assert.equal(centerTouch.steerInput, 0);
  assert.equal(centerTouch.thumbOffsetPx, 0);
  assert.equal(centerTouch.steerDegrees, 0);
  assert.equal(centerTouch.zone, 'center');

  const leftTouch = computeAnalogSteerFromPad(30, 20, 200);
  assert.equal(leftTouch.steerInput, 1);
  assert.ok(leftTouch.thumbOffsetPx < -45);
  assert.equal(leftTouch.ariaValueNow, -100);
  assert.equal(leftTouch.steerDegrees, -45);
  assert.equal(leftTouch.zone, 'left');

  const halfRightTouch = computeAnalogSteerFromPad(162, 20, 200);
  assert.ok(halfRightTouch.steerInput < -0.35 && halfRightTouch.steerInput > -0.65);
  assert.ok(halfRightTouch.thumbOffsetPx > 20);
  assert.ok(halfRightTouch.steerDegrees > 15);
  assert.equal(halfRightTouch.zone, 'right');
});

test('effective touch drive input supports Auto-Gas cruise, Brake override, and Nitro auto-throttle', () => {
  const baseInput = {
    keyForward: false,
    keyBackward: false,
    keyLeft: false,
    keyRight: false,
    keyHandbrake: false,
    keyNitro: false,
    touchGas: false,
    touchBrake: false,
    touchLeft: false,
    touchRight: false,
    touchSteerAnalog: 0,
    touchDrift: false,
    touchNitro: false,
    autoGasEnabled: false,
    nitroCharge: 1.0,
  };

  // Auto-gas enabled applies full throttle automatically
  const autoGasDrive = computeEffectiveTouchDriveInput({
    ...baseInput,
    autoGasEnabled: true,
    touchSteerAnalog: 0.42,
  });
  assert.equal(autoGasDrive.throttle, 1);
  assert.equal(autoGasDrive.effectiveGasActive, true);
  assert.equal(autoGasDrive.steerInput, 0.42);

  // Pressing Brake while Auto-Gas is on overrides throttle to -1 (brake/reverse)
  const brakeOverride = computeEffectiveTouchDriveInput({
    ...baseInput,
    autoGasEnabled: true,
    touchBrake: true,
  });
  assert.equal(brakeOverride.throttle, -1);
  assert.equal(brakeOverride.effectiveGasActive, false);

  // Pressing Touch Nitro alone automatically engages forward throttle + boost
  const nitroTouch = computeEffectiveTouchDriveInput({
    ...baseInput,
    touchNitro: true,
  });
  assert.equal(nitroTouch.throttle, 1);
  assert.equal(nitroTouch.isBoosting, true);
});

test('scene exposure profile keeps ACESFilmic exposure and bloom calibrated against overexposure', () => {
  const profile = computeSceneExposureProfile(2.0);
  assert.ok(profile.toneMappingExposure >= 0.4 && profile.toneMappingExposure <= 0.58);
  assert.ok(profile.bloomThreshold >= 2.2);
  assert.ok(profile.bloomStrength <= 0.25);
});

test('Day, Sunset, and Night lighting profiles configure calibrated exposure, sky, and headlights', () => {
  const day = getTimeOfDayLightingProfile('day');
  const sunset = getTimeOfDayLightingProfile('sunset');
  const night = getTimeOfDayLightingProfile('night');

  assert.equal(day.showStarsAndMoon, false);
  assert.equal(day.headlightSpotIntensity, 0);
  assert.ok(day.sunElevationDeg > 10);

  assert.equal(sunset.showStarsAndMoon, false);
  assert.ok(sunset.sunElevationDeg > 1 && sunset.sunElevationDeg < 10);
  assert.ok(sunset.toneMappingExposure <= 0.58);

  assert.equal(night.showStarsAndMoon, true);
  assert.ok(night.headlightSpotIntensity >= 20);
});

test('stabilized Nitro impulse applies planar horizontal thrust, aerodynamic downforce, and speed cap', () => {
  const boost = computeSafeNitroImpulse({
    forwardX: 0,
    forwardZ: -1,
    kmh: 140,
    dt: 1 / 60,
    isGrounded: true,
    maxNitroKmh: 295,
  });
  assert.ok(boost.dvz < -0.1, 'Expected forward -Z boost velocity delta');
  assert.ok(boost.dvy < 0, 'Expected downward aerodynamic downforce during nitro');
  assert.equal(boost.dvx, 0);

  const airborneBoost = computeSafeNitroImpulse({
    forwardX: 0,
    forwardZ: -1,
    kmh: 140,
    dt: 1 / 60,
    isGrounded: false,
  });
  assert.equal(airborneBoost.dvx, 0);
  assert.equal(airborneBoost.dvy, 0);
  assert.equal(airborneBoost.dvz, 0);

  const cappedBoost = computeSafeNitroImpulse({
    forwardX: 0,
    forwardZ: -1,
    kmh: 300,
    dt: 1 / 60,
    isGrounded: true,
    maxNitroKmh: 295,
  });
  assert.equal(cappedBoost.dvx, 0);
  assert.equal(cappedBoost.dvz, 0);
});

test('car catalog defines 3 distinct playable vehicles (Apex V10 GT, Kaze RX-79 Drift, Titan Baja 4x4)', () => {
  assert.equal(CAR_ORDER.length, 3);
  assert.deepEqual(CAR_ORDER, ['apex', 'kaze', 'titan']);

  const apex = getCarSpec('apex');
  const kaze = getCarSpec('kaze');
  const titan = getCarSpec('titan');

  // Apex has highest top speed, Kaze has highest drift rating, Titan has highest offroad rating & ride height
  assert.ok(apex.maxSpeedKmh > kaze.maxSpeedKmh && kaze.maxSpeedKmh > titan.maxSpeedKmh);
  assert.ok(kaze.driftScoreMultiplier > apex.driftScoreMultiplier);
  assert.ok(titan.rideHeightOffset > apex.rideHeightOffset);
  assert.ok(titan.wheelRadius > apex.wheelRadius);
  assert.equal(getCarSpec('invalid').id, 'apex');
});

test('scenery catalog defines 3 distinct festival worlds (Red Rock Canyon, Alpine Glacier Pass, Neon Cyber Bay)', () => {
  assert.equal(SCENERY_ORDER.length, 3);
  assert.deepEqual(SCENERY_ORDER, ['canyon', 'alpine', 'cyber']);

  for (const sceneryId of SCENERY_ORDER) {
    const spec = SCENERY_CATALOG[sceneryId];
    assert.ok(spec.name.length > 3);
    assert.ok(spec.treePaletteHex.length >= 4);
    assert.ok(spec.rockPaletteHex.length >= 4);
    assert.ok(spec.terrainPalette.mesaTopHex !== spec.terrainPalette.valleyLowHex);
  }
  assert.equal(getScenerySpec('unknown').id, 'canyon');
});

test('high-speed (200-340 km/h) vertical ground physics stays 100% planted on highway slopes without false airborne oscillation', () => {
  let isGrounded = true;
  let y = 0.56;
  let vy = 0;
  let slopeRate = 0;
  const dt = 1 / 60;

  // Simulate 180 frames at 240 km/h (66.7 m/s) climbing a 7.5% highway grade (slopeRate = +5.0 m/s > 3.5 m/s)
  for (let i = 0; i < 180; i++) {
    const prevGroundY = y;
    const targetGroundY = prevGroundY + 5.0 * dt;
    const step = stepVerticalGroundPhysics({
      wasGrounded: isGrounded,
      currentY: y,
      currentVy: vy,
      prevGroundY,
      targetGroundY,
      prevSlopeRate: slopeRate,
      kmh: 240,
      isOffroad: false,
      dt,
    });
    isGrounded = step.isGrounded;
    y = step.y;
    vy = step.vy;
    slopeRate = step.slopeRate;
    assert.equal(isGrounded, true, `Car falsely detached on uphill highway at frame ${i}`);
    assert.equal(y, targetGroundY);
  }

  // Simulate sharp off-road dune crest at 210 km/h -> should cleanly launch and then land
  const launchStep = stepVerticalGroundPhysics({
    wasGrounded: true,
    currentY: 14.0,
    currentVy: 6.5,
    prevGroundY: 14.0,
    targetGroundY: 13.8, // Sudden downward slope (-12 m/s) after +6.5 m/s climb
    prevSlopeRate: 6.5,
    kmh: 210,
    isOffroad: true,
    dt,
  });
  assert.equal(launchStep.isGrounded, false, 'Expected car to launch off sharp off-road dune crest at 210 km/h');
  assert.ok(launchStep.y > 13.8);
});

test('car-relative chase camera maintains steady follow distance at 200-300 km/h across fluctuating frame times without rubber-banding', () => {
  let camState = createInitialChaseCameraState(0, -1);
  let carZ = 0;
  const carX = 0;
  const carY = 0.56;
  const speedMs = 250 / 3.6; // ~69.44 m/s (250 km/h)

  // Vary dt aggressively between 8ms, 16.6ms, 33.3ms, and 45ms to simulate frame-time jitter at 250 km/h
  const frameDts = [0.016, 0.033, 0.008, 0.042, 0.016, 0.028, 0.012, 0.038];
  const distances: number[] = [];

  for (let i = 0; i < 40; i++) {
    const dt = frameDts[i % frameDts.length];
    carZ -= speedMs * dt;
    const res = stepChaseCameraRelative(camState, {
      carX,
      carY,
      carZ,
      carFwdX: 0,
      carFwdZ: -1,
      visualYawOffset: 0,
      kmh: 250,
      throttle: 1,
      isBoosting: true,
      isTitan: false,
      dt,
    });
    camState = res.state;
    const distToCar = Math.hypot(res.camX - carX, res.camZ - carZ);
    distances.push(distToCar);
    // Distance must always remain tightly bounded between 6.4m and 7.5m (never lagging 11-13m behind)
    assert.ok(
      distToCar >= 6.39 && distToCar <= 7.5,
      `Expected camera distance in [6.4, 7.5], got ${distToCar} at step ${i}`,
    );
  }

  // Once settled, frame-to-frame distance variation under heavy dt jitter is near zero (< 0.02m)
  const settledDiff = Math.abs(distances[distances.length - 1] - distances[distances.length - 2]);
  assert.ok(settledDiff < 0.02, `Expected zero camera rubber-banding at 250 km/h, got diff=${settledDiff}`);
});

test('surface physics profile differentiates asphalt, dirt, alpine snow, and Oasis water with Titan 4x4 off-road advantage', () => {
  const asphalt = computeSurfacePhysicsProfile({
    groundY: 0,
    roadDistance: 4,
    carId: 'apex',
    sceneryId: 'canyon',
  });
  assert.equal(asphalt.surface, 'asphalt');
  assert.ok(asphalt.gripMu > 1.0);
  assert.equal(asphalt.waterDragCoeff, 0);

  const snowApex = computeSurfacePhysicsProfile({
    groundY: 12,
    roadDistance: 35,
    carId: 'apex',
    sceneryId: 'alpine',
  });
  const snowTitan = computeSurfacePhysicsProfile({
    groundY: 12,
    roadDistance: 35,
    carId: 'titan',
    sceneryId: 'alpine',
  });
  assert.equal(snowApex.surface, 'snow');
  assert.ok(snowTitan.gripMu > snowApex.gripMu, 'Titan 4x4 should have superior snow grip');

  const oasisWater = computeSurfacePhysicsProfile({
    groundY: -12.6,
    roadDistance: 95,
    carId: 'kaze',
    sceneryId: 'canyon',
  });
  assert.equal(oasisWater.surface, 'water');
  assert.ok(oasisWater.waterDepth > 0.8);
  assert.ok(oasisWater.waterDragCoeff > 0.4);
});

test('real automotive horizontal physics applies slope gravity, trail-braking turn-in, and counter-steer drift dynamics', () => {
  const apex = getCarSpec('apex');
  const asphalt = computeSurfacePhysicsProfile({
    groundY: 0,
    roadDistance: 0,
    carId: 'apex',
    sceneryId: 'canyon',
  });

  // Uphill climb (+0.22 rad pitch) should produce lower longitudinal speed gain than downhill (-0.22 rad pitch)
  const uphill = stepAutomotiveHorizontalPhysics({
    vx: 0,
    vz: -35,
    yaw: 0,
    yawRate: 0,
    visualYawOffset: 0,
    steerAngle: 0,
    throttle: 1,
    handbrake: false,
    isGrounded: true,
    terrainPitch: 0.22,
    terrainRoll: 0,
    carSpec: apex,
    surfaceProfile: asphalt,
    dt: 1 / 60,
  });
  const downhill = stepAutomotiveHorizontalPhysics({
    vx: 0,
    vz: -35,
    yaw: 0,
    yawRate: 0,
    visualYawOffset: 0,
    steerAngle: 0,
    throttle: 1,
    handbrake: false,
    isGrounded: true,
    terrainPitch: -0.22,
    terrainRoll: 0,
    carSpec: apex,
    surfaceProfile: asphalt,
    dt: 1 / 60,
  });
  assert.ok(downhill.vLong > uphill.vLong, 'Downhill slope gravity must accelerate faster than uphill climb');

  // Trail-braking (throttle < 0) shifts weight forward and increases turn-in yaw rate vs full-throttle understeer
  const brakeTurn = stepAutomotiveHorizontalPhysics({
    vx: 0,
    vz: -40,
    yaw: 0,
    yawRate: 0,
    visualYawOffset: 0,
    steerAngle: 0.35,
    throttle: -1,
    handbrake: false,
    isGrounded: true,
    terrainPitch: 0,
    terrainRoll: 0,
    carSpec: apex,
    surfaceProfile: asphalt,
    dt: 1 / 60,
  });
  const accelTurn = stepAutomotiveHorizontalPhysics({
    vx: 0,
    vz: -40,
    yaw: 0,
    yawRate: 0,
    visualYawOffset: 0,
    steerAngle: 0.35,
    throttle: 1,
    handbrake: false,
    isGrounded: true,
    terrainPitch: 0,
    terrainRoll: 0,
    carSpec: apex,
    surfaceProfile: asphalt,
    dt: 1 / 60,
  });
  assert.ok(Math.abs(brakeTurn.yawRate) > Math.abs(accelTurn.yawRate), 'Front weight shift under braking should increase turn-in yaw rate');
});

test('map collider impact solver resolves rigid boulder rebounds and launches smashable props airborne', () => {
  // 1) Rigid boulder head-on impact at 180 km/h (50 m/s along -Z)
  const rigidHit = resolveMapColliderImpact({
    carX: 0.4,
    carZ: 2.0,
    carVx: 0,
    carVz: -50,
    carYaw: 0,
    carRadius: 1.4,
    colX: 0,
    colZ: 0,
    colRadius: 1.8,
    kind: 'rigid',
  });
  assert.equal(rigidHit.collided, true);
  assert.ok(rigidHit.carZ > 2.0, 'Car should be pushed out of penetration');
  assert.ok(rigidHit.carVz > -10, 'Normal velocity into rigid boulder must rebound/arrest');
  assert.ok(rigidHit.impactSpeedKmh > 120);

  // 2) Smashable highway bollard impact at 144 km/h (40 m/s along -Z)
  const smashHit = resolveMapColliderImpact({
    carX: 0,
    carZ: 1.0,
    carVx: 0,
    carVz: -40,
    carYaw: 0,
    carRadius: 1.3,
    colX: 0,
    colZ: 0,
    colRadius: 0.5,
    kind: 'smashable',
  });
  assert.equal(smashHit.collided, true);
  assert.ok(smashHit.carVz < -38, 'Smashing a bollard should preserve 98% of forward speed');
  assert.ok(smashHit.propLaunchVy > 8, 'Smashed bollard should launch airborne');
  assert.ok(smashHit.propLaunchVz < -35, 'Smashed bollard should fly forward ahead of the car');
});


