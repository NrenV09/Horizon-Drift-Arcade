import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CAR_CATALOG,
  CAR_ORDER,
  computeAnalogSteerFromPad,
  computeFourWheelSurfacePose,
  computeSafeNitroImpulse,
  computeSceneExposureProfile,
  computeTerrainHeight,
  computeVehicleTelemetry,
  createInitialSkillState,
  getCarSpec,
  getHighwayPointAtAngle,
  getRoadDistanceInfo,
  getScenerySpec,
  getTimeOfDayLightingProfile,
  ROAD_HALF_WIDTH,
  SCENERY_CATALOG,
  SCENERY_ORDER,
  stepSkillChain,
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

test('analog touch steering pad maps horizontal touch coordinates to proportional steering', () => {
  const centerTouch = computeAnalogSteerFromPad(120, 20, 200);
  assert.equal(centerTouch.steerInput, 0);
  assert.equal(centerTouch.thumbOffsetPx, 0);

  const leftTouch = computeAnalogSteerFromPad(30, 20, 200);
  assert.equal(leftTouch.steerInput, 1);
  assert.ok(leftTouch.thumbOffsetPx < -45);
  assert.equal(leftTouch.ariaValueNow, -100);

  const halfRightTouch = computeAnalogSteerFromPad(162, 20, 200);
  assert.ok(halfRightTouch.steerInput < -0.35 && halfRightTouch.steerInput > -0.65);
  assert.ok(halfRightTouch.thumbOffsetPx > 20);
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

