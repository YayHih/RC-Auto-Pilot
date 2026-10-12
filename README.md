# rcsim - noisy-sensor RC flight sim (open source, all Python)

Run:      `.venv\Scripts\python viewer.py`           (buttons: TAKEOFF / HOVER / LOOP / LAND)
Headless: `.venv\Scripts\python scenario.py "0:takeoff,25:loop,45:hover,70:loop,95:hover,115:land" 260 3 out.png`

| file | what it is |
|---|---|
| rcsim/airframe.py | 6-DOF physics, 0.85 kg 3D foamie, T/W 2.4, prop-wash control authority (hover), stall/flat-plate aero, gear contact |
| rcsim/sensors.py  | IMU, QMC5883L compass, BMP280 baro, u-blox M10 GPS (SEQURE M10-25Q) noise models - edit the numbers here |
| rcsim/estimator.py| Mahony attitude filter + position/velocity/accel-bias Kalman filters, GPS latency + thrust/centripetal compensation |
| rcsim/autopilot.py| state machine: TAKEOFF, CRUISE(orbit), LOOP, TO_HOVER/HOVER/FROM_HOVER, LAND/FLARE/ROLLOUT |
| rcsim/world.py    | 200 Hz loop: physics -> sensors -> estimator -> autopilot -> servos, wind + gusts |

Sensor model (rcsim/sensors.py)
- GPS 10 Hz, 100 ms latency; position error is time-correlated (Gauss-Markov, ~1 m horizontal / 1.8 m vertical 1-sigma) + white noise; velocity noise ~0.05 m/s
- Compass: ~0.25 uT noise, ~0.8 uT residual hard-iron, motor-current interference that grows with throttle^2 (80 % compensated)
- IMU / baro: vibration-level gyro/accel noise, turn-on bias, bias random walk; baro drifts +-0.8 m

NOT in your real hardware: the HTCC-AB01 + M10 GPS/compass has no gyro, accelerometer or barometer.
The sim assumes an MPU-class IMU and BMP280 (you already have a BMP280 STL in Downloads). Without
an IMU a plane cannot be stabilised from GPS + compass alone.
