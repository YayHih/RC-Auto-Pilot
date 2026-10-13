# rcsim - RC flight sim + the Heltec flight controller that flies in it

The autopilot is C++ in `firmware/RCAutopilot/src/fc/` and is the same code on the Heltec CubeCell (HTCC-AB01) and in
the simulator.  The simulator feeds it raw sensor register values on a 100 Hz microsecond clock and takes back the
PCA9685 counts that drive the servos, exactly as the board does; `--fc arm` even runs the Cortex-M0+ machine code.

Run:      `.venv/bin/python viewer.py`  (Windows: `.venv\Scripts\python viewer.py`)
          buttons TAKEOFF / HOVER / LOOP / LAND, MANUAL / ASSIST, wind sliders - see the docstring in viewer.py
          options: `--seed N --wind M/S --wind-from DEG --turb X --no-random --fc native|arm --plane ail=0.6,ele=1.4,...`
Install:  `python3 -m venv .venv && .venv/bin/pip install numpy matplotlib pygame unicorn`  (+ a C++ compiler; the
          CubeCell board package in the Arduino IDE / arduino-cli provides the ARM toolchain)
Headless: `.venv/bin/python scenario.py "0:takeoff,25:loop,45:hover,70:loop,95:hover,115:land" 320 3 out.png`
          `[--fc native|arm|lockstep] [--plane ail=0.6,ele=1.4,rud=1,mass=1.2,thrust=1] [--cfg adapt_enable=0,rwy_hdg_deg=0]`
Checks:   `.venv/bin/python tests/physics_check.py`  physics (energy / momentum conservation, glide, phugoid, ...)
          `.venv/bin/python tests/fc_check.py`       flash/RAM, ARM vs PC bit-identical for a whole mission, cycles per tick
Batch:    `caffeinate -is .venv/bin/python tools/batch.py 16 --mission short`  many random aeroplanes (size, control-surface
          size, per-axis control power, thrust, mass, wing area) in random wind, one worker per performance core
Tuning:   `caffeinate -is .venv/bin/python tools/tune.py 12 10 10`  hyper-parameter search over those aeroplanes
          (`caffeinate` keeps the Mac from sleeping during long runs; idle sleep otherwise suspends them)
Flash:    open `firmware/RCAutopilot/RCAutopilot.ino`, board "CubeCell-Board (HTCC-AB01)", or
          `arduino-cli compile --fqbn CubeCell:CubeCell:CubeCell-Board firmware/RCAutopilot`

| file | what it is |
|---|---|
| firmware/RCAutopilot/RCAutopilot.ino | Heltec hardware layer: MPU6050, BMP280, QMC5883L, PCA9685 (I2C), u-blox M10 UBX, CPPM receiver, LoRa commands, 100 Hz micros() scheduler |
| firmware/.../src/fc/fc.cpp | `fc_tick(raw sensors -> PCA9685 counts)`, register conversions, scheduling of the modules below |
| firmware/.../src/fc/estimator.cpp | attitude (gyro + compass + GPS-velocity-aided tilt, matched windows), position/velocity Kalman filters |
| firmware/.../src/fc/wind_adapt.cpp | wind + airspeed + compass-error estimator (no airspeed sensor); in-flight control-authority identification |
| firmware/.../src/fc/autopilot.cpp | state machine: TAKEOFF, CRUISE, LOOP, HOVER, LAND (holding circle) / FINAL / FLARE / ROLLOUT / GO_AROUND, MANUAL |
| firmware/.../src/fc/fc_math.h | deterministic float math (no libm): the PC and the M0+ produce identical bits |
| rcsim/fcbridge.py | builds and runs the flight controller: native (ctypes), ARM (Unicorn, cycle-counted), lockstep |
| rcsim/hw.py       | sensor chips at register level (MPU6050 DLPF + int16, QMC5883L, BMP280 Pa*256, NAV-PVT) and the PCA9685 pulse train |
| rcsim/airframe.py | 6-DOF rigid body in air: mass/CG/inertia tensor from a parts list, gyroscopic rotor, added (apparent) air mass, strip aero with slipstream jet + swirl, Viterna post-stall, downwash lag, ground effect, LiPo/ESC/motor/prop, tyre stick/slip contact, energy bookkeeping |
| rcsim/sensors.py  | noise models: IMU, QMC5883L compass, BMP280 baro, u-blox M10 GPS - edit the numbers here |
| rcsim/world.py    | 200 Hz physics, 100 Hz flight controller, PCA9685 frames -> servos/ESC; wind, turbulence, gusts; manual flying |
| rcsim/render.py, viewer.py | software renderer and pygame front end |
| rcsim/legacy/     | the old Python estimator/autopilot (replaced by the C++ flight controller) |
| tests/            | physics_check.py, fc_check.py |

Hardware (beyond the HTCC-AB01 + M10 GPS/compass you have)
- MPU6050 (or any MPU-class IMU): the board has no gyro/accelerometer and a plane cannot be stabilised without one.
- BMP280 barometer.
- PCA9685 16-channel servo board: the AB01 has only two hardware PWM pins and the plane needs 4-5 outputs.
- An RC receiver with CPPM output for manual control / the mode switch (strongly recommended: it is your safety pilot).
- The AB01 has one UART: it goes to the GPS, so there is no USB serial while flying (`#define BENCH_DEBUG` for the bench).

How the controller copes with different aeroplanes and with wind
- Control authority: a small dither on each surface (roll 5 Hz, pitch 4 Hz, yaw 3 Hz, < 1 deg of wobble) and a lock-in
  detector measure rad/s^2 per unit stick in steady flight; the gains follow.  Learned scales for surfaces at x0.4 ... x2.5
  of the nominal plane land within ~25 % of the truth.  The hover throttle and cruise throttle trim also learn.
- Wind: from GPS ground velocity and the nose direction while the plane turns (no airspeed sensor), airspeed propagated
  with the accelerometer; also learns the compass heading error.  Typically within 0.4-1.3 m/s of the true wind.
- Landing: picks the runway direction with the headwind; if it arrives high it spirals down in the loiter orbit out on
  the flying side, then flies a compact circuit: a 37.5 m holding circle 8 m off the extended centreline at 12.5 m, a
  short 8.4 deg final (~100 m from the circle to the aim point; pitch holds airspeed = 12.3 m/s + half the headwind,
  throttle holds the path), go-around if not stabilised at 45 % of circuit height, sink-rate flare, roll-out steered to
  the centreline, brakes, stops on GPS speed.  Near the ground the airspeed uses the wind scaled down by the log
  boundary-layer profile (the wind is learned up at circuit height; without this the airspeed reads high on short
  final and the aeroplane stalls in the flare).  The runway axis comes from
  the GPS track of the first metres of the take-off roll (or set `cfg.rwy_hdg_deg`), never from the compass alone.

Safety: one side of the runway
- The pilot stands on one side; `cfg.fly_side` (+1 = right of the runway heading, -1 = left) picks the other.  The loiter
  orbit, the loop line (35 m out, parallel to the runway), the hover and the landing holding circle all stay at least
  `cfg.fence_m` (15 m) out on the flying side.  Only the take-off/climb-out, the final approach and a go-around are flown
  over the runway itself.  A soft fence turns the aeroplane away from the pilot's side, and big turns near the runway
  always go the way that first carries it towards the flying side.  tools/batch.py reports any incursion past the
  runway edge onto the pilot's side while airborne.

Sensor model (rcsim/sensors.py)
- GPS 10 Hz, 100 ms latency; position error is time-correlated (Gauss-Markov, ~1 m horizontal / 1.8 m vertical 1-sigma) + white noise; velocity noise ~0.05 m/s
- Compass: ~0.25 uT noise, ~0.8 uT residual hard-iron, motor-current interference that grows with throttle^2 (80 % compensated)
- IMU / baro: vibration-level gyro/accel noise, turn-on bias, bias random walk; baro drifts +-0.8 m


Manual flying (viewer): MANUAL [m] takes over.  ASSIST on = stick commands roll/pitch angle (the autopilot's attitude loop does
the rest - easiest); ASSIST off = sticks go straight to the control surfaces (ramp the throttle up, it has T/W 2.4 and
swings left on the take-off roll).  Arrows = elevator/aileron (down arrow = pull back = climb), a/d rudder, w/s throttle,
or drag in the CONTROLS panel.  Any auto button hands control back to the autopilot.

Wind: speed / direction (blows FROM) / turbulence / gusts sliders, live.  A windsock beside the runway and the compass in
the view show it.  Every run is random (airframe tolerances, mean wind, wander, gusts) but fully replayable via the seed.

Results (randomised aeroplanes: size x0.85-1.2, control-surface size x0.7-1.4, per-axis control power x0.8-1.25, thrust
x0.8-1.25, mass x0.9-1.2, wing area x0.85-1.15; random wind up to 4.5 m/s + turbulence; the FC is not told the plane):
- full mission (take-off, loop, hover, loop, hover, land), 32 unseen planes: 32/32 landed and stopped on the runway,
  0 crashes, mean lateral stop error ~1.6 m
- tuning (tools/tune.py, 21 candidates x 12 planes) picked the defaults in fc_default_config; on 48 unseen planes the
  tuned set landed 48/48 with a ~30 % shorter final than the hand-set values

Known limits: the tail-sitter hover is marginal in strong wind (the small fin weathervanes the nose into a sideways drift)
and a weak aeroplane can drift a long way downwind in it.  In the last 32-plane run 4 flights crossed the runway edge onto
the pilot's side while airborne: three by 0.4-3.5 m (end of the climb-out / joining final) and one far out (~70 m) whose
cause has not been traced yet - check tools/batch.py output before trusting the fence near people.
Gain priors (auth_q0 / auth_w0 in fc.cpp) come from ident.py on the nominal model; the lock-in calibration
(auth_lockin_cal*) was taken on the nominal sim plane.  t2.py, t3.py and t5.py poke at the old Python autopilot and no
longer run.


Airframe numbers (checked against typical 3D-foamie practice)
- Wing 1.0 m x 0.20 m = 0.20 m^2; ailerons are ~28 % chord over ~65 % of the semi-span (3D planes use 25-30 % chord, +-35-45 deg throw).
- Horizontal tail 0.068 m^2 = 34 % of wing area (flat-plate 3D stabs 25-35 %; sport planes ~20 %), elevator ~48 % chord (typical 3D 40-50 %).
- Fin + rudder 0.05 m^2 = 25 % of wing area (3D 15-25 %, sport 8-12 %), rudder ~50 % chord.  It used to be 0.09 m^2 (45 %), far too big.
- Thrust: 2.3 kg static at full throttle on a 0.82 kg airframe = T/W 2.85 (hover-capable 3D planes 2-3:1, sport planes 0.8-1.2:1).
- Ailerons run from 5 cm off the fuselage to the tip, 40 deg throw (effectiveness falls ~35 % at full throw); the roots sit in the prop wash for hover roll control.
Parked with the brakes on, tyre/brake friction (stick-slip bristle model) holds it in ~4 m/s of crosswind; in a strong
crosswind a light foamie weathervanes on its wheels, as a real one does.

Physics model (rcsim/airframe.py, v3)
- Mass: 0.82 kg summed from a parts list (motor, 4S 2200 pack, ESC, servos, foam, gear...).  CG ~28 % MAC.  Inertia tensor
  incl. Ixz from the same list: Ixx 0.013, Iyy 0.031, Izz 0.043 kg m^2 (the old hand-set values were about half that in
  pitch and yaw, which is why the old model rotated so readily).  Moving the battery moves the CG and the inertia.
- Rigid body: Newton-Euler in body axes incl. centripetal and gyroscopic (rotor angular momentum) terms.  Rotor is its own
  degree of freedom: motor torque, friction and prop torque act between rotor and airframe.
- Added mass: the flat plates drag air with them when they accelerate (+0.05 kg heave, +24 % roll inertia).  Only the
  acceleration part is used; the steady part is already in the plate coefficients.
- Aero: strip theory per wing/tail/fin panel with local flow (w x r, slipstream, swirl), finite-span lift slope, induced drag,
  Viterna-Corrigan post-stall (CN90 ~ 1.2), c.p. moving aft in stall, McCormick ground effect, lagged tail downwash.
- Slipstream: momentum-theory jet that contracts then spreads by mixing; each panel is split into the part in the jet and
  the part outside.  A 10" prop washes only ~25 cm of span (the old model assumed 68 cm), so hover authority is now
  aileron 26 / elevator 29 / rudder 14 rad/s^2 per unit stick (old: 85 / 168 / 110).  Swirl torque recovered by the
  surfaces is capped by angular-momentum conservation (<= 60 % of prop torque).
- Propulsion: battery sag solved together with the ESC (battery current = duty x motor current), motor friction always
  acts, prop windmills when the motor is off.  Static thrust 22.9 N (T/W 2.85), top speed ~25 m/s, best glide L/D ~5-7.
- Energy: `Airframe.energy(x)` gives translational, rotational, rotor, entrained-air and potential energy;
  `info['P_ext']` is the power of all external forces, so d(E)/dt = P_ext is checked in tests/physics_check.py.

