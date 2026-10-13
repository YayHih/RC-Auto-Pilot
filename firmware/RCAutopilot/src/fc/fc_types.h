/* fc_types.h -- the flight controller's whole interface to the outside world.
 *
 * The same structs are filled by the Heltec sketch (from the real sensor registers), by the Python
 * simulator (from emulated registers) and by the ARM emulator harness.  Only fixed-size types, no
 * pointers, natural alignment -> identical byte layout on the Cortex-M0+ and on a 64-bit PC.
 * Python mirrors these in rcsim/fcbridge.py; keep both in step (fc_struct_sizes() checks it).
 */
#ifndef FC_TYPES_H
#define FC_TYPES_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FC_TICK_HZ      100u          /* control loop rate                        */
#define FC_TICK_US      10000u
#define FC_NCH          5u            /* PCA9685 channels used: ail ele rud esc brake */

/* sensor-present flags in fc_in_t.flags */
#define FC_IN_IMU   0x01u
#define FC_IN_MAG   0x02u
#define FC_IN_BARO  0x04u
#define FC_IN_GPS   0x08u

/* ground-station commands (one LoRa byte) */
enum { FC_CMD_NONE = 0, FC_CMD_TAKEOFF, FC_CMD_HOVER, FC_CMD_LOOP, FC_CMD_LAND, FC_CMD_CRUISE };

/* RC receiver mode switch (CPPM channel 5) */
enum { FC_RC_AUTO = 0, FC_RC_DIRECT = 1, FC_RC_ASSIST = 2, FC_RC_NONE = 255 };

/* autopilot states (fc_out_t.state) -- names in fcbridge.STATE_NAMES */
enum {
    ST_INIT = 0, ST_GROUND, ST_TAKEOFF, ST_CRUISE, ST_LOOP_PREP, ST_LOOP, ST_TO_HOVER, ST_HOVER,
    ST_FROM_HOVER, ST_LAND, ST_FINAL, ST_FLARE, ST_ROLLOUT, ST_GO_AROUND, ST_MANUAL, ST_COUNT
};

typedef struct {
    uint32_t t_us;          /* micros() at the start of this tick                         */
    uint8_t  flags;         /* FC_IN_* : which sensor blocks below are new this tick      */
    uint8_t  cmd;           /* FC_CMD_*                                                   */
    uint8_t  rc_mode;       /* FC_RC_*                                                    */
    uint8_t  _pad0;
    int16_t  acc[3];        /* MPU6050 ACCEL_XOUT..: +-8 g, 4096 LSB/g, sensor axes = body FRD */
    int16_t  gyro[3];       /* MPU6050 GYRO_XOUT..: +-1000 dps, 32.8 LSB/(deg/s)          */
    int16_t  mag[3];        /* QMC5883L X,Y,Z: +-8 G, 3000 LSB/G, body FRD                 */
    int16_t  rc_us[4];      /* CPPM pulse widths us: ail, ele, thr, rud                   */
    uint32_t baro_q8;       /* BMP280 compensated pressure, Pa * 256                     */
    int32_t  lat_e7;        /* u-blox UBX-NAV-PVT subset                                  */
    int32_t  lon_e7;
    int32_t  hmsl_mm;
    int32_t  vel_mm_s[3];   /* N, E, D                                                    */
    uint32_t hacc_mm;
    uint8_t  fix_type;      /* 3 = 3D                                                     */
    uint8_t  num_sv;
    uint16_t _pad1;
} fc_in_t;

typedef struct {
    uint16_t pca_off[8];    /* PCA9685 LEDn_OFF counts (ON = 0), ch0..FC_NCH-1 used        */
    uint8_t  state, goal, rc_mode, flags;   /* flags: bit0 armed, bit1 gps ok, bit2 wind ok */
    float    pos[3], vel[3];        /* NED from home, m and m/s                          */
    float    q[4];                  /* attitude estimate, body->NED                      */
    float    rates[3];              /* bias-corrected gyro, rad/s                        */
    float    wind[2];               /* estimated wind N,E (m/s, direction air moves to)  */
    float    wind_sd;               /* 1-sigma of the wind estimate                      */
    float    airspeed;              /* estimated true airspeed m/s                       */
    float    auth_q[3];             /* learned control power per Pa / nominal (1 = as nominal) */
    float    auth_w[3];             /* learned prop-wash control power / nominal         */
    float    cmd[5];                /* da de dr thr brake, normalised (what goes out)    */
    float    land_dir;              /* landing direction, rad (NED azimuth)              */
    float    rwy_hdg;               /* runway heading captured at take-off               */
    uint32_t tick;                  /* tick counter                                       */
} fc_out_t;

typedef struct {
    /* PCA9685 */
    float    pca_osc_hz;            /* internal oscillator (nominal 25 MHz; measure yours)     */
    uint8_t  pca_prescale;          /* 121 -> ~50 Hz                                           */
    uint8_t  _pad0[3];
    /* per output channel: centre / full-throw pulse, us; reverse; ch3 (ESC) uses min..max   */
    int16_t  ch_center_us[8];
    int16_t  ch_throw_us[8];        /* us for full (+-1) deflection                            */
    int8_t   ch_reverse[8];         /* +1 / -1                                                 */
    /* sensors */
    float    mag_decl_deg;          /* magnetic declination at the field                      */
    float    mag_hard_iron_uT[3];   /* calibration offsets                                    */
    float    mag_motor_uT[3];       /* motor-current field at full throttle (calibrated)       */
    /* airframe priors (from ident.py on the nominal model) -- adapted in flight            */
    float    auth_q0[3];            /* rad/s^2 per Pa per unit command                         */
    float    auth_w0[3];            /* rad/s^2 per unit command in hover                       */
    float    auth_lockin_cal[3];    /* what the in-flight lock-in reads on a plane whose authority
                                       equals the priors (servo lag/backlash, damping at the dither
                                       frequency); calibrated once in the simulator: cruise...  */
    float    auth_lockin_cal_w[3];  /* ...and hover (prop-wash) conditions                     */
    float    thrust_max_n;          /* static thrust at full throttle                          */
    float    weight_n;
    float    thr_hover;             /* initial guess of the hover throttle                     */
    /* field */
    float    rwy_len_ahead_m;       /* runway length ahead of the take-off point               */
    float    rwy_len_behind_m;      /* and behind it                                           */
    float    rwy_hdg_deg;           /* runway heading if known (true north), < 0 = take the
                                       line-up heading at take-off (compass, corrected in flight) */
    float    cruise_alt_m, cruise_v;
    float    pattern_alt_m;         /* height the landing circuit is flown at                 */
    float    v_approach;            /* final approach airspeed                                */
    float    glide_deg;             /* final approach path angle                              */
    /* safety: the pilot stands on one side of the runway; all manoeuvring stays on the other  */
    float    fence_m;               /* min distance from the centreline for orbits, loops, hover */
    float    hold_r_m;              /* landing holding-circle radius                          */
    float    flare_h_m;             /* height the flare starts at                             */
    float    flare_tau_s;           /* sink-rate time constant in the flare                   */
    float    bw_fwd[3];             /* rate-loop bandwidths in forward flight (rad/s)         */
    float    bw_hover[3];           /* ... and in the hover                                   */
    float    att_kp;                /* attitude loop gain (1/s)                               */
    uint8_t  adapt_enable;          /* learn control authority in flight                      */
    uint8_t  brake_enable;          /* ch4 drives a wheel-brake servo                         */
    int8_t   fly_side;              /* +1: fly right of the runway heading, -1: left          */
    uint8_t  _pad1;
} fc_config_t;

#ifdef __cplusplus
}
#endif
#endif
