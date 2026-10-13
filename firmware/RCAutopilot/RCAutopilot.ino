/*
 * RCAutopilot -- Heltec CubeCell HTCC-AB01 (ASR6502, Cortex-M0+ 48 MHz) flight controller.
 *
 * This file is only the hardware layer.  All estimation, guidance and control is in src/fc/ -- the
 * very same code the simulator runs (rcsim/fcbridge.py), so what flew in the sim is what flies here.
 * fc_tick() gets raw sensor register values and the micros() clock and returns PCA9685 counts.
 *
 * Wiring
 *   I2C SDA/SCL (400 kHz)  MPU6050 0x68 | BMP280 0x76 | QMC5883L 0x0D (on the GPS module) | PCA9685 0x40
 *   UART (the AB01's only one, also the USB bridge)  u-blox M10 GPS, 38400 baud
 *        -> no USB serial while the GPS is connected; build with BENCH_DEBUG for bench prints
 *   GPIO5                  CPPM from the RC receiver: ch1 ail, ch2 ele, ch3 thr, ch4 rud,
 *                          ch5 mode switch  (<1300 us auto, 1300..1700 assist, >1700 direct)
 *   PCA9685 ch0..ch4       aileron (Y-lead), elevator, rudder + nose wheel, ESC, wheel-brake servo
 *   on-board SX1262 LoRa   ground-station buttons, one byte: 1 takeoff 2 hover 3 loop 4 land 5 cruise
 *   Vext                   switched 3.3 V for the sensors (LOW = on)
 *
 * The AB01 has only two hardware PWM pins, hence the PCA9685: it makes the servo pulses by itself
 * (50 Hz, 12-bit) so the CPU only writes 20 bytes per tick.
 *
 * Mounting: the FC expects body axes x forward, y right, z down.  Set the remaps below to match how
 * the IMU and compass boards are mounted.
 */
#include <Arduino.h>
#include <Wire.h>
#include "LoRaWan_APP.h"
#include "src/fc/fc.h"

// #define BENCH_DEBUG          // no GPS: prints loop timing on USB serial instead

/* ------------------------------------------------------------------ sensor axis remaps */
/* MPU6050 board flat, component side up, chip X forward: chip Y points left, chip Z up */
#define IMU_X(a) ((int16_t)(a)[0])
#define IMU_Y(a) ((int16_t)-(a)[1])
#define IMU_Z(a) ((int16_t)-(a)[2])
/* QMC5883L on the GPS module, arrow (X) forward, module flat */
#define MAG_X(m) ((int16_t)(m)[0])
#define MAG_Y(m) ((int16_t)-(m)[1])
#define MAG_Z(m) ((int16_t)-(m)[2])

#define ADDR_MPU  0x68
#define ADDR_BMP  0x76
#define ADDR_QMC  0x0D
#define ADDR_PCA  0x40
#define PIN_CPPM  GPIO5

#define RF_FREQUENCY   915000000
#define LORA_BANDWIDTH 0
#define LORA_SF        7
#define LORA_CODERATE  1
#define LORA_PREAMBLE  8

static fc_config_t cfg;
static fc_in_t fin;
static fc_out_t fout;

/* ------------------------------------------------------------------ I2C helpers */
static void i2c_w8(uint8_t addr, uint8_t reg, uint8_t v)
{
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.write(v);
    Wire.endTransmission();
}

static bool i2c_read(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t n)
{
    Wire.beginTransmission(addr);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((int)addr, (int)n) != n) return false;
    for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
    return true;
}

/* ------------------------------------------------------------------ MPU6050 */
static void mpu_init()
{
    i2c_w8(ADDR_MPU, 0x6B, 0x01);   /* wake, PLL on gyro X */
    delay(10);
    i2c_w8(ADDR_MPU, 0x19, 0x00);   /* 1 kHz sample rate (DLPF on) */
    i2c_w8(ADDR_MPU, 0x1A, 0x03);   /* DLPF 44 Hz accel / 42 Hz gyro */
    i2c_w8(ADDR_MPU, 0x1B, 0x10);   /* +-1000 dps -> 32.8 LSB/(deg/s) */
    i2c_w8(ADDR_MPU, 0x1C, 0x10);   /* +-8 g      -> 4096 LSB/g */
}

static bool mpu_read(int16_t acc[3], int16_t gyro[3])
{
    uint8_t b[14];
    if (!i2c_read(ADDR_MPU, 0x3B, b, 14)) return false;
    int16_t a[3] = {(int16_t)(b[0] << 8 | b[1]), (int16_t)(b[2] << 8 | b[3]), (int16_t)(b[4] << 8 | b[5])};
    int16_t g[3] = {(int16_t)(b[8] << 8 | b[9]), (int16_t)(b[10] << 8 | b[11]), (int16_t)(b[12] << 8 | b[13])};
    acc[0] = IMU_X(a); acc[1] = IMU_Y(a); acc[2] = IMU_Z(a);
    gyro[0] = IMU_X(g); gyro[1] = IMU_Y(g); gyro[2] = IMU_Z(g);
    return true;
}

/* ------------------------------------------------------------------ QMC5883L */
static void qmc_init()
{
    i2c_w8(ADDR_QMC, 0x0B, 0x01);   /* SET/RESET period */
    i2c_w8(ADDR_QMC, 0x09, 0x15);   /* continuous, 50 Hz, +-8 G (3000 LSB/G), OSR 512 */
}

static bool qmc_read(int16_t mag[3])
{
    uint8_t b[6];
    if (!i2c_read(ADDR_QMC, 0x00, b, 6)) return false;
    int16_t m[3] = {(int16_t)(b[1] << 8 | b[0]), (int16_t)(b[3] << 8 | b[2]), (int16_t)(b[5] << 8 | b[4])};
    mag[0] = MAG_X(m); mag[1] = MAG_Y(m); mag[2] = MAG_Z(m);
    return true;
}

/* ------------------------------------------------------------------ BMP280 (Bosch integer compensation) */
static uint16_t dig_T1, dig_P1;
static int16_t dig_T2, dig_T3, dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9;

static void bmp_init()
{
    uint8_t c[24];
    i2c_read(ADDR_BMP, 0x88, c, 24);
    dig_T1 = c[1] << 8 | c[0]; dig_T2 = c[3] << 8 | c[2]; dig_T3 = c[5] << 8 | c[4];
    dig_P1 = c[7] << 8 | c[6]; dig_P2 = c[9] << 8 | c[8]; dig_P3 = c[11] << 8 | c[10];
    dig_P4 = c[13] << 8 | c[12]; dig_P5 = c[15] << 8 | c[14]; dig_P6 = c[17] << 8 | c[16];
    dig_P7 = c[19] << 8 | c[18]; dig_P8 = c[21] << 8 | c[20]; dig_P9 = c[23] << 8 | c[22];
    i2c_w8(ADDR_BMP, 0xF5, 0x08);   /* standby 0.5 ms, IIR filter x4 */
    i2c_w8(ADDR_BMP, 0xF4, 0x57);   /* T x2, P x16, normal mode */
}

/* pressure in Pa * 256 (datasheet 64-bit routine) */
static bool bmp_read(uint32_t *p_q8)
{
    uint8_t b[6];
    if (!i2c_read(ADDR_BMP, 0xF7, b, 6)) return false;
    int32_t adc_P = (int32_t)b[0] << 12 | (int32_t)b[1] << 4 | b[2] >> 4;
    int32_t adc_T = (int32_t)b[3] << 12 | (int32_t)b[4] << 4 | b[5] >> 4;
    int32_t v1 = ((((adc_T >> 3) - ((int32_t)dig_T1 << 1))) * (int32_t)dig_T2) >> 11;
    int32_t v2 = (((((adc_T >> 4) - (int32_t)dig_T1) * ((adc_T >> 4) - (int32_t)dig_T1)) >> 12) * (int32_t)dig_T3) >> 14;
    int32_t t_fine = v1 + v2;
    int64_t a1 = (int64_t)t_fine - 128000;
    int64_t a2 = a1 * a1 * (int64_t)dig_P6;
    a2 = a2 + ((a1 * (int64_t)dig_P5) << 17);
    a2 = a2 + ((int64_t)dig_P4 << 35);
    a1 = ((a1 * a1 * (int64_t)dig_P3) >> 8) + ((a1 * (int64_t)dig_P2) << 12);
    a1 = (((((int64_t)1) << 47) + a1)) * (int64_t)dig_P1 >> 33;
    if (a1 == 0) return false;
    int64_t p = 1048576 - adc_P;
    p = (((p << 31) - a2) * 3125) / a1;
    a1 = ((int64_t)dig_P9 * (p >> 13) * (p >> 13)) >> 25;
    a2 = ((int64_t)dig_P8 * p) >> 19;
    p = ((p + a1 + a2) >> 8) + ((int64_t)dig_P7 << 4);
    *p_q8 = (uint32_t)p;
    return true;
}

/* ------------------------------------------------------------------ PCA9685 */
static void pca_init(uint8_t prescale)
{
    i2c_w8(ADDR_PCA, 0x00, 0x10);           /* sleep to set the prescaler */
    i2c_w8(ADDR_PCA, 0xFE, prescale);
    i2c_w8(ADDR_PCA, 0x00, 0x20);           /* wake, register auto-increment */
    delay(1);
    i2c_w8(ADDR_PCA, 0x00, 0xA0);           /* restart */
    i2c_w8(ADDR_PCA, 0x01, 0x04);           /* totem-pole outputs */
}

static void pca_write(const uint16_t *off, uint8_t n)
{
    Wire.beginTransmission(ADDR_PCA);
    Wire.write(0x06);                       /* LED0_ON_L, auto-increment through the channels */
    for (uint8_t i = 0; i < n; i++) {
        Wire.write(0); Wire.write(0);
        Wire.write(off[i] & 0xFF); Wire.write(off[i] >> 8);
    }
    Wire.endTransmission();
}

/* ------------------------------------------------------------------ u-blox M10 (UBX) */
static void ubx_send(uint8_t cls, uint8_t id, const uint8_t *pl, uint16_t len)
{
    uint8_t ca = 0, cb = 0, h[4] = {cls, id, (uint8_t)len, (uint8_t)(len >> 8)};
    Serial.write(0xB5); Serial.write(0x62);
    for (int i = 0; i < 4; i++) { Serial.write(h[i]); ca += h[i]; cb += ca; }
    for (uint16_t i = 0; i < len; i++) { Serial.write(pl[i]); ca += pl[i]; cb += ca; }
    Serial.write(ca); Serial.write(cb);
}

static void gps_init()
{
    Serial.begin(38400);
    delay(200);
    /* UBX-CFG-VALSET (RAM): 10 Hz, NAV-PVT on UART1, NMEA off, airborne <4 g dynamic model */
    static const uint8_t v[] = {
        0x00, 0x01, 0x00, 0x00,
        0x01, 0x00, 0x21, 0x30, 100, 0,     /* CFG-RATE-MEAS = 100 ms           */
        0x07, 0x00, 0x91, 0x20, 1,          /* CFG-MSGOUT-UBX_NAV_PVT_UART1 = 1 */
        0x02, 0x00, 0x74, 0x10, 0,          /* CFG-UART1OUTPROT-NMEA = 0        */
        0x21, 0x00, 0x11, 0x20, 8,          /* CFG-NAVSPG-DYNMODEL = airborne4g */
    };
    ubx_send(0x06, 0x8A, v, sizeof(v));
}

static uint8_t ubx_buf[100];
static uint16_t ubx_len, ubx_pos;
static uint8_t ubx_state, ubx_cls, ubx_id, ubx_ca, ubx_cb;
static volatile bool gps_new;

static int32_t rd32(const uint8_t *p) { return (int32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24); }

static void ubx_frame()
{
    if (ubx_cls != 0x01 || ubx_id != 0x07 || ubx_len < 92) return;      /* NAV-PVT */
    const uint8_t *p = ubx_buf;
    fin.fix_type = (p[21] & 0x01) ? p[20] : 0;                           /* gnssFixOK */
    fin.num_sv = p[23];
    fin.lon_e7 = rd32(p + 24);
    fin.lat_e7 = rd32(p + 28);
    fin.hmsl_mm = rd32(p + 36);
    fin.hacc_mm = (uint32_t)rd32(p + 40);
    fin.vel_mm_s[0] = rd32(p + 48);
    fin.vel_mm_s[1] = rd32(p + 52);
    fin.vel_mm_s[2] = rd32(p + 56);
    gps_new = true;
}

static void gps_poll()
{
    while (Serial.available()) {
        uint8_t c = Serial.read();
        switch (ubx_state) {
        case 0: ubx_state = (c == 0xB5); break;
        case 1: ubx_state = (c == 0x62) ? 2 : 0; break;
        case 2: ubx_cls = c; ubx_ca = c; ubx_cb = c; ubx_state = 3; break;
        case 3: ubx_id = c; ubx_ca += c; ubx_cb += ubx_ca; ubx_state = 4; break;
        case 4: ubx_len = c; ubx_ca += c; ubx_cb += ubx_ca; ubx_state = 5; break;
        case 5: ubx_len |= (uint16_t)c << 8; ubx_ca += c; ubx_cb += ubx_ca; ubx_pos = 0;
                ubx_state = ubx_len > sizeof(ubx_buf) ? 0 : (ubx_len ? 6 : 7); break;
        case 6: ubx_buf[ubx_pos++] = c; ubx_ca += c; ubx_cb += ubx_ca; if (ubx_pos >= ubx_len) ubx_state = 7; break;
        case 7: ubx_state = (c == ubx_ca) ? 8 : 0; break;
        case 8: if (c == ubx_cb) ubx_frame(); ubx_state = 0; break;
        }
    }
}

/* ------------------------------------------------------------------ CPPM receiver input */
static volatile uint16_t cppm[8];
static volatile uint8_t cppm_ch;
static volatile uint32_t cppm_last, cppm_frame_t;

static void cppm_isr()
{
    uint32_t now = micros();
    uint32_t w = now - cppm_last;
    cppm_last = now;
    if (w > 3000) { cppm_ch = 0; cppm_frame_t = now; return; }      /* sync gap */
    if (cppm_ch < 8 && w > 800 && w < 2200) cppm[cppm_ch] = (uint16_t)w;
    cppm_ch++;
}

/* ------------------------------------------------------------------ LoRa ground-station commands */
static RadioEvents_t radio_ev;
static volatile uint8_t lora_cmd;

static void on_rx(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr)
{
    (void)rssi; (void)snr;
    if (size >= 1 && payload[0] >= FC_CMD_TAKEOFF && payload[0] <= FC_CMD_CRUISE) lora_cmd = payload[0];
    Radio.Rx(0);
}
static void on_rx_err() { Radio.Rx(0); }

static void lora_init()
{
    radio_ev.RxDone = on_rx;
    radio_ev.RxTimeout = on_rx_err;
    radio_ev.RxError = on_rx_err;
    Radio.Init(&radio_ev);
    Radio.SetChannel(RF_FREQUENCY);
    Radio.SetRxConfig(MODEM_LORA, LORA_BANDWIDTH, LORA_SF, LORA_CODERATE, 0, LORA_PREAMBLE,
                      0, false, 0, true, 0, 0, false, true);
    Radio.Rx(0);
}

/* ------------------------------------------------------------------ main */
static uint32_t next_tick, tick_n, loop_max_us;

void setup()
{
    pinMode(Vext, OUTPUT);
    digitalWrite(Vext, LOW);                /* sensor power */
    delay(100);
    Wire.begin();
    Wire.setClock(400000);
    fc_default_config(&cfg);
    /* field / aeroplane specific settings go here, e.g. cfg.rwy_hdg_deg = 87.0f; */
    pca_init(cfg.pca_prescale);
    mpu_init();
    qmc_init();
    bmp_init();
#ifdef BENCH_DEBUG
    Serial.begin(115200);
#else
    gps_init();
#endif
    pinMode(PIN_CPPM, INPUT);
    attachInterrupt(digitalPinToInterrupt(PIN_CPPM), cppm_isr, RISING);
    lora_init();
    fc_init(&cfg);
    memset(&fin, 0, sizeof(fin));
    next_tick = micros() + FC_TICK_US;
}

static uint8_t rc_mode()
{
    if ((uint32_t)(micros() - cppm_frame_t) > 100000) return FC_RC_NONE;   /* receiver lost: autopilot */
    uint16_t sw = cppm[4];
    return sw < 1300 ? FC_RC_AUTO : (sw > 1700 ? FC_RC_DIRECT : FC_RC_ASSIST);
}

static void fc_step(uint32_t now)
{
    uint8_t flags = 0;
    if (mpu_read(fin.acc, fin.gyro)) flags |= FC_IN_IMU;
    if ((tick_n & 1) == 0 && qmc_read(fin.mag)) flags |= FC_IN_MAG;
    if ((tick_n & 3) == 0 && bmp_read(&fin.baro_q8)) flags |= FC_IN_BARO;
    if (gps_new) { gps_new = false; flags |= FC_IN_GPS; }
    fin.t_us = now;
    fin.flags = flags;
    fin.cmd = lora_cmd;
    lora_cmd = 0;
    fin.rc_mode = rc_mode();
    fin.rc_us[0] = cppm[0]; fin.rc_us[1] = cppm[1]; fin.rc_us[2] = cppm[2]; fin.rc_us[3] = cppm[3];
    fc_tick(&fin, &fout);
    pca_write(fout.pca_off, FC_NCH);
    tick_n++;
}

void loop()
{
    uint32_t now = micros();
    if ((int32_t)(now - next_tick) >= 0) {
        next_tick += FC_TICK_US;
        if ((int32_t)(now - next_tick) > (int32_t)FC_TICK_US) next_tick = now + FC_TICK_US;   /* overrun: resync */
        fc_step(now);
        uint32_t dt = micros() - now;
        if (dt > loop_max_us) loop_max_us = dt;
#ifdef BENCH_DEBUG
        if ((tick_n % 100) == 0) {
            Serial.printf("tick %lu  loop max %lu us  state %u  pca %u %u %u %u\n", (unsigned long)tick_n,
                          (unsigned long)loop_max_us, fout.state, fout.pca_off[0], fout.pca_off[1], fout.pca_off[2], fout.pca_off[3]);
            loop_max_us = 0;
        }
#endif
    }
#ifndef BENCH_DEBUG
    gps_poll();
#endif
    Radio.IrqProcess();
}
