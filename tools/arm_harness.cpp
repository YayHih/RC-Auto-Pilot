/* arm_harness.cpp -- entry points for running the Cortex-M0+ build of the flight controller in the
 * Unicorn emulator (rcsim/fcbridge.py ArmFC).  Not part of the Arduino sketch. */
#include "fc.h"

extern "C" {
fc_config_t fc_arm_cfg;
fc_in_t fc_arm_in;
fc_out_t fc_arm_out;

void fc_arm_defaults(void) { fc_default_config(&fc_arm_cfg); }
void fc_arm_init(void) { fc_init(&fc_arm_cfg); }
void fc_arm_tick(void) { fc_tick(&fc_arm_in, &fc_arm_out); }
}
