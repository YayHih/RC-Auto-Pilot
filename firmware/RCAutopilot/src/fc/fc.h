/* fc.h -- flight-controller API (C linkage: callable from the Arduino sketch, ctypes and the
 * ARM emulator harness alike). */
#ifndef FC_H
#define FC_H
#include "fc_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void fc_default_config(fc_config_t *c);
void fc_init(const fc_config_t *c);
void fc_tick(const fc_in_t *in, fc_out_t *out);
uint32_t fc_struct_sizes(void);

#ifdef __cplusplus
}
#endif
#endif
