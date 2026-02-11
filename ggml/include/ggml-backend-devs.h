#include "ggml.h"

#define XPU_PLUGIN_NAME_SIZE 10

typedef struct card_id {
        unsigned int vendor_id;
        unsigned int device_id;
       	char plugin_name[XPU_PLUGIN_NAME_SIZE];
} card_id_t;

GGML_API int find_devs_via_sysfs(char *plugin);
