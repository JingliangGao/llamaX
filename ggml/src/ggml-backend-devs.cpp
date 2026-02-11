#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>

#include "ggml-backend-devs.h"

static const card_id_t card_ids[] = {
        { 0x6766,    0x3d06,    "ponn"},    //Glenfly Arise-GT-10C0t
        { 0x6766,    0x3d07,    "ponn"},    //Glenfly Arise-GT-2030
        { 0x10de,    0x2231,    "cuda"},    //NVIDIA RTX A5000
        { 0x10de,    0x20f1,    "cuda"},    //NVIDIA A100
        { 0x10de,    0x2685,    "cuda"},    //NVIDIA GeForce RTX 4090 D
        { 0x10de,    0x2702,    "cuda"},    //NVIDIA GeForce RTX 4080
        { 0x10de,    0x2704,    "cuda"},    //NVIDIA GeForce RTX 4080
        { 0x10de,    0x2504,    "cuda"},    //NVIDIA GeForce RTX 3060
        { 0x10de,    0x21c4,    "cuda"},    //NVIDIA GeForce GTX 1660 SUPER
        { 0x1002,    0x744c,    "hip"},     //AMD Radeon RX 7900 XTX
        { 0x1002,    0x7550,    "hip"},     //AMD Radeon RX 9070
        { 0x8086,    0x7d55,    "sycl"},    //Intel Arc Graphics
        { 0x0,       0x0,       ""},
};

int find_devs_via_sysfs(char *plugin) {
    DIR *dir;
    struct dirent *entry;
    char path[256], vendor_path[256], device_path[256];
    unsigned int vendor_id, device_id;
    FILE *fp;

    const card_id_t *p = card_ids;

    // 打开PCI设备目录
    dir = opendir("/sys/bus/pci/devices/");
    if (!dir) return 0;

    while ((entry = readdir(dir)) != NULL) {
        // 跳过 "." 和 ".."
        if (entry->d_name[0] == '.') continue;

        // 构建 vendor 文件路径
        snprintf(vendor_path, sizeof(vendor_path),
                 "/sys/bus/pci/devices/%s/vendor", entry->d_name);
        // 构建 device 文件路径
        snprintf(device_path, sizeof(device_path),
                "/sys/bus/pci/devices/%s/device", entry->d_name);

        // 读取 vendor ID
        fp = fopen(vendor_path, "r");
        if (!fp) {
	    fprintf(stderr, "ERROR: open file %s failed\n", vendor_path);
            continue;
        }
        if (fscanf(fp, "%x", &vendor_id) != 1) {
            fclose(fp);
            continue;
        }
        fclose(fp);

	// 读取 device ID
        fp = fopen(device_path, "r");
        if (!fp) {
            fprintf(stderr, "ERROR: open file %s failed\n", device_path);
            continue;
        }
        if (fscanf(fp, "%x", &device_id) != 1) {
            fclose(fp);
            continue;
        }
        fclose(fp);

        // 与名单匹配
        p = card_ids;
        while (p && p->vendor_id != 0) {
            if (vendor_id == p->vendor_id && device_id == p->device_id) {
                memcpy(plugin, p->plugin_name, XPU_PLUGIN_NAME_SIZE);
                closedir(dir);
                return 1;
            }
            p++;
        }
    }
    closedir(dir);
    return 0;
}
