#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "./header.h"

#pragma pack(push, 1)
typedef struct {
    uint16_t type;          // Magic identifier: "BM"
    uint32_t size;          // File size in bytes
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;        // Offset to image data
} BMPHeader;

typedef struct {
    uint32_t size;          // Header size
    int32_t  width;
    int32_t  height;
    uint16_t planes;
    uint16_t bit_count;     // Bits per pixel (24 for RGB)
    uint32_t compression;
    uint32_t image_size;
    int32_t  x_resolution;
    int32_t  y_resolution;
    uint32_t colors_used;
    uint32_t colors_important;
} BMPInfoHeader;
#pragma pack(pop)

void save_bmp(const char *filename, int width, int height, cell_t **data) {
    // 1. Calculate padding: Each row must be a multiple of 4 bytes
    int row_size = (width * 3 + 3) & ~3; 
    int padding = row_size - (width * 3);
    uint32_t image_size = row_size * height;

    BMPHeader header = {0x4D42, 54 + image_size, 0, 0, 54};
    BMPInfoHeader info = {40, width, height, 1, 24, 0, image_size, 2835, 2835, 0, 0};

    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fwrite(&header, sizeof(header), 1, f);
    fwrite(&info, sizeof(info), 1, f);

    // 2. Write pixel data (Bottom-to-top)
    uint8_t pad_bytes[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            // BMP uses BGR order
            uint8_t bgr[3] = { data[y][x].B, data[y][x].G, data[y][x].R };
            fwrite(bgr, 3, 1, f);
        }
        // Write padding
        fwrite(pad_bytes, 1, padding, f);
    }

    fclose(f);
}