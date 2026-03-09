#include "./header.h"
#include "./bmp_writer.c"

#define FIELD_HEIGHT 1024
#define FIELD_WIDTH 1024

cell_t ** create_field(uint32_t h, uint32_t w) {
    cell_t **field = calloc(h, sizeof(cell_t*));
    for(uint32_t i=0; i<w; i++) {
        field[i] = calloc(w, sizeof(cell_t));
    }
    return field;
}

void fill_field(cell_t **field, uint32_t h, uint32_t w) {
    for(uint32_t y=0; y<h; y++) {
        for(uint32_t x=0; x<w; x++) {
            field[y][x].R = 0xFF;
        }
    }
}

int main() {
    cell_t **field = create_field(FIELD_HEIGHT, FIELD_WIDTH);
    fill_field(field, FIELD_HEIGHT, FIELD_WIDTH);
    save_bmp("output.bmp", FIELD_WIDTH, FIELD_HEIGHT, field);
    return 0;
}
