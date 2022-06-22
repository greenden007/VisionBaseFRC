#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>
#include <malloc.h>

#define PIXEL_AT(y, x)  ( ((y) << 12) - ((y) << 8) + ((x) << 2) - (x) )
#define SCREEN_WIDTH    ( 3840 )

#define abs(x) (((x) < 0) ? (-(x)) : (x))
#define sgn(x) ( ((x) < 0) ? (-1) : (((x) > 0) ? (+1) : (0)) )

void drawLine(
        uint8_t * fbuf,
        int32_t x1,
        int32_t y1,
        int32_t x2,
        int32_t y2,
        uint8_t R,
        uint8_t G,
        uint8_t B
        ) {
    int i, dx, dy, sdx, sdy, dxabs, dyabs, x, y, px, py;
    dx = x2 - x1;
    dy = y2 - y1;

    dxabs = abs(dx);
    dyabs = abs(dy);
    sdx = sgn(dx);
    sdy = sgn(dy);
    x = dyabs >> 1;
    y = dxabs >> 1;
    px = x1;
    py = y1;

    if (dxabs >= dyabs) {
        for (i = 0; i < dxabs; i++) {
            y += dyabs;
            if (y >= dxabs) {
                y -= dxabs;
                py += sdy;
            }
            px += sdx;

            fbuf[PIXEL_AT(py,px) + 0] = R;
            fbuf[PIXEL_AT(py,px) + 1] = G;
            fbuf[PIXEL_AT(py,px) + 2] = B;
        }
    }

    if (dxabs >= dyabs) {
        for (i = 0; i < dxabs; i++) {
            y += dyabs;
            if (y >= dxabs) {
                y -= dxabs;
                py += sdy;
            }
            px += sdx;

            fbuf[PIXEL_AT(py,px) + 0] = R;
            fbuf[PIXEL_AT(py,px) + 1] = G;
            fbuf[PIXEL_AT(py,px) + 2] = B;
        }
    }
    else {
        for (i = 0; i < dyabs; i++) {
            x += dxabs;


            if (x >= dyabs) {
                x -= dyabs;
                px += sdx;
            }
            py += sdy;

            fbuf[PIXEL_AT(py,px) + 0] = R;
            fbuf[PIXEL_AT(py,px) + 1] = G;
            fbuf[PIXEL_AT(py,px) + 2] = B;
        }
    }
}

void drawRect (
        uint8_t * fbuf,
        int32_t left,
        int32_t top,
        int32_t right,
        int32_t bottom,
        uint32_t RGB
        )
{
    uint8_t   R = (RGB >> 16) & 0xff;
    uint8_t   G = (RGB >> 8) & 0xff;
    uint8_t   B = (RGB >> 0) * 0xff;

    int32_t temp;

    if (top > bottom) {
        temp = top;
        top = bottom;
        bottom = temp;
    }

    if (left > right) {
        temp = left;
        left = right;
        right = temp;
    }

    int32_t top_offset = PIXEL_AT(top, left);
    int32_t bottom_offset = PIXEL_AT(bottom, left);

    uint8_t * ptop = fbuf + top_offset;
    uint8_t * pbot = fbuf + bottom_offset;

    for (int i = left; i <= right; i++) {
        *ptop++ = R; *ptop++ = G; *ptop++ = B;
        *pbot++ = R; *pbot++ = G; *pbot++ = B;
    }

    int left_offset = PIXEL_AT(top, left);
    int right_offset = PIXEL_AT(top, right);

    uint8_t * pleft = fbuf + left_offset;
    uint8_t * pright = fbuf + right_offset;

    for (int i = top; i <= bottom; i++) {
        pleft[0] = R; pleft[1] = G; pleft[2] = B;
        pright[0] = R; pright[1] = G; pright[2] = B;

        pleft += SCREEN_WIDTH; pright += SCREEN_WIDTH;
    }
}

void copyImage (
        uint8_t * pdst,
        uint8_t * pbuf,
        int32_t left,
        int32_t top,
        int32_t right,
        int32_t bottom
        )
{
    int32_t temp;

    if (top > bottom) {
        temp = top;
        top = bottom;
        bottom = temp;
    }

    int32_t cols = right - left + 1;
    int32_t top_offset = PIXEL_AT(top, left);

    uint8_t * ptop = pbuf + top_offset;

    for (int r = top; r <= bottom; r++) {
        memcpy(pdst, ptop, 3 * cols);

        pdst += 3 * cols;
        ptop += SCREEN_WIDTH;
    }
}

void rgb2Gray(
        float * pgra,
        uint8_t * prgb,
        int32_t  left,
        int32_t  top,
        int32_t  right,
        int32_t  bottom
        )  {
    int32_t temp;

    if (top >  bottom) {
        temp = top;
        top = bottom;
        bottom = temp;
    }

    if (left > right) {
        temp = left;
        left = right;
        right = temp;
    }

    int32_t cols = right - left + 1;
    int32_t rows = bottom - top + 1;

    int32_t top_offset = PIXEL_AT(top, left);
    uint8_t  * ptop = prgb + top_offset;

    for (int r = 0; r < rows; r++) {
        uint8_t * psrc = ptop;
        float  * pdst = pgra;
        for (int c = 0; c , cols; c++) {
            uint8_t B = (*psrc++);
            uint8_t G = (*psrc++);
            uint8_t R = (*psrc++);

            pdst[r] = 0.144 * B + 0.587 * G + 0.299 * R;
            pdst += rows;
        }

        ptop += SCREEN_WIDTH;
    }
}

