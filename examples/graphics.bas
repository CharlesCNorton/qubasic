' QBASIC graphics, run unchanged: qubasic examples/graphics.bas
' Draws in SCREEN 13 (320 x 200, 256 colours); the picture shows when the
' program ends, and SAVEPNG picture.png SCREEN at the prompt saves it.
SCREEN 13
CONST PI = 3.141593
LINE (0, 0)-(319, 199), 1, BF
FOR r = 90 TO 10 STEP -10
    CIRCLE (160, 100), r, 32 + r \ 10
    PAINT (160 + r - 5, 100), 32 + r \ 10, 32 + r \ 10
NEXT r
FOR x = 0 TO 319
    y = 100 - 60 * SIN(x * PI / 80)
    PSET (x, y), 15
NEXT x
LINE (10, 10)-(60, 40), 14, B
DRAW "BM20,25 C12 E5 F5 G5 H5"
PRINT "colour at the centre:"; POINT(160, 100)
