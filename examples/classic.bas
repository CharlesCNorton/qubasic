' A QBASIC program, run unchanged: qubasic examples/classic.bas
DECLARE SUB SortList (a() AS INTEGER, n AS INTEGER)
DECLARE FUNCTION IsPrime% (n AS INTEGER)
DEFINT A-Z
CONST COUNT = 8

DIM SHARED values(1 TO COUNT)
RESTORE Numbers
FOR i = 1 TO COUNT
    READ values(i)
NEXT i

SortList
PRINT "Sorted:";
FOR i = 1 TO COUNT
    PRINT values(i);
NEXT i
PRINT

PRINT "Primes:";
FOR i = 1 TO COUNT
    IF IsPrime%(values(i)) THEN PRINT values(i);
NEXT i
PRINT

total& = 0
FOR i = 1 TO COUNT: total& = total& + values(i): NEXT i
PRINT USING "Mean: ##.##"; total& / COUNT
GOSUB Verdict
END

Verdict:
SELECT CASE total&
    CASE IS > 100: PRINT "large"
    CASE 50 TO 100: PRINT "medium"
    CASE ELSE: PRINT "small"
END SELECT
RETURN

Numbers:
DATA 31, 4, 15, 9, 26, 5, 3, 7

SUB SortList
    FOR pass = 1 TO COUNT - 1
        FOR j = 1 TO COUNT - pass
            IF values(j) > values(j + 1) THEN SWAP values(j), values(j + 1)
        NEXT j
    NEXT pass
END SUB

FUNCTION IsPrime% (n)
    IsPrime% = 0
    IF n < 2 THEN EXIT FUNCTION
    FOR d = 2 TO SQR(n)
        IF n MOD d = 0 THEN EXIT FUNCTION
    NEXT d
    IsPrime% = -1
END FUNCTION
