"""Pre-compiled regex patterns for the QBASIC parser."""

import re

# ═══════════════════════════════════════════════════════════════════════
# Pre-compiled regexes
# ═══════════════════════════════════════════════════════════════════════

RE_LINE_NUM = re.compile(r'^(\d+)\s*(.*)')
RE_DEF_SINGLE = re.compile(r'(\w+)(?:\(([^)]*)\))?\s*=\s*(.*)')
RE_DEF_BEGIN = re.compile(r'DEF\s+BEGIN\s+(\w+)(?:\(([^)]*)\))?', re.IGNORECASE)
RE_REG_INDEX = re.compile(r'(\w+)\[(\d+)\]')
RE_AT_REG = re.compile(r'@([A-Z])\s+', re.IGNORECASE)
RE_AT_REG_LINE = re.compile(r'@([A-Z])\s+(.*)', re.IGNORECASE)
RE_SEND = re.compile(r'SEND\s+([A-Z])\s+(\S+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SHARE = re.compile(r'SHARE\s+([A-Z])\s+(\d+)\s*,?\s*([A-Z])\s+(\d+)', re.IGNORECASE)
RE_MEAS = re.compile(r'MEAS\s+(\S+)\s*->\s*(\w+)', re.IGNORECASE)
RE_RESET = re.compile(r'RESET\s+(\S+)', re.IGNORECASE)
RE_UNITARY = re.compile(r'UNITARY\s+(\w+)\s*=\s*(\[.+\])', re.IGNORECASE)
RE_DIM = re.compile(r'DIM\s+(\w+)\((\d+)\)', re.IGNORECASE)
RE_REDIM = re.compile(r'REDIM\s+(\w+)\((\d+)\)', re.IGNORECASE)
RE_ERASE = re.compile(r'ERASE\s+(\w+)', re.IGNORECASE)
RE_GET = re.compile(r'GET\s+(\w+\$?)', re.IGNORECASE)
RE_INPUT = re.compile(r'INPUT\s+(?:"([^"]*)"\s*,\s*)?(\w+)', re.IGNORECASE)
RE_CTRL = re.compile(r'CTRL\s+(\w+)\s+(.*)', re.IGNORECASE)
RE_INV = re.compile(r'INV\s+(\w+)\s+(.*)', re.IGNORECASE)
RE_LET_ARRAY = re.compile(r'LET\s+(\w+)\((.+?)\)\s*=\s*(.*)', re.IGNORECASE)
RE_LET_VAR = re.compile(r'LET\s+(\w+)\s*=\s*(.*)', re.IGNORECASE)
RE_PRINT = re.compile(r'PRINT\s+(.*)', re.IGNORECASE)
RE_GOTO = re.compile(r'GOTO\s+(\d+)\s*$', re.IGNORECASE)
RE_GOSUB = re.compile(r'GOSUB\s+(\d+)\s*$', re.IGNORECASE)
RE_FOR = re.compile(
    r'FOR\s+(\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?\s*$', re.IGNORECASE)
RE_NEXT = re.compile(r'NEXT\s+(\w+)\s*$', re.IGNORECASE)
RE_WHILE = re.compile(r'WHILE\s+(.+)$', re.IGNORECASE)
RE_IF_THEN = re.compile(
    r'IF\s+(.+?)\s+THEN(?:\s+(.*?))?(?:\s+ELSE\s+(.*))?$', re.IGNORECASE)
RE_ELSEIF = re.compile(
    r'IF\s+(.+?)\s+THEN\s+(.*?)\s+(?:ELSEIF|ELSE\s+IF)\s+(.+)$', re.IGNORECASE)
RE_GOTO_GOSUB_TARGET = re.compile(r'(GOTO|GOSUB)\s+(\d+)', re.IGNORECASE)
RE_MEASURE_BASIS = re.compile(
    r'MEASURE_(X|Y|Z)\s+(\S+)', re.IGNORECASE)
RE_SYNDROME = re.compile(
    r'SYNDROME\s+(.*)', re.IGNORECASE)

# ── Classic BASIC, memory, SUB/FUNCTION, debug ──────────────────────

RE_DATA = re.compile(r'DATA\s+(.*)', re.IGNORECASE)
RE_READ = re.compile(r'READ\s+(.*)', re.IGNORECASE)
RE_ON_GOTO = re.compile(r'ON\s+(.+?)\s+GOTO\s+([\d\s,]+)', re.IGNORECASE)
RE_ON_GOSUB = re.compile(r'ON\s+(.+?)\s+GOSUB\s+([\d\s,]+)', re.IGNORECASE)
RE_SELECT_CASE = re.compile(r'SELECT\s+CASE\s+(.*)', re.IGNORECASE)
RE_CASE = re.compile(r'CASE\s+(.*)', re.IGNORECASE)
RE_DO = re.compile(r'DO(?:\s+(WHILE|UNTIL)\s+(.+))?\s*$', re.IGNORECASE)
RE_LOOP_STMT = re.compile(r'LOOP(?:\s+(WHILE|UNTIL)\s+(.+))?\s*$', re.IGNORECASE)
RE_EXIT = re.compile(r'EXIT\s+(FOR|WHILE|DO|SUB|FUNCTION)\s*$', re.IGNORECASE)
RE_SUB = re.compile(r'SUB\s+(\w+)(?:\(([^)]*)\))?\s*$', re.IGNORECASE)
RE_END_SUB = re.compile(r'END\s+SUB\s*$', re.IGNORECASE)
RE_FUNCTION = re.compile(r'FUNCTION\s+(\w+)(?:\(([^)]*)\))?\s*$', re.IGNORECASE)
RE_END_FUNCTION = re.compile(r'END\s+FUNCTION\s*$', re.IGNORECASE)
RE_CALL = re.compile(r'CALL\s+(\w+)(?:\(([^)]*)\))?\s*$', re.IGNORECASE)
RE_LOCAL = re.compile(r'LOCAL\s+(.*)', re.IGNORECASE)
RE_STATIC_DECL = re.compile(r'STATIC\s+(.*)', re.IGNORECASE)
RE_SHARED = re.compile(r'SHARED\s+(.*)', re.IGNORECASE)
RE_ON_ERROR = re.compile(r'ON\s+ERROR\s+GOTO\s+(\d+)', re.IGNORECASE)
RE_RESUME = re.compile(r'RESUME(?:\s+(.+))?\s*$', re.IGNORECASE)
RE_ERROR_STMT = re.compile(r'ERROR\s+(\d+)', re.IGNORECASE)
RE_ASSERT = re.compile(r'ASSERT\s+(.*)', re.IGNORECASE)
RE_SWAP = re.compile(r'SWAP\s+(\w+\$?)\s*,\s*(\w+\$?)', re.IGNORECASE)
RE_POKE = re.compile(r'POKE\s+(.+?)\s*,\s*(.+)', re.IGNORECASE)
RE_SYS = re.compile(r'SYS\s+(.+)', re.IGNORECASE)
RE_OPEN = re.compile(
    r'OPEN\s+"?([^"]+)"?\s+FOR\s+(INPUT|OUTPUT|APPEND|RANDOM)\s+AS\s+#?(\d+)'
    r'(?:\s+ENCODING\s+"?([^"]*)"?)?',
    re.IGNORECASE)
RE_CLOSE = re.compile(r'CLOSE\s+#?(\d+)', re.IGNORECASE)
RE_PRINT_FILE = re.compile(r'PRINT\s+#(\d+)\s*,\s*(.*)', re.IGNORECASE)
RE_INPUT_FILE = re.compile(r'INPUT\s+#(\d+)\s*,\s*(\w+\$?)', re.IGNORECASE)
RE_LINE_INPUT = re.compile(
    r'LINE\s+INPUT\s+(?:"([^"]*)"\s*,\s*)?(\w+\$?)', re.IGNORECASE)
RE_OPTION_BASE = re.compile(r'OPTION\s+BASE\s+([01])', re.IGNORECASE)
RE_IMPORT = re.compile(r'IMPORT\s+"?([^"]+)"?', re.IGNORECASE)
RE_SAVE_EXPECT = re.compile(r'SAVE_EXPECT\s+(\w+)\s+([\d\s,]+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SAVE_PROBS = re.compile(r'SAVE_PROBS\s+([\d\s,]+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SAVE_AMPS = re.compile(r'SAVE_AMPS\s+([\d\s,]+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SET_STATE = re.compile(r'SET_STATE\s+(.*)', re.IGNORECASE)
RE_TYPE_BEGIN = re.compile(r'TYPE\s+(\w+)', re.IGNORECASE)
RE_TYPE_FIELD = re.compile(r'(\w+)\s+AS\s+(INTEGER|FLOAT|STRING|QUBIT)', re.IGNORECASE)
RE_END_TYPE = re.compile(r'END\s+TYPE', re.IGNORECASE)
RE_DIM_TYPE = re.compile(r'DIM\s+(\w+)\s+AS\s+(\w+)', re.IGNORECASE)
RE_CHAIN = re.compile(r'CHAIN\s+"?([^"]+)"?', re.IGNORECASE)
RE_MERGE = re.compile(r'MERGE\s+"?([^"]+)"?', re.IGNORECASE)
RE_DEF_FN = re.compile(
    r'DEF\s+FN\s*(\w+)\s*\(([^)]*)\)\s*=\s*(.*)', re.IGNORECASE)
RE_PRINT_USING = re.compile(
    r'PRINT\s+USING\s+"([^"]+)"\s*;\s*(.*)', re.IGNORECASE)
RE_COLOR = re.compile(r'COLOR\s+(\w+)(?:\s*,\s*(\w+))?', re.IGNORECASE)
RE_LOCATE = re.compile(r'LOCATE\s+(\d+)\s*,\s*(\d+)', re.IGNORECASE)
RE_SCREEN = re.compile(r'SCREEN\s+(\d+)', re.IGNORECASE)
RE_LPRINT = re.compile(r'LPRINT\s+(.*)', re.IGNORECASE)
RE_ON_MEASURE = re.compile(r'ON\s+MEASURE\s+GOSUB\s+(\d+)', re.IGNORECASE)
RE_ON_TIMER = re.compile(r'ON\s+TIMER\s*\((\d+)\)\s+GOSUB\s+(\d+)', re.IGNORECASE)
RE_DIM_MULTI = re.compile(r'DIM\s+(\w+)\((\d+(?:\s*,\s*\d+)*)\)', re.IGNORECASE)
RE_LET_STR = re.compile(r'LET\s+(\w+\$)\s*=\s*(.*)', re.IGNORECASE)

__all__ = [
    "RE_LINE_NUM",
    "RE_DEF_SINGLE",
    "RE_DEF_BEGIN",
    "RE_REG_INDEX",
    "RE_AT_REG",
    "RE_AT_REG_LINE",
    "RE_SEND",
    "RE_SHARE",
    "RE_MEAS",
    "RE_RESET",
    "RE_UNITARY",
    "RE_DIM",
    "RE_REDIM",
    "RE_ERASE",
    "RE_GET",
    "RE_INPUT",
    "RE_CTRL",
    "RE_INV",
    "RE_LET_ARRAY",
    "RE_LET_VAR",
    "RE_PRINT",
    "RE_GOTO",
    "RE_GOSUB",
    "RE_FOR",
    "RE_NEXT",
    "RE_WHILE",
    "RE_IF_THEN",
    "RE_GOTO_GOSUB_TARGET",
    "RE_MEASURE_BASIS",
    "RE_SYNDROME",
    "RE_DATA",
    "RE_READ",
    "RE_ON_GOTO",
    "RE_ON_GOSUB",
    "RE_SELECT_CASE",
    "RE_CASE",
    "RE_DO",
    "RE_LOOP_STMT",
    "RE_EXIT",
    "RE_SUB",
    "RE_END_SUB",
    "RE_FUNCTION",
    "RE_END_FUNCTION",
    "RE_CALL",
    "RE_LOCAL",
    "RE_STATIC_DECL",
    "RE_SHARED",
    "RE_ON_ERROR",
    "RE_RESUME",
    "RE_ERROR_STMT",
    "RE_ASSERT",
    "RE_SWAP",
    "RE_POKE",
    "RE_SYS",
    "RE_OPEN",
    "RE_CLOSE",
    "RE_PRINT_FILE",
    "RE_INPUT_FILE",
    "RE_LINE_INPUT",
    "RE_OPTION_BASE",
    "RE_IMPORT",
    "RE_SAVE_EXPECT",
    "RE_SAVE_PROBS",
    "RE_SAVE_AMPS",
    "RE_SET_STATE",
    "RE_TYPE_BEGIN",
    "RE_TYPE_FIELD",
    "RE_END_TYPE",
    "RE_DIM_TYPE",
    "RE_CHAIN",
    "RE_MERGE",
    "RE_DEF_FN",
    "RE_PRINT_USING",
    "RE_COLOR",
    "RE_LOCATE",
    "RE_SCREEN",
    "RE_LPRINT",
    "RE_ON_MEASURE",
    "RE_ON_TIMER",
    "RE_DIM_MULTI",
    "RE_LET_STR",
]
