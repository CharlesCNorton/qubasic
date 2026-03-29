"""QUBASIC built-in demo circuits with self-verification."""


class DemoMixin:
    """Built-in quantum computing demos.

    Requires: TerminalProtocol — uses self.program, self.num_qubits,
    self.shots, self.cmd_new(), self.cmd_run(), self.cmd_list(), self.cmd_locc().
    """

    def cmd_demo(self, name):
        demos = {
            'BELL':        self._demo_bell,
            'GHZ':         self._demo_ghz,
            'TELEPORT':    self._demo_teleport,
            'GROVER':      self._demo_grover,
            'QFT':         self._demo_qft,
            'DEUTSCH':     self._demo_deutsch,
            'BERNSTEIN':   self._demo_bernstein,
            'SUPERDENSE':  self._demo_superdense,
            'RANDOM':      self._demo_random,
            'STRESS':      self._demo_stress,
            'LOCC-TELEPORT': self._demo_locc_teleport,
            'LOCC-COORD':    self._demo_locc_coord,
        }
        name = name.upper().strip()
        if not name or name == 'LIST':
            self.io.writeln("  Available demos:")
            for d in demos:
                prefix = "  (LOCC)" if d.startswith('LOCC') else ""
                self.io.writeln(f"    DEMO {d}{prefix}")
            return
        if name not in demos:
            self.io.writeln(f"?UNKNOWN DEMO: {name}")
            self.io.writeln(f"  Try: DEMO LIST")
            return
        demos[name]()

    def _verify_demo(self, expected_states: list[str], min_frac: float = 0.85,
                     label: str = '') -> bool:
        """Check that expected states dominate the output.

        expected_states: list of bitstrings that should capture >= min_frac of shots.
        Returns True if verification passes.
        """
        if not self.last_counts:
            self.io.writeln(f"  VERIFY: no results")
            return False
        total = sum(self.last_counts.values())
        hit = sum(self.last_counts.get(s, 0) for s in expected_states)
        frac = hit / total if total > 0 else 0
        tag = f" ({label})" if label else ""
        if frac >= min_frac:
            self.io.writeln(f"  VERIFY PASS{tag}: {frac:.1%} in expected states "
                           f"(threshold {min_frac:.0%})")
            return True
        else:
            self.io.writeln(f"  VERIFY FAIL{tag}: {frac:.1%} in expected states "
                           f"(threshold {min_frac:.0%})")
            return False

    def _demo_bell(self):
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10: "REM === Bell State ===",
            20: "REM The simplest entanglement. Two qubits, perfectly correlated.",
            30: "H 0",
            40: "CX 0,1",
            50: "MEASURE",
        }
        self.io.writeln("LOADED: Bell State (2 qubits)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['00', '11'], 0.95, 'Bell: only |00> and |11>')

    def _demo_ghz(self):
        self.cmd_new()
        n = min(self.num_qubits, 8)
        self.num_qubits = n
        self.program = {
            10: "REM === GHZ State ===",
            20: f"REM {n}-qubit Greenberger-Horne-Zeilinger state",
            30: "REM All qubits entangled: 50% all-zeros, 50% all-ones.",
            40: "REM Einstein called this 'spooky action at a distance.'",
            50: "H 0",
        }
        line = 60
        for i in range(1, n):
            self.program[line] = f"CX 0,{i}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: GHZ State ({n} qubits)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['0' * n, '1' * n], 0.95, f'GHZ: only |{"0"*n}> and |{"1"*n}>')

    def _demo_teleport(self):
        self.cmd_new()
        self.num_qubits = 3
        self.program = {
            10:  "REM === Quantum Teleportation (circuit only) ===",
            20:  "REM Qubit 0: state to teleport (prepared as |+>)",
            30:  "REM Qubit 1-2: entangled Bell pair (the 'channel')",
            40:  "REM Shows the entangling stage of teleportation.",
            50:  "REM Classical correction requires mid-circuit measurement",
            60:  "REM with feedforward, which needs LOCC mode.",
            70:  "REM >>> For full teleportation: DEMO LOCC-TELEPORT <<<",
            80:  "REM",
            90:  "REM Prepare state to teleport (|+> on qubit 0)",
            100: "H 0",
            110: "REM Create Bell pair on qubits 1,2",
            120: "H 1",
            130: "CX 1,2",
            140: "REM Teleportation circuit (without classical correction)",
            150: "CX 0,1",
            160: "H 0",
            170: "MEASURE",
        }
        self.io.writeln("LOADED: Quantum Teleportation — circuit stage (3 qubits)")
        self.io.writeln("  Without correction: uniform over all 8 outcomes.")
        self.io.writeln("  For full protocol with correction: DEMO LOCC-TELEPORT")
        self.cmd_list()
        self.cmd_run()

    def _demo_grover(self):
        self.cmd_new()
        self.num_qubits = 3
        self.shots = 1024
        self.program = {
            10:  "REM === Grover's Search Algorithm ===",
            20:  "REM Searching for |101> among 8 states",
            30:  "REM Classically: ~4 queries. Quantumly: 2 iterations, ~95%.",
            40:  "REM Quadratic speedup over classical unstructured search.",
            50:  "REM",
            60:  "REM Superposition",
            70:  "H 0 : H 1 : H 2",
            80:  "REM",
            90:  "REM === Grover iteration 1 ===",
            100: "REM Oracle: flip phase of |101> (X on q1, CCZ, undo X)",
            110: "X 1",
            120: "H 2 : CCX 0,1,2 : H 2",
            130: "X 1",
            140: "REM Diffusion: 2|s><s| - I",
            150: "H 0 : H 1 : H 2",
            160: "X 0 : X 1 : X 2",
            170: "H 2 : CCX 0,1,2 : H 2",
            180: "X 0 : X 1 : X 2",
            190: "H 0 : H 1 : H 2",
            200: "REM",
            210: "REM === Grover iteration 2 ===",
            220: "X 1",
            230: "H 2 : CCX 0,1,2 : H 2",
            240: "X 1",
            250: "H 0 : H 1 : H 2",
            260: "X 0 : X 1 : X 2",
            270: "H 2 : CCX 0,1,2 : H 2",
            280: "X 0 : X 1 : X 2",
            290: "H 0 : H 1 : H 2",
            300: "REM",
            310: "MEASURE",
        }
        self.io.writeln("LOADED: Grover's Search (3 qubits, target=|101>)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['101'], 0.85, 'Grover: target |101>')

    def _demo_qft(self):
        self.cmd_new()
        self.num_qubits = 4
        self.program = {
            10:  "REM === Quantum Fourier Transform ===",
            20:  "REM The quantum analog of the discrete Fourier transform.",
            30:  "REM Exponentially faster than classical FFT.",
            40:  "REM QFT is typically used as a subroutine (phase info lost on measure).",
            50:  "REM",
            60:  "REM Prepare input state |5> = |0101>",
            70:  "X 0 : X 2",
            80:  "REM",
            90:  "REM QFT on 4 qubits",
            100: "H 3",
            110: "CP PI/2, 2, 3",
            120: "CP PI/4, 1, 3",
            130: "CP PI/8, 0, 3",
            140: "H 2",
            150: "CP PI/2, 1, 2",
            160: "CP PI/4, 0, 2",
            170: "H 1",
            180: "CP PI/2, 0, 1",
            190: "H 0",
            200: "REM Swap for bit reversal",
            210: "SWAP 0,3",
            220: "SWAP 1,2",
            230: "MEASURE",
        }
        self.io.writeln("LOADED: QFT on |0101> (4 qubits)")
        self.cmd_list()
        self.cmd_run()

    def _demo_deutsch(self):
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10:  "REM === Deutsch-Jozsa Algorithm ===",
            20:  "REM Determines if a function is constant or balanced",
            30:  "REM in ONE query. Classically needs 2.",
            40:  "REM Published 1992. The first to demonstrate quantum advantage.",
            50:  "REM",
            60:  "REM Oracle qubit in |->",
            70:  "X 1",
            80:  "H 0 : H 1",
            90:  "REM Oracle (balanced function: f(x) = x)",
            100: "CX 0,1",
            110: "REM Interfere and measure",
            120: "H 0",
            130: "MEASURE",
        }
        self.io.writeln("LOADED: Deutsch-Jozsa (2 qubits, balanced oracle)")
        self.io.writeln("  Expect: qubit 0 = 1 (balanced)")
        self.cmd_list()
        self.cmd_run()
        # q0=1 means bit pattern x1 (states 01 or 11)
        self._verify_demo(['01', '11'], 0.95, 'Deutsch: q0=1 (balanced)')

    def _demo_bernstein(self):
        self.cmd_new()
        self.num_qubits = 5
        self.program = {
            10:  "REM === Bernstein-Vazirani Algorithm ===",
            20:  "REM Finds a hidden bit string s in ONE query.",
            30:  "REM Secret: s = 1011 (4 bits, display order q3q2q1q0)",
            40:  "REM Classically you'd need 4 queries. Quantum: 1.",
            50:  "REM",
            60:  "REM Ancilla in |->",
            70:  "X 4",
            80:  "H 0 : H 1 : H 2 : H 3 : H 4",
            90:  "REM Oracle: CX from each bit where s=1",
            100: "CX 0,4",
            110: "CX 1,4",
            120: "CX 3,4",
            130: "REM Interfere",
            140: "H 0 : H 1 : H 2 : H 3",
            150: "MEASURE",
        }
        self.io.writeln("LOADED: Bernstein-Vazirani (5 qubits, secret=1011)")
        self.io.writeln("  Expect: measurement shows ...1011 (q3q2q1q0)")
        self.cmd_list()
        self.cmd_run()
        # Ancilla (q4) is random; data qubits should read 1011
        self._verify_demo(['01011', '11011'], 0.95, 'BV: secret=1011')

    def _demo_superdense(self):
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10:  "REM === Superdense Coding ===",
            20:  "REM Send 2 classical bits using 1 qubit.",
            30:  "REM Encoding: 00->I, 01->X, 10->Z, 11->ZX (=iY)",
            40:  "REM Sending message '11'",
            50:  "REM",
            60:  "REM Create Bell pair",
            70:  "H 0",
            80:  "CX 0,1",
            90:  "REM Encode '11': apply Z then X to qubit 0",
            100: "Z 0",
            110: "X 0",
            120: "REM Decode: reverse Bell circuit",
            130: "CX 0,1",
            140: "H 0",
            150: "MEASURE",
        }
        self.io.writeln("LOADED: Superdense Coding (message='11')")
        self.io.writeln("  Expect: |11> with high probability")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['11'], 0.95, 'Superdense: message=11')

    def _demo_random(self):
        self.cmd_new()
        n = min(self.num_qubits, 8)
        self.num_qubits = n
        self.shots = 1
        self.program = {
            10: "REM === Quantum Random Number Generator ===",
            20: f"REM {n} qubits = {n}-bit truly random number",
            30: "REM (as random as a simulator's PRNG allows)",
        }
        line = 40
        for i in range(n):
            self.program[line] = f"H {i}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: Quantum RNG ({n} bits)")
        self.cmd_list()
        self.cmd_run()
        self.shots = 1024

    def _demo_stress(self):
        self.cmd_new()
        n = 20
        self.num_qubits = n
        self.program = {
            10: f"REM === {n}-Qubit Stress Test ===",
            20: f"REM H on all {n} qubits + nearest-neighbor CX chain",
            30: f"REM Creates a highly entangled {n}-qubit state",
        }
        line = 40
        for i in range(n):
            self.program[line] = f"H {i}"
            line += 10
        for i in range(n - 1):
            self.program[line] = f"CX {i},{i+1}"
            line += 10
        for i in range(n):
            self.program[line] = f"RZ PI/4,{i}"
            line += 10
        for i in range(0, n - 1, 2):
            self.program[line] = f"CX {i},{i+1}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: {n}-Qubit Stress Test")
        self.cmd_run()

    def _demo_locc_teleport(self):
        """Quantum teleportation across A/B boundary (JOINT mode)."""
        self.cmd_new()
        self.cmd_locc('JOINT 3 3')
        self.shots = 1024
        self.program = {
            10:  "REM === LOCC Teleportation ===",
            20:  "REM Alice (A) teleports qubit 0 to Bob (B) qubit 0",
            30:  "REM Pre-shared Bell pair: A[2] <-> B[0]",
            40:  "REM",
            50:  "REM Prepare state to teleport: |+> on A[0]",
            60:  "@A H 0",
            70:  "REM Create shared entanglement",
            80:  "SHARE A 2, B 0",
            90:  "REM",
            100: "REM === Teleportation protocol ===",
            110: "REM Alice entangles her data qubit with her half of the pair",
            120: "@A CX 0, 2",
            130: "@A H 0",
            140: "REM Alice measures and sends classical bits to Bob",
            150: "SEND A 0 -> m0",
            160: "SEND A 2 -> m1",
            170: "REM Bob applies corrections based on Alice's results",
            180: "IF m1 THEN @B X 0",
            190: "IF m0 THEN @B Z 0",
            200: "REM Bob's qubit 0 now holds Alice's original state",
            210: "MEASURE",
        }
        self.io.writeln("LOADED: LOCC Teleportation (JOINT 3+3)")
        self.io.writeln("  Alice sends |+> to Bob. Expect Bob's qubit ~ 50/50.")
        self.cmd_list()
        self.cmd_run()

    def _demo_locc_coord(self):
        """Classical coordination between independent registers (SPLIT mode)."""
        self.cmd_new()
        self.cmd_locc('SPLIT 4 4')
        self.shots = 1024
        self.program = {
            10:  "REM === LOCC Classical Coordination ===",
            20:  "REM Two independent quantum processors sharing classical bits",
            30:  "REM Alice generates a random 2-bit key, Bob copies it",
            40:  "REM",
            50:  "REM Alice: generate random bits",
            60:  "@A H 0 : @A H 1",
            70:  "SEND A 0 -> k0",
            80:  "SEND A 1 -> k1",
            90:  "REM Bob: set qubits to match Alice's key",
            100: "IF k0 THEN @B X 0",
            110: "IF k1 THEN @B X 1",
            120: "REM Both sides now agree on a 2-bit key",
            130: "REM Alice uses her key to prepare an entangled state",
            140: "@A H 2",
            150: "@A CX 2,3",
            160: "REM Bob does the same — they independently create matching states",
            170: "@B H 2",
            180: "@B CX 2,3",
            190: "MEASURE",
        }
        self.io.writeln("LOADED: LOCC Classical Coordination (SPLIT 4+4)")
        self.io.writeln("  Alice and Bob share a random key, then prepare matching states.")
        self.cmd_list()
        self.cmd_run()

