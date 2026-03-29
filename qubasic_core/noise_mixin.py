"""QUBASIC noise model configuration mixin."""


class NoiseMixin:
    """Noise model configuration command for QBasicTerminal.

    Requires: TerminalProtocol — uses self._noise_model, self.io.
    """

    def cmd_noise(self, rest: str) -> None:
        """NOISE <type> <params> — set noise model.

        Types:
          off                          Disable noise
          depolarizing <p>             Depolarizing channel (all gates)
          amplitude_damping <p>        T1-like decay
          phase_flip <p>               T2-like dephasing
          thermal <T1> <T2> <t_gate>   Physical decoherence (microseconds)
          readout <p0> <p1>            Measurement bit-flip error
          combined <p_amp> <p_phase>   Amplitude + phase damping
          pauli <px> <py> <pz>         General Pauli channel
          reset <p0> <p1>              Spontaneous reset error
        """
        if not rest or rest.strip().upper() == 'OFF':
            self._noise_model = None
            self._noise_depol_p = 0.0
            # Update LOCC engine if active
            if getattr(self, 'locc', None) is not None:
                self.locc.noise_param = 0.0
            self.io.writeln("NOISE OFF")
            return
        parts = rest.split()
        ntype = parts[0].lower()
        try:
            from qiskit_aer.noise import (
                NoiseModel, depolarizing_error, amplitude_damping_error,
                phase_damping_error, thermal_relaxation_error,
                phase_amplitude_damping_error, ReadoutError,
                pauli_error, reset_error,
            )
            nm = NoiseModel()
            _1q = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
                   'sx', 'rx', 'ry', 'rz', 'p', 'u', 'id']
            _2q = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                   'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            _3q = ['ccx', 'cswap']
            if ntype == 'depolarizing':
                p = float(parts[1]) if len(parts) > 1 else 0.01
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), _1q)
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), _2q)
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 3), _3q)
                self.io.writeln(f"NOISE depolarizing p={p}")
            elif ntype == 'amplitude_damping':
                p = float(parts[1]) if len(parts) > 1 else 0.01
                nm.add_all_qubit_quantum_error(amplitude_damping_error(p), _1q)
                self.io.writeln(f"NOISE amplitude_damping p={p}")
            elif ntype == 'phase_flip':
                p = float(parts[1]) if len(parts) > 1 else 0.01
                nm.add_all_qubit_quantum_error(phase_damping_error(p), _1q)
                self.io.writeln(f"NOISE phase_flip p={p}")
            elif ntype == 'thermal':
                t1 = float(parts[1]) if len(parts) > 1 else 50e-6
                t2 = float(parts[2]) if len(parts) > 2 else 70e-6
                tg = float(parts[3]) if len(parts) > 3 else 1e-6
                err = thermal_relaxation_error(t1, t2, tg)
                nm.add_all_qubit_quantum_error(err, _1q)
                self.io.writeln(f"NOISE thermal T1={t1} T2={t2} t_gate={tg}")
            elif ntype == 'readout':
                p0 = float(parts[1]) if len(parts) > 1 else 0.05
                p1 = float(parts[2]) if len(parts) > 2 else 0.1
                re = ReadoutError([[1 - p0, p0], [p1, 1 - p1]])
                nm.add_all_qubit_readout_error(re)
                self.io.writeln(f"NOISE readout p0={p0} p1={p1}")
            elif ntype == 'combined':
                pa = float(parts[1]) if len(parts) > 1 else 0.01
                pp = float(parts[2]) if len(parts) > 2 else 0.01
                nm.add_all_qubit_quantum_error(
                    phase_amplitude_damping_error(pa, pp), _1q)
                self.io.writeln(f"NOISE combined amp={pa} phase={pp}")
            elif ntype == 'pauli':
                px = float(parts[1]) if len(parts) > 1 else 0.01
                py = float(parts[2]) if len(parts) > 2 else 0.01
                pz = float(parts[3]) if len(parts) > 3 else 0.01
                pi = max(0, 1.0 - px - py - pz)
                err = pauli_error([('X', px), ('Y', py), ('Z', pz), ('I', pi)])
                nm.add_all_qubit_quantum_error(err, _1q)
                self.io.writeln(f"NOISE pauli px={px} py={py} pz={pz}")
            elif ntype == 'reset':
                p0 = float(parts[1]) if len(parts) > 1 else 0.01
                p1 = float(parts[2]) if len(parts) > 2 else 0.01
                nm.add_all_qubit_quantum_error(reset_error(p0, p1), _1q)
                self.io.writeln(f"NOISE reset p0={p0} p1={p1}")
            else:
                self.io.writeln(f"?UNKNOWN NOISE TYPE: {ntype}")
                self.io.writeln("  Types: depolarizing, amplitude_damping, phase_flip, thermal,")
                self.io.writeln("         readout, combined, pauli, reset")
                return
            self._noise_model = nm
            # Store scalar depolarizing param for LOCC numpy path
            if ntype == 'depolarizing':
                self._noise_depol_p = float(parts[1]) if len(parts) > 1 else 0.01
            else:
                self._noise_depol_p = 0.0
            # Propagate to LOCC engine if active
            if getattr(self, 'locc', None) is not None:
                self.locc.noise_param = self._noise_depol_p
        except ImportError:
            self.io.writeln("?Noise model requires qiskit-aer noise module")
