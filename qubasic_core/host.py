"""QUBASIC host contract: what the terminal's mixins share, for type checking.

QBasicTerminal composes Engine and its mixins, which read and write one
shared set of instance attributes and call one another's methods. HostState
declares every one of them for the type checker (annotations only), and
every mixin derives from it, so mypy's attr-defined check holds each
`self.name` to this list: a new shared attribute or method is declared here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    class HostState:
        # Instance attributes
        _array_dims: dict[str, list[int]]
        _array_lb: dict[str, list[int]]
        _array_types: dict[str, str]
        _auto_line: int
        _auto_mode: bool
        _auto_step: int
        _basis_gates: list[str] | None
        _bcut: Any
        _bmodes: Any
        _breakpoints: set[int]
        _bsv: Any
        _build_qc: Any
        _build_saw_measure: Any
        _call_stack: list[dict[str, Any]]
        _channels: Any
        _circuit_cache: Any
        _circuit_cache_key: Any
        _classical_bits: Any
        _color_bg: str
        _color_fg: str
        _compiled_cache: Any
        _consts: set[str]
        _cont_skip_stop_ip: int | None
        _coupling_map: list[list[int]] | None
        _current_slot: int
        _custom_gates: dict[str, Any]
        _data_cache_key: tuple | None
        _data_line_starts: list[tuple[int, int]]
        _data_pool: list[Any]
        _data_ptr: int
        _deftypes: dict[str, str]
        _device_spec: str | None
        _dimmed_arrays: set[str]
        _endian_big: bool
        _err_code: int
        _err_ip: int | None
        _err_line: int
        _error_target: int | None
        _es_approx_error: Any
        _file_fields: dict[int, list[str]]
        _file_handles: dict[int, Any]
        _func_call_depth: int
        _func_defs: dict[str, dict[str, Any]]
        _fusion_enable: Any
        _global_shared: set[str]
        _gosub_stack: list[int]
        _hamiltonians: dict
        _imported_lines: set[int]
        _in_error_handler: bool
        _include_depth: int
        _include_stack: list[str]
        _last_density: Any
        _last_density_qc: Any
        _last_sv: Any
        _last_transpiled: Any
        _live: Any
        _live_needed: Any
        _live_saves: dict[str, tuple[str, list]]
        _locc_shot_cap: Any
        _logical_mode: dict | None
        _logical_pl_cache: dict
        _lprint_path: str | None
        _max_iterations: int
        _measure_subset: list[int] | None
        _mps_truncation: Any
        _noise_depol_p: float
        _noise_locc_param: float
        _noise_locc_type: str
        _noise_model: Any
        _noise_spec: str | None
        _on_measure_fired: bool
        _on_measure_target: int | None
        _on_timer_interval: float
        _on_timer_last: float
        _on_timer_target: int | None
        _option_base: int
        _parsed: dict[int, Any]
        _pending_set_density: Any
        _pending_set_state: Any
        _pending_type: dict | None
        _poke_state_prep: dict[Any, Any]
        _profile_data: dict[int, dict[str, float]]
        _profile_mode: bool
        _profile_start: float
        _program_slots: dict[int, dict[int, str]]
        _prompt: str
        _qd: Any
        _qn: Any
        _qsv: Any
        _qubit_noise: dict[int, tuple[int, float]]
        _rb_cliffords: Any
        _rnd_last: float
        _rng: Any
        _run_manifest: Any
        _scope_stack: list[dict[str, Any]]
        _screen_mode: int
        _seed: int | None
        _shared_stack: list[set[str]]
        _start_time: float
        _state_source: Any
        _static_vars: dict[str, dict[str, Any]]
        _stats_runs: list[dict[str, int]]
        _status: dict[int, float]
        _step_auto: Any
        _step_delay: Any
        _stopped_ip: int | None
        _strict_names: Any
        _sub_defs: dict[str, dict[str, Any]]
        _sv_checkpoints: list[tuple[int, Any]]
        _sv_parallel_threshold: Any
        _theme: dict[str, str]
        _theme_name: str
        _trace_mode: bool
        _tt_position: int
        _undo_stack: list[dict[int, str]]
        _user_fns: dict[str, dict[str, Any]]
        _user_sys: dict[int, str]
        _user_types: dict[str, list[tuple[str, str]]]
        _var_types: dict[str, str]
        _watches: list[str]
        _zero_page: list[float]
        agent_mode: bool
        arrays: dict[str, Any]
        io: Any
        last_circuit: Any
        last_counts: dict[str, int] | None
        locc: Any
        locc_mode: bool
        num_qubits: int
        program: dict[int, str]
        registers: Any
        shots: int
        sim_device: str
        sim_method: str
        sim_precision: str
        step_mode: bool
        subroutines: dict[str, Any]
        variables: dict[str, Any]

        # Class-level constants and tables
        LOOKUP_MAX_SYNDROMES: Any
        MAX_EXPORT_AMPLITUDES: Any
        MEAS_CIRCUIT_MODE_VALUE: Any
        SHOR_MAX_BITS: Any
        _ANSI_FG: Any
        _CASE_OPS: Any
        _CF_DISPATCH: Any
        _CMD_NO_ARG: Any
        _CMD_WITH_ARG: Any
        _COLOR_NAMES: Any
        _DECODER_FLAGS: Any
        _DECODER_LABEL: Any
        _EXPERIMENTAL_CMDS: Any
        _GATE_DISPATCH: Any
        _MAPPED_RANGES: Any
        _MATRIX_OPS: Any
        _PARTIAL_CMDS: Any
        _PAULI_TO_CONTROLLED: Any
        _PRECISION_METHODS: Any
        _QUANTUM_STATES: Any
        _RESERVED_CONST_NAMES: Any
        _RE_PRINT_ENT: Any
        _RE_PRINT_QUBIT: Any
        _SAFE_CONSTS: Any
        _SAFE_FUNCS: Any
        _SC_A: Any
        _SC_THRESHOLD: Any
        _STIM_TASKS: Any
        _STMT_KEYWORDS: Any
        _SYS_BACKENDS: Any
        _USR_BLOCKED_CMDS: Any
        _gpu_cache: Any

        # Methods, defined across the mixins and resolved through the MRO
        _accumulate_type_field: Callable[..., Any]
        _active_sv_for_reg: Callable[..., Any]
        _add_final_measurement: Callable[..., Any]
        _advance: Callable[..., Any]
        _aer_options: Callable[..., Any]
        _alg_eval_cost: Callable[..., Any]
        _alg_eval_cost_parametric: Callable[..., Any]
        _alg_guard_standard: Callable[..., Any]
        _alg_parametric_setup: Callable[..., Any]
        _alg_qubit_list: Callable[..., Any]
        _amplitude_rows: Callable[..., Any]
        _annihilation: Callable[..., Any]
        _ansatz_state: Callable[..., Any]
        _apply_gate: Callable[..., Any]
        _apply_gate_str: Callable[..., Any]
        _array_bound: Callable[..., Any]
        _array_flat_index: Callable[..., Any]
        _array_kind: Callable[..., Any]
        _array_read: Callable[..., Any]
        _array_write: Callable[..., Any]
        _as_program: Callable[..., Any]
        _assert_assignable: Callable[..., Any]
        _ast_eval: Callable[..., Any]
        _ast_eval_matrix: Callable[..., Any]
        _auto_display: Callable[..., Any]
        _basic_number: Callable[..., Any]
        _bcheck: Callable[..., Any]
        _bench_backend: Callable[..., Any]
        _bind_params: Callable[..., Any]
        _bit_names: Callable[..., Any]
        _bit_order_note: Callable[..., Any]
        _bits: Callable[..., Any]
        _bloch_vector: Callable[..., Any]
        _bos_apply: Callable[..., Any]
        _bound_pair: Callable[..., Any]
        _build_backend_opts: Callable[..., Any]
        _build_hamiltonian: Callable[..., Any]
        _build_lines: Callable[..., Any]
        _build_loop: Callable[..., Any]
        _build_qubit_noise: Callable[..., Any]
        _builtin: Callable[..., Any]
        _call_user_fn_expr: Callable[..., Any]
        _capture_mode: Callable[..., Any]
        _case_matches: Callable[..., Any]
        _cf_assert: Callable[..., Any]
        _cf_call: Callable[..., Any]
        _cf_case: Callable[..., Any]
        _cf_data: Callable[..., Any]
        _cf_def_fn: Callable[..., Any]
        _cf_do: Callable[..., Any]
        _cf_end_function: Callable[..., Any]
        _cf_end_select: Callable[..., Any]
        _cf_end_sub: Callable[..., Any]
        _cf_error: Callable[..., Any]
        _cf_exit: Callable[..., Any]
        _cf_for: Callable[..., Any]
        _cf_function: Callable[..., Any]
        _cf_gosub: Callable[..., Any]
        _cf_goto: Callable[..., Any]
        _cf_if_jump: Callable[..., Any]
        _cf_let_array: Callable[..., Any]
        _cf_let_str: Callable[..., Any]
        _cf_let_var: Callable[..., Any]
        _cf_local: Callable[..., Any]
        _cf_loop: Callable[..., Any]
        _cf_mid_assign: Callable[..., Any]
        _cf_next: Callable[..., Any]
        _cf_on_error: Callable[..., Any]
        _cf_on_gosub: Callable[..., Any]
        _cf_on_goto: Callable[..., Any]
        _cf_on_measure: Callable[..., Any]
        _cf_on_timer: Callable[..., Any]
        _cf_option_base: Callable[..., Any]
        _cf_option_endian: Callable[..., Any]
        _cf_print: Callable[..., Any]
        _cf_print_using: Callable[..., Any]
        _cf_read: Callable[..., Any]
        _cf_resume: Callable[..., Any]
        _cf_return: Callable[..., Any]
        _cf_select_case: Callable[..., Any]
        _cf_shared: Callable[..., Any]
        _cf_static: Callable[..., Any]
        _cf_stop: Callable[..., Any]
        _cf_sub: Callable[..., Any]
        _cf_swap: Callable[..., Any]
        _cf_wend: Callable[..., Any]
        _cf_while: Callable[..., Any]
        _check_agent_path: Callable[..., Any]
        _check_breakpoint: Callable[..., Any]
        _check_fits: Callable[..., Any]
        _check_timer_callback: Callable[..., Any]
        _check_type: Callable[..., Any]
        _checkpoint_sv: Callable[..., Any]
        _choose_decoder: Callable[..., Any]
        _circuit_complexity: Callable[..., Any]
        _circuit_to_qb: Callable[..., Any]
        _classical_if_target: Callable[..., Any]
        _close_file: Callable[..., Any]
        _code_rate: Callable[..., Any]
        _coerce: Callable[..., Any]
        _collect_data: Callable[..., Any]
        _compiled: Callable[..., Any]
        _conjugate_term: Callable[..., Any]
        _controlled_increment: Callable[..., Any]
        _current_state: Callable[..., Any]
        _date_str: Callable[..., Any]
        _decode_batch: Callable[..., Any]
        _def_multiline: Callable[..., Any]
        _default_of_kind: Callable[..., Any]
        _default_value: Callable[..., Any]
        _defer_state: Callable[..., Any]
        _defer_state_extraction: Callable[..., Any]
        _demo_bell: Callable[..., Any]
        _demo_bernstein: Callable[..., Any]
        _demo_deutsch: Callable[..., Any]
        _demo_ghz: Callable[..., Any]
        _demo_grover: Callable[..., Any]
        _demo_locc_coord: Callable[..., Any]
        _demo_locc_teleport: Callable[..., Any]
        _demo_qft: Callable[..., Any]
        _demo_random: Callable[..., Any]
        _demo_stress: Callable[..., Any]
        _demo_superdense: Callable[..., Any]
        _demo_teleport: Callable[..., Any]
        _density_from_qc: Callable[..., Any]
        _device_lines: Callable[..., Any]
        _dim_record: Callable[..., Any]
        _draw_braille_bloch: Callable[..., Any]
        _drop_array: Callable[..., Any]
        _drop_loop: Callable[..., Any]
        _emit_feature_map: Callable[..., Any]
        _emit_mcz: Callable[..., Any]
        _emit_phase_flip: Callable[..., Any]
        _emit_poke_state_prep: Callable[..., Any]
        _emit_qadd_phases: Callable[..., Any]
        _emit_qft: Callable[..., Any]
        _emit_using: Callable[..., Any]
        _encodable: Callable[..., Any]
        _eof: Callable[..., Any]
        _eval_condition: Callable[..., Any]
        _eval_int: Callable[..., Any]
        _eval_operand: Callable[..., Any]
        _eval_print_item: Callable[..., Any]
        _eval_string_expr: Callable[..., Any]
        _eval_with_vars: Callable[..., Any]
        _exec_control_flow: Callable[..., Any]
        _exec_input_file: Callable[..., Any]
        _exec_line: Callable[..., Any]
        _exec_line_input_file: Callable[..., Any]
        _exec_lprint: Callable[..., Any]
        _exec_print_file: Callable[..., Any]
        _expand_statement: Callable[..., Any]
        _export_state: Callable[..., Any]
        _extract_save_results: Callable[..., Any]
        _extract_state_now: Callable[..., Any]
        _feedforward_if: Callable[..., Any]
        _finalize_run: Callable[..., Any]
        _find_qb_file: Callable[..., Any]
        _finish_live_saves: Callable[..., Any]
        _fit_decay: Callable[..., Any]
        _format_dirac: Callable[..., Any]
        _format_print: Callable[..., Any]
        _freefile: Callable[..., Any]
        _function_lines: Callable[..., Any]
        _gate_info: Callable[..., Any]
        _gate_object: Callable[..., Any]
        _get_parsed: Callable[..., Any]
        _get_string_ns: Callable[..., Any]
        _grad_descent: Callable[..., Any]
        _handle_error: Callable[..., Any]
        _hhl_solve: Callable[..., Any]
        _histogram_agrees: Callable[..., Any]
        _ideal_distribution: Callable[..., Any]
        _ideal_samples: Callable[..., Any]
        _init_bosonic: Callable[..., Any]
        _init_classic: Callable[..., Any]
        _init_debug: Callable[..., Any]
        _init_dynamics: Callable[..., Any]
        _init_file_handles: Callable[..., Any]
        _init_live: Callable[..., Any]
        _init_logical: Callable[..., Any]
        _init_memory: Callable[..., Any]
        _init_profiler: Callable[..., Any]
        _init_program_mgmt: Callable[..., Any]
        _init_qol: Callable[..., Any]
        _init_qudits: Callable[..., Any]
        _init_screen: Callable[..., Any]
        _init_subs: Callable[..., Any]
        _init_variables: Callable[..., Any]
        _inkey: Callable[..., Any]
        _input_values: Callable[..., Any]
        _internal_bits: Callable[..., Any]
        _invalidate_run_state: Callable[..., Any]
        _invoke_function: Callable[..., Any]
        _is_clifford: Callable[..., Any]
        _is_function_name: Callable[..., Any]
        _is_mapped: Callable[..., Any]
        _is_string_var: Callable[..., Any]
        _jump_to_line: Callable[..., Any]
        _keep_state: Callable[..., Any]
        _kind: Callable[..., Any]
        _lattice_cnot: Callable[..., Any]
        _lattice_merge: Callable[..., Any]
        _lattice_model: Callable[..., Any]
        _lattice_rate: Callable[..., Any]
        _lattice_run: Callable[..., Any]
        _lindblad_traj: Callable[..., Any]
        _live_noise: Callable[..., Any]
        _live_prefix: Callable[..., Any]
        _load_bas: Callable[..., Any]
        _load_lines_with_defs: Callable[..., Any]
        _local_pauli_matrix: Callable[..., Any]
        _locc_apply_gate: Callable[..., Any]
        _locc_bloch: Callable[..., Any]
        _locc_display_results: Callable[..., Any]
        _locc_exec_line: Callable[..., Any]
        _locc_execute_program: Callable[..., Any]
        _locc_noise_param: Callable[..., Any]
        _locc_run: Callable[..., Any]
        _locc_run_vectorized: Callable[..., Any]
        _locc_run_with_send: Callable[..., Any]
        _locc_state: Callable[..., Any]
        _locc_try_special: Callable[..., Any]
        _lof: Callable[..., Any]
        _logical_error_rate: Callable[..., Any]
        _logical_pl: Callable[..., Any]
        _logical_trailer: Callable[..., Any]
        _lookup_feasible: Callable[..., Any]
        _loop_body: Callable[..., Any]
        _lvalue: Callable[..., Any]
        _lvalue_kind: Callable[..., Any]
        _make_array: Callable[..., Any]
        _make_backend: Callable[..., Any]
        _match_graph: Callable[..., Any]
        _match_recovery: Callable[..., Any]
        _matchable: Callable[..., Any]
        _measure_into_bit: Callable[..., Any]
        _merge_circuit: Callable[..., Any]
        _mlae: Callable[..., Any]
        _mode_rdm_diag: Callable[..., Any]
        _modifier_operands: Callable[..., Any]
        _module_wide: Callable[..., Any]
        _needs_live: Callable[..., Any]
        _nelder_mead: Callable[..., Any]
        _np_operand: Callable[..., Any]
        _offset_qubits: Callable[..., Any]
        _on_terminal: Callable[..., Any]
        _open_file: Callable[..., Any]
        _optimize: Callable[..., Any]
        _parse_hhl_args: Callable[..., Any]
        _parse_matrix: Callable[..., Any]
        _parse_opt_args: Callable[..., Any]
        _parse_pauli_sum: Callable[..., Any]
        _parse_range_to: Callable[..., Any]
        _parse_syndrome: Callable[..., Any]
        _pauli_label: Callable[..., Any]
        _peek: Callable[..., Any]
        _peek_config: Callable[..., Any]
        _peek_qubit: Callable[..., Any]
        _poke: Callable[..., Any]
        _pop_scope: Callable[..., Any]
        _prepare_run: Callable[..., Any]
        _print_bloch_single: Callable[..., Any]
        _print_histogram_plain: Callable[..., Any]
        _print_histogram_rich: Callable[..., Any]
        _print_probs: Callable[..., Any]
        _print_statevector: Callable[..., Any]
        _print_sv_compact: Callable[..., Any]
        _print_watches: Callable[..., Any]
        _profile_line_end: Callable[..., Any]
        _profile_line_start: Callable[..., Any]
        _program_has_measure: Callable[..., Any]
        _push_scope: Callable[..., Any]
        _qaoa_optimize: Callable[..., Any]
        _qaoa_problem: Callable[..., Any]
        _qaoa_program: Callable[..., Any]
        _qaoa_state: Callable[..., Any]
        _qb_search_dirs: Callable[..., Any]
        _qcheck: Callable[..., Any]
        _qec_bb_rate: Callable[..., Any]
        _qec_binary: Callable[..., Any]
        _qec_circuit_level: Callable[..., Any]
        _qec_code: Callable[..., Any]
        _qec_code_build: Callable[..., Any]
        _qec_css_matrices: Callable[..., Any]
        _qec_decoder: Callable[..., Any]
        _qec_lookup_table: Callable[..., Any]
        _qec_matching_decode: Callable[..., Any]
        _qec_mwpm_rate: Callable[..., Any]
        _qec_phenomenological: Callable[..., Any]
        _qec_show_bb: Callable[..., Any]
        _qudit_apply: Callable[..., Any]
        _quit: Callable[..., Any]
        _random_circuit: Callable[..., Any]
        _rb_survival: Callable[..., Any]
        _read_lvalue: Callable[..., Any]
        _readable: Callable[..., Any]
        _record_save: Callable[..., Any]
        _report_n: Callable[..., Any]
        _report_width: Callable[..., Any]
        _reset_declarations: Callable[..., Any]
        _resolve_analysis_target: Callable[..., Any]
        _resolve_qubit: Callable[..., Any]
        _restore_to: Callable[..., Any]
        _restore_var: Callable[..., Any]
        _rich_print: Callable[..., Any]
        _rnd: Callable[..., Any]
        _run_encoded: Callable[..., Any]
        _run_kwargs: Callable[..., Any]
        _run_live: Callable[..., Any]
        _run_no_measure: Callable[..., Any]
        _run_steane: Callable[..., Any]
        _run_timer: Callable[..., Any]
        _run_with_fallback: Callable[..., Any]
        _safe_eval: Callable[..., Any]
        _sanitize_path: Callable[..., Any]
        _scan_subs: Callable[..., Any]
        _select_method: Callable[..., Any]
        _setup_readline: Callable[..., Any]
        _shor_circuit_phases: Callable[..., Any]
        _shor_oracle_phases: Callable[..., Any]
        _show_profile: Callable[..., Any]
        _show_stats: Callable[..., Any]
        _silenced: Callable[..., Any]
        _single_qubit_cliffords: Callable[..., Any]
        _split_arg_list: Callable[..., Any]
        _split_colon_stmts: Callable[..., Any]
        _split_decoder_flags: Callable[..., Any]
        _split_print_items: Callable[..., Any]
        _spsa: Callable[..., Any]
        _state_status: Callable[..., Any]
        _stats_export_csv: Callable[..., Any]
        _status_dict: Callable[..., Any]
        _status_prompt: Callable[..., Any]
        _steane_encode: Callable[..., Any]
        _step_lines: Callable[..., Any]
        _stmt_accept: Callable[..., Any]
        _stmt_beep: Callable[..., Any]
        _stmt_cls: Callable[..., Any]
        _stmt_color: Callable[..., Any]
        _stmt_const: Callable[..., Any]
        _stmt_deftype: Callable[..., Any]
        _stmt_locate: Callable[..., Any]
        _stmt_randomize: Callable[..., Any]
        _stmt_screen: Callable[..., Any]
        _stmt_sleep: Callable[..., Any]
        _stmt_write: Callable[..., Any]
        _store_var: Callable[..., Any]
        _suggest_command: Callable[..., Any]
        _surface_code: Callable[..., Any]
        _syndromes: Callable[..., Any]
        _target: Callable[..., Any]
        _time_str: Callable[..., Any]
        _tokenize_gate: Callable[..., Any]
        _trace_line: Callable[..., Any]
        _transpile_kwargs: Callable[..., Any]
        _transpile_routed: Callable[..., Any]
        _try_exec_amplify: Callable[..., Any]
        _try_exec_apply_circuit: Callable[..., Any]
        _try_exec_applychannel: Callable[..., Any]
        _try_exec_diffuse: Callable[..., Any]
        _try_exec_dim: Callable[..., Any]
        _try_exec_dim_type: Callable[..., Any]
        _try_exec_erase: Callable[..., Any]
        _try_exec_evolve: Callable[..., Any]
        _try_exec_featuremap: Callable[..., Any]
        _try_exec_get: Callable[..., Any]
        _try_exec_graphstate: Callable[..., Any]
        _try_exec_input: Callable[..., Any]
        _try_exec_let_str: Callable[..., Any]
        _try_exec_line_input: Callable[..., Any]
        _try_exec_mcgate: Callable[..., Any]
        _try_exec_meas: Callable[..., Any]
        _try_exec_measure_basis: Callable[..., Any]
        _try_exec_open_close: Callable[..., Any]
        _try_exec_poke: Callable[..., Any]
        _try_exec_print_using: Callable[..., Any]
        _try_exec_qadd: Callable[..., Any]
        _try_exec_qaddc: Callable[..., Any]
        _try_exec_qft: Callable[..., Any]
        _try_exec_qpe: Callable[..., Any]
        _try_exec_redim: Callable[..., Any]
        _try_exec_reset: Callable[..., Any]
        _try_exec_save_amps: Callable[..., Any]
        _try_exec_save_expect: Callable[..., Any]
        _try_exec_save_probs: Callable[..., Any]
        _try_exec_set_state: Callable[..., Any]
        _try_exec_syndrome: Callable[..., Any]
        _try_exec_sys: Callable[..., Any]
        _try_exec_unitary: Callable[..., Any]
        _try_quantum_print: Callable[..., Any]
        _try_stmt_handlers: Callable[..., Any]
        _update_status: Callable[..., Any]
        _using_text: Callable[..., Any]
        _usr_fn: Callable[..., Any]
        _var_options: Callable[..., Any]
        _verify_demo: Callable[..., Any]
        _vqe_program: Callable[..., Any]
        _writable: Callable[..., Any]
        _write_lvalue: Callable[..., Any]
        _write_program: Callable[..., Any]
        _write_to_handle: Callable[..., Any]
        build_circuit: Callable[..., Any]
        clear: Callable[..., Any]
        cmd_ampest: Callable[..., Any]
        cmd_animate: Callable[..., Any]
        cmd_apply_circuit: Callable[..., Any]
        cmd_auto: Callable[..., Any]
        cmd_bank: Callable[..., Any]
        cmd_basis: Callable[..., Any]
        cmd_bench: Callable[..., Any]
        cmd_bloch: Callable[..., Any]
        cmd_bosonic: Callable[..., Any]
        cmd_breakpoint: Callable[..., Any]
        cmd_bs: Callable[..., Any]
        cmd_bstate: Callable[..., Any]
        cmd_cat: Callable[..., Any]
        cmd_catalog: Callable[..., Any]
        cmd_chain: Callable[..., Any]
        cmd_channel: Callable[..., Any]
        cmd_checksum: Callable[..., Any]
        cmd_circuit: Callable[..., Any]
        cmd_circuit_def: Callable[..., Any]
        cmd_clear: Callable[..., Any]
        cmd_clip: Callable[..., Any]
        cmd_close: Callable[..., Any]
        cmd_cls: Callable[..., Any]
        cmd_color: Callable[..., Any]
        cmd_compare: Callable[..., Any]
        cmd_concurrence: Callable[..., Any]
        cmd_consistency: Callable[..., Any]
        cmd_cont: Callable[..., Any]
        cmd_copy: Callable[..., Any]
        cmd_coupling: Callable[..., Any]
        cmd_csv: Callable[..., Any]
        cmd_decompose: Callable[..., Any]
        cmd_def: Callable[..., Any]
        cmd_defs: Callable[..., Any]
        cmd_delete: Callable[..., Any]
        cmd_demo: Callable[..., Any]
        cmd_density: Callable[..., Any]
        cmd_device: Callable[..., Any]
        cmd_diff: Callable[..., Any]
        cmd_dir: Callable[..., Any]
        cmd_displace: Callable[..., Any]
        cmd_distill: Callable[..., Any]
        cmd_draw: Callable[..., Any]
        cmd_dump: Callable[..., Any]
        cmd_edit: Callable[..., Any]
        cmd_entropy: Callable[..., Any]
        cmd_estimate: Callable[..., Any]
        cmd_expect: Callable[..., Any]
        cmd_explain: Callable[..., Any]
        cmd_export: Callable[..., Any]
        cmd_fidelity: Callable[..., Any]
        cmd_find: Callable[..., Any]
        cmd_forward: Callable[..., Any]
        cmd_gradient: Callable[..., Any]
        cmd_grover: Callable[..., Any]
        cmd_gst: Callable[..., Any]
        cmd_hamiltonian: Callable[..., Any]
        cmd_heatmap: Callable[..., Any]
        cmd_help: Callable[..., Any]
        cmd_hhl: Callable[..., Any]
        cmd_hist: Callable[..., Any]
        cmd_history: Callable[..., Any]
        cmd_import: Callable[..., Any]
        cmd_include: Callable[..., Any]
        cmd_iqpe: Callable[..., Any]
        cmd_lattice: Callable[..., Any]
        cmd_let: Callable[..., Any]
        cmd_let_str: Callable[..., Any]
        cmd_lindblad: Callable[..., Any]
        cmd_list: Callable[..., Any]
        cmd_list_arrays: Callable[..., Any]
        cmd_list_colored: Callable[..., Any]
        cmd_list_subs: Callable[..., Any]
        cmd_list_vars: Callable[..., Any]
        cmd_load: Callable[..., Any]
        cmd_loadqasm: Callable[..., Any]
        cmd_locate: Callable[..., Any]
        cmd_locc: Callable[..., Any]
        cmd_loccinfo: Callable[..., Any]
        cmd_logical_error_rate: Callable[..., Any]
        cmd_lqubits: Callable[..., Any]
        cmd_map: Callable[..., Any]
        cmd_merge: Callable[..., Any]
        cmd_method: Callable[..., Any]
        cmd_minimize: Callable[..., Any]
        cmd_mirror: Callable[..., Any]
        cmd_monitor: Callable[..., Any]
        cmd_move: Callable[..., Any]
        cmd_negativity: Callable[..., Any]
        cmd_new: Callable[..., Any]
        cmd_noise: Callable[..., Any]
        cmd_open: Callable[..., Any]
        cmd_optimize: Callable[..., Any]
        cmd_pauliprop: Callable[..., Any]
        cmd_peek: Callable[..., Any]
        cmd_play: Callable[..., Any]
        cmd_plot: Callable[..., Any]
        cmd_poke: Callable[..., Any]
        cmd_precision: Callable[..., Any]
        cmd_probe: Callable[..., Any]
        cmd_probs: Callable[..., Any]
        cmd_profile: Callable[..., Any]
        cmd_prompt: Callable[..., Any]
        cmd_ptomography: Callable[..., Any]
        cmd_qaoa: Callable[..., Any]
        cmd_qec: Callable[..., Any]
        cmd_qf: Callable[..., Any]
        cmd_qkernel: Callable[..., Any]
        cmd_qmeasure: Callable[..., Any]
        cmd_qstate: Callable[..., Any]
        cmd_qsum: Callable[..., Any]
        cmd_qubits: Callable[..., Any]
        cmd_qudit: Callable[..., Any]
        cmd_quiz: Callable[..., Any]
        cmd_qvolume: Callable[..., Any]
        cmd_qwalk: Callable[..., Any]
        cmd_qx: Callable[..., Any]
        cmd_qz: Callable[..., Any]
        cmd_ram: Callable[..., Any]
        cmd_rb: Callable[..., Any]
        cmd_rbint: Callable[..., Any]
        cmd_reg: Callable[..., Any]
        cmd_regs: Callable[..., Any]
        cmd_renum: Callable[..., Any]
        cmd_replace: Callable[..., Any]
        cmd_resources: Callable[..., Any]
        cmd_restore: Callable[..., Any]
        cmd_rewind: Callable[..., Any]
        cmd_run: Callable[..., Any]
        cmd_sample: Callable[..., Any]
        cmd_save: Callable[..., Any]
        cmd_savepng: Callable[..., Any]
        cmd_screen: Callable[..., Any]
        cmd_seed: Callable[..., Any]
        cmd_send: Callable[..., Any]
        cmd_set_density: Callable[..., Any]
        cmd_set_state: Callable[..., Any]
        cmd_share: Callable[..., Any]
        cmd_shor: Callable[..., Any]
        cmd_shots: Callable[..., Any]
        cmd_simon: Callable[..., Any]
        cmd_squeeze: Callable[..., Any]
        cmd_state: Callable[..., Any]
        cmd_stats: Callable[..., Any]
        cmd_status: Callable[..., Any]
        cmd_step: Callable[..., Any]
        cmd_sweep: Callable[..., Any]
        cmd_sys: Callable[..., Any]
        cmd_theme: Callable[..., Any]
        cmd_threshold: Callable[..., Any]
        cmd_tomography: Callable[..., Any]
        cmd_troff: Callable[..., Any]
        cmd_tron: Callable[..., Any]
        cmd_type: Callable[..., Any]
        cmd_undo: Callable[..., Any]
        cmd_undo_preview: Callable[..., Any]
        cmd_vars: Callable[..., Any]
        cmd_version: Callable[..., Any]
        cmd_vqe: Callable[..., Any]
        cmd_wait: Callable[..., Any]
        cmd_watch: Callable[..., Any]
        cmd_xeb: Callable[..., Any]
        dispatch: Callable[..., Any]
        eval_expr: Callable[..., Any]
        print_banner: Callable[..., Any]
        print_histogram: Callable[..., Any]
        process: Callable[..., Any]
        repl: Callable[..., Any]
        result: Callable[..., Any]
        run_immediate: Callable[..., Any]

        # Properties
        @property
        def _active_nqubits(self) -> Any: ...
        @property
        def _active_sv(self) -> Any: ...
        @property
        def _transpile_opt_level(self) -> Any: ...
        @property
        def last_sv(self) -> Any: ...
        @last_sv.setter
        def last_sv(self, value: Any) -> None: ...
else:
    class HostState:
        """Base of every mixin; its declarations exist for the type checker."""
