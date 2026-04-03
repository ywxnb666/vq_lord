from _eval_final_sourceless import load_compiled_module

_module = load_compiled_module('mm_eval_suite_utils')
globals().update(_module.__dict__)
