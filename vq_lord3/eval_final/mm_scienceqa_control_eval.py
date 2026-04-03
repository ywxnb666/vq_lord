from _eval_final_sourceless import load_compiled_module

_module = load_compiled_module('mm_scienceqa_control_eval')
globals().update(_module.__dict__)
