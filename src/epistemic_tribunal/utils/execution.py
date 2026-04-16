"""Execution sandbox for synthesized ARC transformation programs."""
import sys
import traceback
from typing import Any, Optional, Callable

from epistemic_tribunal.tasks.base import Grid, grid_shape, crop_to_content, rotate_90, flip_h, flip_v
import math
import collections
import inspect

def get_sandbox_globals() -> dict[str, Any]:
    """Returns the globals dictionary used for executing synthesized code."""
    import numpy as np
    return {
        "__builtins__": __builtins__,
        "math": math,
        "collections": collections,
        "Counter": collections.Counter,
        "deque": collections.deque,
        "defaultdict": collections.defaultdict,
        "grid_shape": grid_shape,
        "crop_to_content": crop_to_content,
        "rotate_90": rotate_90,
        "flip_h": flip_h,
        "flip_v": flip_v,
        "np": np,
        "numpy": np,
    }

def get_sandbox_docs() -> str:
    """Returns a formatted string of docstrings for primitive Grid utilities."""
    globs = get_sandbox_globals()
    docs = []
    
    # We explicitly highlight only our custom grid functions
    custom_funcs = ["grid_shape", "crop_to_content", "rotate_90", "flip_h", "flip_v"]
    
    for fname in custom_funcs:
        func = globs.get(fname)
        if func and callable(func):
            sig = inspect.signature(func)
            doc = func.__doc__.strip().split('\n')[0] if func.__doc__ else "No documentation."
            docs.append(f"- `{fname}{sig}`: {doc}")
            
    docs.append("- Standard Python modules: `math`, `collections`, `numpy (as np)`")
    return "\n".join(docs)

def execute_transformation(
    code_string: str, 
    input_grid: Grid,
    function_name: str = "solve",
    timeout_seconds: int = 5
) -> Grid:
    """Execute *code_string* as a Python function and apply it to *input_grid*.

    Parameters
    ----------
    code_string:
        The Python source code containing the transformation logic.
    input_grid:
        The grid to transform.
    function_name:
        The name of the function to call within the code.
    timeout_seconds:
        Maximum execution time (not strictly enforced via OS signals in this simple version).

    Returns
    -------
    Grid
        The resulting transformed grid.

    Raises
    ------
    RuntimeError
        If execution fails or the function is not found.
    """
    # 1. Prepare sandboxed globals
    # We include basic math and grid utilities
    safe_globals = get_sandbox_globals()

    # 2. Execute the code block to define the function
    try:
        # We wrap in a try block to handle syntax errors or logic crashes
        exec(code_string, safe_globals)
    except Exception as exc:
        raise RuntimeError(f"Code execution failed during definition: {exc}\n{traceback.format_exc()}")

    # 3. Retrieve and call the specific function
    if function_name not in safe_globals:
        # Fallback search if 'solve' isn't there but there is only one function
        funcs = [k for k, v in safe_globals.items() if callable(v) and not k.startswith("__") and k not in ["grid_shape", "crop_to_content"]]
        if len(funcs) == 1:
            function_name = funcs[0]
        else:
            raise RuntimeError(f"Function {function_name!r} not found in generated code. Available: {funcs}")

    solve_fn = safe_globals[function_name]
    
    try:
        # ARC tasks often use deep copies to avoid mutating input grid if they want
        import copy
        result = solve_fn(copy.deepcopy(input_grid))
        
        # Basic validation: must return a list of lists of ints
        if not isinstance(result, list) or (result and not isinstance(result[0], list)):
            raise TypeError(f"Transformation must return a Grid (list[list[int]]), got {type(result)}")
            
        return result
    except Exception as exc:
        raise RuntimeError(f"Code execution failed during execution on input: {exc}")
