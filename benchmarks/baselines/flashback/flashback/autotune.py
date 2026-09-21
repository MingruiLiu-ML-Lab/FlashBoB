import numpy as np
import tqdm
import jax
import shutil
from pathlib import Path
import random
import timeit
import jax.numpy as jnp
from itertools import product
import dill
import multiprocessing as mp
import os

DEBUG = os.environ.get('DEBUG_AUTOTUNER', 'false') == 'true'
SKIP_AUTOTUNE = os.environ.get('SKIP_AUTOTUNER', 'false') == 'true'
AUTOTUNE = 2138929384

def print_log(msg, *args, **kwargs):
    msg = '🎛 🦉: ' + msg.strip()
    return print(msg, *args, **kwargs)

class MaybeArray:
    def __init__(self, isarray, shape, dtype, value):
        self.isarray = isarray
        self.shape = shape
        self.dtype = dtype
        self.value = value


def hashable_for_arguments(*args, **kwargs):
    if kwargs:
        keys, values = zip(*kwargs.items())
    else:
        keys, values = tuple(), tuple()

    all_values = list(args) + list(values)
    signatures = []

    for value in all_values:
        # check if numpy array
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            signature = (value.shape, value.dtype)
            signatures.append(signature)
            # ensure that this is hashable
            try:
                hash(signature)
            except:
                raise ValueError(f"Signature {signature} is not hashable")
        elif isinstance(value, (tuple, list)):
            this_sig = hashable_for_arguments(*value)
            signatures.append(this_sig)
        elif isinstance(value, dict):
            values = list(value.values())
            keys = list(value.keys())
            this_sig = hashable_for_arguments(*values)
            this_sig2 = hashable_for_arguments(*keys)
            signatures.append((this_sig, this_sig2))
        else:
            try:
                hash(value)
            except:
                raise ValueError(f"Value {value} is not hashable")

            signatures.append(value)

    signature = tuple(signatures)
    args_signature = signature[:len(args)]
    kwargs_signature = signature[len(args):]
    ret = (keys, args_signature, kwargs_signature)
    return ret

def time_it(fn, *args, **kwargs):
    fn(*args, **kwargs)

    def run_it():
        res = fn(*args, **kwargs)
        jax.block_until_ready(res)

    run_it()
    run_it()
    return timeit.timeit(run_it, number=5)

# get optimal configuration for a given function
def autotune_instance(fn_path, args_tr, kwargs_tr, autotune_kws):
    # first check that all autotune guys are auto
    # spawn a new process and run the function a bunch of times with dummy data
    with open(fn_path, 'rb') as f:
        fn = dill.load(f)

    args = jax.tree_util.tree_map_with_path(from_shape_and_dtype, args_tr)
    kwargs = jax.tree_util.tree_map_with_path(from_shape_and_dtype, kwargs_tr)
    for k in autotune_kws[0].keys():
        # assert k in kwargs, f"autotune kw {k} not in kwargs"
        if k in kwargs:
            assert (v := kwargs[k]) == AUTOTUNE, f"autotune kw {k} must be AUTOTUNE, got {v}"

    for k, v in kwargs.items():
        if k in autotune_kws[0]:
            assert v == AUTOTUNE, f"autotune kw {k} must be None, got {v}"

    # try each config
    best_cfg = None
    best_time = float('inf')
    from collections import defaultdict
    max_block = defaultdict(lambda:float('inf'))
    assert 'block' in autotune_kws[0], "Must have block in autotune_kws"
    assert 'num_stages' in autotune_kws[0], "Must have num_stages in autotune_kws"
    from jaxlib.xla_extension import XlaRuntimeError

    iterator = tqdm.tqdm(autotune_kws, desc="🎛️ 🦉:  Autotuning progress", leave=True)
    for cfg in iterator:
        new_kwargs = kwargs | cfg
        num_stages = cfg['num_stages']
        if cfg['block'] >= max_block[num_stages]:
            continue

        try:
            cfg_time = time_it(fn, *args, **new_kwargs)
        except Exception as e: # XlaRuntimeError as e:
            if 'shared memory' in str(e).lower():
                cfg_time = float('inf')
                max_block[num_stages] = min(cfg['block'], max_block[num_stages])
                continue

            raise e

        iterator.set_postfix({'time': cfg_time})
        if cfg_time < best_time:
            best_time = cfg_time
            best_cfg = cfg
            if DEBUG:
                print_log('Found better config', best_cfg, best_time)

    assert best_cfg is not None, "🎛 🦉: No best config found, all failed"
    print_log('FOUND BEST CONFIG', best_cfg, best_time)
    return best_cfg

def _make_kw_grid(autotune_kw):
    at_keys = autotune_kw.keys()
    at_vals = autotune_kw.values()
    prod = list(product(*at_vals))
    kw_grid = [dict(zip(at_keys, p)) for p in prod]
    return kw_grid

def to_shape_and_dtype(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return MaybeArray(True, x.shape, x.dtype, None)

    return MaybeArray(False, None, None, x)

def from_shape_and_dtype(path, x):
    assert isinstance(x, MaybeArray), f"Expected MaybeArray, got {x}"
    # make random matrix
    if x.isarray:
        shape, dtype = x.shape, x.dtype
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape, dtype=dtype)
    else:
        return x.value

def autotune_wrapper(queue, autotuner, fn, args_tr, kwargs_tr, grid):
    # Call the function and put the result in the queue
    result = autotuner(fn, args_tr, kwargs_tr, grid)
    queue.put(result)

def run_in_spawned_process(autotune_instance, fn, args_tr, kwargs_tr, grid):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    tmp_file = f'/tmp/{random.randint(0, 1000000)}.pkl'
    with open(tmp_file, 'wb') as f:
        dill.dump(fn, f)

    args = (result_queue, autotune_instance, tmp_file, args_tr, kwargs_tr, grid)
    process = ctx.Process(target=autotune_wrapper, args=args)

    process.start()
    result = result_queue.get()
    process.join()

    return result

import hashlib
def deterministic_hash(obj):
    """Compute a deterministic hash for a hashable Python object."""
    data = str(obj)
    data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def _cfg_path(function_name, hashable):
    user = os.environ.get('USER', 'default')
    return f'/tmp/pallas_autotune_{user}/{function_name}_{deterministic_hash(hashable)}.pkl'

def get_cached_cfg(function_name, hashable):
    p = _cfg_path(function_name, hashable)
    if not os.path.exists(p):
        return None

    with open(p, 'rb') as f:
        cfg = dill.load(f)

    return cfg

def save_cached_cfg(function_name, hashable, opt_cfg):
    p = _cfg_path(function_name, hashable)
    p = Path(p)
    if not p.parent.exists():
        p.parent.mkdir(parents=True)

    assert opt_cfg is not None
    with open(p, 'wb') as f:
        dill.dump(opt_cfg, f)

# automatically test fn over grid of autotune_kw values
# given the other input dimensions
def autotuned(fn, _filter=None, **_kw_grid):
    full_grid = _make_kw_grid(_kw_grid)
    metadata_to_opt_cfg = {}

    def autotuned_fn(*args, **kwargs):
        if _filter is not None:
            grid = [cfg for cfg in full_grid if _filter(cfg, args, kwargs)]
        else:
            grid = full_grid

        if SKIP_AUTOTUNE:
            choice = grid[0]
            new_kw = kwargs | choice
            return fn(*args, **new_kw)

        hashable = hashable_for_arguments(*args, **kwargs)
        hash(hashable)
        grid_hash = deterministic_hash(grid)
        hashable = hashable + (grid_hash,)
        hash(hashable)
        if hashable in metadata_to_opt_cfg:
            opt_cfg = metadata_to_opt_cfg[hashable]
        else:
            function_name = fn.__name__
            opt_cfg = None
            if not DEBUG:
                # check if we have a cached version of the autotuned function
                # if so, load it and return it
                opt_cfg = get_cached_cfg(function_name, hashable)
            else:
                # clear out cache if it exists
                p = _cfg_path(function_name, hashable)
                print_log('DEBUG MODE: Clearing cache for ', p)
                parent = Path(p).parent
                # assert that parent is not /tmp
                assert parent != Path('/tmp')
                shutil.rmtree(parent, ignore_errors=True)

            # if it dont exist
            if opt_cfg is None:
                # grid: override
                if not DEBUG:
                    print_log(f'Cache miss: {function_name} @ {hashable}')

                args_tr, kwargs_tr = jax.tree.map(to_shape_and_dtype, (args, kwargs))
                # opt_cfg = autotune_instance(fn, args_tr, kwargs_tr, grid)
                # spawn multiprocessing process to run autotune_instance
                # p = multiprocessing.Process(target=autotune_instance, args=(fn, args_tr, kwargs_tr, grid))
                # check if curr process is the main
                is_main = mp.current_process().name == 'MainProcess'
                if not is_main:
                    raise ValueError("you cannot run `autotuned` in a child process.. common cause: your autotune code runs automatically on import. solution: use if __name__ == '__main__' then [your code here]")

                opt_cfg = run_in_spawned_process(autotune_instance, fn, args_tr,
                                                kwargs_tr, grid)
                if not DEBUG:
                    # save the optimal config to disk
                    save_cached_cfg(function_name, hashable, opt_cfg)
            else:
                print_log(f'Found cached config for {function_name} @ {hashable}')

            metadata_to_opt_cfg[hashable] = opt_cfg

        new_kw = kwargs | opt_cfg
        return fn(*args, **new_kw)

    return autotuned_fn
