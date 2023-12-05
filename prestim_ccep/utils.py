from functools import wraps

def run_once(fn):
    # ensure function only runs once (e.g. downloads, preprocess, etc.)
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return fn(*args, **kwargs)
        else:
            print(f"{fn.__name__} already completed")
    wrapper.has_run = False
    return wrapper
