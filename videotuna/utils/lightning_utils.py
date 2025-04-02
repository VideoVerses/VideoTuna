import inspect
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import pytorch_lightning as pl

_ARGPARSE_CLS = Union[Type["pl.LightningDataModule"], Type["pl.Trainer"]]


def _get_abbrev_qualified_cls_name(cls: _ARGPARSE_CLS) -> str:
    assert isinstance(cls, type), repr(cls)
    if cls.__module__.startswith("pytorch_lightning."):
        # Abbreviate.
        return f"pl.{cls.__name__}"
    # Fully qualified.
    return f"{cls.__module__}.{cls.__qualname__}"


def get_init_arguments_and_types(cls: _ARGPARSE_CLS) -> List[Tuple[str, Tuple, Any]]:
    r"""Scans the class signature and returns argument names, types and default values.

    Returns:
        List with tuples of 3 values:
        (argument name, set with argument types, argument default value).

    Examples:

        >>> from pytorch_lightning import Trainer
        >>> args = get_init_arguments_and_types(Trainer)

    """
    cls_default_params = inspect.signature(cls).parameters
    name_type_default = []
    for arg in cls_default_params:
        arg_type = cls_default_params[arg].annotation
        arg_default = cls_default_params[arg].default
        try:
            arg_types = tuple(arg_type.__args__)
        except (AttributeError, TypeError):
            arg_types = (arg_type,)

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default


def _parse_args_from_docstring(docstring: str) -> Dict[str, str]:
    arg_block_indent = None
    current_arg = ""
    parsed = {}
    for line in docstring.split("\n"):
        stripped = line.lstrip()
        if not stripped:
            continue
        line_indent = len(line) - len(stripped)
        if stripped.startswith(("Args:", "Arguments:", "Parameters:")):
            arg_block_indent = line_indent + 4
        elif arg_block_indent is None:
            continue
        elif line_indent < arg_block_indent:
            break
        elif line_indent == arg_block_indent:
            current_arg, arg_description = stripped.split(":", maxsplit=1)
            parsed[current_arg] = arg_description.lstrip()
        elif line_indent > arg_block_indent:
            parsed[current_arg] += f" {stripped}"
    return parsed


def _gpus_allowed_type(x: str) -> Union[int, str]:
    if "," in x:
        return str(x)
    return int(x)


def _precision_allowed_type(x: Union[int, str]) -> Union[int, str]:
    """
    >>> _precision_allowed_type("32")
    32
    >>> _precision_allowed_type("bf16")
    'bf16'
    """
    try:
        return int(x)
    except ValueError:
        return x


def str_to_bool_or_str(val: str) -> Union[str, bool]:
    """Possibly convert a string representation of truth to bool. Returns the input otherwise. Based on the python
    implementation distutils.utils.strtobool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    """
    lower = val.lower()
    if lower in ("y", "yes", "t", "true", "on", "1"):
        return True
    if lower in ("n", "no", "f", "false", "off", "0"):
        return False
    return val


def str_to_bool(val: str) -> bool:
    """Convert a string representation of truth to bool.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises:
        ValueError:
            If ``val`` isn't in one of the aforementioned true or false values.
    >>> str_to_bool('YES')
    True
    >>> str_to_bool('FALSE')
    False
    """
    val_converted = str_to_bool_or_str(val)
    if isinstance(val_converted, bool):
        return val_converted
    raise ValueError(f"invalid truth value {val_converted}")


def str_to_bool_or_int(val: str) -> Union[bool, int, str]:
    """Convert a string representation to truth of bool if possible, or otherwise try to convert it to an int.
    >>> str_to_bool_or_int("FALSE")
    False
    >>> str_to_bool_or_int("1")
    True
    >>> str_to_bool_or_int("2")
    2
    >>> str_to_bool_or_int("abc")
    'abc'
    """
    val_converted = str_to_bool_or_str(val)
    if isinstance(val_converted, bool):
        return val_converted
    try:
        return int(val_converted)
    except ValueError:
        return val_converted


def _int_or_float_type(x: Union[int, float, str]) -> Union[int, float]:
    if "." in str(x):
        return float(x)
    return int(x)


def add_trainer_args_to_parser(cls, parent_parser, use_argument_group=True):
    if not isinstance(parent_parser, ArgumentParser):
        raise RuntimeError("Please only pass an `ArgumentParser` instance.")
    if use_argument_group:
        group_name = _get_abbrev_qualified_cls_name(cls)
        parser = parent_parser.add_argument_group(group_name)
    else:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

    ignore_arg_names = ["self", "args", "kwargs"]

    allowed_types = (str, int, float, bool)

    # Get symbols from cls or init function.
    for symbol in (cls, cls.__init__):
        args_and_types = get_init_arguments_and_types(symbol)  # type: ignore[arg-type]
        args_and_types = [x for x in args_and_types if x[0] not in ignore_arg_names]
        if len(args_and_types) > 0:
            break

    args_help = _parse_args_from_docstring(cls.__init__.__doc__ or cls.__doc__ or "")

    for arg, arg_types, arg_default in args_and_types:
        arg_types = tuple(at for at in allowed_types if at in arg_types)
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs: Dict[str, Any] = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type: Callable[[str], Union[bool, int, float, str]] = str_to_bool
            elif int in arg_types:
                use_type = str_to_bool_or_int
            elif str in arg_types:
                use_type = str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == "gpus" or arg == "tpu_cores":
            use_type = _gpus_allowed_type

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == "track_grad_norm":
            use_type = float

        # hack for precision
        if arg == "precision":
            use_type = _precision_allowed_type

        try:
            parser.add_argument(
                f"--{arg}",
                dest=arg,
                default=arg_default,
                type=use_type,
                help=args_help.get(arg),
                required=(arg_default == inspect._empty),
                **arg_kwargs,
            )
        except Exception:
            # TODO: check the argument appending to the parser
            pass

    if use_argument_group:
        return parent_parser
    return parser
