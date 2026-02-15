"""Generic frozen ordered container for named elements.

IdList provides dict-like access by key (string) or position (int),
with helpful error messages including close-match suggestions.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import TYPE_CHECKING, Generic, TypeVar

from . import io as fx_io

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

T = TypeVar('T')


# Precompiled regex pattern for natural sorting
_NATURAL_SPLIT = re.compile(r'(\d+)')


def _natural_sort_key(text: str) -> list:
    """Sort key for natural ordering (e.g., bus1, bus2, bus10 instead of bus1, bus10, bus2)."""
    return [int(c) if c.isdigit() else c.lower() for c in _NATURAL_SPLIT.split(text)]


class IdList(Generic[T]):
    """Generic frozen ordered container for named elements.

    Backed by ``dict[str, T]`` internally. Provides dict-like access by
    primary key, optional short-key fallback, or positional index.

    Args:
        elements: Initial elements to add.
        key_fn: Callable extracting the primary key from an element.
        short_key_fn: Optional callable for fallback lookup (e.g. short id).
        display_name: Name shown in repr/error messages (e.g. 'inputs', 'components').

    Examples:
        >>> il = IdList([bus_a, bus_b], key_fn=lambda b: b.id_full, display_name='buses')
        >>> il['BusA']
        >>> il[0]
        >>> len(il)
        2
    """

    __slots__ = ('_data', '_key_fn', '_short_key_fn', '_display_name', '_truncate_repr')

    def __init__(
        self,
        elements: list[T] | None = None,
        *,
        key_fn: Callable[[T], str],
        short_key_fn: Callable[[T], str] | None = None,
        display_name: str = 'elements',
        truncate_repr: int | None = None,
    ) -> None:
        self._data: dict[str, T] = {}
        self._key_fn = key_fn
        self._short_key_fn = short_key_fn
        self._display_name = display_name
        self._truncate_repr = truncate_repr
        if elements:
            for elem in elements:
                self.add(elem)

    # --- mutation (build phase) -------------------------------------------

    def add(self, element: T) -> None:
        """Add *element* to the container (build phase).

        Raises:
            ValueError: If the key already exists.
        """
        key = self._key_fn(element)
        if key in self._data:
            item_name = element.__class__.__name__
            raise ValueError(
                f'{item_name} with id "{key}" already exists in {self._display_name}. '
                f'Each {item_name.lower()} must have a unique id.'
            )
        self._data[key] = element

    # --- read access ------------------------------------------------------

    def __getitem__(self, key: str | int) -> T:
        """Get element by primary key, short key, or positional index."""
        if isinstance(key, int):
            try:
                return list(self._data.values())[key]
            except IndexError:
                raise IndexError(
                    f'{self._display_name.capitalize()} index {key} out of range '
                    f'(container has {len(self._data)} items)'
                ) from None

        # Primary key lookup
        if key in self._data:
            return self._data[key]

        # Short-key fallback
        if self._short_key_fn is not None:
            for elem in self._data.values():
                if self._short_key_fn(elem) == key:
                    return elem

        # Error with suggestions
        self._raise_key_error(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._data:
            return True
        if self._short_key_fn is not None:
            return any(self._short_key_fn(elem) == key for elem in self._data.values())
        return False

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over primary keys."""
        return iter(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def values(self) -> list[T]:
        return list(self._data.values())

    def items(self) -> list[tuple[str, T]]:
        return list(self._data.items())

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get element by primary key, returning *default* if not found."""
        if key in self._data:
            return self._data[key]
        if self._short_key_fn is not None:
            for elem in self._data.values():
                if self._short_key_fn(elem) == key:
                    return elem
        return default

    # --- combination ------------------------------------------------------

    def __add__(self, other: IdList[T]) -> IdList[T]:
        """Return a new IdList combining elements from both lists."""
        result = IdList(
            key_fn=self._key_fn,
            short_key_fn=self._short_key_fn,
            display_name=self._display_name,
        )
        for elem in self._data.values():
            result.add(elem)
        for elem in other._data.values():
            result.add(elem)
        return result

    # --- repr -------------------------------------------------------------

    def _get_repr(self, max_items: int | None = None) -> str:
        limit = max_items if max_items is not None else self._truncate_repr
        count = len(self._data)
        title = f'{self._display_name.capitalize()} ({count} item{"s" if count != 1 else ""})'

        if not self._data:
            r = fx_io.format_title_with_underline(title)
            r += '<empty>\n'
        else:
            r = fx_io.format_title_with_underline(title)
            sorted_names = sorted(self._data.keys(), key=_natural_sort_key)
            if limit is not None and limit > 0 and len(sorted_names) > limit:
                for name in sorted_names[:limit]:
                    r += f' * {name}\n'
                r += f' ... (+{len(sorted_names) - limit} more)\n'
            else:
                for name in sorted_names:
                    r += f' * {name}\n'
        return r

    def __repr__(self) -> str:
        return self._get_repr()

    # --- helpers ----------------------------------------------------------

    def _raise_key_error(self, key: str) -> None:
        """Raise a KeyError with helpful suggestions."""
        suggestions = get_close_matches(key, self._data.keys(), n=3, cutoff=0.6)
        # Also check short keys for suggestions
        if self._short_key_fn is not None:
            short_keys = [self._short_key_fn(e) for e in self._data.values()]
            suggestions += get_close_matches(key, short_keys, n=3, cutoff=0.6)

        error_msg = f'"{key}" not found in {self._display_name}.'
        if suggestions:
            error_msg += f' Did you mean: {", ".join(suggestions)}?'
        else:
            available = list(self._data.keys())
            if len(available) <= 5:
                error_msg += f' Available: {", ".join(available)}'
            else:
                error_msg += f' Available: {", ".join(available[:5])} ... (+{len(available) - 5} more)'
        raise KeyError(error_msg) from None


# --- factory helpers -------------------------------------------------------


def flow_id_list(flows: list | None = None, **kw) -> IdList:
    """Create an IdList keyed by ``flow.id`` with short-key fallback to ``flow.flow_id``."""
    return IdList(flows, key_fn=lambda f: f.id, short_key_fn=lambda f: f.flow_id, **kw)


def element_id_list(elements: list | None = None, **kw) -> IdList:
    """Create an IdList keyed by ``element.id``."""
    return IdList(elements, key_fn=lambda e: e.id, **kw)
