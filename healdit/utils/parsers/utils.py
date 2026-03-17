from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Type


def comma_list_to_list(comma_list: str, type_: Type = str) -> List[int | str]:
    return [type_(item.strip()) for item in comma_list.split(",") if item.strip()]