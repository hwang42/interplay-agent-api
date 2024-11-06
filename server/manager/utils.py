from __future__ import annotations

import os.path
from typing import Optional

from jinja2 import Environment, PackageLoader, Template


class TemplateManager:
    def __init__(
            self,
            name: str,
            *,
            template_base: str = "template",
            template_file: str = "jinja2"
    ) -> None:
        self.extension = template_file
        self.environment = Environment(
            loader=PackageLoader(name, os.path.join(template_base))
        )

    def get(self, template: str, *, delimiter: str = ".") -> Template:
        filename = f"{template}{delimiter}{self.extension}"
        return self.environment.get_template(filename)

    def sys(
            self,
            prefix: Optional[str] = None,
            *,
            suffix: str = "sys",
            delimiter: str = "_"
    ) -> Template:
        if prefix is None:
            return self.get(suffix)

        return self.get(f"{prefix}{delimiter}{suffix}")

    def usr(
            self,
            prefix: Optional[str] = None,
            *,
            suffix: str = "usr",
            delimiter: str = "_"
    ) -> Template:
        return self.sys(prefix, suffix=suffix, delimiter=delimiter)
