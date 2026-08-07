"""
Helper utilities for TATVA CLI output formatting.
"""

import json
from typing import Any, Sequence

import click


def print_json(data: Any) -> None:
    """
    Print machine-readable JSON to stdout.
    """
    click.echo(json.dumps(data, indent=2))


def print_header(title: str) -> None:
    """
    Print a consistent, clean CLI header section.
    """
    click.echo(f"=== {title} ===")


def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], col_widths: Sequence[int]) -> None:
    """
    Print a text table with consistent column widths.
    """
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    click.echo(header_str)

    divider = "-" * (sum(col_widths) + 3 * (len(headers) - 1))
    click.echo(divider)

    for row in rows:
        formatted_row = []
        for cell, width in zip(row, col_widths):
            # Check for click style components
            cell_str = str(cell)
            formatted_row.append(f"{cell_str:<{width}}")
        click.echo(" | ".join(formatted_row))
