from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Matrix = list[list[int]]

VERIFICATION_INDICES: tuple[int, ...] = (0, 1, 2, 5, 10, 20)
COMPARISON_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    20_000,
)
# Each pair is (index, number of timed repetitions). The reported times are
# measurements of those repetitions, not analytic estimates.
TIMING_CASES: tuple[tuple[int, int], ...] = (
    (500, 7),
    (2_000, 5),
    (10_000, 3),
)


@dataclass(frozen=True)
class FibonacciComputation:
    """Computed Fibonacci value and the matrix power that produced it.

    ``matrix_multiplications`` counts calls to the 2 by 2 product. It does
    not count the eight scalar products inside each matrix product.
    """

    value: int
    matrix_multiplications: int
    power: Matrix


@dataclass(frozen=True)
class IndexCheck:
    """Agreement of both matrix methods with the defining recurrence."""

    n: int
    linear: FibonacciComputation
    logarithmic: FibonacciComputation
    reference: int
    values_match: bool
    matrices_match: bool
    multiplication_counts_match: bool

    @property
    def passed(self) -> bool:
        """Return whether every check for this index succeeded."""
        return (
            self.values_match
            and self.matrices_match
            and self.multiplication_counts_match
        )


def validate_index(n: int) -> None:
    """Raise if ``n`` is not a non-negative integer.

    Boolean values are rejected because ``bool`` is a subclass of ``int``.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be a non-negative integer")


def _require_2x2(matrix: Matrix, name: str) -> None:
    """Raise if ``matrix`` is not a 2 by 2 matrix of integers."""
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError(f"{name} must be a 2 by 2 matrix")
    for row in matrix:
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(f"{name} entries must be integers")


def copy_matrix(matrix: Matrix) -> Matrix:
    """Return a shallow copy of a 2 by 2 matrix."""
    return [row[:] for row in matrix]


def identity_2x2() -> Matrix:
    """Return the 2 by 2 multiplicative identity."""
    return [[1, 0], [0, 1]]


def fibonacci_base_matrix() -> Matrix:
    """Return the Fibonacci matrix ``[[0, 1], [1, 1]]``."""
    return [[0, 1], [1, 1]]


def multiply_2x2(left: Matrix, right: Matrix) -> Matrix:
    """Return the product of two 2 by 2 integer matrices.

    The product uses eight integer multiplications and four additions.
    Python integers grow as needed, so the entries are not reduced modulo
    a fixed word size.
    """
    _require_2x2(left, "left")
    _require_2x2(right, "right")
    return [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]


def matrix_power_linear(base: Matrix, exponent: int) -> tuple[Matrix, int]:
    """Raise ``base`` to ``exponent`` by successive multiplication.

    The exponent 0 returns the identity and performs no multiplication.
    A positive exponent starts at ``base`` and multiplies by it
    ``exponent - 1`` times, so the multiplication count is linear in the
    exponent.
    """
    validate_index(exponent)
    _require_2x2(base, "base")
    if exponent == 0:
        return identity_2x2(), 0

    result = copy_matrix(base)
    multiplications = 0
    for _ in range(1, exponent):
        result = multiply_2x2(result, base)
        multiplications += 1
    return result, multiplications


def matrix_power_logarithmic(base: Matrix, exponent: int) -> tuple[Matrix, int]:
    """Raise ``base`` to ``exponent`` by exponentiation by squaring.

    Even exponents square the power whose exponent is ``exponent // 2``.
    Odd exponents greater than 1 do the same and then multiply by ``base``.
    For an odd exponent, ``exponent // 2`` equals ``(exponent - 1) // 2``,
    so that branch is ``(base ** ((exponent - 1) // 2)) ** 2 * base``.
    The exponent 0 returns the identity. Each recursive call receives an
    exponent at most half of the current one.
    """
    validate_index(exponent)
    _require_2x2(base, "base")
    if exponent == 0:
        return identity_2x2(), 0
    if exponent == 1:
        return copy_matrix(base), 0

    half, half_multiplications = matrix_power_logarithmic(base, exponent // 2)
    squared = multiply_2x2(half, half)
    if exponent % 2 == 0:
        return squared, half_multiplications + 1
    return multiply_2x2(squared, base), half_multiplications + 2


def logarithmic_multiplication_count(exponent: int) -> int:
    """Count squarings and extra base products without building matrices.

    The recurrence mirrors ``matrix_power_logarithmic`` and is used only to
    check that the instrumented counter increments once per product.
    """
    validate_index(exponent)
    if exponent <= 1:
        return 0
    extra = 1 if exponent % 2 == 0 else 2
    return logarithmic_multiplication_count(exponent // 2) + extra


def fibonacci_entry(n: int, power: Matrix) -> int:
    """Return F_n from ``power``, which must be M^n.

    For n >= 1 the identity M^n = [[F_{n-1}, F_n], [F_n, F_{n+1}]] places
    F_n in row 0, column 1. For n = 0 that identity would refer to F_{-1},
    so F_0 is defined directly as 0 and the matrix entry is not read.
    """
    validate_index(n)
    if n == 0:
        return 0
    _require_2x2(power, "power")
    return power[0][1]


def fibonacci_linear(n: int) -> FibonacciComputation:
    """Compute F_n with a linear number of matrix multiplications."""
    validate_index(n)
    power, multiplications = matrix_power_linear(fibonacci_base_matrix(), n)
    return FibonacciComputation(
        value=fibonacci_entry(n, power),
        matrix_multiplications=multiplications,
        power=power,
    )


def fibonacci_logarithmic(n: int) -> FibonacciComputation:
    """Compute F_n with a logarithmic number of matrix multiplications."""
    validate_index(n)
    power, multiplications = matrix_power_logarithmic(fibonacci_base_matrix(), n)
    return FibonacciComputation(
        value=fibonacci_entry(n, power),
        matrix_multiplications=multiplications,
        power=power,
    )


def fibonacci_by_recurrence(n: int) -> int:
    """Compute F_n from F_0 = 0, F_1 = 1 and F_k = F_{k-1} + F_{k-2}.

    This walk is an independent check. It is not a matrix algorithm.
    """
    validate_index(n)
    if n == 0:
        return 0
    previous = 0
    current = 1
    for _ in range(1, n):
        previous, current = current, previous + current
    return current


def expected_power(n: int) -> Matrix:
    """Return the matrix that the Fibonacci identity assigns to M^n.

    For n = 0 the result is the identity. The closed form that mentions
    F_{n-1} is applied only for n >= 1.
    """
    validate_index(n)
    if n == 0:
        return identity_2x2()
    previous = fibonacci_by_recurrence(n - 1)
    current = fibonacci_by_recurrence(n)
    following = fibonacci_by_recurrence(n + 1)
    return [[previous, current], [current, following]]


def check_index(n: int) -> IndexCheck:
    """Compare both matrix methods with the recurrence for one index."""
    linear = fibonacci_linear(n)
    logarithmic = fibonacci_logarithmic(n)
    reference = fibonacci_by_recurrence(n)
    expected = expected_power(n)
    linear_count = 0 if n == 0 else n - 1
    values_match = linear.value == logarithmic.value == reference
    matrices_match = linear.power == logarithmic.power == expected
    counts_match = (
        linear.matrix_multiplications == linear_count
        and logarithmic.matrix_multiplications
        == logarithmic_multiplication_count(n)
    )
    return IndexCheck(
        n=n,
        linear=linear,
        logarithmic=logarithmic,
        reference=reference,
        values_match=values_match,
        matrices_match=matrices_match,
        multiplication_counts_match=counts_match,
    )


def format_matrix(matrix: Matrix) -> str:
    """Format a 2 by 2 matrix on one line."""
    return f"[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]"


def format_yes_no(value: bool) -> str:
    """Return a Spanish yes or no."""
    return "sí" if value else "no"


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Print an aligned text table."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: Sequence[str]) -> str:
        return " | ".join(
            cell.rjust(widths[index]) for index, cell in enumerate(cells)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def print_environment() -> None:
    """Print the interpreter and machine used for this execution."""
    print("Entorno de ejecución")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Ejecutable: {sys.executable}")
    print(f"  Plataforma: {platform.platform()}")
    print(f"  Arquitectura: {platform.machine()}")
    processor = platform.processor()
    if processor:
        print(f"  Procesador: {processor}")


def print_single(n: int) -> int:
    """Print both methods for one index. Return 0 when they agree."""
    try:
        result = check_index(n)
    except (TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"n = {n}")
    print(
        "  Método lineal:      "
        f"F_{n} = {result.linear.value} "
        f"(multiplicaciones de matrices: {result.linear.matrix_multiplications})"
    )
    print(
        "  Método logarítmico: "
        f"F_{n} = {result.logarithmic.value} "
        f"(multiplicaciones de matrices: {result.logarithmic.matrix_multiplications})"
    )
    print(f"  Referencia por recurrencia: F_{n} = {result.reference}")
    print(f"  Los tres valores coinciden: {format_yes_no(result.values_match)}")
    print(f"  M^{n} = {format_matrix(result.logarithmic.power)}")
    print(f"  Ambas potencias coinciden con la matriz esperada: {format_yes_no(result.matrices_match)}")
    if n == 0:
        print(
            "  El caso n = 0 se resuelve de forma explícita: F_0 = 0. "
            "No se lee una entrada como F_{-1}."
        )
        print(
            "  M^0 es la identidad. Su esquina superior izquierda vale 1; "
            "la fórmula con F_{n-1} no se aplica en este caso."
        )
    else:
        print(
            f"  La entrada superior derecha de M^{n} es F_{n}. "
            "La fórmula matricial se usa solo para n >= 1."
        )
    if not result.passed:
        print("  La comprobación de este índice falló.", file=sys.stderr)
        return 1
    return 0


def print_verification(indices: Sequence[int] = VERIFICATION_INDICES) -> int:
    """Check the required indices and print each matrix. Return 0 on success."""
    print("Verificación")
    print(
        "La referencia aplica la recurrencia F_0 = 0, F_1 = 1 y "
        "F_k = F_{k-1} + F_{k-2}. No usa potencias de matrices."
    )
    exit_code = 0
    rows: list[list[str]] = []
    for n in indices:
        result = check_index(n)
        rows.append(
            [
                str(n),
                str(result.reference),
                str(result.linear.matrix_multiplications),
                str(result.logarithmic.matrix_multiplications),
                format_yes_no(result.passed),
            ]
        )
        if not result.passed:
            exit_code = 1
        print()
        print(f"n = {n}")
        print(f"  F_n = {result.reference}")
        print(f"  M^{n} = {format_matrix(result.linear.power)}")
        print(
            "  Potencias lineal y logarítmica iguales a la matriz esperada: "
            f"{format_yes_no(result.matrices_match)}"
        )
        if n == 0:
            print("  F_0 se devuelve como 0 sin usar F_{-1}.")
    print()
    print_table(
        ["n", "F_n", "mult. lineal", "mult. logarítmica", "comprobación"],
        rows,
    )
    if exit_code == 0:
        print("Resultado: todas las comprobaciones pasaron.")
    else:
        print("Resultado: alguna comprobación falló.", file=sys.stderr)
    return exit_code


def print_comparison(indices: Sequence[int] = COMPARISON_INDICES) -> int:
    """Print exact multiplication counts for several indices."""
    print("Comparación del número de multiplicaciones de matrices")
    print(
        "Estos conteos los registra el programa en cada llamada a multiply_2x2. "
        "No son estimaciones."
    )
    exit_code = 0
    rows: list[list[str]] = []
    for n in indices:
        result = check_index(n)
        if not result.passed:
            exit_code = 1
        logarithmic_count = result.logarithmic.matrix_multiplications
        if logarithmic_count == 0:
            ratio = "no aplica"
        else:
            ratio = f"{result.linear.matrix_multiplications / logarithmic_count:.2f}"
        rows.append(
            [
                str(n),
                str(result.linear.matrix_multiplications),
                str(logarithmic_count),
                ratio,
                str(result.reference.bit_length()),
                str(len(str(result.reference))),
                format_yes_no(result.values_match),
            ]
        )
    print_table(
        [
            "n",
            "mult. lineal",
            "mult. logarítmica",
            "cociente",
            "bits de F_n",
            "dígitos",
            "valores iguales",
        ],
        rows,
    )
    return exit_code


def sample_seconds(
    function: Callable[[int], FibonacciComputation],
    n: int,
    repeats: int,
) -> list[float]:
    """Time ``repeats`` calls after one untimed warm-up call.

    The clock is ``time.perf_counter_ns``. Garbage collection stays enabled,
    as it would be in a normal execution.
    """
    function(n)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function(n)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
        samples.append(elapsed)
    return samples


def median(samples: Sequence[float]) -> float:
    """Return the median of a non-empty sequence."""
    ordered = sorted(samples)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def print_timings(cases: Sequence[tuple[int, int]] = TIMING_CASES) -> int:
    """Measure both methods and print minimum and median times."""
    print("Tiempos medidos")
    print(
        "Cada celda es un tiempo observado en este proceso, en segundos. "
        "Antes de las repeticiones hay una llamada de calentamiento que no se mide. "
        "Se muestra el mínimo y la mediana. No son estimaciones y pueden cambiar "
        "con el equipo, la carga del sistema y la versión de Python."
    )
    exit_code = 0
    rows: list[list[str]] = []
    for n, repeats in cases:
        result = check_index(n)
        if not result.passed:
            exit_code = 1
        linear_samples = sample_seconds(fibonacci_linear, n, repeats)
        logarithmic_samples = sample_seconds(fibonacci_logarithmic, n, repeats)
        rows.append(
            [
                str(n),
                str(repeats),
                f"{min(linear_samples):.6f}",
                f"{median(linear_samples):.6f}",
                f"{min(logarithmic_samples):.6f}",
                f"{median(logarithmic_samples):.6f}",
            ]
        )
    print_table(
        [
            "n",
            "repeticiones",
            "lineal mín. (s)",
            "lineal mediana (s)",
            "logarítmico mín. (s)",
            "logarítmico mediana (s)",
        ],
        rows,
    )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calcula F_n con la matriz [[0, 1], [1, 1]] "
            "por multiplicación sucesiva y por exponenciación rápida."
        )
    )
    parser.add_argument(
        "n",
        nargs="?",
        type=_non_negative_argument,
        help="Índice no negativo. Muestra ambos métodos para ese valor.",
    )
    parser.add_argument(
        "--verificar",
        action="store_true",
        help="Comprueba n = 0, 1, 2, 5, 10 y 20.",
    )
    parser.add_argument(
        "--comparar",
        action="store_true",
        help=(
            "Compara el número de multiplicaciones de matrices "
            "y mide tiempos de ejecución."
        ),
    )
    return parser


def _non_negative_argument(text: str) -> int:
    """Parse a command-line index. Reject signs, decimals and negatives."""
    if text.startswith("+") or not text.isdigit():
        raise argparse.ArgumentTypeError("n debe ser un entero no negativo")
    return int(text)


def main(argv: list[str] | None = None) -> int:
    """Run the requested reports. Return a process status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_single = args.n is not None
    run_verification = args.verificar
    run_comparison = args.comparar
    if not run_single and not run_verification and not run_comparison:
        run_verification = True
        run_comparison = True

    print_environment()
    print()
    exit_code = 0
    if run_single:
        exit_code = max(exit_code, print_single(args.n))
        print()
    if run_verification:
        exit_code = max(exit_code, print_verification())
        print()
    if run_comparison:
        exit_code = max(exit_code, print_comparison())
        print()
        exit_code = max(exit_code, print_timings())
        print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
