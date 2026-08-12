"""
Command Line Interface for TATVA.
"""

import os
import sys
from typing import Any, NoReturn

import click

from tatva import __version__
from tatva._output import print_header, print_json, print_table
from tatva.compiler import (
    DEFAULT_TARGET,
    TARGETS,
    ImportInProgressError,
    UnsupportedOperatorError,
    analyze_graph,
    import_model,
)
from tatva.config import get_anthropic_api_key
from tatva.runner import (
    establish_baseline,
    find_qemu,
    find_riscv_gcc,
    verify_target,
)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_target(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
    """
    Validate the selected target configuration and gate experimental targets.
    """
    if value is None:
        return None
    value_upper = value.upper()
    if value_upper not in TARGETS:
        raise click.BadParameter(
            f"Unknown target variant '{value}'. Supported targets: {', '.join(TARGETS.keys())}"
        )

    variant = TARGETS[value_upper]
    allow_experimental = ctx.params.get("allow_experimental", False)
    if variant.experimental and not allow_experimental:
        raise click.UsageError(
            f"Target variant '{value}' is experimental. Use '--allow-experimental' to enable this configuration."
        )

    return variant


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="tatva")
@click.option("-v", "--verbose", count=True, help="Increase verbosity level (-v for INFO, -vv for DEBUG).")
@click.option("--debug", is_flag=True, help="Enable detailed debug logging and tracebacks.")
@click.option("--log-file", type=click.Path(), help="Write structured logs to specified file.")
@click.option("--json-log", is_flag=True, help="Format log output as JSON lines.")
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: int = 0,
    debug: bool = False,
    log_file: Any = None,
    json_log: bool = False,
) -> None:
    """
    TATVA: RISC-V Transformer Optimization Toolchain.

    Provides commands to import, analyze, compile, optimize, and simulate
    Transformer models for bare-metal RISC-V targets.
    """
    from tatva.config import load_dotenv_file
    from tatva.logging_setup import configure_logging

    configure_logging(verbosity=verbose, debug=debug, log_file=log_file, json_log=json_log)

    # Before anything reads a key. Nothing used to load .env despite every error
    # message telling people to put their key there.
    load_dotenv_file()

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    ctx.obj["log_file"] = log_file

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def handle_cli_exception(
    e: Exception, debug: bool = False, json_format: bool = False
) -> NoReturn:
    """
    Unified CLI error boundary. Always exits non-zero.

    This existed and was never called: baseline-test, optimize and diagnose each had
    their own near-copy, and two of them checked os.environ["ANTHROPIC_API_KEY"]
    directly -- so a user who had set TATVA_ANTHROPIC_KEY was told to go set a key
    they had already set.
    """
    from tatva.diagnostics import classify_failure, explain
    from tatva.logging_setup import get_logger

    logger = get_logger("cli")
    # exc_info is gated on --debug on purpose. This logger has a console handler, so
    # logger.exception() here would dump a Python traceback at every user who mistyped
    # a path -- the exact thing the friendly explanation below exists to replace.
    logger.error("%s", e, exc_info=debug)

    context = classify_failure(e)

    if json_format:
        print_json(
            {
                "status": "error",
                "error_type": context.error_type,
                "metadata": context.metadata,
            }
        )
        sys.exit(1)

    if context.error_type != "unknown":
        if not get_anthropic_api_key():
            click.secho(
                "Note: Configure TATVA_ANTHROPIC_KEY or ANTHROPIC_API_KEY environment variable to retrieve richer, AI-generated Claude API explanations.",
                fg="cyan",
            )
        click.secho("\nDiagnostics Explanation:", fg="red", bold=True)
        click.echo(explain(context))
        sys.exit(1)
    else:
        # Unknown / unexpected exception
        click.secho(
            "\nAn unexpected error occurred while processing your request.",
            fg="red",
            bold=True,
        )
        click.echo(f"Error Summary: {e}")
        if debug:
            import traceback

            click.secho("\nTraceback (Debug Mode Enabled):", fg="yellow")
            traceback.print_exc()
        else:
            click.echo(
                "To enable full debug tracing or record log output to a file, re-run with:"
            )
            click.secho(
                "  tatva --debug --log-file tatva.log <command>", fg="cyan"
            )
        sys.exit(1)


def _searched_paths(exe: str, install_name: str, legacy_dir: str) -> list[str]:
    """
    The places a toolchain binary is looked for, in order.

    "not found" is not a useful diagnosis on its own -- the next question is always
    "where did you look?", and until now the answer lived only in the source.
    """
    from tatva.runner import _search_bin_dirs

    return ["PATH", *(os.path.join(d, exe) for d in _search_bin_dirs(install_name, legacy_dir))]


@cli.command("doctor")
@click.option("--json", "json_format", is_flag=True, help="Print machine-readable JSON status.")
def doctor(json_format: bool) -> None:
    """
    Run environment and toolchain health checks.

    Usage Example:
      tatva doctor
    """
    status: dict[str, dict[str, Any]] = {}
    all_ok = True

    # 1. Check Python
    python_ver = sys.version.split()[0]
    status["python"] = {"status": "ok", "version": python_ver}

    # 2. Check TVM
    try:
        import tvm  # type: ignore
        status["tvm"] = {"status": "ok", "version": tvm.__version__}
    except ImportError:
        all_ok = False
        status["tvm"] = {
            "status": "error",
            "version": None,
            "error": "apache-tvm package is not installed. Run `pip install apache-tvm` or refer to docs/TOOLCHAIN.md.",
        }

    # 3. Check ONNX Runtime
    try:
        import onnxruntime  # type: ignore
        status["onnxruntime"] = {"status": "ok", "version": onnxruntime.__version__}
    except ImportError:
        all_ok = False
        status["onnxruntime"] = {
            "status": "error",
            "version": None,
            "error": "onnxruntime package is not installed. Run `pip install onnxruntime`.",
        }

    # 4. Check RISC-V GCC compiler
    _gcc_name, gcc_path = find_riscv_gcc()
    if gcc_path:
        try:
            # Eagerly fetch version
            import subprocess
            res = subprocess.run([gcc_path, "--version"], capture_output=True, text=True, timeout=5, check=True)
            gcc_version = res.stdout.splitlines()[0].strip()
            status["riscv_gcc"] = {"status": "ok", "path": gcc_path, "version": gcc_version}
        except Exception as e:
            all_ok = False
            status["riscv_gcc"] = {
                "status": "error",
                "path": gcc_path,
                "version": None,
                "error": f"Failed to execute RISC-V GCC: {e}",
            }
    else:
        all_ok = False
        status["riscv_gcc"] = {
            "status": "error",
            "path": None,
            "version": None,
            "error": "RISC-V GCC cross-compiler binary not found. Run `tatva setup` to install it, "
                     "or see docs/TOOLCHAIN.md to use your own.",
            "searched": _searched_paths("riscv-none-elf-gcc", "riscv-none-elf-gcc", "riscv-toolchain"),
        }

    # 5. Check QEMU emulator
    _qemu_name, qemu_path = find_qemu(64)
    if qemu_path:
        try:
            import subprocess
            res = subprocess.run([qemu_path, "--version"], capture_output=True, text=True, timeout=5, check=True)
            qemu_version = res.stdout.splitlines()[0].strip()
            status["qemu"] = {"status": "ok", "path": qemu_path, "version": qemu_version}
        except Exception as e:
            all_ok = False
            status["qemu"] = {
                "status": "error",
                "path": qemu_path,
                "version": None,
                "error": f"Failed to execute QEMU: {e}",
            }
    else:
        all_ok = False
        status["qemu"] = {
            "status": "error",
            "path": None,
            "version": None,
            "error": "QEMU emulator binary not found. Run `tatva setup` to install it, "
                     "or see docs/TOOLCHAIN.md to use your own.",
            "searched": _searched_paths("qemu-system-riscv64", "qemu-riscv", "qemu"),
        }

    if json_format:
        print_json(status)
    else:
        print_header("TATVA Environment & Toolchain Doctor")
        for tool, info in status.items():
            if info["status"] == "ok":
                version_str = f" (version: {info['version']})" if info.get("version") else ""
                click.echo(f"[OK] {tool}{version_str}")
            else:
                click.echo(f"[ERROR] {tool} check failed!")
                click.echo(f"  Reason: {info['error']}")
                for location in info.get("searched", []):
                    click.echo(f"  Looked in: {location}")

    if not all_ok:
        sys.exit(1)


@cli.command("setup")
@click.option(
    "--component",
    "-c",
    type=click.Choice(["all", "gcc", "qemu"]),
    default="all",
    help="Which toolchain component to install.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be downloaded and where, then stop.")
@click.option("--force", is_flag=True, help="Reinstall even if the component is already present.")
@click.option("--yes", "-y", is_flag=True, help="Do not prompt before downloading.")
def setup(component: str, dry_run: bool, force: bool, yes: bool) -> None:
    """
    Install the pinned RISC-V cross-compiler and QEMU emulator.

    Downloads prebuilt xPack binaries for this platform into a per-user tools directory
    (override with TATVA_TOOLS_DIR). Already have your own toolchain on PATH? You do not
    need this -- run `tatva doctor` to confirm TATVA can see it.

    Usage Example:
      tatva setup --dry-run
      tatva setup --component qemu
    """
    from tatva.toolchain import (
        ToolchainUnavailableError,
        install_component,
        plan_install,
        tools_dir,
    )

    keys = ["gcc", "qemu"] if component == "all" else [component]

    try:
        plans = [plan_install(key) for key in keys]
    except ToolchainUnavailableError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    print_header("TATVA Toolchain Setup")
    click.echo(f"Tools directory: {tools_dir()}\n")
    for plan in plans:
        click.echo(plan.describe())
        click.echo("")

    pending = [p for p in plans if force or not p.already_installed]
    if not pending:
        click.secho("Everything is already installed. Re-run with --force to reinstall.", fg="green")
        return

    total_mb = sum(p.component.approx_size_mb for p in pending)

    if dry_run:
        click.secho(f"Dry run: nothing downloaded. {len(pending)} component(s), ~{total_mb} MB.", fg="yellow")
        return

    # Downloading half a gigabyte is not something to do because someone typed a command
    # they were guessing at.
    if not yes and not click.confirm(f"Download and install {len(pending)} component(s) (~{total_mb} MB)?"):
        click.echo("Aborted.")
        sys.exit(1)

    failures = []
    for plan in pending:
        click.secho(f"Installing {plan.component.label}...", fg="cyan")
        try:
            path = install_component(plan.component.key, force=force)
            click.secho(f"  OK: {path}", fg="green")
        except ToolchainUnavailableError as e:
            failures.append((plan.component.key, str(e)))
            click.secho(f"  FAILED: {e}", fg="red", err=True)

    if failures:
        click.secho(
            f"\n{len(failures)} component(s) failed to install. "
            "TATVA also accepts any riscv-none-elf-gcc and qemu-system-riscv64 on PATH.",
            fg="red",
            err=True,
        )
        sys.exit(1)

    click.secho("\nDone. Run `tatva doctor` to confirm.", fg="green")


@cli.command("targets")
@click.argument("name", required=False)
@click.option("--verify", is_flag=True, help="Verify target configurations using compiler and QEMU tests.")
@click.option("--json", "json_format", is_flag=True, help="Print machine-readable JSON status.")
def targets(name: str | None, verify: bool, json_format: bool) -> None:
    """
    List and verify supported RISC-V target architectures.

    Usage Example:
      tatva targets --verify
    """
    targets_to_process = {}
    if name:
        name_upper = name.upper()
        if name_upper not in TARGETS:
            click.echo(f"Error: Unknown target variant '{name}'.", err=True)
            click.echo(f"Supported targets: {', '.join(TARGETS.keys())}", err=True)
            sys.exit(1)
        targets_to_process = {name_upper: TARGETS[name_upper]}
    else:
        targets_to_process = TARGETS

    if verify:
        results = {}
        all_passed = True

        for tgt_name, variant in targets_to_process.items():
            if variant.experimental:
                results[tgt_name] = {"status": "skipped", "error": "Experimental target verification bypassed by default."}
                continue

            res = verify_target(variant)
            results[tgt_name] = res
            if res["status"] != "ok":
                all_passed = False

        if json_format:
            print_json(results)
        else:
            print_header("Target Verification Results")
            headers = ["Target Name", "Status", "Details / Errors"]
            rows = []
            for tgt_name, res in results.items():
                status_str = res["status"].upper()
                details = "Verified successfully" if status_str == "OK" else res.get("error", "Verification failed")
                rows.append([tgt_name, status_str, details])
            print_table(headers, rows, [15, 10, 40])

        if not all_passed:
            sys.exit(1)

    else:
        if json_format:
            output_list = []
            for tgt_name, variant in targets_to_process.items():
                output_list.append({
                    "name": variant.name,
                    "gcc_march": variant.gcc_march,
                    "gcc_mabi": variant.gcc_mabi,
                    "bitness": variant.bitness,
                    "default": tgt_name == DEFAULT_TARGET,
                    "experimental": variant.experimental,
                    "notes": variant.notes
                })
            print_json(output_list)
        else:
            print_header("Supported RISC-V Target Architectures")
            headers = ["Target Name", "Bitness", "Default", "Experimental", "GCC Flags"]
            rows = []
            for tgt_name, variant in targets_to_process.items():
                is_default = "Yes" if tgt_name == DEFAULT_TARGET else "No"
                is_experimental = "Yes" if variant.experimental else "No"
                gcc_flags = f"-march={variant.gcc_march} -mabi={variant.gcc_mabi}"
                rows.append([tgt_name, variant.bitness, is_default, is_experimental, gcc_flags])
            print_table(headers, rows, [15, 8, 8, 12, 30])
            click.echo("\nUse 'tatva targets --verify [NAME]' to run toolchain verification tests.")


@cli.command("analyze")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--json", "json_format", is_flag=True, help="Print machine-readable JSON summary.")
def analyze(model_path: str, json_format: bool) -> None:
    """
    Import and analyze an ONNX model's computation graph.

    Usage Example:
      tatva analyze models/model.onnx
    """
    try:
        model_ir = import_model(model_path)
        report = analyze_graph(model_ir)
    except ImportInProgressError as e:
        if json_format:
            print_json({"status": "error", "error": str(e)})
        else:
            click.echo(f"Import In Progress: {e}")
        sys.exit(1)
    except UnsupportedOperatorError as e:
        if json_format:
            print_json({"status": "error", "error": str(e)})
        else:
            click.echo(f"Unsupported Operator Error: {e}")
        sys.exit(1)
    except Exception as e:
        if json_format:
            print_json({"status": "error", "error": f"Failed to analyze model: {e}"})
        else:
            click.echo(f"Error: {e}")
        sys.exit(1)

    if json_format:
        output_dict = {
            "format": model_ir.metadata.get("format", "Unknown"),
            "file_size_bytes": model_ir.metadata.get("file_size_bytes", 0),
            "total_ops": report.total_ops,
            "has_transformer_bottleneck": report.has_transformer_bottleneck,
            "op_histogram": report.op_histogram,
            "unsupported_ops": report.unsupported_ops,
        }
        print_json(output_dict)
    else:
        print_header("Model Computation-Graph Analysis")
        click.echo(f"Model Format: {model_ir.metadata.get('format', 'Unknown')}")
        file_size_kb = model_ir.metadata.get("file_size_bytes", 0) / 1024.0
        click.echo(f"File Size: {file_size_kb:.2f} KB")
        click.echo(f"Total Operators: {report.total_ops}")
        click.echo("")
        click.echo("Operator Histogram:")
        click.echo("-" * 50)
        sorted_ops = sorted(report.op_histogram.items(), key=lambda x: x[1], reverse=True)
        for op, count in sorted_ops:
            click.echo(f"  {op:<25}: {count}")

        click.echo("")
        bottleneck_status = "DETECTED" if report.has_transformer_bottleneck else "NOT DETECTED"
        click.echo(f"Transformer Bottleneck (Attention/Softmax): {bottleneck_status}")

        click.echo("")
        click.echo("Unsupported Operators for RISC-V:")
        click.echo("-" * 50)
        if report.unsupported_ops:
            for op in report.unsupported_ops:
                click.echo(f"  - {op}")
        else:
            click.echo("  None (All operators supported!)")


@cli.command("baseline-test")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--allow-experimental",
    is_flag=True,
    is_eager=True,
    help="Allow experimental target configurations.",
)
@click.option(
    "--target",
    "-t",
    default="RV64GC",
    callback=validate_target,
    help="Target architecture configuration (e.g. RV64GC).",
)
@click.option(
    "--json",
    "json_format",
    is_flag=True,
    help="Print machine-readable JSON status.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output logging.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show raw traceback for compilation/runtime failures.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(),
    default="BASELINE.md",
    show_default=True,
    help="Where to write the baseline report. Relative paths are resolved from the current directory.",
)
def baseline_test(
    model_path: str,
    allow_experimental: bool,
    target: Any,
    json_format: bool,
    verbose: bool,
    debug: bool,
    out: str,
) -> None:
    """
    Establish a verified FP32 baseline in QEMU and generate a baseline report.

    Usage Example:
      tatva baseline-test models/model.onnx --target RV64GC
    """

    if verbose:
        click.echo(
            f"Target selected: {target.name} (march={target.gcc_march}, mabi={target.gcc_mabi})"
        )

    try:
        baseline_res = establish_baseline(model_path, target)
    except Exception as e:
        handle_cli_exception(e, debug=debug, json_format=json_format)

    latency = baseline_res.latency_result

    if json_format:
        output_dict = {
            "status": "ok",
            "parity_passed": baseline_res.parity_passed,
            "ref_logits": baseline_res.ref_logits,
            "target_logits": baseline_res.target_logits,
            "latency": {
                "environment": latency.environment,
                "simulated": latency.simulated,
                "mean_ms": latency.mean_ms,
                "median_ms": latency.median_ms,
                "p95_ms": latency.p95_ms,
                "raw_samples_ms": latency.raw_samples_ms,
            },
        }
        print_json(output_dict)
    else:
        click.echo("\nNumerical Parity Verification: SUCCESS")
        click.echo(f"  Host Logits:   {baseline_res.ref_logits}")
        click.echo(f"  RISC-V Logits: {baseline_res.target_logits}")

        click.echo("\nMeasured Latency Statistics:")
        click.echo(
            f"  Environment: {latency.environment} (simulated: {latency.simulated})"
        )
        click.echo(f"  Mean Latency:   {latency.mean_ms:.4f} ms")
        click.echo(f"  Median Latency: {latency.median_ms:.4f} ms")
        click.echo(f"  p95 Latency:    {latency.p95_ms:.4f} ms")

    # Write next to the user, not next to the source tree. PROJECT_DIR is derived from
    # this file's location, so for a pip-installed wheel the report landed inside
    # site-packages, where nobody would ever find it.
    baseline_md_path = os.path.abspath(out)
    if verbose:
        click.echo(f"Writing baseline report markdown to {baseline_md_path}...")

    model_filename = os.path.basename(model_path)
    reproduce_cmd = f"tatva baseline-test {model_path} --target {target.name}"

    baseline_md_content = f"""# TATVA M1 Baseline Metrics

> [!IMPORTANT]
> **Environment Verification:**
> All values listed below were measured using **QEMU system-mode simulation** on the host CPU. They do **NOT** represent real, physical RISC-V hardware execution.

## Model Details
* **Source Model:** `{model_filename}`
* **Architecture:** Transformer (BERT-derived)
* **Configuration:** FP32 Baseline Execution (No quantization, standard softmax scheduling)
* **Target Variant:** `{target.name}` (RISC-V Bare-metal, openSBI Virt)
* **GCC Compiler Flags:** `-march={target.gcc_march} -mabi={target.gcc_mabi} -O3`

## Measured Latency Parameters
* **Target CPU frequency:** 100 MHz (Nominal cycle-to-time conversion frequency)
* **Emulation Environment:** QEMU simulated: True
* **Mean Inference Latency:** **`{latency.mean_ms:.4f}` ms**
* **Median Inference Latency:** **`{latency.median_ms:.4f}` ms**
* **p95 Inference Latency:** **`{latency.p95_ms:.4f}` ms**

---

## Parity Verification Results

Out-of-box correctness was validated by performing host-side ONNX Runtime inference and comparing output logits against target-side RISC-V bare-metal output.

* **Parity Check Output Status:** **{"PASS" if baseline_res.parity_passed else "FAIL"}**
* **Allowed Tolerance Threshold:** `1e-4`
* **Reference Host Logits (Top 5):** `{baseline_res.ref_logits}`
* **RISC-V Bare-Metal Logits (Top 5):** `{baseline_res.target_logits}`

---

## Replication Command
```bash
{reproduce_cmd}
```
"""
    try:
        parent = os.path.dirname(baseline_md_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(baseline_md_path, "w", encoding="utf-8") as f:
            f.write(baseline_md_content)
        if not json_format:
            click.echo(f"\nBaseline report written to {baseline_md_path}")
    except Exception as e:
        click.echo(f"Error writing baseline report to '{baseline_md_path}': {e}", err=True)
        sys.exit(1)


@cli.command("optimize")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--allow-experimental",
    is_flag=True,
    is_eager=True,
    help="Allow experimental target configurations.",
)
@click.option(
    "--target",
    "-t",
    default="RV64GC",
    callback=validate_target,
    help="Target architecture configuration.",
)
@click.option(
    "--passes",
    "-p",
    default="fuse",
    help="Comma-separated list of passes to run (e.g. fuse,quantize).",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(),
    default="build",
    help="Output directory for optimized artifact.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Replace --out if it already exists. Without this, an existing directory is an error.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show raw traceback for compilation/runtime failures.",
)
def optimize(
    model_path: str,
    allow_experimental: bool,
    target: Any,
    passes: str,
    out: str,
    force: bool,
    debug: bool,
) -> None:
    """
    Optimize model precision and schedules.

    This command compiles, quantizes, and generates custom TVM optimization passes.
    """
    import json
    import shutil

    from tatva.diagnostics import AccuracyDropError
    from tatva.optimizer import compare_configs

    # 1. Parse and validate passes
    selected_passes = [p.strip().lower() for p in passes.split(",")]
    for p in selected_passes:
        if p not in ("fuse", "quantize"):
            click.secho(
                f"Error: Invalid pass '{p}'. Valid passes are 'fuse' and 'quantize'.",
                fg="red",
                err=True,
            )
            sys.exit(1)

    # Print regression warning if quantize is requested. The warning states the
    # structural reason rather than a latency figure: the pass round-trips values
    # through INT8 and then computes in FP32, so it can only add work. Quoting a
    # fixed millisecond delta here would be inventing a number -- the real one
    # depends on the model and is measured and printed by the run below.
    if "quantize" in selected_passes:
        click.secho(
            "Warning: 'quantize' simulates INT8 numerics -- values are round-tripped through "
            "INT8 but the matmuls still execute in FP32 -- so it measures the accuracy cost of "
            "quantization and is expected to be SLOWER than the FP32 baseline on scalar RISC-V "
            "targets. The measured latency for this model is reported below.",
            fg="yellow",
            err=True,
        )

    # 2. Never silently overwrite. Refusing was the right default; the missing half was
    # any way to say yes, so the only way to re-run a build was to delete the directory
    # by hand. --force does that for you, and says so.
    if os.path.exists(out):
        if not force:
            click.secho(
                f"Error: Output directory '{out}' already exists. "
                f"Re-run with --force to replace it, or pass a different --out.",
                fg="red",
                err=True,
            )
            sys.exit(1)
        click.secho(f"Replacing existing output directory '{out}' (--force).", fg="yellow", err=True)
        shutil.rmtree(out, ignore_errors=True)

    # Print header
    print_header(f"TATVA Optimization: {os.path.basename(model_path)}")
    click.echo(f"Target:      {target.name}")
    click.echo(f"Passes:      {', '.join(selected_passes)}")
    click.echo(f"Output:      {out}\n")

    click.echo("Running baseline vs optimized comparison in QEMU...")

    # Run the comparison
    try:
        res = compare_configs(
            model_path, target, ["baseline", "optimized"], passes=selected_passes
        )
        comp = res["comparison"]

        # Check accuracy parity verification
        if not comp.get("opt_accuracy_ok", False):
            raise AccuracyDropError(
                mse=comp.get("opt_accuracy_delta_mse", 0.0),
                tolerance=0.05,
                details="Parity verification failed: MSE exceeds threshold.",
            )
    except Exception as e:
        handle_cli_exception(e, debug=debug)

    # Copy files to --out directory
    build_dir = res["results"]["optimized"]["build_dir"]
    try:
        shutil.copytree(build_dir, out)
    except Exception as e:
        click.secho(
            f"Error copying compilation artifact to '{out}': {e}",
            fg="red",
            err=True,
        )
        sys.exit(1)

    # Write report.json alongside the binary inside the output directory
    report = {
        "config": {"target": target.name, "passes": selected_passes},
        "environment": {
            "name": res["results"]["optimized"]["latency"].environment,
            "simulated": res["results"]["optimized"]["latency"].simulated,
        },
        "measurements": {
            "baseline": {
                "mean_ms": res["results"]["baseline"]["latency"].mean_ms,
                "median_ms": res["results"]["baseline"]["latency"].median_ms,
                "p95_ms": res["results"]["baseline"]["latency"].p95_ms,
            },
            "optimized": {
                "mean_ms": comp["opt_mean_ms"],
                "median_ms": comp["opt_median_ms"],
                "p95_ms": comp["opt_p95_ms"],
            },
            "latency_delta_ms": comp["opt_mean_delta_ms"],
        },
        "parity_result": {
            "accuracy_delta_mse": comp["opt_accuracy_delta_mse"],
            "accuracy_ok": comp["opt_accuracy_ok"],
        },
    }

    report_path = os.path.join(out, "report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        click.secho(f"Error writing report.json: {e}", fg="red", err=True)
        sys.exit(1)

    # Print baseline vs optimized comparison table
    env_label = (
        "QEMU_SIM"
        if res["results"]["optimized"]["latency"].simulated
        else "REAL_HW"
    )

    headers = [
        "Metric",
        f"Baseline ({env_label})",
        f"Optimized ({env_label})",
        "Delta",
    ]
    data = [
        [
            "Mean Latency",
            f"{res['results']['baseline']['latency'].mean_ms:.4f} ms",
            f"{comp['opt_mean_ms']:.4f} ms",
            f"{comp['opt_mean_delta_ms']:.4f} ms",
        ],
        [
            "Median Latency",
            f"{res['results']['baseline']['latency'].median_ms:.4f} ms",
            f"{comp['opt_median_ms']:.4f} ms",
            f"{comp['opt_median_delta_ms']:.4f} ms",
        ],
        [
            "p95 Latency",
            f"{res['results']['baseline']['latency'].p95_ms:.4f} ms",
            f"{comp['opt_p95_ms']:.4f} ms",
            f"{comp['opt_p95_delta_ms']:.4f} ms",
        ],
        [
            "Accuracy MSE",
            "Reference",
            f"{comp['opt_accuracy_delta_mse']:.6f}",
            "PASS" if comp["opt_accuracy_ok"] else "FAIL",
        ],
    ]

    click.echo("\nBenchmark Results:")
    print_table(headers, data, [18, 25, 25, 12])

    click.secho(f"\nSuccess: Optimized artifact written to '{out}'", fg="green")
    click.echo(f"Report JSON written to: {report_path}")


@cli.command("diagnose")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--allow-experimental",
    is_flag=True,
    is_eager=True,
    help="Allow experimental target configurations.",
)
@click.option(
    "--target",
    "-t",
    default="RV64GC",
    callback=validate_target,
    help="Target architecture configuration.",
)
@click.option(
    "--passes",
    "-p",
    default="fuse",
    help="Comma-separated list of passes to run (e.g. fuse,quantize).",
)
@click.option(
    "--json",
    "json_format",
    is_flag=True,
    help="Emit structured context as JSON.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show raw traceback for compilation/runtime failures.",
)
def diagnose(
    model_path: str,
    allow_experimental: bool,
    target: Any,
    passes: str,
    json_format: bool,
    debug: bool,
) -> None:
    """
    Run compiler/optimize paths and explain structured failures in plain English.

    Can also accept a saved failure JSON report to re-explain it.
    """
    import json

    from tatva.diagnostics import DiagnosisContext, classify_failure, explain

    # Handle saved JSON reports directly
    if model_path.endswith(".json"):
        try:
            with open(model_path) as f:
                data = json.load(f)

            # Determine context
            if "error_type" in data:
                context = DiagnosisContext(
                    error_type=data["error_type"],
                    metadata=data.get("metadata", {}),
                )
            elif "error" in data:
                # Classify from string
                context = classify_failure(Exception(data["error"]))
            else:
                context = classify_failure(Exception(str(data)))

            if json_format:
                print_json(
                    {
                        "error_type": context.error_type,
                        "metadata": context.metadata,
                    }
                )
            else:
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    click.secho(
                        "Note: Configure ANTHROPIC_API_KEY environment variable to retrieve richer, AI-generated Claude API explanations.",
                        fg="cyan",
                    )
                click.secho("\nDiagnostics Explanation:", fg="red", bold=True)
                click.echo(explain(context))
            sys.exit(0 if context.error_type != "unknown" else 1)
        except Exception as e:
            click.secho(f"Error parsing saved JSON report: {e}", fg="red", err=True)
            sys.exit(1)

    # Attempt the compile/optimize path on a model
    try:
        # Import model once
        model_ir = import_model(model_path)

        # Apply passes
        selected_passes = [p.strip().lower() for p in passes.split(",")]
        opt_model_ir = model_ir
        if "fuse" in selected_passes:
            from tatva.optimizer import select_fast_softmax_kernel
            opt_model_ir = select_fast_softmax_kernel(opt_model_ir)
        if "quantize" in selected_passes:
            from tatva.optimizer import quantize
            opt_model_ir = quantize(opt_model_ir)

        # Compile
        from tatva.runner import compile_model
        compile_model(opt_model_ir, target)

        # Parity check
        from tatva.runner import establish_baseline
        establish_baseline(model_path, target)

        click.echo("No errors detected. Model compiled successfully.")
        sys.exit(0)
    except Exception as e:
        handle_cli_exception(e, debug=debug, json_format=json_format)


@cli.command("validate")
@click.option("--models-dir", default="models", help="Directory containing ONNX models for validation matrix.")
@click.option("--target", "-t", default="RV64GC", help="Target architecture variant (e.g. RV64GC, RV64GCV).")
def validate_cmd(models_dir: str, target: str) -> None:
    """
    Run multi-model validation matrix across ONNX models.

    Usage Example:
      tatva validate --models-dir models --target RV64GCV
    """
    import importlib.util
    val_script = os.path.join(PROJECT_DIR, "scratch", "validate_models.py")
    if not os.path.exists(val_script):
        click.echo(f"Error: Validation script not found at {val_script}", err=True)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("validate_models", val_script)
    if spec is None or spec.loader is None:
        click.echo("Error loading validate_models module.", err=True)
        sys.exit(1)

    val_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(val_module)

    matrix = val_module.run_validation_matrix(models_dir, variant_name=target)
    failures = [r for r in matrix if r["status"] in ("FAIL", "ERROR")]
    if failures:
        sys.exit(1)
    sys.exit(0)


@cli.command("profile")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--allow-experimental",
    is_flag=True,
    is_eager=True,
    help="Allow experimental target configurations.",
)
@click.option(
    "--target",
    "-t",
    default="RV64GC",
    callback=validate_target,
    help="Target architecture variant (e.g. RV64GC, RV64GCV).",
)
@click.option(
    "--passes",
    "-p",
    default="",
    help="Comma-separated passes to apply before profiling (fuse, quantize). Default: none.",
)
@click.option("--timed-count", default=10, show_default=True, help="Number of timed iterations to attribute.")
@click.option("--json", "json_format", is_flag=True, help="Print machine-readable JSON breakdown.")
def profile_cmd(
    model_path: str,
    allow_experimental: bool,
    target: Any,
    passes: str,
    timed_count: int,
    json_format: bool,
) -> None:
    """
    Attribute simulated cycles to individual generated kernels.

    Builds the model with rdcycle instrumentation around every kernel call and
    reports where the time actually goes. The instrumented build is used ONLY for
    this breakdown -- it costs roughly 9 cycles per call site, so the latency it
    reports is not the latency `tatva baseline-test` reports, and this command
    prints both so the difference is visible rather than assumed.

    Usage Example:
      tatva profile models/model.onnx
      tatva profile models/model.onnx --passes quantize --json
    """
    from tatva.runner import compile_model, run_and_measure

    selected_passes = [p.strip().lower() for p in passes.split(",") if p.strip()]

    try:
        model_ir = import_model(model_path)
        for name in selected_passes:
            if name == "fuse":
                from tatva.optimizer import select_fast_softmax_kernel

                model_ir = select_fast_softmax_kernel(model_ir)
            elif name == "quantize":
                from tatva.optimizer import quantize

                model_ir = quantize(model_ir)
            else:
                raise click.BadParameter(f"Unknown pass '{name}'. Supported: fuse, quantize.")

        artifact = compile_model(model_ir, target, timed_count=timed_count, profile=True)
        result = run_and_measure(artifact)
    except click.ClickException:
        raise
    except Exception as e:
        if json_format:
            print_json({"status": "error", "error": str(e)})
            sys.exit(1)
        handle_cli_exception(e)
        return

    if not result.kernel_profiles:
        # Never present an empty breakdown as a successful profile.
        msg = "The instrumented build produced no KERNEL_CYCLES output; no attribution is available."
        if json_format:
            print_json({"status": "error", "error": msg})
        else:
            click.secho(msg, fg="red", err=True)
        sys.exit(1)

    if json_format:
        print_json(
            {
                "status": "ok",
                "model": model_path,
                "target": target.name,
                "passes": selected_passes,
                "timed_count": timed_count,
                "instrumented_mean_ms": result.mean_ms,
                "total_cycles": result.total_cycles,
                "attributed_cycles": result.attributed_cycles,
                "attribution_coverage": result.attribution_coverage,
                "kernels": [k.to_dict() for k in result.kernel_profiles],
            }
        )
        return

    print_header("Per-Kernel Cycle Attribution")
    click.echo(f"Model:  {model_path}")
    click.echo(f"Target: {target.name}")
    click.echo(f"Passes: {', '.join(selected_passes) if selected_passes else 'none (FP32 baseline)'}")
    click.echo("")
    click.secho(
        "Note: these cycles come from an INSTRUMENTED build (~9 cycles per kernel call).\n"
        "      Use 'tatva baseline-test' for the latency figure to report.",
        fg="yellow",
    )
    click.echo("")
    click.echo(f"{'KERNEL':<28}{'CALLS':>8}{'CYCLES':>16}{'CYC/CALL':>14}{'SHARE':>9}")
    click.echo("-" * 75)
    for k in result.kernel_profiles:
        share = (k.cycles / result.attributed_cycles * 100) if result.attributed_cycles else 0.0
        click.echo(f"{k.name:<28}{k.calls:>8}{k.cycles:>16,}{k.cycles_per_call:>14,.1f}{share:>8.2f}%")
    click.echo("-" * 75)
    click.echo(
        f"Attributed {result.attributed_cycles:,} of {result.total_cycles:,} measured cycles "
        f"({result.attribution_coverage * 100:.3f}%)."
    )
    click.echo(
        "The unattributed remainder is the harness itself: the run prologue, the DLTensor\n"
        "pointer stores, and the rdcycle pair bracketing each call."
    )


@cli.command("gui")
def gui_cmd() -> None:
    """
    Launch the interactive TATVA Desktop GUI application.

    Usage Example:
      tatva gui
    """
    from tatva.gui import launch_gui
    launch_gui()


if __name__ == "__main__":
    cli()


