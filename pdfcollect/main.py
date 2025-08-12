import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import shutil
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
import json
import cProfile
import pstats

from vision_action_agent.src.vision.element_detector import ElementType


def click_image_action(agent, params):
    image_templates = params.get("image_template")
    if not image_templates:
        logger.error("image_template not specified for click_image action")
        return False

    if isinstance(image_templates, str):
        image_templates = [image_templates]  # Convert single string to list

    for template in image_templates:
        elements = agent.find_element(
            image_template=template, element_type=ElementType.ICON
        )
        if elements:
            # Click the center of the first found element
            agent.mouse.click(elements[0].center[0], elements[0].center[1])
            logger.info(f"Clicked image: {template}")
            return True

    logger.error(
        f"Could not find any of the specified images {image_templates} on the screen"
    )
    return False


def pause_for_manual_action(agent, params):
    message = params.get(
        "message", "Manual action required. Press Enter to " "continue..."
    )
    print(f"\n--- {message} ---")
    input()
    return True


def _resolve_templates_dir() -> str:
    """Resolve the templates directory so image matching works when run from pdfcollect/.

    Prefer `vision_action_agent/templates/` relative to this file; fallback to `./templates`.
    """
    here = Path(__file__).resolve().parent
    repo_templates = (here / ".." / "vision_action_agent" / "templates").resolve()
    if repo_templates.exists():
        return str(repo_templates)
    local_templates = (here / "templates").resolve()
    return str(local_templates)


def _wait_for_latest_pdf(downloads_dir: Path, timeout: float = 60.0, stable_checks: int = 2, poll_interval: float = 1.5) -> Optional[Path]:
    """Wait for the newest PDF to appear and stabilize in size.

    Skips partial extensions (e.g., .crdownload/.part). Returns the stable Path or None.
    """
    end = time.time() + timeout
    last_size = None
    last_file: Optional[Path] = None
    stable_count = 0

    def latest_pdf() -> Optional[Path]:
        candidates = [p for p in downloads_dir.glob("*.pdf") if p.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    while time.time() < end:
        f = latest_pdf()
        if f is None:
            time.sleep(poll_interval)
            continue
        try:
            size = f.stat().st_size
        except FileNotFoundError:
            time.sleep(poll_interval)
            continue
        if last_file is None or f != last_file:
            last_file = f
            last_size = size
            stable_count = 0
        else:
            if size == last_size:
                stable_count += 1
                if stable_count >= stable_checks:
                    return f
            else:
                last_size = size
                stable_count = 0
        time.sleep(poll_interval)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Automate PDF downloads using a DOI."
    )
    parser.add_argument("doi", type=str, help="The DOI of the article to download.")
    parser.add_argument(
        "--downloads-path",
        type=str,
        default=os.environ.get("DOWNLOADS_DIR") or str(Path.home() / "Downloads"),
        help="Directory to scan for downloaded PDFs (default: ~/Downloads or env DOWNLOADS_DIR)",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=str,
        default=None,
        help="Directory to save step screenshots (default: pdfcollect/screenshots)",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the PDF to appear and stabilize (default: 60)",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Do not save screenshots after each step (useful if screen capture fails)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip UI initialization and step execution; simulate for performance measurement.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Collect timing information and write a benchmark summary.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=str(Path(__file__).resolve().parent / "benchmark.json"),
        help="Path to write benchmark JSON (with --benchmark)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with cProfile and write stats to profile.prof",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=str(Path(__file__).resolve().parent / "profile.prof"),
        help="Path to write cProfile stats (with --profile)",
    )
    args = parser.parse_args()
    timings = {}
    t0 = time.perf_counter()
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()

    agent = None
    if not args.dry_run:
        t_agent_start = time.perf_counter()
        # Lazy import to avoid GUI deps during dry-run
        from vision_action_agent.src.core.agent import VisionActionAgent  # noqa: WPS433
        agent = VisionActionAgent()
        timings["agent_init_s"] = time.perf_counter() - t_agent_start

    # Ensure templates are loaded from the repo location
    templates_dir = _resolve_templates_dir()
    try:
        agent.element_detector._load_templates(templates_dir)
        logger.info(f"Template directory set to: {templates_dir}")
    except Exception as e:
        logger.warning(f"Unable to reload templates from {templates_dir}: {e}")

    if agent is not None:
        # Precheck: verify screen capture works before running workflow
        try:
            test_img = agent.capture_screen()
            if test_img is None or getattr(test_img, 'size', 0) == 0:
                raise RuntimeError("Empty screenshot returned")
            logger.info("Precheck: screen capture is available.")
        except Exception as e:
            logger.error(
                "Precheck failed: cannot capture the screen. "
                "Ensure a graphical session is available and DISPLAY is set.\n"
                "Tips: On GNOME/Wayland, use an Xorg session; on X11, run `xhost +SI:localuser:$USER`.\n"
                f"Underlying error: {e}"
            )
            sys.exit(1)

        # Register custom actions
        agent.workflow_executor.register_custom_action(
            "click_image",
            lambda params, exec_agent=agent: click_image_action(exec_agent, params),
        )
        agent.workflow_executor.register_custom_action(
            "pause_for_manual_action",
            lambda params, exec_agent=agent: pause_for_manual_action(exec_agent, params),
        )

    exit_code = 0
    try:
        # Load the workflow from the config file
        t_load = time.perf_counter()
        if agent is not None:
            workflow_steps = agent.load_workflow_from_file("config.yaml")
            if not workflow_steps:
                logger.error("No workflow steps loaded. Check config.yaml for invalid action_type values.")
                exit_code = 1
                return
        else:
            # Dry run: parse YAML directly
            import yaml
            with open(Path(__file__).resolve().parent / "config.yaml", "r") as f:
                data = yaml.safe_load(f)
            workflow_steps = data.get("steps", [])
        timings["workflow_load_s"] = time.perf_counter() - t_load

        # Substitute the DOI into the workflow steps
        for step in workflow_steps:
            if agent is not None:
                if getattr(step, 'id', None) == "enter_doi":
                    step.parameters["text"] = step.parameters["text"].replace(
                        "{{doi}}", args.doi
                    )
            else:
                if isinstance(step, dict) and step.get("id") == "enter_doi":
                    params = step.get("parameters", {})
                    if isinstance(params.get("text"), str):
                        params["text"] = params["text"].replace("{{doi}}", args.doi)
                        step["parameters"] = params

        logger.info(f"Starting PDF download for DOI: {args.doi}")

        # Create screenshots directory (only in real run)
        screenshots_dir = (
            Path(args.screenshots_dir)
            if args.screenshots_dir
            else Path(__file__).resolve().parent / "screenshots"
        )
        if agent is not None:
            screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Execute workflow step by step with screenshots
        workflow_successful = True
        for i, step in enumerate(workflow_steps):
            if agent is not None:
                sid = step.id
                sact = step.action_type.value
            else:
                sid = step.get("id", f"step_{i+1}")
                sact = step.get("action_type", "dry")
            logger.info(
                f"Executing step {i+1}/{len(workflow_steps)}: {sid} ({sact})"
            )

            step_t0 = time.perf_counter()
            if agent is not None:
                # Execute the step using the workflow_executor
                success = agent.workflow_executor.execute_step(step)

                # Save screenshot after each step
                if not args.no_screenshots:
                    screenshot_filename = (
                        screenshots_dir
                        / f"step_{i+1}_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    )
                    if not agent.save_screenshot(str(screenshot_filename)):
                        logger.error(f"Failed to save screenshot to {screenshot_filename}")
                    else:
                        logger.info(f"Screenshot saved: {screenshot_filename}")

                if not success and not step.continue_on_failure:
                    logger.error(f"Workflow stopped at step {sid} due to failure.")
                    workflow_successful = False
                    break
            else:
                # Dry run: simulate minimal per-step latency
                time.sleep(0.01)
                success = True
            step_dt = time.perf_counter() - step_t0
            if args.benchmark:
                timings.setdefault("steps", []).append({
                    "index": i + 1,
                    "id": sid,
                    "action": sact,
                    "duration_s": round(step_dt, 4),
                })

        if agent is None:
            logger.info("Dry-run completed.")
        elif workflow_successful:
            logger.info("Workflow completed successfully.")

            # --- File Management Logic ---
            downloads_dir = Path(args.downloads_path).expanduser().resolve()
            if not downloads_dir.exists():
                logger.error(f"Downloads directory not found: {downloads_dir}")
                exit_code = 1
            else:
                latest_pdf = _wait_for_latest_pdf(
                    downloads_dir, timeout=args.download_timeout
                )
                if latest_pdf is None:
                    logger.error(
                        f"No stable PDF found in {downloads_dir} within {args.download_timeout}s"
                    )
                    exit_code = 1
                else:
                    logger.info(f"Found downloaded file: {latest_pdf}")
                    # Sanitize the DOI to create a valid filename
                    sanitized_doi = args.doi.replace("/", "_").replace(":", "-")
                    destination_path = Path(__file__).resolve().parent / f"{sanitized_doi}.pdf"
                    try:
                        shutil.move(str(latest_pdf), str(destination_path))
                        logger.info(f"File moved and renamed to: {destination_path}")
                    except Exception as e:
                        logger.error(f"Failed to move file: {e}")
                        exit_code = 1
        elif agent is not None:
            logger.error("Workflow failed.")
            exit_code = 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        exit_code = 1
    finally:
        # Final result summary
        if exit_code == 0:
            logger.info("RESULT: SUCCESS")
        else:
            logger.error("RESULT: FAILED")

        # Profiling & Benchmark output
        if prof:
            prof.disable()
            ps = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.dump_stats(args.profile_output)
            logger.info(f"cProfile stats written to {args.profile_output}")
        total_time = time.perf_counter() - t0
        timings["total_wall_time_s"] = round(total_time, 4)
        if args.benchmark:
            try:
                with open(args.benchmark_output, "w") as f:
                    json.dump(timings, f, indent=2)
                logger.info(f"Benchmark written to {args.benchmark_output}")
            except Exception as e:
                logger.error(f"Failed to write benchmark: {e}")
        if agent is not None:
            try:
                agent.cleanup()
            except Exception:
                pass
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
