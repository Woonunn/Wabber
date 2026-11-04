"""Plot cumulative USD cost for four fixed providers and Wabber from paired requests."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


# ATSuite Fig. 11 cloud colors; Wabber uses the dark red from Fig. 9.
SERIES = {
    "ali_fc_usd": ("Aliyun FC", "#FACEB1", "--", "o"),
    "ali_agentrun_usd": ("Aliyun AgentRun", "#FACEB1", "-", "s"),
    "aws_lambda_usd": ("AWS Lambda", "#ACC4E8", "--", "o"),
    "aws_agentcore_usd": ("AWS AgentCore", "#ACC4E8", "-", "s"),
    "wabber_usd": ("Wabber", "#B2182B", "-", "D"),
}
COLUMNS = ["request_index", *SERIES]
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}


def load_costs(source: Path) -> tuple[list[int], dict[str, list[float]]]:
    """Validate paired incremental costs, then accumulate in the supplied request order."""
    requests = [0]
    cumulative = {key: [0.0] for key in SERIES}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != COLUMNS:
            raise ValueError("Expected CSV header: " + ",".join(COLUMNS))
        for index, row in enumerate(reader, start=1):
            if len(row) != len(COLUMNS):
                raise ValueError(f"CSV row {index + 1}: expected {len(COLUMNS)} fields")
            if int(row[0]) != index:
                raise ValueError(f"CSV row {index + 1}: request_index must be {index}")
            for key, raw in zip(SERIES, row[1:]):
                try:
                    cost = float(raw)
                except ValueError as error:
                    raise ValueError(f"Request {index}, {key}: expected a nonnegative USD cost") from error
                total = cumulative[key][-1] + cost
                if not math.isfinite(cost) or cost < 0 or not math.isfinite(total):
                    raise ValueError(f"Request {index}, {key}: cost and cumulative total must be finite and nonnegative")
                cumulative[key].append(total)
            requests.append(index)
    if len(requests) == 1:
        raise ValueError("No request costs yet; fill the CSV template before plotting")
    return requests, cumulative


def plot(source: Path, output: Path) -> list[Path]:
    """Save a single-column figure as PDF, SVG and PNG, preserving previous outputs."""
    requests, cumulative = load_costs(source)
    paths = [output.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a new --output stem to preserve existing figures")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.74)
        try:
            # Keep markers sparse without dropping any measured request from the lines.
            marker_indices = sorted({round((len(requests) - 1) * i / 8) for i in range(9)})
            for key, (label, color, linestyle, marker) in SERIES.items():
                highlighted = key == "wabber_usd"
                ax.plot(
                    requests, cumulative[key], label=label, color=color, linestyle=linestyle,
                    linewidth=2.2 if highlighted else 1.1, marker=marker,
                    markersize=3.1 if highlighted else 2.7, markevery=marker_indices,
                    markeredgecolor=color if highlighted else "#333333", markeredgewidth=0.4,
                    zorder=4 if highlighted else 2,
                )
            ax.set_xlabel("Request sequence")
            ax.set_ylabel("Cumulative cost (USD)")
            ax.set_xlim(0, requests[-1])
            maximum = max(values[-1] for values in cumulative.values())
            ax.set_ylim(0, maximum * 1.06 if maximum else 1)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_powerlimits((-3, 4))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_offset_position("right")
            ax.tick_params(direction="in", length=2.5)
            ax.set_axisbelow(True)
            ax.grid(color="#B0B0B0", linestyle=":", linewidth=0.45, alpha=0.65)
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 2, 4, 1, 3]
            legend = ax.legend(
                [handles[i] for i in order], [labels[i] for i in order],
                loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False,
                fontsize=7, handlelength=2.4, columnspacing=1.2, borderaxespad=0,
            )
            for label in legend.get_texts():
                if label.get_text() == "Wabber":
                    label.set_fontweight("bold")
            output.parent.mkdir(parents=True, exist_ok=True)
            for path in paths:
                fig.savefig(path, dpi=400)
        finally:
            plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV of incremental per-request USD costs")
    parser.add_argument("--output", required=True, type=Path, help="Output path stem, without an extension")
    args = parser.parse_args()
    try:
        print("\n".join(str(path) for path in plot(args.input, args.output)))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
