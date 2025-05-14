"""
This function summarizes a test from a log file.
Output is a markdown file.
"""

import sys


def main(log_file: str, out_file: str):
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect important lines

    # Summary line: the last of the log file
    tests_summary = "\n".join(lines[-2:])  # Last line is empty
    # If there is "failed" in the summary line, it means that there are failed tests.
    # Fetch every from the ==== summary === to the end
    some_failed = "failed" in tests_summary or "error" in tests_summary
    if some_failed:
        summary_index = 2
        for i, line in enumerate(reversed(lines)):
            if "== FAILURES ==" in line:
                summary_index = i + 2
                break
        tests_summary = "".join(lines[-summary_index:])

    performance_line_30Hz: str = ""
    performance_line_500Hz: str = ""
    performance_recording_line_500Hz: str = ""

    # Parse the log file
    for line in lines:
        if "[TEST_PERFORMANCE_30Hz]" in line:
            # I Want to select the line from "TEST_PERFORMANCE" to the end of the line
            performance_line_30Hz = line[line.find("[TEST_PERFORMANCE_30Hz]") :]
        if "[TEST_PERFORMANCE_500Hz]" in line:
            # I Want to select the line from "TEST_PERFORMANCE" to the end of the line
            performance_line_500Hz = line[line.find("[TEST_PERFORMANCE_500Hz]") :]

        if "[TEST_RECORDING_PERFORMANCE_500Hz]" in line:
            performance_recording_line_500Hz = line[
                line.find("[TEST_RECORDING_PERFORMANCE_500Hz]") :
            ]

    # Build a simple markdown summary
    emoji = "✅" if some_failed == 0 else "❌"
    summary = []
    summary.append(f"## {emoji} API integrations tests")

    summary.append("### Performance")
    summary.append(performance_line_30Hz)
    summary.append(performance_line_500Hz)
    summary.append(performance_recording_line_500Hz)

    summary.append("### Pytests logs summary")

    if len(tests_summary) >= 500:
        # show only the last 500 characters
        tests_summary = tests_summary[-500:]
    summary.append(f"```{tests_summary}```")

    if some_failed == 0:
        summary.append(":tada: **All tests passed!** ")
    else:
        summary.append(
            "❌ **Some tests failed**. Check the logs in Github Actions for details."
        )

    # Write out the summary
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python summarize_test_results.py <log_file> <out_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
