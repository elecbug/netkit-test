import os
import re
import matplotlib.pyplot as plt

# 로그 파일 이름 패턴: hopwave_result-[INTERVAL]-[COUNT].log
FILE_PATTERN = re.compile(r"hopwave_result-(\d+)-(\d+)\.log")


def parse_duplicate_messages(filepath):
    """로그 파일에서 Duplicate Messages 값을 읽어 반환"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.lower().startswith("duplicate messages"):
                value = line.split("\t")[-1].strip()
                return float(value)
    return None


def plot_from_folder(folder_path, output_path):
    results = {}  # { count: [(interval, duplicate), ...] }
    interval_group = {}  # { interval: [(count, duplicate), ...] }

    # 폴더 내부 파일 순회
    for filename in os.listdir(folder_path):
        match = FILE_PATTERN.match(filename)
        if not match:
            continue

        interval = int(match.group(1))
        count = int(match.group(2))
        full_path = os.path.join(folder_path, filename)

        duplicate = parse_duplicate_messages(full_path)
        if duplicate is None:
            continue

        # COUNT 기준으로 그룹화
        if count not in results:
            results[count] = []
        results[count].append((interval, duplicate))

        # INTERVAL 기준으로 그룹화
        if interval not in interval_group:
            interval_group[interval] = []
        interval_group[interval].append((count, duplicate))

    # -----------------------------
    # ⑴ 전체 Figure 생성
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes

    # -----------------------------
    # ⑵ 그래프 1: x=interval, count별 구분
    # -----------------------------
    ax1.set_title("Duplicate Messages vs Interval (Grouped by Count)")
    ax1.set_xlabel("Interval")
    ax1.set_ylabel("Duplicate Messages")
    ax1.grid(True)

    # count 정렬 (중요)
    for count in sorted(results.keys()):
        values = results[count]
        values.sort()  # interval 기준 정렬
        intervals = [v[0] for v in values]
        duplicates = [v[1] for v in values]
        ax1.plot(intervals, duplicates, marker="o", label=f"count={count}", markersize=5)

    ax1.legend()

    # -----------------------------
    # ⑶ 그래프 2: x=count, interval별 구분
    # -----------------------------
    ax2.set_title("Duplicate Messages vs Count (Grouped by Interval)")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Duplicate Messages")
    ax2.grid(True)

    # interval 정렬
    for interval in sorted(interval_group.keys()):
        values = interval_group[interval]
        values.sort()  # count 기준 정렬
        counts = [v[0] for v in values]
        duplicates = [v[1] for v in values]
        ax2.plot(counts, duplicates, marker="o", label=f"interval={interval}", markersize=5)

    ax2.legend()

    # 저장
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved to {output_path}")


# --------------------------------------
# 메인
# --------------------------------------
if __name__ == "__main__":
    folder_name = os.sys.argv[1] if len(os.sys.argv) > 1 else "results"
    output_name = os.sys.argv[2] if len(os.sys.argv) > 2 else "combined_plot.png"
    plot_from_folder(folder_name, output_name)
