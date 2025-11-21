import os
import re
import matplotlib.pyplot as plt

# 로그 파일 이름 패턴: hopwave_result-[INTERVAL]-[COUNT].log
FILE_PATTERN = re.compile(r"hopwave_result-(\d+)-(\d+)\.log")


def parse_log(filepath):
    """로그 파일에서 Duplicate Messages와 Reachability 값을 읽어 반환"""
    duplicate = None
    reachability = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            lower = line.lower()

            if lower.startswith("duplicate messages"):
                duplicate = float(line.split("\t")[-1].strip())

            elif lower.startswith("reachability"):
                reachability = float(line.split("\t")[-1].strip())

    return duplicate, reachability


def reachability_to_size(r):
    """점 크기 계산: 90%~100% 영역을 0~100 크기로 스케일"""
    if r is None:
        return 10
    # r = 0.90 ~ 1.00 → 0 ~ 100 점 크기
    # (r * 100 - 90) = 0 ~ 10 → *10 해서 0~100
    s = (r * 100 - 90) * 10
    if s < 10:
        s = 10  # 최소 크기
    return s


def plot_from_folder(folder_path, output_path):
    # count → (interval, duplicate, reachability)
    results = {}
    # interval → (count, duplicate, reachability)
    interval_group = {}

    # -----------------------------
    # ⑴ 로그 파일 읽기
    # -----------------------------
    for filename in os.listdir(folder_path):
        match = FILE_PATTERN.match(filename)
        if not match:
            continue

        interval = int(match.group(1))
        count = int(match.group(2))
        full_path = os.path.join(folder_path, filename)

        duplicate, reachability = parse_log(full_path)
        if duplicate is None:
            continue

        # COUNT 기준
        if count not in results:
            results[count] = []
        results[count].append((interval, duplicate, reachability))

        # INTERVAL 기준
        if interval not in interval_group:
            interval_group[interval] = []
        interval_group[interval].append((count, duplicate, reachability))

    # -----------------------------
    # ⑵ 두 개 subplot 그림 생성
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1, ax2 = axes

    # -----------------------------
    # ⑶ 그래프 1: x=interval, count별 구분
    # -----------------------------
    ax1.set_title("Duplicate Messages vs Interval (Grouped by Count)")
    ax1.set_xlabel("Interval")
    ax1.set_ylabel("Duplicate Messages")
    ax1.grid(True)

    for count in sorted(results.keys()):
        values = sorted(results[count])  # interval 기준 정렬
        intervals = [v[0] for v in values]
        duplicates = [v[1] for v in values]
        sizes = [reachability_to_size(v[2]) for v in values]

        ax1.scatter(intervals, duplicates, s=sizes, label=f"count={count}")
        ax1.plot(intervals, duplicates, linewidth=0.7)

    ax1.legend()

    # -----------------------------
    # ⑷ 그래프 2: x=count, interval별 구분
    # -----------------------------
    ax2.set_title("Duplicate Messages vs Count (Grouped by Interval)")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Duplicate Messages")
    ax2.grid(True)

    for interval in sorted(interval_group.keys()):
        values = sorted(interval_group[interval])  # count 기준 정렬
        counts = [v[0] for v in values]
        duplicates = [v[1] for v in values]
        sizes = [reachability_to_size(v[2]) for v in values]

        ax2.scatter(counts, duplicates, s=sizes, label=f"interval={interval}")
        ax2.plot(counts, duplicates, linewidth=0.7)

    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[+] Saved plot → {output_path}")


# --------------------------------------
# main
# --------------------------------------
if __name__ == "__main__":
    folder_name = os.sys.argv[1] if len(os.sys.argv) > 1 else "results"
    output_name = os.sys.argv[2] if len(os.sys.argv) > 2 else "combined_plot.png"
    plot_from_folder(folder_name, output_name)
