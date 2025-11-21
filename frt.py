import matplotlib.pyplot as plt
import os

def plot_cumulative_from_file(path, output_path):
    times = []
    counts = []

    # 파일 읽기
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 쉼표 또는 공백 자동 인식
                if "," in line:
                    t, c = line.split(",")
                else:
                    t, c = line.split()
            except ValueError:
                continue

            try:
                float(t)
                float(c)
            except ValueError:
                continue

            times.append(float(t))
            counts.append(float(c))

    # 시간으로 정렬
    data = sorted(zip(times, counts), key=lambda x: x[0])
    times_sorted = [t for t, _ in data]
    counts_sorted = [c for _, c in data]

    # 누적합 계산
    cumulative = []
    s = 0
    for v in counts_sorted:
        s += v
        cumulative.append(s)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(times_sorted, cumulative, marker="o", markersize=2)
    plt.xlabel("Time (sec)")
    plt.ylabel("Cumulative Count")
    plt.title("Cumulative Sum Over Time")
    plt.grid(True)
    plt.xlim(left=0, right=5.25)
    plt.ylim(bottom=0, top=1000)
    plt.tight_layout()
    plt.savefig(output_path)

# 실행 예시
# plot_cumulative_from_file("data.txt")

if __name__ == "__main__":
    file_name = os.sys.argv[1] if len(os.sys.argv) > 1 else "result.log"
    output_name = os.sys.argv[2] if len(os.sys.argv) > 2 else "cumulative_plot.png"
    plot_cumulative_from_file(file_name, output_name)