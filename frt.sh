python3 frt.py temp/flooding_result.log temp/flooding.png

intervals=(2 3 4 5)
counts=(1 2 3 4 5 6 7)

for interval in "${intervals[@]}"; do
    for count in "${counts[@]}"; do
        infile="temp/hopwave_result-${interval}-${count}.log"
        outfile="temp/hopwave-${interval}-${count}.png"
        python3 frt.py "$infile" "$outfile"
    done
done