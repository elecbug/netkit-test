package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/elecbug/netkit/graph/standard_graph"
	"github.com/elecbug/netkit/p2p"
)

func main() {
	try := 30

	for _, networkSize := range []int{500, 1000, 2000} {
		for _, degree := range []int{6, 10, 14, 20} {
			for _, hopwaveInterval := range []int{2, 3, 4} {
				for _, hopwaveCount := range []int{1, 2, 4, 6, 8, 11, 14, 17} {
					if hopwaveCount >= degree {
						continue
					}

					hopwaveLoop(try, networkSize, degree, hopwaveInterval, hopwaveCount)
				}
			}

			hopwaveLoop(try, networkSize, degree, 1, degree) // equivalent to flooding
		}
	}
}

func customHopWaveProtocol(msg p2p.Message, known []p2p.PeerID, sent []p2p.PeerID, received []p2p.PeerID, params map[string]any) *[]p2p.PeerID {
	// fmt.Printf("debug %v %v %v %v\n", known, sent, received, params)
	interval := params["hopwave_interval"].(int)
	count := params["hopwave_count"].(int)

	result := make([]p2p.PeerID, 0)

	for _, peerID := range known {
		// fmt.Printf("debug peerID: %v\n", peerID)
		for _, s := range sent {
			if peerID == s {
				goto NEXT_PEER
			}
		}
		for _, r := range received {
			if peerID == r {
				goto NEXT_PEER
			}
		}
	NEXT_PEER:
		result = append(result, peerID)
	}

	// fmt.Printf("debug hopcount: %d, interval: %d\n", msg.HopCount, interval)

	if msg.HopCount%interval == 0 {
		return &result
	} else {
		rand.Shuffle(len(result), func(i, j int) {
			result[i], result[j] = result[j], result[i]
		})

		if len(result) > count {
			result = result[:count]
		}

		return &result
	}
}

func hopwaveLoop(try, networkSize, degree int, hopwaveInterval int, hopwaveCount int) {
	sg := standard_graph.NewStandardGraph()
	sg.SetSeedRandom()

	mu := sync.Mutex{}
	counter := make(map[string]float64)
	otherData := make(map[string]float64)

	wg := sync.WaitGroup{}
	sem := make(chan struct{}, 5) // limit to 5 concurrent goroutines
	wg.Add(try)

	for i := 0; i < try; i++ {
		sem <- struct{}{}

		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()

			println("Round:", i)

			er := sg.ErdosRenyiGraph(networkSize, float64(degree)/float64(networkSize), true)

			p, err := p2p.GenerateNetwork(
				er,
				func() float64 {
					return float64(p2p.NormalRandom(100, 50))
				},
				func() float64 {
					return float64(p2p.ParetoRandom(50, 2.0))
					// return float64(p2p.ExponentialRandom(0.002))
				},
				&p2p.Config{
					CustomParams: map[string]any{
						"hopwave_interval": hopwaveInterval,
						"hopwave_count":    hopwaveCount,
					},
				},
			)

			if err != nil {
				panic(err)
			}

			ctx, cancel := context.WithCancel(context.Background())

			p.RunNetworkSimulation(ctx)

			err = p.Publish(p.PeerIDs()[0], "Hello, world!", p2p.Custom, customHopWaveProtocol)

			if err != nil {
				panic(err)
			}

			time.Sleep(5 * time.Second)
			cancel()

			ts := p.FirstMessageReceptionTimes("Hello, world!")
			sort.Slice(ts, func(i, j int) bool {
				return ts[i].Before(ts[j])
			})

			for _, t := range ts {
				sec := float32(t.Sub(ts[0]).Seconds())
				secStr := fmt.Sprintf("%.3f", sec)

				mu.Lock()
				counter[secStr] += 1.0 / float64(try)
				otherData["rcv"] += float64(sec) / float64(len(ts)) / float64(try)
				mu.Unlock()
			}
			mu.Lock()
			otherData["dup"] += float64(p.NumberOfDuplicateMessages("Hello, world!")) / float64(try)
			otherData["reach"] += p.Reachability("Hello, world!") / float64(try)
			mu.Unlock()
		}(i)
	}

	fmt.Println("Waiting for all goroutines to finish... [hopwave interval:", hopwaveInterval, "count:", hopwaveCount, "]")
	wg.Wait()

	fs, err := os.Create(fmt.Sprintf("temp/hopwave_result-%d-%d-%d-%d.log", networkSize, degree, hopwaveInterval, hopwaveCount))
	if err != nil {
		panic(err)
	}
	defer fs.Close()

	fmt.Fprintf(fs, "Duplicate Messages\t%.2f\n", otherData["dup"])
	fmt.Fprintf(fs, "Reachability\t%.4f\n", otherData["reach"])
	fmt.Fprintf(fs, "Average Reception Time\t%.4f\n", otherData["rcv"])
	fmt.Fprintf(fs, "Time(sec)\tCount\n")

	for secStr, count := range counter {
		fmt.Fprintf(fs, "%s\t\t%.2f\n", secStr, count)
	}
}
