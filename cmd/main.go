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
	wg := sync.WaitGroup{}
	wg.Add(2)
	defer wg.Wait()

	go func() {
		defer wg.Done()
		hopwaveLoop()
	}()

	go func() {
		defer wg.Done()
		floodingLoop()
	}()

	wg.Wait()
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

func hopwaveLoop() {
	sg := standard_graph.NewStandardGraph()
	sg.SetSeedRandom()

	mu := sync.Mutex{}
	counter := make(map[string]float64)
	try := 1000
	networkSize := 1000

	wg := sync.WaitGroup{}
	wg.Add(try)
	for i := 0; i < try; i++ {
		go func() {
			defer wg.Done()

			println("Round:", i)

			er := sg.ErdosRenyiGraph(networkSize, 6.0/float64(networkSize), true)

			p, err := p2p.GenerateNetwork(
				er,
				func() float64 {
					return float64(p2p.NormalRandom(1000, 500))
				},
				func() float64 {
					return float64(p2p.ParetoRandom(500, 2.0))
					// return float64(p2p.ExponentialRandom(0.002))
				},
				&p2p.Config{
					CustomParams: map[string]any{
						"hopwave_interval": 2,
						"hopwave_count":    3,
					},
				},
			)

			if err != nil {
				panic(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			p.RunNetworkSimulation(ctx)

			err = p.Publish(p.PeerIDs()[0], "Hello, world!", p2p.Custom, customHopWaveProtocol)

			if err != nil {
				panic(err)
			}

			time.Sleep(20 * time.Second)
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
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	fs, err := os.Create("temp/hopwave_result.log")
	if err != nil {
		panic(err)
	}
	defer fs.Close()

	fmt.Fprintf(fs, "Time(sec)\tCount\n")

	for secStr, count := range counter {
		fmt.Fprintf(fs, "%s\t\t%.2f\n", secStr, count)
	}
}

func floodingLoop() {
	sg := standard_graph.NewStandardGraph()
	sg.SetSeedRandom()

	mu := sync.Mutex{}
	counter := make(map[string]float64)
	try := 1000
	networkSize := 1000

	wg := sync.WaitGroup{}
	wg.Add(try)
	for i := 0; i < try; i++ {
		go func() {
			defer wg.Done()

			println("Round:", i)

			er := sg.ErdosRenyiGraph(networkSize, 6.0/float64(networkSize), true)

			p, err := p2p.GenerateNetwork(
				er,
				func() float64 {
					return float64(p2p.NormalRandom(1000, 500))
				},
				func() float64 {
					return float64(p2p.ParetoRandom(500, 2.0))
					// return float64(p2p.ExponentialRandom(0.002))
				},
				&p2p.Config{},
			)

			if err != nil {
				panic(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			p.RunNetworkSimulation(ctx)

			err = p.Publish(p.PeerIDs()[0], "Hello, world!", p2p.Flooding, nil)

			if err != nil {
				panic(err)
			}

			time.Sleep(20 * time.Second)
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
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	fs, err := os.Create("temp/flooding_result.log")
	if err != nil {
		panic(err)
	}
	defer fs.Close()

	fmt.Fprintf(fs, "Time(sec)\tCount\n")

	for secStr, count := range counter {
		fmt.Fprintf(fs, "%s\t\t%.2f\n", secStr, count)
	}
}
