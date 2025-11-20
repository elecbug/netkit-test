package main

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/elecbug/netkit/graph/standard_graph"
	"github.com/elecbug/netkit/p2p"
)

func main() {
	sg := standard_graph.NewStandardGraph()
	sg.SetSeedRandom()

	mu := sync.Mutex{}
	couter := make(map[string]float64)
	try := 1000

	wg := sync.WaitGroup{}
	wg.Add(try)
	for i := 0; i < try; i++ {
		go func() {
			defer wg.Done()

			println("Round:", i)

			er := sg.ErdosRenyiGraph(1000, 6.0/1000.0, true)

			p, err := p2p.GenerateNetwork(
				er,
				func() float64 {
					return float64(p2p.NormalRand(100, 50, rand.NewSource(time.Now().UnixNano())))
					// return float64(p2p.UniformRand(10, 500, rand.NewSource(time.Now().UnixNano())))
				},
				func() float64 {
					return float64(p2p.LogNormalRand(500, 0.1, rand.NewSource(time.Now().UnixNano())))
				},
				&p2p.Config{},
			)

			if err != nil {
				panic(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			p.RunNetworkSimulation(ctx)

			err = p.Publish(p.PeerIDs()[0], "Hello, world!", p2p.Flooding)

			if err != nil {
				panic(err)
			}

			time.Sleep(10 * time.Second)
			cancel()

			ts := p.FirstMessageReceptionTimes("Hello, world!")
			sort.Slice(ts, func(i, j int) bool {
				return ts[i].Before(ts[j])
			})

			// for peerID, t := range ts {
			// 	println(peerID, t.String())
			// }

			// println("test")

			for _, t := range ts {
				sec := float32(t.Sub(ts[0]).Seconds())
				secStr := fmt.Sprintf("%.3f", sec)

				// println(secStr)
				mu.Lock()
				couter[secStr] += 1.0 / float64(try)
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	fmt.Println("Time(sec)\tCount")
	for secStr, count := range couter {
		fmt.Printf("%s\t\t%.2f\n", secStr, count)
	}
}
