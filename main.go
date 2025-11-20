package main

import (
	"context"
	"fmt"
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
	counter := make(map[string]float64)
	try := 2000
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

			err = p.Publish(p.PeerIDs()[0], "Hello, world!", p2p.Flooding)

			if err != nil {
				panic(err)
			}

			time.Sleep(20 * time.Second)
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
				counter[secStr] += 1.0 / float64(try)
				mu.Unlock()
			}

			// if i == 0 {
			// 	println("Sample Data")

			// 	for j := 0; j < int(len(p.PeerIDs())); j++ {

			// 		info, err := p.MessageInfo(p.PeerIDs()[j], "Hello, world!")

			// 		if err != nil {
			// 			panic(err)
			// 		}

			// 		fmt.Printf("%v\n", info)
			// 	}
			// }
		}()
	}

	wg.Wait()

	fmt.Println("Time(sec)\tCount")
	for secStr, count := range counter {
		fmt.Printf("%s\t\t%.2f\n", secStr, count)
	}
}
