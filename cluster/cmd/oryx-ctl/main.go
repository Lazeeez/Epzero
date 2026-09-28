// oryx-ctl: operate an Oryx cluster through its router.
//
//	oryx-ctl load   -router URL -token-file F -data web/public/data/openimages   # int8 CLIP vectors
//	oryx-ctl crash  -router URL -token-file F -node node2
//	oryx-ctl status -router URL
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: oryx-ctl load|crash|status [flags]")
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	routerURL := fs.String("router", "http://localhost:8080", "router base URL")
	tokenFile := fs.String("token-file", "", "file holding the bearer token")
	data := fs.String("data", "web/public/data/openimages", "dataset directory (manifest.json, vectors.i8, scales.f32)")
	batch := fs.Int("batch", 500, "vectors per upsert request")
	nodeName := fs.String("node", "", "node to crash")
	fs.Parse(args)

	token := ""
	if *tokenFile != "" {
		b, err := os.ReadFile(*tokenFile)
		check(err)
		token = strings.TrimSpace(string(b))
	}
	switch cmd {
	case "load":
		load(*routerURL, token, *data, *batch)
	case "crash":
		post(*routerURL+"/v1/chaos/crash", token, map[string]string{"node": *nodeName}, os.Stdout)
	case "status":
		resp, err := http.Get(*routerURL + "/v1/cluster")
		check(err)
		defer resp.Body.Close()
		io.Copy(os.Stdout, resp.Body)
	default:
		fmt.Fprintln(os.Stderr, "unknown command", cmd)
		os.Exit(2)
	}
}

// load dequantizes the exported int8 vectors and upserts them in batches; labels are row numbers,
// matching the explorer's metadata.
func load(routerURL, token, dir string, batch int) {
	var manifest struct {
		Count int `json:"count"`
		Dim   int `json:"dim"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	check(err)
	check(json.Unmarshal(raw, &manifest))
	codes, err := os.ReadFile(filepath.Join(dir, "vectors.i8"))
	check(err)
	scaleBytes, err := os.ReadFile(filepath.Join(dir, "scales.f32"))
	check(err)
	scales := make([]float32, manifest.Dim)
	check(binary.Read(bytes.NewReader(scaleBytes), binary.LittleEndian, scales))
	if len(codes) != manifest.Count*manifest.Dim {
		check(fmt.Errorf("vectors.i8 has %d bytes, expected %d", len(codes), manifest.Count*manifest.Dim))
	}

	start := time.Now()
	for lo := 0; lo < manifest.Count; lo += batch {
		hi := min(lo+batch, manifest.Count)
		body := struct {
			Labels  []uint64    `json:"labels"`
			Vectors [][]float32 `json:"vectors"`
		}{}
		for i := lo; i < hi; i++ {
			v := make([]float32, manifest.Dim)
			for d := range v {
				v[d] = float32(int8(codes[i*manifest.Dim+d])) * scales[d] / 127
			}
			normalize(v)
			body.Labels = append(body.Labels, uint64(i))
			body.Vectors = append(body.Vectors, v)
		}
		var out bytes.Buffer
		post(routerURL+"/v1/upsert", token, body, &out)
		fmt.Printf("\rloaded %d/%d (%.0f/s)", hi, manifest.Count, float64(hi)/time.Since(start).Seconds())
	}
	fmt.Println()
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum > 0 {
		inv := float32(1 / math.Sqrt(sum))
		for i := range v {
			v[i] *= inv
		}
	}
}

func post(url, token string, body any, out io.Writer) {
	payload, err := json.Marshal(body)
	check(err)
	for attempt := 0; ; attempt++ {
		req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := http.DefaultClient.Do(req)
		if err == nil && resp.StatusCode < 300 {
			io.Copy(out, resp.Body)
			resp.Body.Close()
			return
		}
		msg := ""
		if err != nil {
			msg = err.Error()
		} else {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			msg = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
		}
		if attempt >= 4 {
			check(fmt.Errorf("%s: %s", url, msg))
		}
		time.Sleep(time.Duration(attempt+1) * time.Second)
	}
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "oryx-ctl:", err)
		os.Exit(1)
	}
}
