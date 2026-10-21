//go:build js

package router

import "errors"

// In the browser there is no network to dial: Config.Dial must supply the nodes (package sim).
func defaultDial(NodeConfig) (Node, error) {
	return nil, errors.New("no network transport in WebAssembly: set Config.Dial")
}
