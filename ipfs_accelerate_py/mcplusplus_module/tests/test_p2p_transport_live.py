"""Live, loopback tests for the MCP++ libp2p transport."""

import trio

from ipfs_accelerate_py.mcplusplus_module.p2p_transport import MCPp2pNode


def test_two_nodes_can_call_a_tool_over_libp2p():
    async def run():
        server = MCPp2pNode(
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
            bootstrap_peers=[],
        )
        client = MCPp2pNode(
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
            bootstrap_peers=[],
        )

        async def handler(method, params):
            assert params.pop("_sender_peer_id")
            return {"method": method, "params": params}

        async with trio.open_nursery() as nursery:
            await server.start(nursery)
            server.set_tool_handler(handler)
            await client.start(nursery)
            await client._connect_bootstrap(server.multiaddrs[0])

            result = await client.call_tool(
                server.peer_id,
                "model_list_served",
                {"endpoint_url": "http://127.0.0.1:8080/v1"},
                timeout=5.0,
                max_retries=1,
            )

            assert result["method"] == "model_list_served"
            assert result["params"]["endpoint_url"].endswith("/v1")
            await client.stop()
            await server.stop()

    trio.run(run)
