import argparse
import sys
import multiaddr
import trio
import libp2p
from libp2p import (
    new_host,
)
from libp2p.network.stream.net_stream_interface import (
    INetStream,
)
from libp2p.peer.peerinfo import (
    info_from_p2p_addr,
)
from libp2p.typing import (
    TProtocol,
)

class libp2p_kit():
    def __init__(self,  resources=None, metadata=None):
        self.PROTOCOL_ID = TProtocol("/chat/1.0.0")
        self.MAX_READ_LEN = 2**32 - 1


    async def read_data(self, stream: INetStream) -> None:
        while True:
            read_bytes = await stream.read(self.MAX_READ_LEN)
            if read_bytes is not None:
                read_string = read_bytes.decode()
                if read_string != "\n":
                    # Green console colour: 	\x1b[32m
                    # Reset console colour: 	\x1b[0m
                    print("\x1b[32m %s\x1b[0m " % read_string, end="")


    async def write_data(self, stream: INetStream) -> None:
        async_f = trio.wrap_file(sys.stdin)
        while True:
            line = await async_f.readline()
            await stream.write(line.encode())

    async def run(self, port: int, destination: str) -> None:
        localhost_ip = "127.0.0.1"
        listen_addr = multiaddr.Multiaddr(f"/ip4/0.0.0.0/tcp/{port}")
        host = new_host()
        async with host.run(listen_addrs=[listen_addr]), trio.open_nursery() as nursery:
            if not destination:  # its the server

                async def stream_handler(stream: INetStream) -> None:
                    nursery.start_soon(self.read_data, stream)
                    nursery.start_soon(self.write_data, stream)

                host.set_stream_handler(self.PROTOCOL_ID, stream_handler)

                print(
                    "Run this from the same folder in another console:\n\n"
                    f"python chat.py -p {int(port) + 1} "
                    f"-d /ip4/{localhost_ip}/tcp/{port}/p2p/{host.get_id().pretty()}\n"
                )
                print("Waiting for incoming connection...")

            else:  # its the client
                maddr = multiaddr.Multiaddr(destination)
                info = info_from_p2p_addr(maddr)
                # Associate the peer with local ip address
                await host.connect(info)
                # Start a stream with the destination.
                # Multiaddress of the destination peer is fetched from the peerstore
                # using 'peerId'.
                stream = await host.new_stream(info.peer_id, [self.PROTOCOL_ID])

                nursery.start_soon(self.read_data, stream)
                nursery.start_soon(self.write_data, stream)
                print(f"Connected to peer {info.addrs[0]}")

            await trio.sleep_forever()


    def main(self, source_port, maddr, destination, **kwargs):
        args = {}
        if source_port != None:
            args.port = source_port
        else:
            args.port = 8000
        
        if maddr != None:
            args.destination = maddr
        else:
            args.destination = None
        
        if destination != None:
            args.destination = destination
        else:
            args.destination = None

        if not args.port:
            raise RuntimeError("was not able to determine a local port")

        try:
            trio.run(self.run, *(args.port, args.destination))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    test_libp2p_kit = libp2p_kit()
    test_libp2p_kit.main()
