def handle_ipfs_gateway_url(args):
    """Get the gateway URL for an IPFS hash."""
    ipfs_hash = args.get("ipfs_hash")
    if not ipfs_hash:
        return {"error": "Missing required argument: ipfs_hash", "success": False}
    
    return {"gateway_url": f"https://ipfs.io/ipfs/{ipfs_hash}"}
