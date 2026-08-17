# LPC-160 Documentation and migration

Updated or generated from live campaign declarations:

- `ipfs_datasets_py.logic` public contracts remain the semantic authority.
- Catalog snapshot, orthogonal axes, DomainLogicSlice@2, LogicProviderProtocol@2,
  manifest handshake, and SupervisorLogicPlatformClient are documented in the
  LPC notes directory.
- UI/UX IR is pinned from `origin/agent/ui-ux-ir` @ `9d558ad70` onto the
  campaign datasets gitlink; it is not yet merged to datasets `main`.
- Feature labels: catalog/axes/protocol/client = campaign-stable; UI/UX IR
  package = beta on the campaign pin; declaration-only families remain
  declaration-only.

Do not hardcode test counts, coverage percentages, submillisecond proof
claims, provider availability, or production-readiness.
