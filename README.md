# com.arisen.vegetation.generic-renderpipeline

Generic Render Pipeline adapter for `com.arisen.vegetation`.

The package registers one optional feature through the Generic RP feature
registry and caches package-neutral vegetation services during package load.
It does not depend on Vulkan or any concrete RHI backend.

The feature prepares exact residency-held mesh/material/instance resources at
the frame boundary, runs deterministic setup-owned hierarchical culling and LOD
with reusable storage, and contributes direct-indexed instanced opaque and
cascaded-shadow passes. Command recording consumes only prepared buffers,
pipelines, bindings, constants, and draw ranges; it performs no service lookup,
asset discovery, allocation, upload, or pipeline creation. The adapter remains
backend-neutral and keeps device-resource release submission-ticket deferred.
