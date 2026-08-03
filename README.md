# com.arisen.vegetation.generic-renderpipeline

Generic Render Pipeline adapter for `com.arisen.vegetation`.

The package registers one optional feature through the Generic RP feature
registry and caches package-neutral vegetation services during package load.
It does not depend on Vulkan or any concrete RHI backend.

The initial feature contributes no draw passes while vegetation assets and
cooked cluster pages are being defined. Its stable lifecycle already exercises
registration, activation, device-resource release, and reverse-order teardown.
