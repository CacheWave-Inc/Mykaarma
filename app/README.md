# CacheWave Inspector (Android)

Camera app that guides a technician to photograph a license plate, battery and brake, using guidance returned by an edge node (JEPA on Triton, port 8000, or RT-DETR, port 8100). The two engines speak the same KServe-v2 binary protocol; the app finds the node by mDNS (`_cachewave-triton._tcp`) and falls back to a subnet scan.

Build: Android Studio, or Gradle 8.7 with JDK 17 and the Android SDK (create `local.properties` with `sdk.dir=...`), then `gradle assembleDebug`.

`EDGE_URL` in `app/build.gradle.kts` is only the first-run default address; the app saves the address it finds.
