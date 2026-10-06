package com.cachewave.inspector

import android.content.Context
import android.net.ConnectivityManager
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import java.net.HttpURLConnection
import java.net.Inet4Address
import java.net.InetSocketAddress
import java.net.Socket
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/** One inference engine the phone can talk to. Both speak the same KServe-v2 request/response format. */
data class Engine(val id: String, val label: String, val port: Int, val model: String)

/**
 * Finds the edge node on the local network, for the engine the user picked (JEPA on Triton, or RT-DETR).
 * Order: (1) last address that worked for that engine, (2) mDNS service "_cachewave-triton._tcp" whose TXT record says
 *        engine=<id>, (3) scan of the phone's own subnet on the engine's port, (4) the address compiled into the app.
 * Every candidate must answer the readiness check for the engine's model before it is used.
 */
object EdgeDiscovery {
    private const val SERVICE_TYPE = "_cachewave-triton._tcp."

    val ENGINES = listOf(
        Engine("jepa", "JEPA", 8000, "vjepa_inspect"),
        Engine("rtdetr", "RT-DETR", 8100, "rtdetr_inspect"),
    )

    @Volatile var engine: Engine = ENGINES[0]
    @Volatile var baseUrl: String = BuildConfig.EDGE_URL
    @Volatile var how: String = "built-in default"
    private val running = AtomicBoolean(false)

    val modelName: String get() = engine.model

    private fun prefs(ctx: Context) = ctx.applicationContext.getSharedPreferences("edge", Context.MODE_PRIVATE)

    fun loadEngine(ctx: Context) {
        val id = prefs(ctx).getString("engine", "jepa")
        engine = ENGINES.firstOrNull { it.id == id } ?: ENGINES[0]
    }

    fun setEngine(ctx: Context, e: Engine) {
        prefs(ctx).edit().putString("engine", e.id).apply()
        engine = e
        baseUrl = prefs(ctx).getString("url_${e.id}", null) ?: BuildConfig.EDGE_URL.substringBeforeLast(":") + ":${e.port}"
    }

    fun healthy(url: String, model: String = engine.model, timeoutMs: Int = 1200): Boolean = try {
        val c = URL("$url/v2/models/$model/ready").openConnection() as HttpURLConnection
        c.connectTimeout = timeoutMs; c.readTimeout = timeoutMs
        (c.responseCode == 200).also { c.disconnect() }
    } catch (e: Exception) { false }

    private fun use(ctx: Context, url: String, source: String): Boolean {
        baseUrl = url; how = source
        prefs(ctx).edit().putString("url_${engine.id}", url).apply()
        return true
    }

    /** Blocking, so call it off the main thread. Returns false if the chosen engine answered nowhere. */
    fun find(ctx: Context): Boolean {
        if (!running.compareAndSet(false, true)) return false
        try {
            val app = ctx.applicationContext
            val e = engine
            val t0 = System.currentTimeMillis()
            fun ms() = System.currentTimeMillis() - t0
            val saved = prefs(app).getString("url_${e.id}", null)
            if (saved != null && healthy(saved, e.model)) { Log.i("EdgeDiscovery", "[${e.id}] saved address ok $saved (${ms()} ms)"); return use(app, saved, "saved address") }
            Log.i("EdgeDiscovery", "[${e.id}] saved address $saved not answering (${ms()} ms); trying mDNS")
            val m = discoverMdns(app, e, 4000)
            Log.i("EdgeDiscovery", "[${e.id}] mDNS result=$m (${ms()} ms)")
            m?.let { return use(app, it, "auto-discovered (mDNS)") }
            val sc = scanSubnet(app, e)
            Log.i("EdgeDiscovery", "[${e.id}] scan result=$sc (${ms()} ms)")
            sc?.let { return use(app, it, "found by network scan") }
            val def = BuildConfig.EDGE_URL.substringBeforeLast(":") + ":${e.port}"
            if (healthy(def, e.model)) return use(app, def, "built-in default")
            return false
        } finally {
            running.set(false)
        }
    }

    @Suppress("DEPRECATION")
    private fun discoverMdns(ctx: Context, e: Engine, timeoutMs: Long): String? {
        val nsd = ctx.getSystemService(Context.NSD_SERVICE) as? NsdManager ?: return null
        val result = arrayOfNulls<String>(1)
        val done = CountDownLatch(1)
        val queue = java.util.concurrent.LinkedBlockingQueue<NsdServiceInfo>()
        val resolving = AtomicBoolean(false)

        fun resolveNext() {
            if (result[0] != null || queue.isEmpty() || !resolving.compareAndSet(false, true)) return
            val info = queue.poll() ?: run { resolving.set(false); return }
            nsd.resolveService(info, object : NsdManager.ResolveListener {
                override fun onResolveFailed(si: NsdServiceInfo?, errorCode: Int) {
                    Log.i("EdgeDiscovery", "NSD resolve failed $errorCode"); resolving.set(false); resolveNext()
                }

                override fun onServiceResolved(si: NsdServiceInfo) {
                    val eng = si.attributes["engine"]?.toString(Charsets.UTF_8)
                    Log.i("EdgeDiscovery", "NSD resolved ${si.serviceName} -> ${si.host} port ${si.port} engine=$eng")
                    val hosts = if (android.os.Build.VERSION.SDK_INT >= 34) si.hostAddresses else listOf(si.host)
                    val host = hosts.firstOrNull { it is Inet4Address }
                    if (host != null && (eng == null || eng == e.id)) {
                        val url = "http://${host.hostAddress}:${if (si.port > 0) si.port else e.port}"
                        if (healthy(url, e.model)) { result[0] = url; done.countDown() }
                    }
                    resolving.set(false); resolveNext()
                }
            })
        }

        val discovery = object : NsdManager.DiscoveryListener {
            override fun onStartDiscoveryFailed(serviceType: String?, errorCode: Int) { Log.i("EdgeDiscovery", "NSD start failed $errorCode"); done.countDown() }
            override fun onStopDiscoveryFailed(serviceType: String?, errorCode: Int) {}
            override fun onDiscoveryStarted(serviceType: String?) {}
            override fun onDiscoveryStopped(serviceType: String?) {}
            override fun onServiceLost(info: NsdServiceInfo?) {}
            override fun onServiceFound(info: NsdServiceInfo) {
                Log.i("EdgeDiscovery", "NSD found ${info.serviceName}")
                queue.add(info)          // NSD resolves one service at a time
                resolveNext()
            }
        }
        try {
            nsd.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, discovery)
            done.await(timeoutMs, TimeUnit.MILLISECONDS)
        } catch (ex: Exception) {
        } finally {
            try { nsd.stopServiceDiscovery(discovery) } catch (ex: Exception) {}
        }
        return result[0]
    }

    /** For networks that block multicast: try the engine's port on every host of the phone's subnet (capped at 2046 hosts). */
    private fun scanSubnet(ctx: Context, e: Engine): String? {
        val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return null
        val lp = cm.getLinkProperties(cm.activeNetwork ?: return null) ?: return null
        val la = lp.linkAddresses.firstOrNull { it.address is Inet4Address } ?: return null
        val me = (la.address as Inet4Address).address.fold(0L) { acc, b -> (acc shl 8) or (b.toLong() and 0xff) }
        val prefix = maxOf(la.prefixLength, 21)
        val mask = (0xFFFFFFFFL shl (32 - prefix)) and 0xFFFFFFFFL
        val first = (me and mask) + 1
        val last = (me and mask) + (1L shl (32 - prefix)) - 2
        val pool = Executors.newFixedThreadPool(64)
        val found = java.util.concurrent.ConcurrentLinkedQueue<String>()
        try {
            val tasks = (first..last).filter { it != me }.map { ip ->
                pool.submit {
                    val addr = "${(ip shr 24) and 255}.${(ip shr 16) and 255}.${(ip shr 8) and 255}.${ip and 255}"
                    try {
                        Socket().use { s -> s.connect(InetSocketAddress(addr, e.port), 250) }
                        if (healthy("http://$addr:${e.port}", e.model, 800)) found.add("http://$addr:${e.port}")
                    } catch (ex: Exception) {}
                }
            }
            val deadline = System.currentTimeMillis() + 9000
            for (t in tasks) {
                val left = deadline - System.currentTimeMillis()
                if (left <= 0 || found.isNotEmpty()) break
                try { t.get(left, TimeUnit.MILLISECONDS) } catch (ex: Exception) {}
            }
        } finally {
            pool.shutdownNow()
        }
        return found.peek()
    }
}
