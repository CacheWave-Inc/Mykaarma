package com.cachewave.inspector

import android.os.SystemClock
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One answer from the edge node's vjepa_inspect model. */
data class Guidance(
    val code: Int,
    val ready: Boolean,
    val dx: Float,
    val dy: Float,
    val brightness: Float,
    val sharpness: Float,
    val motion: Float,
    val inferMs: Float,
    val framesUsed: Int,
    val roundTripMs: Long,
    val plateProb: Float = 0f,
    val width: Float = 0f,
    val detected: Int = -1,      // auto mode only: which item the server recognised (0 brake, 1 battery, 2 plate), -1 = none
) {
    companion object Codes {
        const val OK = 0; const val DARK = 1; const val BRIGHT = 2; const val BLURRY = 3
        const val SHAKY = 4; const val LEFT = 5; const val RIGHT = 6; const val UP = 7
        const val DOWN = 8; const val CLOSER = 9; const val BACK = 10; const val WARMING = 11
        const val ERROR = 12; const val NO_PLATE = 13; const val ADJUST = 14; const val COVERED = 15
    }
}

/** Talks to Triton over plain HTTP using the KServe v2 binary-tensor extension. */
object EdgeClient {
    private const val OUTPUT_FLOATS = 12

    fun modelFor(task: Int) = EdgeDiscovery.modelName

    fun health(): Boolean = try {
        val c = URL("${EdgeDiscovery.baseUrl}/v2/models/${EdgeDiscovery.modelName}/ready").openConnection() as HttpURLConnection
        c.connectTimeout = 1500; c.readTimeout = 1500
        (c.responseCode == 200).also { c.disconnect() }
    } catch (e: Exception) { false }

    /** [jpegs] are 256x256 JPEG frames, oldest first. [task]: 0 brake, 1 battery, 2 plate, 3 = auto (the server decides which of the three is in view). [verify]: re-check of a captured still, judged more tolerantly. */
    fun score(task: Int, jpegs: List<ByteArray>, verify: Boolean = false): Guidance {
        val payload = ByteArrayOutputStream().apply {
            write(if (verify) task or 0x80 else task); write(jpegs.size)
            for (j in jpegs) {
                write(ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putInt(j.size).array())
                write(j)
            }
        }.toByteArray()
        val header = ("{\"inputs\":[{\"name\":\"FRAMES\",\"shape\":[${payload.size}],\"datatype\":\"UINT8\"," +
            "\"parameters\":{\"binary_data_size\":${payload.size}}}]," +
            "\"outputs\":[{\"name\":\"GUIDANCE\",\"parameters\":{\"binary_data\":true}}]}").toByteArray()

        val t0 = SystemClock.elapsedRealtime()
        val conn = URL("${EdgeDiscovery.baseUrl}/v2/models/${modelFor(task)}/infer").openConnection() as HttpURLConnection
        try {
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.connectTimeout = 2000
            conn.readTimeout = 6000
            conn.setRequestProperty("Inference-Header-Content-Length", header.size.toString())
            conn.setRequestProperty("Content-Type", "application/octet-stream")
            conn.setFixedLengthStreamingMode(header.size + payload.size)
            conn.outputStream.use { it.write(header); it.write(payload) }
            if (conn.responseCode != 200) throw IOException("HTTP ${conn.responseCode}")
            val headerLen = conn.getHeaderField("Inference-Header-Content-Length")?.toIntOrNull()
                ?: throw IOException("missing response header length")
            val body = conn.inputStream.use { it.readBytes() }
            val floats = ByteBuffer.wrap(body, headerLen, body.size - headerLen)
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val v = FloatArray(OUTPUT_FLOATS).also { floats.get(it) }
            return Guidance(
                code = v[0].toInt(), ready = v[1] > 0.5f, dx = v[2], dy = v[3],
                brightness = v[5], sharpness = v[6], motion = v[7],
                inferMs = v[10], framesUsed = v[11].toInt(),
                roundTripMs = SystemClock.elapsedRealtime() - t0,
                plateProb = v[8], width = v[4], detected = if (task == 3) v[9].toInt() else -1,
            )
        } finally {
            conn.disconnect()
        }
    }
}
