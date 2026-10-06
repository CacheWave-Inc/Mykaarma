package com.cachewave.inspector

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.display.DisplayManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class InspectActivity : AppCompatActivity() {
    private val labels = arrayOf("Brake inspection", "Battery inspection", "License plate")
    private val subjects = arrayOf("wheel", "battery", "plate")
    private val names = arrayOf("brake assembly", "battery", "license plate")

    private var task = 0
    private var autoMode = false          // started from the home page "Start": the app recognises which item is in view, in any order
    private var cooldownTask = -1         // item just captured: ignore it until the camera has moved to something else
    private var candidate = -1
    private var candidateCount = 0
    private lateinit var titlePill: TextView
    private val chips = arrayOfNulls<TextView>(3)
    private val frames = ArrayDeque<ByteArray>()      // last <=16 JPEG frames, 256x256
    private var lastSampleMs = 0L
    @Volatile private var inFlight = false
    private var readyStreak = 0
    private var missStreak = 0
    private var netFailures = 0
    private var autoBlockedUntil = 0L    // after a rejected photo: pause auto-capture briefly, then try again by itself
    private var holdUntil = 0L            // keep a rejection message on screen for a few seconds
    private var rejects = 0              // consecutive rejected photos; after 3 wait until the view changes (guidance not ready)
    private var isReady = false
    private var isInvalid = false
    @Volatile private var capturing = false
    @Volatile private var probeTop: ByteArray? = null
    @Volatile private var probeBottom: ByteArray? = null

    private val main = Handler(Looper.getMainLooper())
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val netExecutor = Executors.newSingleThreadExecutor()
    private var imageCapture: ImageCapture? = null

    private lateinit var previewView: PreviewView
    private lateinit var overlay: GuideOverlayView
    private lateinit var guidanceText: TextView
    private lateinit var statusPill: TextView
    private lateinit var shutter: FrameLayout
    private lateinit var doneLayer: FrameLayout
    private lateinit var doneLabel: TextView
    private var autoPending = false

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else guidanceText.text = "Camera permission is required"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        task = intent.getIntExtra("task", 0).coerceIn(0, 2)
        autoMode = intent.getBooleanExtra("auto", false)
        collecting = getSharedPreferences("edge", MODE_PRIVATE).getBoolean("collect", false)
        buildUi()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun pill(argb: Int, radiusDp: Int) = GradientDrawable().apply {
        cornerRadius = dp(radiusDp).toFloat(); setColor(argb)
    }

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        rootLayout = root
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))

        overlay = GuideOverlayView(this)
        overlay.configure(task, false, GuideOverlayView.Dir.NONE)
        root.addView(overlay, FrameLayout.LayoutParams(-1, -1))

        val back = TextView(this).apply {
            text = "‹"; setTextColor(Color.WHITE); textSize = 30f; gravity = Gravity.CENTER
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.argb(115, 0, 0, 0)) }
            setOnClickListener { finish() }
        }
        root.addView(back, FrameLayout.LayoutParams(dp(42), dp(42)).apply {
            gravity = Gravity.TOP or Gravity.START; topMargin = dp(40); leftMargin = dp(16)
        })
        titlePill = TextView(this).apply {
            text = if (autoMode) "Show any item" else labels[task]; setTextColor(Color.WHITE); textSize = 13f; setTypeface(typeface, Typeface.BOLD)
            setPadding(dp(16), dp(8), dp(16), dp(8)); background = pill(Color.argb(115, 0, 0, 0), 20)
        }
        root.addView(titlePill, FrameLayout.LayoutParams(-2, -2).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL; topMargin = dp(44) })
        if (autoMode) {
            val chipRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
            val short = arrayOf("Brake", "Battery", "Plate")
            for (i in 0..2) {
                chips[i] = TextView(this).apply {
                    text = short[i]; textSize = 12f; setTypeface(typeface, Typeface.BOLD); setPadding(dp(14), dp(6), dp(14), dp(6))
                }
                chipRow.addView(chips[i], LinearLayout.LayoutParams(-2, -2).apply { leftMargin = dp(4); rightMargin = dp(4) })
            }
            root.addView(chipRow, FrameLayout.LayoutParams(-2, -2).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL; topMargin = dp(92) })
            chipRowView = chipRow
            refreshChips()
        }

        val bottom = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_HORIZONTAL }
        bottomStack = bottom
        statusPill = TextView(this).apply {
            text = "Edge node · connecting…"; setTextColor(Color.parseColor("#C9D6EA")); textSize = 12f
            setPadding(dp(14), dp(6), dp(14), dp(6)); background = pill(Color.argb(115, 0, 0, 0), 20)
        }
        guidanceText = TextView(this).apply {
            text = if (autoMode) "Point the camera at a wheel, battery or license plate" else "Point the camera at the ${subjects[task]}"; setTextColor(Color.WHITE); textSize = 16f
            setTypeface(typeface, Typeface.BOLD); gravity = Gravity.CENTER
            setPadding(dp(18), dp(12), dp(18), dp(12)); background = pill(Color.argb(150, 0, 0, 0), 14)
        }
        shutter = FrameLayout(this)
        shutter.setOnClickListener {
            if (capturing) return@setOnClickListener
            if (isReady) capturePhoto()
            else guidanceText.text = if (isInvalid) "Not valid — no ${names[task]}, nothing captured" else "Not ready — follow the guidance"
        }
        setShutter(false)

        bottom.addView(statusPill, LinearLayout.LayoutParams(-2, -2))
        bottom.addView(guidanceText, LinearLayout.LayoutParams(-2, -2).apply { topMargin = dp(10) })
        bottom.addView(shutter, LinearLayout.LayoutParams(dp(78), dp(78)).apply { topMargin = dp(16); bottomMargin = dp(34) })
        root.addView(bottom, FrameLayout.LayoutParams(-1, -2).apply { gravity = Gravity.BOTTOM })

        root.addView(TextView(this).apply {
            text = "Stop"; setTextColor(Color.WHITE); textSize = 14f; setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER; setPadding(dp(20), dp(10), dp(20), dp(10))
            background = pill(Color.argb(150, 20, 30, 45), 20)
            setOnClickListener { finish() }
        }, FrameLayout.LayoutParams(-2, -2).apply { gravity = Gravity.TOP or Gravity.END; topMargin = dp(40); rightMargin = dp(16) })

        doneLayer = FrameLayout(this).apply { setBackgroundColor(Color.argb(235, 11, 37, 69)); visibility = View.GONE }
        val doneBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER }
        doneBox.addView(TextView(this).apply {
            text = "✓"; setTextColor(Color.WHITE); textSize = 64f; gravity = Gravity.CENTER
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.parseColor("#2CB67D")) }
        }, LinearLayout.LayoutParams(dp(130), dp(130)))
        doneLabel = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 18f; setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER; setPadding(0, dp(18), 0, 0)
        }
        doneBox.addView(doneLabel)
        doneLayer.addView(doneBox, FrameLayout.LayoutParams(-2, -2, Gravity.CENTER))
        root.addView(doneLayer, FrameLayout.LayoutParams(-1, -1))

        setContentView(root)
        applyOrientationLayout()
        (getSystemService(DISPLAY_SERVICE) as DisplayManager).registerDisplayListener(displayListener, main)
    }

    /** Shutter states: grey (not ready / nothing valid in view), green ✓ (valid, ready). */
    private fun setShutter(ready: Boolean, invalid: Boolean = false) {
        isReady = ready; isInvalid = invalid
        shutter.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.parseColor(if (ready) "#2CB67D" else "#3A4656"))
            setStroke(dp(4), if (ready) Color.WHITE else Color.argb(90, 255, 255, 255))
        }
        shutter.removeAllViews()
        if (ready) shutter.addView(TextView(this).apply {
            text = "✓"; setTextColor(Color.WHITE); textSize = 30f; gravity = Gravity.CENTER
        }, FrameLayout.LayoutParams(-1, -1))
    }

    private var previewUse: Preview? = null
    private var collecting = false
    private var lastCollectMs = 0L
    private var analysisUse: ImageAnalysis? = null
    @Volatile private var probeHorizontal = false     // true when the preview is wider than tall: extra squares sit left/right
    private var lastRotation = -1

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) = applyRotation()
    }

    /** The camera follows the display: when the phone is turned (including 180 degrees) tell every use case. */
    @Suppress("DEPRECATION")
    private fun applyRotation() {
        val rot = windowManager.defaultDisplay.rotation
        if (rot == lastRotation) return
        lastRotation = rot
        previewUse?.targetRotation = rot; imageCapture?.targetRotation = rot; analysisUse?.targetRotation = rot
        synchronized(frames) { frames.clear() }      // frames from the other orientation would confuse the model
        probeTop = null; probeBottom = null
    }

    private lateinit var rootLayout: FrameLayout
    private lateinit var bottomStack: LinearLayout
    private var chipRowView: LinearLayout? = null

    /**
     * Portrait: status pill, guidance text and shutter stacked at the bottom, chips in a row at the top.
     * Landscape: the picture is short, so the chips go down the left edge, the status pill to the bottom-left and the shutter to
     * the right edge (like a camera app), leaving only the guidance text at the bottom so nothing covers the guide box.
     */
    private fun applyOrientationLayout() {
        val land = resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        (statusPill.parent as? android.view.ViewGroup)?.removeView(statusPill)
        (shutter.parent as? android.view.ViewGroup)?.removeView(shutter)
        if (land) {
            rootLayout.addView(statusPill, FrameLayout.LayoutParams(-2, -2).apply { gravity = Gravity.BOTTOM or Gravity.START; leftMargin = dp(16); bottomMargin = dp(14) })
            rootLayout.addView(shutter, FrameLayout.LayoutParams(dp(64), dp(64)).apply { gravity = Gravity.END or Gravity.CENTER_VERTICAL; rightMargin = dp(24) })
        } else {
            bottomStack.addView(statusPill, 0, LinearLayout.LayoutParams(-2, -2))
            bottomStack.addView(shutter, LinearLayout.LayoutParams(dp(78), dp(78)).apply { topMargin = dp(16); bottomMargin = dp(34) })
        }
        statusPill.textSize = if (land) 10f else 12f
        (guidanceText.layoutParams as LinearLayout.LayoutParams).apply { topMargin = if (land) 0 else dp(10); bottomMargin = if (land) dp(12) else 0 }
        guidanceText.textSize = if (land) 14f else 16f
        guidanceText.requestLayout()
        chipRowView?.let { row ->
            row.orientation = if (land) LinearLayout.VERTICAL else LinearLayout.HORIZONTAL
            (row.layoutParams as FrameLayout.LayoutParams).apply {
                gravity = if (land) Gravity.TOP or Gravity.START else Gravity.TOP or Gravity.CENTER_HORIZONTAL
                leftMargin = if (land) dp(16) else 0
                topMargin = dp(if (land) 96 else 92)
            }
            for (i in 0 until row.childCount) (row.getChildAt(i).layoutParams as LinearLayout.LayoutParams).apply {
                leftMargin = if (land) 0 else dp(4); rightMargin = if (land) 0 else dp(4); topMargin = if (land) dp(4) else 0
            }
            row.requestLayout()
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        applyOrientationLayout()
        applyRotation()
        overlay.invalidate()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(720, 960))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(cameraExecutor) { sample(it) }
            previewUse = preview; analysisUse = analysis
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture, analysis)
            applyRotation()
            main.postDelayed(scoreTick, 1000)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Rotate, centre-crop to the square the technician sees, shrink to 256x256, JPEG. */
    private fun squareJpeg(src: Bitmap, rotationDegrees: Int): ByteArray = squareJpegs(src, rotationDegrees, false).first

    /** RT-DETR is a detector that benefits from detail, so it gets 512x512 frames; JEPA's input is fixed at 256x256. */
    private fun bigMode() = EdgeDiscovery.engine.id == "rtdetr"
    private fun frameSize() = if (bigMode()) 512 else 256

    private fun jpeg256(b: Bitmap): ByteArray {
        val out = ByteArrayOutputStream()
        val size = frameSize()
        // shrink in halving steps: each step is a 2x2 average. One big shrink of the 8-megapixel still (about 5x) aliases and looks
        // noisy compared with the live frames, which made the photo check disagree with the live check.
        var bmp = b
        while (bmp.width / 2 >= size) bmp = Bitmap.createScaledBitmap(bmp, bmp.width / 2, bmp.height / 2, true)
        if (bmp.width != size) bmp = Bitmap.createScaledBitmap(bmp, size, size, true)
        bmp.compress(Bitmap.CompressFormat.JPEG, if (bigMode()) 85 else 80, out)
        return out.toByteArray()
    }

    /**
     * (centre, top, bottom) 256x256 JPEGs of the square(s) the technician sees. The on-screen preview is a narrow, tall
     * slice of the sensor image, about two squares high: the centre square is what the model scores, top/bottom are
     * only probed to tell the technician which way to tilt when a plate sits outside the centre square.
     */
    private fun squareJpegs(src: Bitmap, rotationDegrees: Int, withProbes: Boolean): Triple<ByteArray, ByteArray?, ByteArray?> {
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        val rw = rotated.width.toFloat(); val rh = rotated.height.toFloat()
        // the on-screen preview shows a centred crop of the sensor image (FILL_CENTER): find that visible part
        val vw = previewView.width.toFloat(); val vh = previewView.height.toFloat()
        val visW: Float; val visH: Float
        if (vw > 0 && vh > 0) {
            val view = vw / vh; val img = rw / rh
            visW = rw * minOf(1f, view / img); visH = rh * minOf(1f, img / view)
        } else { visW = rw; visH = rh }
        val side = minOf(visW, visH).toInt().coerceAtLeast(64)
        val ox = ((rw - visW) / 2).toInt(); val oy = ((rh - visH) / 2).toInt()
        val cx0 = ox + ((visW - side) / 2).toInt(); val cy0 = oy + ((visH - side) / 2).toInt()
        fun crop(x: Int, y: Int) = jpeg256(Bitmap.createBitmap(rotated, x.coerceIn(0, rotated.width - side), y.coerceIn(0, rotated.height - side), side, side))
        val centre = crop(cx0, cy0)
        if (collecting && withProbes) {                      // live frames only (not the captured still)
            val now = SystemClock.elapsedRealtime()
            if (now - lastCollectMs >= 1000) {
                lastCollectMs = now
                val cb = Bitmap.createBitmap(rotated, cx0.coerceIn(0, rotated.width - side), cy0.coerceIn(0, rotated.height - side), side, side)
                saveCollected(Bitmap.createScaledBitmap(cb, 512, 512, true))
            }
        }
        if (!withProbes) return Triple(centre, null, null)
        return when {
            visH - side >= side / 3f -> { probeHorizontal = false; Triple(centre, crop(cx0, oy), crop(cx0, oy + (visH - side).toInt())) }    // portrait: top and bottom
            visW - side >= side / 3f -> { probeHorizontal = true; Triple(centre, crop(ox, cy0), crop(ox + (visW - side).toInt(), cy0)) }      // landscape: left and right
            else -> Triple(centre, null, null)
        }
    }

    /** Phone side of the pipeline: 4 fps, centre-crop, shrink to 256x256, JPEG. */
    private fun sample(image: ImageProxy) {
        try {
            val now = SystemClock.elapsedRealtime()
            if (now - lastSampleMs < 250 || capturing) return
            lastSampleMs = now
            val (jpeg, top, bottom) = squareJpegs(image.toBitmap(), image.imageInfo.rotationDegrees, true)
            probeTop = top; probeBottom = bottom
            synchronized(frames) {
                frames.addLast(jpeg)
                while (frames.size > 16) frames.removeFirst()
            }
        } finally {
            image.close()
        }
    }

    /** Once a second: send the latest frames to the edge node and apply its answer. */
    private val scoreTick = object : Runnable {
        override fun run() {
            main.postDelayed(this, 1000)
            if (capturing || inFlight) return
            val batch = synchronized(frames) { if (bigMode()) frames.toList().takeLast(2) else frames.toList() }
            if (batch.isEmpty()) return
            inFlight = true
            netExecutor.execute {
                val result = try { EdgeClient.score(if (autoMode) 3 else task, batch) } catch (e: Exception) { null }
                // nothing in the centre square: is the target sitting in the part of the preview above or below it?
                var hint: GuideOverlayView.Dir? = null
                var tooClose = false
                if (result?.code == Guidance.NO_PLATE) {
                    fun prob(j: ByteArray?) = if (j == null) 0f else try { EdgeClient.score(if (autoMode) 3 else task, List(if (bigMode()) 1 else 4) { j }).plateProb } catch (e: Exception) { 0f }
                    val up = prob(probeTop); val down = prob(probeBottom)
                    Log.i("CWTiming", "probe up=${"%.2f".format(up)} down=${"%.2f".format(down)}")
                    if (up >= 0.85f && down >= 0.85f) tooClose = true            // seen both above and below the middle square: it fills the frame
                    else if (maxOf(up, down) >= 0.85f) hint = if (probeHorizontal) (if (up >= down) GuideOverlayView.Dir.LEFT else GuideOverlayView.Dir.RIGHT) else (if (up >= down) GuideOverlayView.Dir.UP else GuideOverlayView.Dir.DOWN)
                }
                main.post { inFlight = false; applyGuidance(result, hint, tooClose) }
            }
        }
    }

    private fun applyGuidance(g: Guidance?, hint: GuideOverlayView.Dir? = null, tooClose: Boolean = false) {
        if (capturing) return
        if (SystemClock.elapsedRealtime() < holdUntil) return
        if (g == null) {
            // the node may have moved to another address (DHCP): look for it again after a few failures in a row
            if (++netFailures == 3) netExecutor.execute { EdgeDiscovery.find(applicationContext) }
            readyStreak = 0; setShutter(false)
            overlay.configure(task, false, GuideOverlayView.Dir.NONE)
            statusPill.text = "Edge node · unreachable"
            statusPill.setTextColor(Color.parseColor("#FFD27A"))
            guidanceText.text = "Can't reach the edge node"
            return
        }
        netFailures = 0
        Log.i("CWTiming", "gpu_ms=${g.inferMs.toInt()} round_trip_ms=${g.roundTripMs} frames=${g.framesUsed} code=${g.code} ready=${g.ready} plate=${"%.2f".format(g.plateProb)} w=${"%.2f".format(g.width)} dx=${"%.2f".format(g.dx)} dy=${"%.2f".format(g.dy)}")
        if (autoMode) {
            val d = g.detected
            if (d < 0) {                                   // nothing recognisable in view
                candidate = -1; candidateCount = 0; cooldownTask = -1
                readyStreak = 0; missStreak = 2
                titlePill.text = "Show any item"
                val msg = when {
                    g.code == Guidance.DARK -> "Too dark — find better light"
                    g.code == Guidance.BRIGHT -> "Too bright — reduce glare"
                    tooClose -> "Move back — the item is too close"
                    hint == GuideOverlayView.Dir.UP -> "Something is above — tilt up"
                    hint == GuideOverlayView.Dir.DOWN -> "Something is below — tilt down"
                    hint == GuideOverlayView.Dir.LEFT -> "Something is to the left — move left"
                    hint == GuideOverlayView.Dir.RIGHT -> "Something is to the right — move right"
                    else -> "Point the camera at a wheel, battery or license plate"
                }
                guidanceText.text = msg; guidanceText.setTextColor(Color.WHITE)
                overlay.configure(task, false, if (hint != null) hint else GuideOverlayView.Dir.NONE, false)
                if (isReady || isInvalid) setShutter(false, false)
                return
            }
            // switch to a newly recognised item only after two answers in a row agree
            if (d != task) {
                if (d == candidate) candidateCount++ else { candidate = d; candidateCount = 1 }
                if (candidateCount >= 2) { task = d; candidate = -1; candidateCount = 0; readyStreak = 0; missStreak = 0; isReady = false }
                else { guidanceText.text = "Looking…"; return }
            }
            titlePill.text = labels[task]
            if (cooldownTask >= 0 && d != cooldownTask) cooldownTask = -1
            if (d == cooldownTask) {                      // this item already has its photo
                guidanceText.text = "${labels[d]} done — aim at the next item"; guidanceText.setTextColor(Color.parseColor("#2CB67D"))
                overlay.configure(task, false, GuideOverlayView.Dir.NONE, false)
                if (isReady) setShutter(false, false)
                readyStreak = 0
                return
            }
        }
        statusPill.text = "${EdgeDiscovery.engine.label} · ${if (EdgeDiscovery.engine.id == "rtdetr") "server" else "GPU"} ${g.inferMs.toInt()} ms · round trip ${g.roundTripMs} ms"
        statusPill.setTextColor(Color.parseColor("#C9D6EA"))
        readyStreak = if (g.ready) readyStreak + 1 else 0
        missStreak = if (g.ready) 0 else missStreak + 1
        if (!g.ready && rejects >= 3) { rejects = 0; autoBlockedUntil = 0L }      // the view changed: allow auto-capture again
        // hysteresis: two OKs in a row to turn green, two misses in a row to turn it off again (no flicker on a borderline frame)
        val ready = if (isReady) missStreak < 2 else readyStreak >= 2
        val none = GuideOverlayView.Dir.NONE
        val (text, dir) = when (g.code) {
            Guidance.DARK -> "Too dark — find better light" to none
            Guidance.BRIGHT -> "Too bright — reduce glare" to none
            Guidance.SHAKY -> "Hold the phone steady" to none
            Guidance.BLURRY -> "Image is blurry — let it focus" to none
            Guidance.LEFT -> "Move left" to GuideOverlayView.Dir.LEFT
            Guidance.RIGHT -> "Move right" to GuideOverlayView.Dir.RIGHT
            Guidance.UP -> "Tilt up" to GuideOverlayView.Dir.UP
            Guidance.DOWN -> "Tilt down" to GuideOverlayView.Dir.DOWN
            Guidance.CLOSER -> "Move closer to the ${names[task]}" to none
            Guidance.BACK -> "Move back — ${names[task]} is too close" to none
            Guidance.NO_PLATE -> when (hint) {
                GuideOverlayView.Dir.UP -> "The ${names[task]} is above — tilt up" to GuideOverlayView.Dir.UP
                GuideOverlayView.Dir.DOWN -> "The ${names[task]} is below — tilt down" to GuideOverlayView.Dir.DOWN
                GuideOverlayView.Dir.LEFT -> "The ${names[task]} is to the left — move left" to GuideOverlayView.Dir.LEFT
                GuideOverlayView.Dir.RIGHT -> "The ${names[task]} is to the right — move right" to GuideOverlayView.Dir.RIGHT
                else -> (if (tooClose) "Move back — ${names[task]} is too close" else "Not valid — no ${names[task]} in view") to none
            }
            Guidance.ADJUST -> "Face the ${names[task]} squarely" to none
            Guidance.COVERED -> "Remove the battery cover — terminals must be visible" to none
            Guidance.WARMING -> "Hold still — reading the scene" to none
            Guidance.OK -> (if (ready) "Good framing — hold still, capturing…" else "Almost there — hold steady") to none
            else -> "Edge node error" to none
        }
        guidanceText.text = text
        val invalid = (g.code == Guidance.NO_PLATE && hint == null) || g.code == Guidance.COVERED
        guidanceText.setTextColor(Color.parseColor(if (ready) "#2CB67D" else "#FFFFFF"))
        overlay.configure(task, ready, dir, invalid)
        if (ready != isReady || invalid != isInvalid) setShutter(ready, invalid)
        if (ready && !autoPending) {
            autoPending = true
            main.postDelayed({
                autoPending = false
                val wait = autoBlockedUntil - SystemClock.elapsedRealtime()
                Log.i("CWAuto", "auto-capture check isReady=$isReady capturing=$capturing waitMs=$wait rejects=$rejects")
                if (isReady && !capturing && wait <= 0) capturePhoto()
            }, 250)
        }
    }

    private fun capturePhoto() {
        val capture = imageCapture ?: return
        capturing = true
        capture.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val rotation = image.imageInfo.rotationDegrees
                val raw = try { image.toBitmap() } catch (e: Exception) { null } finally { image.close() }
                if (raw == null) { reject("Capture failed — try again"); return }
                val bmp = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
                // the photo itself must pass the edge node's check, otherwise it is discarded
                guidanceText.text = "Checking the photo…"
                // RT-DETR: re-detecting on the sharp 8-megapixel still gave different answers from the live frames the shot was approved on,
                // so check the newest live frame instead (same pipeline as the live check, taken a fraction of a second before the still).
                val live = if (bigMode()) synchronized(frames) { frames.lastOrNull() } else null
                val jpeg = live ?: squareJpeg(raw, rotation)
                netExecutor.execute {
                    val verdict = try { EdgeClient.score(task, List(if (bigMode()) 1 else 4) { jpeg }, verify = true) } catch (e: Exception) { null }
                    Log.i("CWTiming", "verify code=${verdict?.code} present=${"%.2f".format(verdict?.plateProb ?: 0f)} size=${"%.2f".format(verdict?.width ?: 0f)} dx=${"%.2f".format(verdict?.dx ?: 0f)} dy=${"%.2f".format(verdict?.dy ?: 0f)}")
                    main.post {
                        if (verdict != null && verdict.code == Guidance.OK) accept(bmp)
                        else if (collecting && verdict != null) { saveCollected(bmp, "CacheWaveRejected", "rej_"); reject("Photo not saved: only ${(verdict.plateProb * 100).toInt()}% sure it shows the ${names[task]}. Hold steady, it will try again") }
                        else reject(if (verdict == null) "Can't reach the server to check the photo — nothing saved"
                                    else "Photo not saved: only ${(verdict.plateProb * 100).toInt()}% sure it shows the ${names[task]}. Hold steady, it will try again")
                    }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                reject("Capture failed — try again")
            }
        })
    }

    private fun refreshChips() {
        val short = arrayOf("Brake", "Battery", "Plate")
        for (i in 0..2) chips[i]?.apply {
            val done = Progress.isDone(this@InspectActivity, i)
            text = if (done) "✓ ${short[i]}" else short[i]
            setTextColor(Color.WHITE)
            background = pill(if (done) Color.parseColor("#2CB67D") else Color.argb(120, 0, 0, 0), 16)
        }
    }

    private fun accept(bmp: Bitmap) {
        rejects = 0; autoBlockedUntil = 0L
        save(bmp)
        Progress.markDone(this, task)
        refreshChips()
        doneLabel.text = "${labels[task]} complete"
        doneLayer.visibility = View.VISIBLE
        main.postDelayed({
            val allDone = (0 until labels.size).all { Progress.isDone(this, it) }
            if (autoMode && !allDone) {
                // stay in the camera: the technician can show the next item in any order
                doneLayer.visibility = View.GONE
                cooldownTask = task
                capturing = false; readyStreak = 0; missStreak = 0
                synchronized(frames) { frames.clear() }
                setShutter(false, false)
                guidanceText.text = "${labels[task]} done — aim at the next item"
                guidanceText.setTextColor(Color.parseColor("#2CB67D"))
            } else {
                finish()
            }
        }, 1300)
    }

    private fun reject(message: String) {
        holdUntil = SystemClock.elapsedRealtime() + 4000
        capturing = false; readyStreak = 0; missStreak = 0
        rejects++
        autoBlockedUntil = SystemClock.elapsedRealtime() + (if (rejects >= 3) Long.MAX_VALUE / 4 else 4000L)
        synchronized(frames) { frames.clear() }
        setShutter(false, true)
        overlay.configure(task, false, GuideOverlayView.Dir.NONE, true)
        guidanceText.text = message
        guidanceText.setTextColor(Color.parseColor("#FFFFFF"))
    }

    private fun saveCollected(bmp: Bitmap, folder: String = "CacheWaveCollect", prefix: String = "col_") {
        try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "${prefix}${System.currentTimeMillis()}.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/$folder")
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
            contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        } catch (_: Exception) { }
    }

    private fun save(bmp: Bitmap) {
        try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "inspection_${System.currentTimeMillis()}.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CacheWaveInspector")
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
            contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        } catch (_: Exception) { }
    }

    override fun onDestroy() {
        (getSystemService(DISPLAY_SERVICE) as DisplayManager).unregisterDisplayListener(displayListener)
        main.removeCallbacksAndMessages(null)
        cameraExecutor.shutdown(); netExecutor.shutdown()
        super.onDestroy()
    }
}
