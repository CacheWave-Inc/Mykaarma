package com.cachewave.inspector

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var summary: TextView
    private lateinit var clearButton: TextView
    private val checks = arrayOfNulls<TextView>(Progress.TASKS)
    private val engineButtons = HashMap<String, TextView>()
    private lateinit var startButton: TextView

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun card(title: String, subtitle: String, task: Int): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(20), dp(16), dp(20))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#13315C")); cornerRadius = dp(16).toFloat()
                setStroke(dp(1), Color.parseColor("#2A4A7A"))
            }
            isClickable = true
            setOnClickListener {
                startActivity(Intent(this@MainActivity, InspectActivity::class.java).putExtra("task", task))
            }
        }
        val texts = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        texts.addView(TextView(this).apply {
            text = title; setTextColor(Color.WHITE); textSize = 18f; setTypeface(typeface, Typeface.BOLD)
        })
        texts.addView(TextView(this).apply {
            text = subtitle; setTextColor(Color.parseColor("#9FB3D1")); textSize = 13.5f
            setPadding(0, dp(4), 0, 0)
        })
        row.addView(texts, LinearLayout.LayoutParams(0, -2, 1f))

        val check = TextView(this).apply {
            gravity = Gravity.CENTER; textSize = 20f; setTypeface(typeface, Typeface.BOLD)
        }
        checks[task] = check
        row.addView(check, LinearLayout.LayoutParams(dp(38), dp(38)).apply { leftMargin = dp(12) })

        row.layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(14) }
        return row
    }

    private fun refreshChecks() {
        for (t in 0 until Progress.TASKS) {
            val done = Progress.isDone(this, t)
            checks[t]?.apply {
                text = if (done) "✓" else ""
                setTextColor(Color.WHITE)
                contentDescription = if (done) "Completed" else "Not done"
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    if (done) setColor(Color.parseColor("#2CB67D"))
                    else { setColor(Color.TRANSPARENT); setStroke(dp(2), Color.parseColor("#5B7BA8")) }
                }
            }
        }
        val n = Progress.doneCount(this)
        summary.text = if (n == Progress.TASKS) "All $n inspections complete" else "$n of ${Progress.TASKS} complete"
        summary.setTextColor(Color.parseColor(if (n == Progress.TASKS) "#2CB67D" else "#9FB3D1"))
        clearButton.alpha = if (n == 0) 0.45f else 1f
        startButton.text = when (n) { 0 -> "Start inspection"; Progress.TASKS -> "Start again"; else -> "Continue inspection" }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(22), dp(48), dp(22), dp(24))
            setBackgroundColor(Color.parseColor("#0B2545"))
        }
        val brand = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        brand.addView(ImageView(this).apply {
            setImageResource(R.drawable.cachewave_logo); adjustViewBounds = true; scaleType = ImageView.ScaleType.FIT_CENTER
            contentDescription = "CacheWave logo"
        }, LinearLayout.LayoutParams(dp(58), dp(40)))
        brand.addView(TextView(this).apply {
            text = "CacheWave Inspector"; setTextColor(Color.parseColor("#FF7A33"))
            textSize = 15f; setTypeface(typeface, Typeface.BOLD); setPadding(dp(10), 0, 0, 0)
        })
        root.addView(brand)
        root.addView(TextView(this).apply {
            text = "What are you inspecting?"; setTextColor(Color.WHITE); textSize = 27f
            setTypeface(typeface, Typeface.BOLD); setPadding(0, dp(8), 0, 0)
        })
        root.addView(TextView(this).apply {
            text = "Pick a task and follow the on-screen guidance until the shutter turns green."
            setTextColor(Color.parseColor("#9FB3D1")); textSize = 14.5f; setPadding(0, dp(8), 0, dp(10))
        })
        startButton = TextView(this).apply {
            textSize = 17f; setTextColor(Color.WHITE); setTypeface(typeface, Typeface.BOLD); gravity = Gravity.CENTER
            setPadding(dp(20), dp(16), dp(20), dp(16)); isClickable = true
            background = GradientDrawable().apply { setColor(Color.parseColor("#FF7A33")); cornerRadius = dp(16).toFloat() }
            setOnClickListener {
                // camera + guidance in any order: the app recognises brake, battery or license plate by itself
                if (Progress.doneCount(this@MainActivity) == Progress.TASKS) Progress.clearAll(this@MainActivity)
                startActivity(Intent(this@MainActivity, InspectActivity::class.java).putExtra("auto", true))
            }
        }
        root.addView(startButton, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(4) })
        root.addView(card("Brake inspection", "Wheel and brake assembly close-up", 0))
        root.addView(card("Battery inspection", "Terminals, case and label", 1))
        root.addView(card("License plate", "Full plate, flat and readable", 2))

        val progressRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(4), dp(20), dp(4), 0)
        }
        summary = TextView(this).apply { textSize = 14f; setTypeface(typeface, Typeface.BOLD) }
        progressRow.addView(summary, LinearLayout.LayoutParams(0, -2, 1f))
        clearButton = TextView(this).apply {
            text = "Clear all"; setTextColor(Color.WHITE); textSize = 14f; setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER; setPadding(dp(20), dp(10), dp(20), dp(10))
            background = GradientDrawable().apply {
                cornerRadius = dp(20).toFloat(); setStroke(dp(1), Color.parseColor("#5B7BA8")); setColor(Color.TRANSPARENT)
            }
            isClickable = true
            setOnClickListener { Progress.clearAll(this@MainActivity); refreshChecks() }
        }
        progressRow.addView(clearButton, LinearLayout.LayoutParams(-2, -2))
        root.addView(progressRow, LinearLayout.LayoutParams(-1, -2))

        root.addView(View(this), LinearLayout.LayoutParams(-1, 0, 1f))
        // which inference engine to use on the edge node: both answer the same requests
        val engineRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
        engineRow.addView(TextView(this).apply {
            text = "Engine"; setTextColor(Color.parseColor("#9FB3D1")); textSize = 12f; setPadding(0, 0, dp(10), 0)
        })
        for (e in EdgeDiscovery.ENGINES) {
            val b = TextView(this).apply {
                text = e.label; textSize = 13f; setTypeface(typeface, Typeface.BOLD); gravity = Gravity.CENTER
                setPadding(dp(18), dp(8), dp(18), dp(8)); isClickable = true
                setOnClickListener {
                    if (EdgeDiscovery.engine.id != e.id) { EdgeDiscovery.setEngine(this@MainActivity, e); refreshEngineButtons(); checkEdge() }
                }
            }
            engineButtons[e.id] = b
            engineRow.addView(b, LinearLayout.LayoutParams(-2, -2).apply { leftMargin = dp(6) })
        }
        root.addView(engineRow, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
        // developer switch: save a 512x512 centre crop about once a second while the camera is open, to build a training set
        val collectSwitch = TextView(this).apply {
            textSize = 12f; gravity = Gravity.CENTER; setPadding(dp(14), dp(8), dp(14), dp(8)); isClickable = true
            fun paint() {
                val on = getSharedPreferences("edge", MODE_PRIVATE).getBoolean("collect", false)
                text = if (on) "Saving frames for training: ON (tap to turn off)" else "Save frames for training: off"
                setTextColor(if (on) Color.parseColor("#FFD27A") else Color.parseColor("#9FB3D1"))
            }
            paint()
            setOnClickListener {
                val p = getSharedPreferences("edge", MODE_PRIVATE)
                p.edit().putBoolean("collect", !p.getBoolean("collect", false)).apply(); paint()
            }
        }
        root.addView(collectSwitch, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(6) })
        status = TextView(this).apply {
            text = "Edge node: checking…"; setTextColor(Color.parseColor("#9FB3D1"))
            textSize = 12f; gravity = Gravity.CENTER
        }
        root.addView(status, LinearLayout.LayoutParams(-1, -2))
        setContentView(android.widget.ScrollView(this).apply { isFillViewport = true; setBackgroundColor(Color.parseColor("#0B2545")); addView(root) })
        refreshChecks()
        refreshEngineButtons()
    }

    private fun refreshEngineButtons() {
        for ((id, b) in engineButtons) {
            val on = id == EdgeDiscovery.engine.id
            b.setTextColor(Color.WHITE)
            b.background = GradientDrawable().apply {
                cornerRadius = dp(18).toFloat()
                if (on) setColor(Color.parseColor("#2C6FBB")) else { setColor(Color.TRANSPARENT); setStroke(dp(1), Color.parseColor("#5B7BA8")) }
            }
            b.alpha = if (on) 1f else 0.7f
        }
    }

    private fun checkEdge() {
        status.text = "Edge node: looking…"; status.setTextColor(Color.parseColor("#9FB3D1"))
        thread {
            val up = EdgeDiscovery.find(this) || EdgeClient.health()
            runOnUiThread {
                status.text = if (up) "${EdgeDiscovery.engine.label} online · ${EdgeDiscovery.baseUrl}\n(${EdgeDiscovery.how})"
                else "${EdgeDiscovery.engine.label} not found on this network"
                status.setTextColor(Color.parseColor(if (up) "#2CB67D" else "#E24B4A"))
            }
        }
    }

    override fun onResume() {
        super.onResume()
        EdgeDiscovery.loadEngine(this)
        refreshEngineButtons()
        refreshChecks()
        checkEdge()
    }
}
