package com.cachewave.inspector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View

/** Dashed framing guide for the chosen task, plus a direction arrow. */
class GuideOverlayView(context: Context) : View(context) {
    enum class Dir { NONE, UP, DOWN, LEFT, RIGHT }

    private val d = resources.displayMetrics.density
    private var task = 0
    private var ready = false
    private var dir = Dir.NONE
    private var invalid = false

    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 3 * d
    }
    private val arrowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 7 * d; color = Color.WHITE
        strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
        setShadowLayer(8 * d, 0f, 2 * d, Color.argb(160, 0, 0, 0))
    }

    init { setLayerType(LAYER_TYPE_SOFTWARE, null) }

    fun configure(task: Int, ready: Boolean, dir: Dir, invalid: Boolean = false) {
        this.task = task; this.ready = ready; this.dir = dir; this.invalid = invalid
        invalidate()
    }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        val cx = w / 2; val cy = h / 2
        val side = minOf(w, h)                                   // the analysed square: as wide as the narrower screen side
        val rect = when (task) {
            0 -> { val r = side * 0.31f; RectF(cx - r, cy - r, cx + r, cy + r) }
            1 -> { val hw = side * 0.36f; val hh = hw * 0.64f; RectF(cx - hw, cy - hh, cx + hw, cy + hh) }
            else -> { val hw = side * 0.29f; val hh = hw * 0.5f; RectF(cx - hw, cy - hh, cx + hw, cy + hh) }
        }
        // corner marks: the square region the edge node analyses
        val left = cx - side / 2; val top = cy - side / 2; val arm = 28 * d
        val frame = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2 * d; color = Color.argb(110, 255, 255, 255) }
        for ((fx, fy) in listOf(left to top, left + side to top, left to top + side, left + side to top + side)) {
            val sx = if (fx == left) 1 else -1; val sy = if (fy == top) 1 else -1
            c.drawLine(fx, fy, fx + sx * arm, fy, frame); c.drawLine(fx, fy, fx, fy + sy * arm, frame)
        }
        guidePaint.color = if (ready) Color.parseColor("#2CB67D") else Color.argb(150, 255, 255, 255)
        guidePaint.pathEffect = if (ready) null else DashPathEffect(floatArrayOf(18 * d, 12 * d), 0f)
        if (task == 0) c.drawOval(rect, guidePaint) else c.drawRoundRect(rect, 14 * d, 14 * d, guidePaint)

        if (dir != Dir.NONE) {
            val s = 30 * d
            val p = Path()
            when (dir) {
                Dir.UP -> { p.moveTo(cx - s, cy + s / 2); p.lineTo(cx, cy - s / 2); p.lineTo(cx + s, cy + s / 2) }
                Dir.DOWN -> { p.moveTo(cx - s, cy - s / 2); p.lineTo(cx, cy + s / 2); p.lineTo(cx + s, cy - s / 2) }
                Dir.LEFT -> { p.moveTo(cx + s / 2, cy - s); p.lineTo(cx - s / 2, cy); p.lineTo(cx + s / 2, cy + s) }
                else -> { p.moveTo(cx - s / 2, cy - s); p.lineTo(cx + s / 2, cy); p.lineTo(cx - s / 2, cy + s) }
            }
            c.drawPath(p, arrowPaint)
        }
    }
}
