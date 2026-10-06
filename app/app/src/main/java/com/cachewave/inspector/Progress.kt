package com.cachewave.inspector

import android.content.Context

/** Which inspections are finished, kept on the phone so the home screen can show check marks. */
object Progress {
    private const val FILE = "inspector"
    const val TASKS = 3

    private fun prefs(c: Context) = c.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    fun isDone(c: Context, task: Int) = prefs(c).getBoolean("done_$task", false)

    fun markDone(c: Context, task: Int) {
        prefs(c).edit().putBoolean("done_$task", true).apply()
    }

    fun clearAll(c: Context) {
        prefs(c).edit().apply { for (t in 0 until TASKS) remove("done_$t") }.apply()
    }

    fun doneCount(c: Context) = (0 until TASKS).count { isDone(c, it) }
}
