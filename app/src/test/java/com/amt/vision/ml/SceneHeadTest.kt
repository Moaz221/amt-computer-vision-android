package com.amt.vision.ml

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import kotlin.math.abs

/** Expected numbers come from tools/train_scene_head.py's numpy maths on the same input. */
class SceneHeadTest {
    @Test
    fun matchesNumpyReference() {
        val head = SceneHead.fromJson(File("src/main/assets/scene_head.json").readText())
        val classes = File("src/main/assets/classes.txt").readLines().map { it.trim() }.filter { it.isNotEmpty() }
        val p = FloatArray(1000)
        p[classes.indexOf("alp")] = 0.6f
        p[classes.indexOf("seashore")] = 0.2f
        p[classes.indexOf("church")] = 0.1f
        val out = head.predict(p)
        val expected = floatArrayOf(0.0437f, 0.0036f, 0.6209f, 0.2237f, 0.0807f, 0.0274f)
        for (i in expected.indices) assertEquals(expected[i].toDouble(), out[i].toDouble(), 2e-3)
        assertEquals(1.0, out.sum().toDouble(), 1e-4)
    }
}
